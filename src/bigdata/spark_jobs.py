"""
PySpark pipeline for financial data analytics.

Demonstrates distributed data processing using Spark DataFrames:
  - select, filter, groupBy, agg, join, withColumn, orderBy, window functions

All operations are annotated with comments explaining their distributed semantics.

Usage:
    from src.bigdata.spark_jobs import SparkPipeline
    pipeline = SparkPipeline(cfg)
    pipeline.run()
"""

import os
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

# Ensure Spark workers and driver use the active Python executable on Windows
if "PYSPARK_PYTHON" not in os.environ:
    os.environ["PYSPARK_PYTHON"] = sys.executable
if "PYSPARK_DRIVER_PYTHON" not in os.environ:
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

# Auto-configure HADOOP_HOME if bundled in repository for Windows compatibility
if "HADOOP_HOME" not in os.environ:
    hadoop_dir = ROOT / "hadoop"
    if hadoop_dir.exists():
        os.environ["HADOOP_HOME"] = str(hadoop_dir)
        bin_dir = str(hadoop_dir / "bin")
        if bin_dir not in os.environ.get("PATH", ""):
            os.environ["PATH"] = f"{bin_dir};{os.environ.get('PATH', '')}"

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DateType,
    DoubleType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
)

from src.utils.logger import logger


# ── Schema for market_features.csv ───────────────────────────
MARKET_SCHEMA = StructType([
    StructField("date", StringType(), True),
    StructField("symbol", StringType(), True),
    StructField("open", DoubleType(), True),
    StructField("high", DoubleType(), True),
    StructField("low", DoubleType(), True),
    StructField("close", DoubleType(), True),
    StructField("volume", LongType(), True),
    StructField("outlier_flag", StringType(), True),
    StructField("daily_return", DoubleType(), True),
    StructField("log_return", DoubleType(), True),
    StructField("ma_7", DoubleType(), True),
    StructField("ma_20", DoubleType(), True),
    StructField("ma_50", DoubleType(), True),
    StructField("volatility", DoubleType(), True),
    StructField("volume_change", DoubleType(), True),
    StructField("rsi_14", DoubleType(), True),
    StructField("macd", DoubleType(), True),
    StructField("macd_signal", DoubleType(), True),
])


class SparkPipeline:
    """
    PySpark pipeline for financial market analytics.

    Reads processed market features CSV, performs distributed aggregations,
    and saves results as Parquet files.
    """

    def __init__(self, cfg: dict, spark: Optional[SparkSession] = None):
        """
        Parameters
        ----------
        cfg : dict
            Full config dict (from config.yaml).
        spark : SparkSession, optional
            Existing SparkSession (useful for testing). If None, one is created.
        """
        self.cfg = cfg
        spark_cfg = cfg.get("spark", {})
        self.app_name = spark_cfg.get("app_name", "FinancialAnalytics")
        self.master = spark_cfg.get("master", "local[*]")
        self.output_path = Path(ROOT / spark_cfg.get("output_path", "data/processed/spark"))

        if spark is not None:
            self.spark = spark
            self._owns_spark = False
        else:
            self.spark = self._create_session()
            self._owns_spark = True

    def _create_session(self) -> SparkSession:
        """Create a SparkSession configured for local execution."""
        logger.info(f"Creating SparkSession: app={self.app_name}, master={self.master}")

        # getOrCreate() avoids conflicts if a session already exists in this JVM
        session = (
            SparkSession.builder
            .appName(self.app_name)
            .master(self.master)
            .config("spark.sql.parquet.compression.codec", "snappy")
            .config("spark.driver.memory", "2g")
            .config("spark.ui.showConsoleProgress", "false")
            .config("spark.sql.session.timeZone", "Asia/Kolkata")
            .getOrCreate()
        )

        # Reduce Spark's verbose logging
        session.sparkContext.setLogLevel("WARN")
        logger.info("SparkSession created successfully")
        return session

    # ─────────────────────────────────────────────────────────
    # Data Loading
    # ─────────────────────────────────────────────────────────

    def load_market_data(self, filepath: Optional[str] = None) -> DataFrame:
        """
        Load market features CSV into a Spark DataFrame.

        DISTRIBUTED: In cluster mode, Spark partitions the CSV across executors.
        Each executor reads a portion of the file in parallel.
        """
        if filepath is None:
            filepath = str(ROOT / self.cfg["paths"]["processed_market"])

        logger.info(f"Loading market data into Spark: {filepath}")

        # READ operation — distributed across partitions
        df = (
            self.spark.read
            .option("header", "true")
            .option("inferSchema", "false")
            .schema(MARKET_SCHEMA)
            .csv(filepath)
        )

        # withColumn — adds/transforms a column (distributed per-partition)
        # Convert date string to proper DateType for downstream operations
        df = df.withColumn("date", F.to_date(F.col("date")))

        row_count = df.count()  # ACTION: triggers full scan across all partitions
        logger.info(f"Loaded {row_count:,} rows into Spark DataFrame")

        return df

    # ─────────────────────────────────────────────────────────
    # Data Quality Checks
    # ─────────────────────────────────────────────────────────

    def run_quality_checks(self, df: DataFrame) -> dict:
        """
        Perform data quality checks on the Spark DataFrame.

        DISTRIBUTED: Each check (count, filter, agg) runs across all partitions.
        Results are collected to the driver for reporting.
        """
        logger.info("Running data quality checks...")
        checks = {}

        # Total row count — ACTION across all partitions
        checks["total_rows"] = df.count()

        # Null counts per column — agg() is distributed, collect() brings to driver
        null_counts = {}
        for col_name in ["close", "volume", "daily_return", "symbol", "date"]:
            # filter — distributed predicate evaluation on each partition
            null_count = df.filter(F.col(col_name).isNull()).count()
            null_counts[col_name] = null_count
        checks["null_counts"] = null_counts

        # Negative prices — filter is evaluated on each partition independently
        neg_prices = df.filter(F.col("close") < 0).count()
        checks["negative_prices"] = neg_prices

        # Date range — agg with min/max, distributed across partitions
        date_stats = df.agg(
            F.min("date").alias("min_date"),
            F.max("date").alias("max_date"),
        ).collect()[0]
        checks["date_range"] = {
            "min": str(date_stats["min_date"]),
            "max": str(date_stats["max_date"]),
        }

        # Distinct symbol count — distinct() causes a shuffle across partitions
        checks["symbol_count"] = df.select("symbol").distinct().count()

        logger.info(f"  Total rows: {checks['total_rows']:,}")
        logger.info(f"  Symbols: {checks['symbol_count']}")
        logger.info(f"  Date range: {checks['date_range']['min']} → {checks['date_range']['max']}")
        logger.info(f"  Negative prices: {checks['negative_prices']}")
        for col, cnt in checks["null_counts"].items():
            if cnt > 0:
                logger.warning(f"  Nulls in {col}: {cnt}")

        return checks

    # ─────────────────────────────────────────────────────────
    # Per-Stock Statistics
    # ─────────────────────────────────────────────────────────

    def compute_stock_statistics(self, df: DataFrame) -> DataFrame:
        """
        Calculate per-stock aggregate statistics.

        DISTRIBUTED: groupBy causes a shuffle — all rows for the same symbol
        are moved to the same partition. agg() then computes stats within
        each partition. This is Spark's equivalent of GROUP BY in SQL.
        """
        logger.info("Computing per-stock statistics...")

        # groupBy + agg — SHUFFLE operation: rows are redistributed by symbol
        # Each aggregation function runs in parallel across partitions
        stock_stats = (
            df
            # select — projection, narrows columns (distributed, no shuffle)
            .select(
                "symbol", "close", "volume",
                "daily_return", "log_return", "volatility",
                "high", "low",
            )
            # filter — remove rows with null returns (first row per stock)
            .filter(F.col("daily_return").isNotNull())
            # groupBy — triggers shuffle to co-locate rows by symbol
            .groupBy("symbol")
            # agg — computed on each group in parallel
            .agg(
                F.avg("daily_return").alias("avg_daily_return"),
                F.avg("log_return").alias("avg_log_return"),
                F.avg("volatility").alias("avg_volatility"),
                F.avg("volume").alias("avg_volume"),
                F.sum("volume").alias("total_volume"),
                F.max("close").alias("max_price"),
                F.min("close").alias("min_price"),
                F.count("*").alias("trading_days"),
            )
            # orderBy — triggers another shuffle for global sort
            .orderBy("symbol")
        )

        logger.info(f"  Computed stats for {stock_stats.count()} stocks")
        return stock_stats

    # ─────────────────────────────────────────────────────────
    # Daily Market-Level Statistics
    # ─────────────────────────────────────────────────────────

    def compute_daily_statistics(self, df: DataFrame) -> DataFrame:
        """
        Calculate daily market-wide statistics.

        DISTRIBUTED: groupBy("date") shuffles rows so all stocks for the same
        date end up on the same partition, then aggregates per partition.
        """
        logger.info("Computing daily market statistics...")

        daily_stats = (
            df
            .filter(F.col("daily_return").isNotNull())
            # groupBy date — shuffle by date
            .groupBy("date")
            .agg(
                F.avg("daily_return").alias("market_avg_return"),
                F.sum("volume").alias("market_total_volume"),
                F.count("symbol").alias("stocks_traded"),
                F.avg("close").alias("avg_close_price"),
                F.stddev("daily_return").alias("market_return_std"),
            )
            # orderBy — global sort across all partitions
            .orderBy("date")
        )

        logger.info(f"  Computed daily stats for {daily_stats.count()} trading days")
        return daily_stats

    # ─────────────────────────────────────────────────────────
    # Stock-Level Aggregations with Window Functions
    # ─────────────────────────────────────────────────────────

    def compute_stock_rankings(self, df: DataFrame) -> DataFrame:
        """
        Rank stocks and compute running aggregations using window functions.

        DISTRIBUTED: Window functions partition data by a key and order within
        each partition. The window frame defines which rows to include.
        Unlike groupBy, window functions preserve individual rows.
        """
        logger.info("Computing stock rankings and window aggregations...")

        # ── Window specifications ────────────────────────────
        # Partition by symbol, order by date — for per-stock running calculations
        stock_window = Window.partitionBy("symbol").orderBy("date")

        # Global window ordered by avg volatility — for cross-stock ranking
        # (We'll compute this on stock-level aggregates)

        # withColumn with window — adds running cumulative volume per stock
        # DISTRIBUTED: each partition (by symbol) computes independently
        df_with_running = (
            df
            .filter(F.col("daily_return").isNotNull())
            # withColumn — add running cumulative volume per stock
            .withColumn(
                "cumulative_volume",
                F.sum("volume").over(stock_window)
            )
            # withColumn — add row number within each stock (trading day index)
            .withColumn(
                "trading_day_index",
                F.row_number().over(stock_window)
            )
            # withColumn — 5-day moving average of close via window
            .withColumn(
                "spark_ma_5",
                F.avg("close").over(
                    stock_window.rowsBetween(-4, 0)  # current + 4 previous
                )
            )
        )

        # Compute per-stock volatility rankings
        stock_vol = (
            df
            .filter(F.col("volatility").isNotNull())
            .groupBy("symbol")
            .agg(F.avg("volatility").alias("avg_volatility"))
        )

        # Window for ranking — orderBy descending volatility
        rank_window = Window.orderBy(F.desc("avg_volatility"))

        stock_rankings = (
            stock_vol
            # withColumn — rank by volatility (dense_rank preserves ties)
            .withColumn("volatility_rank", F.dense_rank().over(rank_window))
            .orderBy("volatility_rank")
        )

        logger.info(f"  Rankings computed for {stock_rankings.count()} stocks")
        return stock_rankings

    # ─────────────────────────────────────────────────────────
    # Cross-analysis: Join stock stats with rankings
    # ─────────────────────────────────────────────────────────

    def compute_enriched_statistics(
        self, stock_stats: DataFrame, stock_rankings: DataFrame
    ) -> DataFrame:
        """
        Join stock statistics with volatility rankings.

        DISTRIBUTED: join triggers a shuffle — both DataFrames are
        redistributed by the join key ("symbol") so matching rows
        end up on the same partition for the join computation.
        """
        logger.info("Joining stock statistics with rankings...")

        # Avoid duplicate avg_volatility column from stock_rankings
        rankings_to_join = stock_rankings.select("symbol", "volatility_rank")

        # join — SHUFFLE operation on both sides by "symbol"
        enriched = (
            stock_stats
            .join(rankings_to_join, on="symbol", how="inner")
            .orderBy("volatility_rank")
        )

        logger.info(f"  Enriched stats: {enriched.count()} stocks")
        return enriched

    # ─────────────────────────────────────────────────────────
    # Save Results
    # ─────────────────────────────────────────────────────────

    def save_results(
        self,
        stock_stats: DataFrame,
        daily_stats: DataFrame,
        stock_rankings: DataFrame,
        enriched_stats: DataFrame,
    ) -> dict:
        """
        Save all results as Parquet files.

        DISTRIBUTED: write triggers a full execution of the DAG. Spark
        writes one Parquet file per partition. coalesce(1) merges to a
        single file for easy downstream consumption.
        """
        self.output_path.mkdir(parents=True, exist_ok=True)
        saved = {}

        datasets = {
            "stock_statistics": stock_stats,
            "daily_statistics": daily_stats,
            "stock_rankings": stock_rankings,
            "enriched_statistics": enriched_stats,
        }

        for name, df in datasets.items():
            out_path = str(self.output_path / name)
            logger.info(f"  Saving {name} → {out_path}")

            # coalesce(1) — reduce to 1 partition for single output file
            # In production with large data, you'd skip coalesce for parallelism
            (
                df
                .coalesce(1)
                .write
                .mode("overwrite")
                .parquet(out_path)
            )
            saved[name] = out_path

        logger.info(f"All results saved to {self.output_path}")
        return saved

    # ─────────────────────────────────────────────────────────
    # Full Pipeline
    # ─────────────────────────────────────────────────────────

    def run(self, filepath: Optional[str] = None) -> dict:
        """
        Execute the full Spark analytics pipeline.

        Returns
        -------
        dict
            Summary of quality checks, stats, and output paths.
        """
        logger.info("=" * 60)
        logger.info("Spark Pipeline — Starting")
        logger.info("=" * 60)

        try:
            # 1. Load data
            df = self.load_market_data(filepath)

            # 2. Quality checks
            quality = self.run_quality_checks(df)

            # 3. Per-stock statistics
            stock_stats = self.compute_stock_statistics(df)

            # 4. Daily market statistics
            daily_stats = self.compute_daily_statistics(df)

            # 5. Stock rankings (window functions)
            stock_rankings = self.compute_stock_rankings(df)

            # 6. Enriched stats (join)
            enriched = self.compute_enriched_statistics(stock_stats, stock_rankings)

            # 7. Save all results
            saved = self.save_results(stock_stats, daily_stats, stock_rankings, enriched)

            # 8. Show sample results on driver
            logger.info("\n── Stock Statistics (Top 5) ──")
            stock_stats.show(5, truncate=False)

            logger.info("\n── Daily Statistics (First 5 days) ──")
            daily_stats.show(5, truncate=False)

            logger.info("\n── Volatility Rankings ──")
            stock_rankings.show(10, truncate=False)

            logger.info("=" * 60)
            logger.info("Spark Pipeline — Complete ✨")
            logger.info("=" * 60)

            return {
                "quality_checks": quality,
                "output_paths": saved,
                "stock_count": stock_stats.count(),
                "daily_count": daily_stats.count(),
            }

        finally:
            if self._owns_spark:
                self.spark.stop()
                logger.info("SparkSession stopped")

    def stop(self):
        """Stop the SparkSession if we own it."""
        if self._owns_spark and self.spark is not None:
            self.spark.stop()
