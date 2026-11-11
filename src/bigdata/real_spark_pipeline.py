"""
Apache Spark Distributed Analytics Pipeline for Real Financial Datasets.

Executes distributed Spark DataFrame operations on the 10-year consolidated NIFTY 50
market features dataset (data/real/processed/market/consolidated_market_features.parquet):

1. Ticker-level average daily returns
2. Ticker-level annualized volatility (sigma * sqrt(252))
3. Daily market-wide trading volume totals
4. Benchmark-relative performance summary
5. Yearly market performance summaries
6. Multi-year stock rankings by annualized return and Sharpe ratio
7. Save all 6 aggregated analytical tables as Parquet to data/real/distributed/spark/

Includes complete execution manifest and cluster submission command templates.
"""

from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path
import sys
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Ensure workers and driver use virtualenv python
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

# Auto-configure HADOOP_HOME for Windows compatibility
hadoop_dir = PROJECT_ROOT / "hadoop"
if hadoop_dir.exists():
    os.environ["HADOOP_HOME"] = str(hadoop_dir)
    hadoop_bin = str(hadoop_dir / "bin")
    if hadoop_bin not in os.environ.get("PATH", ""):
        os.environ["PATH"] = f"{hadoop_bin};{os.environ.get('PATH', '')}"


INPUT_PARQUET = PROJECT_ROOT / "data" / "real" / "processed" / "market" / "consolidated_market_features.parquet"
OUTPUT_DIR = PROJECT_ROOT / "data" / "real" / "distributed" / "spark"


class RealSparkPipeline:
    def __init__(self, master: str = "local[*]", app_name: str = "RealFinancialAnalytics"):
        self.master = master
        self.app_name = app_name
        self.output_dir = OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.spark = None

    def start_session(self):
        from pyspark.sql import SparkSession

        logger.info(f"Initializing SparkSession: app={self.app_name}, master={self.master}")
        self.spark = (
            SparkSession.builder
            .appName(self.app_name)
            .master(self.master)
            .config("spark.sql.parquet.compression.codec", "snappy")
            .config("spark.driver.memory", "2g")
            .config("spark.ui.showConsoleProgress", "false")
            .config("spark.sql.session.timeZone", "Asia/Kolkata")
            .getOrCreate()
        )
        self.spark.sparkContext.setLogLevel("WARN")
        logger.info("SparkSession initialized successfully.")
        return self.spark

    def stop_session(self):
        if self.spark:
            self.spark.stop()
            logger.info("SparkSession stopped.")

    def run_all(self) -> dict:
        from pyspark.sql import functions as F
        from pyspark.sql.window import Window

        t0 = time.time()
        spark = self.start_session()

        logger.info(f"Reading consolidated market features from {INPUT_PARQUET}...")
        df = spark.read.parquet(str(INPUT_PARQUET))
        df = df.withColumn("date", F.to_date(F.col("date")))
        total_rows = df.count()
        logger.info(f"Loaded {total_rows:,} rows across {df.select('symbol').distinct().count()} tickers.")

        # Op 1: Ticker-level average daily returns
        logger.info("Op 1: Computing ticker-level average daily returns...")
        df_returns = (
            df.filter(F.col("daily_adjusted_return").isNotNull())
            .groupBy("symbol")
            .agg(
                F.mean("daily_adjusted_return").alias("mean_daily_return"),
                F.count("daily_adjusted_return").alias("observation_count"),
            )
            .withColumn("annualized_return", F.col("mean_daily_return") * 252.0)
            .orderBy("symbol")
        )
        p1 = self.output_dir / "ticker_avg_returns.parquet"
        df_returns.write.mode("overwrite").parquet(str(p1))
        df_returns.toPandas().to_csv(self.output_dir / "ticker_avg_returns.csv", index=False)

        # Op 2: Ticker-level annualized volatility
        logger.info("Op 2: Computing ticker-level annualized volatility...")
        df_vol = (
            df.filter(F.col("daily_adjusted_return").isNotNull())
            .groupBy("symbol")
            .agg(
                F.stddev_samp("daily_adjusted_return").alias("daily_volatility"),
                F.count("daily_adjusted_return").alias("observation_count"),
            )
            .withColumn("annualized_volatility", F.col("daily_volatility") * math.sqrt(252))
            .orderBy("symbol")
        )
        p2 = self.output_dir / "ticker_volatility.parquet"
        df_vol.write.mode("overwrite").parquet(str(p2))
        df_vol.toPandas().to_csv(self.output_dir / "ticker_volatility.csv", index=False)

        # Op 3: Daily market-wide trading-volume totals
        logger.info("Op 3: Computing daily market-wide trading volume...")
        df_volume = (
            df.groupBy("date")
            .agg(
                F.sum("volume").alias("total_market_volume"),
                F.mean("volume").alias("mean_stock_volume"),
                F.count("symbol").alias("active_constituents"),
            )
            .orderBy("date")
        )
        p3 = self.output_dir / "daily_market_volume.parquet"
        df_volume.write.mode("overwrite").parquet(str(p3))
        df_volume.toPandas().to_csv(self.output_dir / "daily_market_volume.csv", index=False)

        # Op 4: Benchmark-relative performance summary
        logger.info("Op 4: Computing benchmark-relative performance summary...")
        df_bench = (
            df.filter(F.col("daily_adjusted_return").isNotNull() & F.col("benchmark_daily_return").isNotNull())
            .groupBy("symbol")
            .agg(
                F.mean("relative_return_vs_nifty").alias("mean_daily_alpha"),
                F.mean("daily_adjusted_return").alias("ticker_mean_return"),
                F.mean("benchmark_daily_return").alias("benchmark_mean_return"),
                F.count("*").alias("aligned_sessions"),
            )
            .withColumn("annualized_alpha", F.col("mean_daily_alpha") * 252.0)
            .orderBy(F.col("annualized_alpha").desc())
        )
        p4 = self.output_dir / "benchmark_relative.parquet"
        df_bench.write.mode("overwrite").parquet(str(p4))
        df_bench.toPandas().to_csv(self.output_dir / "benchmark_relative.csv", index=False)

        # Op 5: Yearly market performance summaries
        logger.info("Op 5: Computing yearly market performance summaries...")
        df_yearly = (
            df.withColumn("year", F.year("date"))
            .groupBy("year")
            .agg(
                F.sum("volume").alias("total_yearly_volume"),
                F.mean("daily_adjusted_return").alias("mean_daily_return"),
                F.countDistinct("date").alias("trading_days"),
                F.count("*").alias("total_records"),
            )
            .withColumn("approx_annual_return", F.col("mean_daily_return") * F.col("trading_days"))
            .orderBy("year")
        )
        p5 = self.output_dir / "yearly_market_summary.parquet"
        df_yearly.write.mode("overwrite").parquet(str(p5))
        df_yearly.toPandas().to_csv(self.output_dir / "yearly_market_summary.csv", index=False)

        # Op 6: Multi-year stock rankings by return and Sharpe ratio
        logger.info("Op 6: Computing multi-year stock rankings (Return and Sharpe)...")
        # Join returns and volatility
        df_perf = (
            df.filter(F.col("daily_adjusted_return").isNotNull())
            .groupBy("symbol")
            .agg(
                F.mean("daily_adjusted_return").alias("daily_mean_return"),
                F.stddev_samp("daily_adjusted_return").alias("daily_stddev"),
                F.count("*").alias("sessions"),
            )
            .withColumn("annualized_return", F.col("daily_mean_return") * 252.0)
            .withColumn("annualized_volatility", F.col("daily_stddev") * math.sqrt(252))
            .withColumn("sharpe_ratio", F.col("annualized_return") / F.col("annualized_volatility"))
        )

        w_ret = Window.orderBy(F.col("annualized_return").desc())
        w_sharpe = Window.orderBy(F.col("sharpe_ratio").desc())

        df_ranked = (
            df_perf
            .withColumn("rank_return", F.dense_rank().over(w_ret))
            .withColumn("rank_sharpe", F.dense_rank().over(w_sharpe))
            .orderBy("rank_sharpe")
        )
        p6 = self.output_dir / "stock_rankings.parquet"
        df_ranked.write.mode("overwrite").parquet(str(p6))
        df_ranked.toPandas().to_csv(self.output_dir / "stock_rankings.csv", index=False)

        elapsed = round(time.time() - t0, 3)
        logger.info(f"All 7 Spark operations finished in {elapsed}s.")

        manifest = {
            "execution_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "input_dataset": str(INPUT_PARQUET.relative_to(PROJECT_ROOT)),
            "master": self.master,
            "app_name": self.app_name,
            "total_records_processed": total_rows,
            "duration_sec": elapsed,
            "operations": [
                {"id": "op1", "name": "ticker_avg_returns", "parquet": str(p1.relative_to(PROJECT_ROOT)), "rows": df_returns.count()},
                {"id": "op2", "name": "ticker_volatility", "parquet": str(p2.relative_to(PROJECT_ROOT)), "rows": df_vol.count()},
                {"id": "op3", "name": "daily_market_volume", "parquet": str(p3.relative_to(PROJECT_ROOT)), "rows": df_volume.count()},
                {"id": "op4", "name": "benchmark_relative", "parquet": str(p4.relative_to(PROJECT_ROOT)), "rows": df_bench.count()},
                {"id": "op5", "name": "yearly_market_summary", "parquet": str(p5.relative_to(PROJECT_ROOT)), "rows": df_yearly.count()},
                {"id": "op6", "name": "stock_rankings", "parquet": str(p6.relative_to(PROJECT_ROOT)), "rows": df_ranked.count()},
            ],
            "cluster_submit_command": (
                "spark-submit \\\n"
                "  --master yarn \\\n"
                "  --deploy-mode client \\\n"
                "  --executor-memory 2G \\\n"
                "  --num-executors 2 \\\n"
                "  src/bigdata/real_spark_pipeline.py"
            ),
        }

        with open(self.output_dir / "spark_execution_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        self.stop_session()
        return manifest


if __name__ == "__main__":
    pipeline = RealSparkPipeline()
    manifest = pipeline.run_all()
    print(json.dumps(manifest, indent=2))

