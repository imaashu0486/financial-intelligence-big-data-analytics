"""
Apache Spark Standalone Cluster Pipeline for Real Financial Analytics.

Executes distributed DataFrame analytics directly on the Spark Standalone Cluster:
- Master: spark://hadoop-master:7077
- Input: hdfs://hadoop-master:9000/financial-intelligence/real/processed/market/consolidated_market_features.parquet
- Output: hdfs://hadoop-master:9000/financial-intelligence/real/distributed/spark/

Explicitly prohibits local execution modes (local[*]).
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

HDFS_NAMENODE = "hdfs://hadoop-master:9000"
HDFS_INPUT = f"{HDFS_NAMENODE}/financial-intelligence/real/processed/market/consolidated_market_features.parquet"
HDFS_OUTPUT = f"{HDFS_NAMENODE}/financial-intelligence/real/distributed/spark"
LOCAL_REPORT_PATH = Path("/opt/project/data/real/distributed/spark/spark_cluster_execution_report.json")


def run_spark_cluster_analytics():
    from pyspark.sql import SparkSession, Window
    from pyspark.sql import functions as F

    logger.info("==========================================================================")
    logger.info(" Starting Spark Cluster Job on Real Financial Dataset")
    logger.info("==========================================================================")

    # Initialize SparkSession configured for Standalone Cluster
    spark = (
        SparkSession.builder
        .appName("RealFinancialAnalyticsCluster")
        .config("spark.sql.parquet.compression.codec", "snappy")
        .config("spark.executor.memory", "768m")
        .config("spark.executor.cores", "1")
        .config("spark.sql.session.timeZone", "Asia/Kolkata")
        .getOrCreate()
    )
    sc = spark.sparkContext
    sc.setLogLevel("WARN")

    master_uri = sc.master
    app_id = sc.applicationId
    logger.info(f"Spark App ID: {app_id}")
    logger.info(f"Connected Master: {master_uri}")

    if master_uri.startswith("local"):
        spark.stop()
        raise RuntimeError(f"PROHIBITED: Spark must run on cluster, but got master='{master_uri}'!")

    t0 = time.time()

    # 1. Read input from HDFS
    logger.info(f"Reading input from HDFS: {HDFS_INPUT}...")
    df = spark.read.parquet(HDFS_INPUT)
    df = df.withColumn("date", F.to_date(F.col("date")))
    total_rows = df.count()
    ticker_count = df.select("symbol").distinct().count()
    logger.info(f"Read {total_rows:,} rows across {ticker_count} tickers from HDFS.")

    # Op 1: Ticker average daily returns
    logger.info("Computing Op 1: Ticker Average Returns...")
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
    out1 = f"{HDFS_OUTPUT}/ticker_avg_returns.parquet"
    df_returns.write.mode("overwrite").parquet(out1)
    logger.info(f"Saved {df_returns.count()} rows to {out1}")

    # Op 2: Ticker annualized volatility
    logger.info("Computing Op 2: Ticker Annualized Volatility...")
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
    out2 = f"{HDFS_OUTPUT}/ticker_volatility.parquet"
    df_vol.write.mode("overwrite").parquet(out2)
    logger.info(f"Saved {df_vol.count()} rows to {out2}")

    # Op 3: Daily market-wide volume
    logger.info("Computing Op 3: Daily Market-Wide Trading Volume...")
    df_volume = (
        df.groupBy("date")
        .agg(
            F.sum("volume").alias("total_market_volume"),
            F.mean("volume").alias("mean_stock_volume"),
            F.count("symbol").alias("active_constituents"),
        )
        .orderBy("date")
    )
    out3 = f"{HDFS_OUTPUT}/daily_market_volume.parquet"
    df_volume.write.mode("overwrite").parquet(out3)
    logger.info(f"Saved {df_volume.count()} rows to {out3}")

    # Op 4: Benchmark relative alpha
    logger.info("Computing Op 4: Benchmark Relative Alpha...")
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
    out4 = f"{HDFS_OUTPUT}/benchmark_relative.parquet"
    df_bench.write.mode("overwrite").parquet(out4)
    logger.info(f"Saved {df_bench.count()} rows to {out4}")

    # Op 5: Yearly market summary
    logger.info("Computing Op 5: Yearly Market Summary...")
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
    out5 = f"{HDFS_OUTPUT}/yearly_market_summary.parquet"
    df_yearly.write.mode("overwrite").parquet(out5)
    logger.info(f"Saved {df_yearly.count()} rows to {out5}")

    # Op 6: Multi-year Sharpe ratio stock rankings
    logger.info("Computing Op 6: Multi-Year Sharpe Ratio Rankings...")
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
    out6 = f"{HDFS_OUTPUT}/stock_rankings.parquet"
    df_ranked.write.mode("overwrite").parquet(out6)
    logger.info(f"Saved {df_ranked.count()} rows to {out6}")

    duration = round(time.time() - t0, 2)
    logger.info(f"Spark cluster execution finished in {duration}s.")

    # Collect top 5 Sharpe stocks
    top_5 = [row["symbol"] for row in df_ranked.select("symbol").limit(5).collect()]
    logger.info(f"Top 5 Sharpe Stocks: {top_5}")

    # Capture cluster report
    report = {
        "execution_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "spark_application_id": app_id,
        "spark_master": master_uri,
        "input_hdfs_path": HDFS_INPUT,
        "output_hdfs_root": HDFS_OUTPUT,
        "total_rows_processed": total_rows,
        "duration_sec": duration,
        "top_5_sharpe_stocks": top_5,
        "operations": [
            {"op": "op1_ticker_returns", "rows": df_returns.count(), "hdfs_path": out1},
            {"op": "op2_ticker_volatility", "rows": df_vol.count(), "hdfs_path": out2},
            {"op": "op3_daily_volume", "rows": df_volume.count(), "hdfs_path": out3},
            {"op": "op4_benchmark_relative", "rows": df_bench.count(), "hdfs_path": out4},
            {"op": "op5_yearly_summary", "rows": df_yearly.count(), "hdfs_path": out5},
            {"op": "op6_stock_rankings", "rows": df_ranked.count(), "hdfs_path": out6},
        ],
    }

    LOCAL_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOCAL_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Saved Spark execution report to {LOCAL_REPORT_PATH}")
    spark.stop()
    return report


if __name__ == "__main__":
    run_spark_cluster_analytics()

