"""
Distributed Apache Spark Advanced Analytics Pipeline (Step 8 Phase C).

Executes distributed DataFrame analytics directly on the Spark Standalone Cluster:
- Master: spark://hadoop-master:7077
- Input: hdfs://hadoop-master:9000/financial-intelligence/real/processed/market/consolidated_market_features.parquet
- Output: hdfs://hadoop-master:9000/financial-intelligence/real/analytics/

Generates:
1. risk_return_metrics.parquet
2. benchmark_relative_metrics.parquet
3. market_regime_breadth.parquet
4. composite_stock_rankings.parquet
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
HDFS_ANALYTICS_ROOT = f"{HDFS_NAMENODE}/financial-intelligence/real/analytics"

LOCAL_OUT_DIR = Path("/opt/project/data/real/analytics") if Path("/opt/project").exists() else Path("data/real/analytics")


def run_distributed_spark_analytics():
    from pyspark.sql import SparkSession, Window
    from pyspark.sql import functions as F

    logger.info("==========================================================================")
    logger.info(" Initializing Spark Standalone Cluster Job for Step 8 Advanced Analytics")
    logger.info("==========================================================================")

    spark = (
        SparkSession.builder
        .appName("RealAdvancedFinancialAnalyticsCluster")
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
    logger.info(f"Connected Spark Master: {master_uri}")
    logger.info(f"Assigned Application ID: {app_id}")

    if master_uri.startswith("local"):
        spark.stop()
        raise RuntimeError(f"PROHIBITED: Local execution detected ('{master_uri}'). Must run on cluster master!")

    t0 = time.time()

    # 1. Read input dataset from HDFS
    logger.info(f"Loading input Parquet dataset from {HDFS_INPUT}...")
    df = spark.read.parquet(HDFS_INPUT)
    df = df.withColumn("date", F.to_date(F.col("date")))
    total_rows = df.count()
    ticker_count = df.select("symbol").distinct().count()
    logger.info(f"Loaded {total_rows:,} rows across {ticker_count} tickers from HDFS.")

    # -------------------------------------------------------------
    # Op 1: Risk and Return Metrics
    # -------------------------------------------------------------
    logger.info("Op 1: Computing distributed risk and return metrics...")
    df_valid = df.filter(F.col("daily_adjusted_return").isNotNull())

    df_risk = (
        df_valid
        .groupBy("symbol")
        .agg(
            F.count("*").alias("observation_count"),
            F.mean("daily_adjusted_return").alias("mean_daily_return"),
            F.stddev_samp("daily_adjusted_return").alias("daily_volatility"),
            F.max("daily_adjusted_return").alias("best_day_return"),
            F.min("daily_adjusted_return").alias("worst_day_return"),
            F.sum(F.when(F.col("daily_adjusted_return") > 0, 1).otherwise(0)).alias("positive_days_count"),
        )
        .withColumn("annualized_return", F.col("mean_daily_return") * 252.0)
        .withColumn("annualized_volatility", F.col("daily_volatility") * math.sqrt(252.0))
        .withColumn("sharpe_ratio_rf0", F.col("annualized_return") / F.col("annualized_volatility"))
        .withColumn("sharpe_ratio_rf6", (F.col("annualized_return") - 0.06) / F.col("annualized_volatility"))
        .withColumn("positive_days_pct", (F.col("positive_days_count") / F.col("observation_count")) * 100.0)
        .orderBy("symbol")
    )
    p_risk = f"{HDFS_ANALYTICS_ROOT}/risk_return_metrics.parquet"
    df_risk.write.mode("overwrite").parquet(p_risk)
    logger.info(f"Saved {df_risk.count()} rows to {p_risk}")

    # -------------------------------------------------------------
    # Op 2: Benchmark Relative Metrics
    # -------------------------------------------------------------
    logger.info("Op 2: Computing distributed benchmark-relative metrics vs NIFTY 50...")
    df_bm_valid = df.filter(F.col("daily_adjusted_return").isNotNull() & F.col("benchmark_daily_return").isNotNull())

    # Calculate covariance and variance
    df_bench = (
        df_bm_valid
        .groupBy("symbol")
        .agg(
            F.count("*").alias("aligned_sessions"),
            F.mean("daily_adjusted_return").alias("stock_mean_ret"),
            F.mean("benchmark_daily_return").alias("bench_mean_ret"),
            F.covar_samp("daily_adjusted_return", "benchmark_daily_return").alias("covariance"),
            F.var_samp("benchmark_daily_return").alias("bench_variance"),
            F.corr("daily_adjusted_return", "benchmark_daily_return").alias("correlation"),
            F.stddev_samp(F.col("daily_adjusted_return") - F.col("benchmark_daily_return")).alias("diff_stddev"),
        )
        .withColumn("beta", F.col("covariance") / F.col("bench_variance"))
        .withColumn("annualized_stock_return", F.col("stock_mean_ret") * 252.0)
        .withColumn("annualized_bench_return", F.col("bench_mean_ret") * 252.0)
        .withColumn("annualized_excess_return", F.col("annualized_stock_return") - F.col("annualized_bench_return"))
        .withColumn("jensen_alpha_rf6", (F.col("annualized_stock_return") - 0.06) - F.col("beta") * (F.col("annualized_bench_return") - 0.06))
        .withColumn("tracking_error", F.col("diff_stddev") * math.sqrt(252.0))
        .withColumn("information_ratio", F.col("annualized_excess_return") / F.col("tracking_error"))
        .orderBy("symbol")
    )
    p_bench = f"{HDFS_ANALYTICS_ROOT}/benchmark_relative_metrics.parquet"
    df_bench.write.mode("overwrite").parquet(p_bench)
    logger.info(f"Saved {df_bench.count()} rows to {p_bench}")

    # -------------------------------------------------------------
    # Op 3: Market Regime and Breadth
    # -------------------------------------------------------------
    logger.info("Op 3: Computing daily market-wide breadth and moving average breadth...")
    df_breadth = (
        df
        .groupBy("date")
        .agg(
            F.count("*").alias("total_stocks"),
            F.sum(F.when(F.col("daily_adjusted_return") > 0, 1).otherwise(0)).alias("advances"),
            F.sum(F.when(F.col("daily_adjusted_return") < 0, 1).otherwise(0)).alias("declines"),
            F.sum(F.when(F.col("close") > F.col("ma_20"), 1).otherwise(0)).alias("above_ma20"),
            F.sum(F.when(F.col("close") > F.col("ma_50"), 1).otherwise(0)).alias("above_ma50"),
            F.sum(F.when(F.col("close") > F.col("ma_200"), 1).otherwise(0)).alias("above_ma200"),
        )
        .withColumn("advance_ratio", F.col("advances") / F.col("total_stocks"))
        .withColumn("pct_above_ma20", F.col("above_ma20") / F.col("total_stocks"))
        .withColumn("pct_above_ma50", F.col("above_ma50") / F.col("total_stocks"))
        .withColumn("pct_above_ma200", F.col("above_ma200") / F.col("total_stocks"))
        .orderBy("date")
    )
    p_breadth = f"{HDFS_ANALYTICS_ROOT}/market_regime_breadth.parquet"
    df_breadth.write.mode("overwrite").parquet(p_breadth)
    logger.info(f"Saved {df_breadth.count()} rows to {p_breadth}")

    # -------------------------------------------------------------
    # Op 4: Multi-Factor Composite Stock Rankings
    # -------------------------------------------------------------
    logger.info("Op 4: Computing multi-factor composite stock rankings...")
    df_joined = df_risk.join(df_bench, on="symbol")

    # Window specs for percentile ranks
    w_sharpe = Window.orderBy(F.col("sharpe_ratio_rf6").asc())
    w_vol = Window.orderBy(F.col("annualized_volatility").desc())  # Lower vol gets higher rank
    w_alpha = Window.orderBy(F.col("jensen_alpha_rf6").asc())
    w_cons = Window.orderBy(F.col("positive_days_pct").asc())

    df_scored = (
        df_joined
        .withColumn("score_sharpe", F.percent_rank().over(w_sharpe) * 100.0)
        .withColumn("score_volatility", F.percent_rank().over(w_vol) * 100.0)
        .withColumn("score_alpha", F.percent_rank().over(w_alpha) * 100.0)
        .withColumn("score_consistency", F.percent_rank().over(w_cons) * 100.0)
        .withColumn(
            "composite_score",
            F.col("score_sharpe") * 0.30
            + F.col("score_volatility") * 0.25
            + F.col("score_alpha") * 0.25
            + F.col("score_consistency") * 0.20
        )
    )

    w_final = Window.orderBy(F.col("composite_score").desc())
    df_ranked = (
        df_scored
        .withColumn("rank", F.dense_rank().over(w_final))
        .select(
            "rank",
            "symbol",
            F.round("composite_score", 2).alias("composite_score"),
            F.round("annualized_return", 4).alias("annualized_return"),
            F.round("annualized_volatility", 4).alias("annualized_volatility"),
            F.round("sharpe_ratio_rf6", 4).alias("sharpe_ratio_rf6"),
            F.round("beta", 4).alias("beta"),
            F.round("jensen_alpha_rf6", 4).alias("jensen_alpha_rf6"),
            F.round("positive_days_pct", 2).alias("positive_days_pct"),
        )
        .orderBy("rank")
    )
    p_rank = f"{HDFS_ANALYTICS_ROOT}/composite_stock_rankings.parquet"
    df_ranked.write.mode("overwrite").parquet(p_rank)
    logger.info(f"Saved {df_ranked.count()} rows to {p_rank}")

    elapsed = round(time.time() - t0, 2)
    logger.info(f"Spark distributed analytics completed in {elapsed}s.")

    # Export local CSV summaries for verification and downstream integration
    LOCAL_OUT_DIR.mkdir(parents=True, exist_ok=True)
    df_risk.toPandas().to_csv(LOCAL_OUT_DIR / "risk_return_metrics.csv", index=False)
    df_bench.toPandas().to_csv(LOCAL_OUT_DIR / "benchmark_relative_metrics.csv", index=False)
    df_breadth.toPandas().to_csv(LOCAL_OUT_DIR / "market_regime_breadth.csv", index=False)
    df_ranked.toPandas().to_csv(LOCAL_OUT_DIR / "composite_stock_rankings.csv", index=False)

    top_5_stocks = [r["symbol"] for r in df_ranked.select("symbol").limit(5).collect()]
    logger.info(f"Top 5 Composite Stocks: {top_5_stocks}")

    report = {
        "execution_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "spark_application_id": app_id,
        "spark_master": master_uri,
        "input_dataset_hdfs": HDFS_INPUT,
        "analytics_output_hdfs_root": HDFS_ANALYTICS_ROOT,
        "total_records_processed": total_rows,
        "duration_seconds": elapsed,
        "top_5_composite_equities": top_5_stocks,
        "generated_hdfs_tables": [
            {"name": "risk_return_metrics", "rows": df_risk.count(), "hdfs_path": p_risk},
            {"name": "benchmark_relative_metrics", "rows": df_bench.count(), "hdfs_path": p_bench},
            {"name": "market_regime_breadth", "rows": df_breadth.count(), "hdfs_path": p_breadth},
            {"name": "composite_stock_rankings", "rows": df_ranked.count(), "hdfs_path": p_rank},
        ],
    }

    report_path = LOCAL_OUT_DIR / "spark_step8_execution_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Execution report saved to {report_path}")
    spark.stop()
    return report


if __name__ == "__main__":
    run_distributed_spark_analytics()

