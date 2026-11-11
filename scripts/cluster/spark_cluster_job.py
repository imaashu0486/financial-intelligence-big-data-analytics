"""
Cluster-mode Spark Job for Financial Analytics.
Reads from HDFS, performs distributed aggregations across cluster workers,
and writes Parquet output back to HDFS.
"""

import sys
from pyspark.sql import SparkSession
from pyspark.sql import functions as F


def run_cluster_job():
    print("=" * 70)
    print("Starting Spark Distributed Financial Aggregation Job")
    print("=" * 70)

    # Initialize SparkSession configured for the cluster
    spark = (
        SparkSession.builder
        .appName("FinancialClusterAnalytics")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    hdfs_input = "hdfs://hadoop-master:9000/financial_analytics/raw/market_features.csv"
    hdfs_output = "hdfs://hadoop-master:9000/financial_analytics/processed/spark_cluster_output"

    print(f"Reading input from HDFS: {hdfs_input}")
    df = spark.read.csv(hdfs_input, header=True, inferSchema=True)

    print(f"Loaded {df.count()} market rows with {len(df.columns)} columns across partitions.")

    # Perform distributed aggregations: mean return, volatility, total volume
    summary_df = (
        df.groupBy("symbol")
        .agg(
            F.count("date").alias("trading_days"),
            F.round(F.mean("daily_return"), 6).alias("mean_daily_return"),
            F.round(F.stddev("daily_return"), 6).alias("daily_return_std"),
            F.round(F.mean("volatility"), 6).alias("mean_volatility"),
            F.round(F.min("close"), 2).alias("min_price"),
            F.round(F.max("close"), 2).alias("max_price"),
            F.sum("volume").alias("total_volume")
        )
        .orderBy(F.desc("total_volume"))
    )

    print("\nDistributed Stock Summary Results:")
    summary_df.show(15, truncate=False)

    print(f"\nWriting summary results to HDFS Parquet: {hdfs_output}")
    (
        summary_df.write
        .mode("overwrite")
        .parquet(hdfs_output)
    )

    print("✅ Cluster Spark job completed successfully!")
    spark.stop()


if __name__ == "__main__":
    run_cluster_job()

