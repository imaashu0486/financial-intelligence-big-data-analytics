"""
Read and validate Parquet files directly from HDFS.
"""

import os
import subprocess
import pandas as pd

HDFS_URL = "hdfs://hadoop-master:9000/financial-intelligence/real/distributed/spark/stock_rankings.parquet"

df = pd.read_parquet(HDFS_URL)
print(f"Successfully read {len(df)} rows from {HDFS_URL}")
print(df[["symbol", "annualized_return", "annualized_volatility", "sharpe_ratio", "rank_sharpe"]].head(10).to_string())

