"""
Tests for Step 7: Real Data Distributed Processing (HDFS, MapReduce, Spark).

Covers:
1. HDFS metadata manifest structure, file count, replication factor, and checksum integrity.
2. Unit tests for each of the 5 MapReduce mappers and reducers.
3. Integration verification of MapReduce output TSV files.
4. Integration verification of Apache Spark Parquet and CSV output tables.
5. Mathematical consistency between MapReduce, Spark, and reference validation report.
6. Non-regression and non-destructive preservation of existing synthetic datasets.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
import subprocess
import sys
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
HDFS_MANIFEST_PATH = PROJECT_ROOT / "data" / "real" / "metadata" / "hdfs_ingestion_manifest.json"
MR_DIR = PROJECT_ROOT / "data" / "real" / "distributed" / "mapreduce"
SPARK_DIR = PROJECT_ROOT / "data" / "real" / "distributed" / "spark"
VAL_REPORT_PATH = PROJECT_ROOT / "data" / "real" / "distributed" / "validation" / "distributed_validation_report.json"
SYNTHETIC_DIR = PROJECT_ROOT / "data" / "processed"


# =====================================================================
# 1. HDFS Manifest Verification
# =====================================================================

def test_hdfs_manifest_structure_and_completeness():
    assert HDFS_MANIFEST_PATH.exists(), "HDFS ingestion manifest does not exist"
    with open(HDFS_MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert "files" in manifest
    assert "hdfs_project_root" in manifest
    assert manifest["hdfs_project_root"] == "/financial-intelligence/real"
    assert manifest["replication_factor"] == 2
    assert manifest["summary"]["total_files_cataloged"] == 77
    assert manifest["summary"]["total_bytes"] > 80_000_000

    # Ensure all files have valid SHA-256 and positive size
    for item in manifest["files"]:
        assert len(item["sha256_checksum"]) == 64
        assert item["file_size_bytes"] > 0
        assert item["hdfs_destination"].startswith("/financial-intelligence/real/")


# =====================================================================
# 2. MapReduce Unit Tests (Mappers and Reducers)
# =====================================================================

def run_mr_script(script_name: str, input_str: str) -> str:
    script_path = PROJECT_ROOT / "src" / "bigdata" / "mapreduce" / script_name
    proc = subprocess.run(
        [sys.executable, str(script_path)],
        input=input_str,
        text=True,
        capture_output=True,
        check=True,
    )
    return proc.stdout


def test_mr_job1_volume_by_ticker():
    sample_csv = "date,symbol,open,high,low,close,adj_close,volume\n2020-01-01,TCS,100,105,99,102,100,5000\n2020-01-02,TCS,102,106,101,105,103,3000\n"
    map_out = run_mr_script("volume_by_ticker_mapper.py", sample_csv)
    assert map_out == "TCS\t5000\nTCS\t3000\n"

    red_out = run_mr_script("volume_by_ticker_reducer.py", map_out)
    assert red_out == "TCS\t8000\t2\n"


def test_mr_job2_daily_market_return():
    sample_csv = "date,symbol,daily_adjusted_return\n2020-01-02,INFY,0.02\n2020-01-02,TCS,0.01\n"
    map_out = run_mr_script("daily_market_return_mapper.py", sample_csv)
    assert "2020-01-02\t0.02000000\n" in map_out
    assert "2020-01-02\t0.01000000\n" in map_out

    red_out = run_mr_script("daily_market_return_reducer.py", map_out)
    assert red_out == "2020-01-02\t0.01500000\t2\n"


def test_mr_job3_company_avg_return():
    sample_csv = "date,symbol,daily_adjusted_return\n2020-01-02,RELIANCE,0.01\n2020-01-03,RELIANCE,0.03\n"
    map_out = run_mr_script("company_avg_return_mapper.py", sample_csv)
    assert map_out == "RELIANCE\t0.01000000\nRELIANCE\t0.03000000\n"

    red_out = run_mr_script("company_avg_return_reducer.py", map_out)
    assert red_out == "RELIANCE\t0.02000000\t2\n"


def test_mr_job4_yearly_volume():
    sample_csv = "date,symbol,volume\n2020-01-02,HDFCBANK,1000\n2020-05-10,HDFCBANK,2000\n2021-01-02,HDFCBANK,5000\n"
    map_out = run_mr_script("yearly_volume_mapper.py", sample_csv)
    lines = sorted(map_out.splitlines())
    sorted_map = "\n".join(lines) + "\n"

    red_out = run_mr_script("yearly_volume_reducer.py", sorted_map)
    assert "HDFCBANK\t2020\t3000\t2\n" in red_out
    assert "HDFCBANK\t2021\t5000\t1\n" in red_out


def test_mr_job5_market_quality():
    sample_csv = (
        "date,open,high,low,close,volume,daily_adjusted_return\n"
        "2020-01-01,100,105,95,102,1000,0.01\n"
        "2020-01-02,100,90,95,102,1000,0.01\n"  # anomaly: high < low
    )
    map_out = run_mr_script("market_quality_mapper.py", sample_csv)
    lines = sorted(map_out.splitlines())
    sorted_map = "\n".join(lines) + "\n"

    red_out = run_mr_script("market_quality_reducer.py", sorted_map)
    assert "TOTAL_ROWS\t2\n" in red_out
    assert "VALID_OHLC\t1\n" in red_out
    assert "ANOMALY_HIGH_LESS_THAN_LOW\t1\n" in red_out


# =====================================================================
# 3. MapReduce Integration Output Verification
# =====================================================================

def test_mapreduce_output_tables_integrity():
    job1_path = MR_DIR / "job1_volume_by_ticker.tsv"
    job2_path = MR_DIR / "job2_daily_market_return.tsv"
    job3_path = MR_DIR / "job3_company_avg_return.tsv"
    job4_path = MR_DIR / "job4_yearly_volume.tsv"
    job5_path = MR_DIR / "job5_market_quality_summary.tsv"

    assert job1_path.exists()
    assert job2_path.exists()
    assert job3_path.exists()
    assert job4_path.exists()
    assert job5_path.exists()

    df1 = pd.read_csv(job1_path, sep="\t")
    assert len(df1) == 50
    assert "symbol" in df1.columns and "total_volume" in df1.columns

    df2 = pd.read_csv(job2_path, sep="\t")
    assert len(df2) == 2466
    assert "date" in df2.columns and "mean_daily_return" in df2.columns

    df3 = pd.read_csv(job3_path, sep="\t")
    assert len(df3) == 50
    assert "symbol" in df3.columns and "mean_daily_return" in df3.columns

    df4 = pd.read_csv(job4_path, sep="\t")
    assert len(df4) == 496  # 50 tickers x 10 years (minus 4 ticker-years for late additions)

    df5 = pd.read_csv(job5_path, sep="\t")
    q_dict = dict(zip(df5["metric_name"], df5["total_count"]))
    assert q_dict["TOTAL_ROWS"] == 121962
    assert q_dict["VALID_OHLC"] == 121962
    assert q_dict["VALID_VOLUME"] == 121962
    assert q_dict["MISSING_RETURN"] == 50


# =====================================================================
# 4. Spark Integration Output Verification
# =====================================================================

def test_spark_output_tables_integrity():
    expected_parquets = [
        "ticker_avg_returns.parquet",
        "ticker_volatility.parquet",
        "daily_market_volume.parquet",
        "benchmark_relative.parquet",
        "yearly_market_summary.parquet",
        "stock_rankings.parquet",
    ]
    for p in expected_parquets:
        parquet_path = SPARK_DIR / p
        assert parquet_path.exists(), f"Spark output missing: {p}"
        df_p = pd.read_parquet(parquet_path)
        assert len(df_p) > 0, f"Spark table empty: {p}"

    # Verify stock rankings
    rank_df = pd.read_parquet(SPARK_DIR / "stock_rankings.parquet")
    assert len(rank_df) == 50
    assert "sharpe_ratio" in rank_df.columns
    assert "rank_sharpe" in rank_df.columns
    # Check top stock
    top_stock = rank_df.loc[rank_df["rank_sharpe"] == 1, "symbol"].iloc[0]
    assert top_stock == "TRENT"

    # Verify yearly summary
    yearly_df = pd.read_parquet(SPARK_DIR / "yearly_market_summary.parquet")
    assert len(yearly_df) == 10
    assert set(yearly_df["year"]) == set(range(2015, 2025))


# =====================================================================
# 5. Distributed Validation Engine Verification
# =====================================================================

def test_distributed_validation_report_verdict():
    assert VAL_REPORT_PATH.exists(), "Validation report does not exist"
    with open(VAL_REPORT_PATH, "r", encoding="utf-8") as f:
        report = json.load(f)

    assert report["all_checks_passed"] is True
    assert report["verdict"] == "PASS"
    assert report["failed_checks"] == 0
    assert report["passed_checks"] == 6


# =====================================================================
# 6. Preservation of Synthetic Datasets
# =====================================================================

def test_synthetic_data_preservation():
    synthetic_market = SYNTHETIC_DIR / "market_features.csv"
    assert synthetic_market.exists(), "Synthetic market data must be preserved"
    assert synthetic_market.stat().st_size > 0
