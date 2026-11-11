"""
Independent Verification and Mathematical Validation Engine.

Compares MapReduce distributed outputs, Apache Spark distributed outputs,
and independent local pandas/numpy calculations on the real financial dataset.
Generates a comprehensive validation report at:
data/real/distributed/validation/distributed_validation_report.json
"""

from __future__ import annotations

import csv
import json
import logging
import math
from pathlib import Path
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
INPUT_PARQUET = PROJECT_ROOT / "data" / "real" / "processed" / "market" / "consolidated_market_features.parquet"
INPUT_CSV = PROJECT_ROOT / "data" / "real" / "processed" / "market" / "consolidated_market_features.csv"

MR_DIR = PROJECT_ROOT / "data" / "real" / "distributed" / "mapreduce"
SPARK_DIR = PROJECT_ROOT / "data" / "real" / "distributed" / "spark"
VAL_DIR = PROJECT_ROOT / "data" / "real" / "distributed" / "validation"
VAL_DIR.mkdir(parents=True, exist_ok=True)


def validate_all() -> dict:
    logger.info("Loading ground-truth real market data with pandas...")
    df = pd.read_parquet(INPUT_PARQUET)
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year

    checks = []

    # -------------------------------------------------------------
    # Check 1: Volume by Ticker (MapReduce Job 1 vs Pandas Ground Truth)
    # -------------------------------------------------------------
    logger.info("Validating Check 1: Volume by Ticker (MapReduce Job 1)...")
    mr1_path = MR_DIR / "job1_volume_by_ticker.tsv"
    mr1_df = pd.read_csv(mr1_path, sep="\t")
    pd_vol = df.groupby("symbol")["volume"].agg(["sum", "count"]).reset_index()

    merged1 = pd.merge(mr1_df, pd_vol, on="symbol", suffixes=("_mr", "_gt"))
    vol_match = (merged1["total_volume"] == merged1["sum"]).all()
    count_match = (merged1["record_count"] == merged1["count"]).all()
    checks.append({
        "check_id": "MR_JOB_1_VOLUME",
        "description": "MapReduce 10-year volume by ticker matches pandas ground truth exactly",
        "passed": bool(vol_match and count_match),
        "total_tickers": len(merged1),
        "mismatches": int((merged1["total_volume"] != merged1["sum"]).sum()),
    })

    # -------------------------------------------------------------
    # Check 2: Cross-Sectional Daily Market Return (MapReduce Job 2 vs Ground Truth)
    # -------------------------------------------------------------
    logger.info("Validating Check 2: Daily Market Return (MapReduce Job 2)...")
    mr2_path = MR_DIR / "job2_daily_market_return.tsv"
    mr2_df = pd.read_csv(mr2_path, sep="\t")
    mr2_df["date"] = pd.to_datetime(mr2_df["date"])

    pd_daily = df.dropna(subset=["daily_adjusted_return"]).groupby("date")["daily_adjusted_return"].agg(["mean", "count"]).reset_index()
    merged2 = pd.merge(mr2_df, pd_daily, on="date", suffixes=("_mr", "_gt"))
    diff2 = (merged2["mean_daily_return"] - merged2["mean"]).abs()
    max_diff2 = float(diff2.max())
    passed2 = max_diff2 < 1e-6 and (merged2["constituent_count"] == merged2["count"]).all()
    checks.append({
        "check_id": "MR_JOB_2_DAILY_RETURN",
        "description": "MapReduce cross-sectional daily market return matches ground truth (tol=1e-6)",
        "passed": bool(passed2),
        "total_dates": len(merged2),
        "max_absolute_error": max_diff2,
    })

    # -------------------------------------------------------------
    # Check 3: Company Average Return (MapReduce Job 3 vs Spark Op 1 vs Ground Truth)
    # -------------------------------------------------------------
    logger.info("Validating Check 3: Company Average Return (MR Job 3 vs Spark Op 1)...")
    mr3_path = MR_DIR / "job3_company_avg_return.tsv"
    mr3_df = pd.read_csv(mr3_path, sep="\t")

    spark1_path = SPARK_DIR / "ticker_avg_returns.parquet"
    spark1_df = pd.read_parquet(spark1_path)

    pd_ret = df.dropna(subset=["daily_adjusted_return"]).groupby("symbol")["daily_adjusted_return"].agg(["mean", "count"]).reset_index()

    m_mr = pd.merge(mr3_df, pd_ret, on="symbol")
    max_diff_mr = float((m_mr["mean_daily_return"] - m_mr["mean"]).abs().max())

    m_sp = pd.merge(spark1_df, pd_ret, on="symbol")
    max_diff_sp = float((m_sp["mean_daily_return"] - m_sp["mean"]).abs().max())

    passed3 = max_diff_mr < 1e-6 and max_diff_sp < 1e-6
    checks.append({
        "check_id": "MR_SPARK_COMPANY_AVG_RETURN",
        "description": "MapReduce Job 3 and Spark Op 1 average returns match ground truth",
        "passed": bool(passed3),
        "max_err_mr_vs_gt": max_diff_mr,
        "max_err_spark_vs_gt": max_diff_sp,
    })

    # -------------------------------------------------------------
    # Check 4: Yearly Volume (MapReduce Job 4 vs Spark Op 5 vs Ground Truth)
    # -------------------------------------------------------------
    logger.info("Validating Check 4: Yearly Volume Aggregations...")
    mr4_path = MR_DIR / "job4_yearly_volume.tsv"
    mr4_df = pd.read_csv(mr4_path, sep="\t")

    spark5_path = SPARK_DIR / "yearly_market_summary.parquet"
    spark5_df = pd.read_parquet(spark5_path)

    pd_yr_market = df.groupby("year")["volume"].sum().reset_index()
    m_yr_sp = pd.merge(spark5_df, pd_yr_market, on="year")
    spark_yr_match = (m_yr_sp["total_yearly_volume"] == m_yr_sp["volume"]).all()

    pd_yr_ticker = df.groupby(["symbol", "year"])["volume"].sum().reset_index()
    m_yr_mr = pd.merge(mr4_df, pd_yr_ticker, on=["symbol", "year"])
    mr_yr_match = (m_yr_mr["total_volume"] == m_yr_mr["volume"]).all()

    checks.append({
        "check_id": "YEARLY_VOLUME_CONSISTENCY",
        "description": "Yearly market totals in Spark and per-ticker yearly totals in MapReduce match ground truth",
        "passed": bool(spark_yr_match and mr_yr_match),
        "spark_yearly_totals_match": bool(spark_yr_match),
        "mapreduce_ticker_yearly_totals_match": bool(mr_yr_match),
    })

    # -------------------------------------------------------------
    # Check 5: Market Data Quality Summary (MapReduce Job 5)
    # -------------------------------------------------------------
    logger.info("Validating Check 5: Data Quality Summary (MapReduce Job 5)...")
    mr5_path = MR_DIR / "job5_market_quality_summary.tsv"
    mr5_dict = dict(zip(pd.read_csv(mr5_path, sep="\t")["metric_name"], pd.read_csv(mr5_path, sep="\t")["total_count"]))

    gt_rows = len(df)
    gt_valid_ohlc = ((df["high"] >= df["low"]) & (df["low"] > 0) & (df["open"] > 0) & (df["close"] > 0)).sum()
    gt_missing_ret = df["daily_adjusted_return"].isna().sum()

    quality_passed = (
        mr5_dict.get("TOTAL_ROWS") == gt_rows
        and mr5_dict.get("VALID_OHLC") == gt_valid_ohlc
        and mr5_dict.get("MISSING_RETURN") == gt_missing_ret
    )
    checks.append({
        "check_id": "MR_DATA_QUALITY_AUDIT",
        "description": "MapReduce data quality summary matches exact dataframe integrity counts",
        "passed": bool(quality_passed),
        "total_rows": int(mr5_dict.get("TOTAL_ROWS", 0)),
        "valid_ohlc": int(mr5_dict.get("VALID_OHLC", 0)),
        "missing_return": int(mr5_dict.get("MISSING_RETURN", 0)),
    })

    # -------------------------------------------------------------
    # Check 6: Spark Volatility and Sharpe Ratio Rankings
    # -------------------------------------------------------------
    logger.info("Validating Check 6: Spark Volatility & Sharpe Rankings...")
    spark_vol_path = SPARK_DIR / "ticker_volatility.parquet"
    spark_vol = pd.read_parquet(spark_vol_path)

    pd_vol_gt = (
        df.dropna(subset=["daily_adjusted_return"])
        .groupby("symbol")["daily_adjusted_return"]
        .std(ddof=1)
        .reset_index()
    )
    pd_vol_gt["annualized_vol"] = pd_vol_gt["daily_adjusted_return"] * math.sqrt(252)

    m_vol = pd.merge(spark_vol, pd_vol_gt, on="symbol")
    vol_err = float((m_vol["annualized_volatility"] - m_vol["annualized_vol"]).abs().max())

    spark_rank_path = SPARK_DIR / "stock_rankings.parquet"
    spark_rank = pd.read_parquet(spark_rank_path)
    top_sharpe_spark = list(spark_rank.sort_values("rank_sharpe")["symbol"].head(5))

    # Calculate GT sharpe
    gt_perf = df.dropna(subset=["daily_adjusted_return"]).groupby("symbol")["daily_adjusted_return"].agg(["mean", "std"]).reset_index()
    gt_perf["ann_ret"] = gt_perf["mean"] * 252.0
    gt_perf["ann_vol"] = gt_perf["std"] * math.sqrt(252)
    gt_perf["sharpe"] = gt_perf["ann_ret"] / gt_perf["ann_vol"]
    top_sharpe_gt = list(gt_perf.sort_values("sharpe", ascending=False)["symbol"].head(5))

    ranking_match = top_sharpe_spark == top_sharpe_gt
    checks.append({
        "check_id": "SPARK_VOLATILITY_AND_SHARPE_RANKING",
        "description": "Spark annualized volatility and top 5 Sharpe ratio rankings match ground truth",
        "passed": bool(vol_err < 1e-6 and ranking_match),
        "max_volatility_error": vol_err,
        "top_5_sharpe_spark": top_sharpe_spark,
        "top_5_sharpe_ground_truth": top_sharpe_gt,
    })

    all_passed = all(c["passed"] for c in checks)
    report = {
        "validation_timestamp": pd.Timestamp.now().isoformat(),
        "all_checks_passed": all_passed,
        "verdict": "PASS" if all_passed else "FAIL",
        "total_checks": len(checks),
        "passed_checks": sum(1 for c in checks if c["passed"]),
        "failed_checks": sum(1 for c in checks if not c["passed"]),
        "checks": checks,
    }

    report_path = VAL_DIR / "distributed_validation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Validation complete: verdict={report['verdict']}. Report written to {report_path}")
    return report


if __name__ == "__main__":
    rep = validate_all()
    print(json.dumps(rep, indent=2))

