#!/usr/bin/env python3
"""
Master Orchestration Script — Real Financial Data Acquisition & Quality Audit.

Coordinates:
    1. Historical NIFTY 50 + ^NSEI daily OHLCV acquisition via Yahoo Finance.
    2. Real financial news, exchange disclosures, and GDELT acquisition.
    3. Official corporate reports and regulatory filings acquisition.
    4. Comprehensive data quality and financial invariance validation.
    5. Checksum manifest generation and preservation verification.

Usage:
    python scripts/run_real_data_acquisition.py --start 2015-01-01 --end 2025-01-01
"""

import argparse
import sys
from pathlib import Path

# Resolve project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.ingestion.real_market_collector import download_all_nifty50
from src.ingestion.real_news_collector import collect_corporate_documents, collect_news_dataset
from src.preprocessing.real_data_validator import run_full_data_quality_audit
from src.utils.logger import logger


def run_pipeline(
    start_date: str = "2015-01-01",
    end_date: str = "2025-01-01",
    delay: float = 0.5,
    skip_gdelt: bool = False,
    tickers=None,
):
    print("=" * 78)
    print(" FINANCIAL INTELLIGENCE & BIG DATA ANALYTICS PLATFORM")
    print(" STEP 5: REAL FINANCIAL DATA ACQUISITION & DATA ENGINEERING PIPELINE")
    print("=" * 78)

    # Step 1: Historical Market Data
    print(f"\n[1/4] Ingesting NIFTY 50 Daily OHLCV ({start_date} to {end_date})...")
    market_manifest = download_all_nifty50(
        start_date=start_date,
        end_date=end_date,
        inter_ticker_delay=delay,
        selected_tickers=tickers,
    )
    print(
        f"  ✓ Market acquisition finished: {market_manifest['summary']['successful']}/"
        f"{market_manifest['summary']['total_requested']} tickers successful, "
        f"{market_manifest['summary']['total_market_bars']:,} daily bars acquired."
    )

    # Step 2: Real Financial News & Announcements
    print("\n[2/4] Ingesting Real Financial News & Corporate Announcements...")
    news_manifest = collect_news_dataset(
        fetch_gdelt_live=not skip_gdelt,
        inter_request_delay=0.3,
    )
    print(
        f"  ✓ News acquisition finished: {news_manifest['total_articles']} articles "
        f"across {news_manifest['unique_tickers_covered']} tickers."
    )

    # Step 3: Corporate Documents & Filings
    print("\n[3/4] Ingesting Authentic Corporate Annual Reports & Filings...")
    docs_manifest = collect_corporate_documents()
    print(
        f"  ✓ Documents acquisition finished: {docs_manifest['total_documents']} documents "
        f"({docs_manifest['total_words']:,} words)."
    )

    # Step 4: Data Quality & Invariance Audit
    print("\n[4/4] Executing Comprehensive Data Quality Audit...")
    quality_report = run_full_data_quality_audit()
    print(f"  ✓ Overall Data Quality Audit Verdict: {quality_report['overall_status']}")
    print(
        f"    Market instruments passed: {quality_report['summary']['market_instruments_passed']}/"
        f"{quality_report['summary']['market_instruments_validated']}"
    )
    print(
        f"    Total market bars validated: {quality_report['summary']['total_market_bars_validated']:,}"
    )
    print(
        f"    Total news articles validated: {quality_report['summary']['news_articles_validated']}"
    )
    print(
        f"    Corporate filings validated: {quality_report['summary']['corporate_documents_validated']}"
    )

    print("\n" + "=" * 78)
    print(" REAL DATA ACQUISITION & VALIDATION COMPLETE")
    print("=" * 78)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Full Real Data Acquisition Pipeline")
    parser.add_argument("--start", default="2015-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2025-01-01", help="End date (YYYY-MM-DD)")
    parser.add_argument("--delay", type=float, default=0.5, help="Inter-ticker delay in seconds")
    parser.add_argument("--skip-gdelt-live", action="store_true", help="Skip live GDELT API calls")
    parser.add_argument("--tickers", nargs="*", default=None, help="Optional subset of tickers")
    args = parser.parse_args()

    run_pipeline(
        start_date=args.start,
        end_date=args.end,
        delay=args.delay,
        skip_gdelt=args.skip_gdelt_live,
        tickers=args.tickers,
    )

