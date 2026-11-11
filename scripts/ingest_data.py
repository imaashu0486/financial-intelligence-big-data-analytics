#!/usr/bin/env python3
"""
Orchestration script — runs the full ingestion + preprocessing pipeline.

Usage:
    python scripts/ingest_data.py --source sample   # uses data/sample/
    python scripts/ingest_data.py --source raw       # uses data/raw/
"""

import argparse
import sys
from pathlib import Path

import yaml

# ── Resolve project root ─────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.ingestion.document_loader import list_documents
from src.ingestion.market_loader import load_market_data
from src.ingestion.news_loader import load_news_data
from src.preprocessing.cleaning import clean_market_data, clean_news_data
from src.preprocessing.feature_engineering import engineer_features
from src.utils.logger import logger


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the data ingestion & preprocessing pipeline"
    )
    parser.add_argument(
        "--source",
        choices=["sample", "raw"],
        default="sample",
        help="Data source directory (default: sample)",
    )
    args = parser.parse_args()

    # Load config
    with open(ROOT / "config" / "config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    paths = cfg["paths"]
    source = args.source

    # Resolve input paths
    market_path = ROOT / paths[f"{source}_market"]
    news_path = ROOT / paths[f"{source}_news"]
    docs_dir = ROOT / paths["documents_dir"]

    # Resolve output paths
    out_dir = ROOT / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"{'=' * 60}")
    logger.info(f"Pipeline started — source: {source}")
    logger.info(f"{'=' * 60}")

    # ── 1. Load ──────────────────────────────────────────────
    logger.info("▸ Step 1/4: Loading data")
    market_df = load_market_data(market_path)
    news_df = load_news_data(news_path)
    docs = list_documents(docs_dir)

    # ── 2. Clean ─────────────────────────────────────────────
    logger.info("▸ Step 2/4: Cleaning data")
    market_df = clean_market_data(market_df, cfg["cleaning"])
    news_df = clean_news_data(news_df)

    # ── 3. Feature Engineering ───────────────────────────────
    logger.info("▸ Step 3/4: Engineering features")
    market_df = engineer_features(market_df, cfg["features"])

    # ── 4. Save ──────────────────────────────────────────────
    logger.info("▸ Step 4/4: Saving processed data")

    market_out = out_dir / "market_features.csv"
    market_df.to_csv(market_out, index=False)
    logger.info(f"  ✅ {market_out}  ({len(market_df):,} rows, {len(market_df.columns)} cols)")

    news_out = out_dir / "news_cleaned.csv"
    news_df.to_csv(news_out, index=False)
    logger.info(f"  ✅ {news_out}  ({len(news_df):,} rows)")

    # Summary
    logger.info(f"{'=' * 60}")
    logger.info("Pipeline complete ✨")
    logger.info(f"  Market features : {market_df.shape}")
    logger.info(f"  News cleaned    : {news_df.shape}")
    logger.info(f"  Documents found : {len(docs)}")
    logger.info(f"  Output dir      : {out_dir}")
    logger.info(f"{'=' * 60}")


if __name__ == "__main__":
    main()
