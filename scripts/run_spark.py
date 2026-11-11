#!/usr/bin/env python3
"""
Run the PySpark analytics pipeline.

Usage:
    python scripts/run_spark.py
    python scripts/run_spark.py --input data/processed/market_features.csv
"""

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.bigdata.spark_jobs import SparkPipeline
from src.utils.logger import logger


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the PySpark financial analytics pipeline"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to market_features.csv (default: from config)",
    )
    parser.add_argument(
        "--master",
        type=str,
        default=None,
        help="Spark master URL (e.g. spark://hadoop-master:7077, local[*], yarn)",
    )
    args = parser.parse_args()

    # Load config
    with open(ROOT / "config" / "config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.master:
        cfg.setdefault("spark", {})["master"] = args.master

    logger.info("=" * 60)
    logger.info("Phase 2 — PySpark Analytics Pipeline")
    logger.info("=" * 60)

    # Check that input data exists
    if args.input:
        input_path = args.input
    else:
        input_path = str(ROOT / cfg["paths"]["processed_market"])

    if not Path(input_path).exists():
        logger.error(
            f"Input file not found: {input_path}\n"
            f"Run Phase 1 first:\n"
            f"  python scripts/generate_sample_data.py\n"
            f"  python scripts/ingest_data.py --source sample"
        )
        sys.exit(1)

    # Run pipeline
    pipeline = SparkPipeline(cfg)
    results = pipeline.run(filepath=input_path)

    # Summary
    logger.info("\n── Pipeline Summary ──")
    logger.info(f"  Stocks analysed  : {results['stock_count']}")
    logger.info(f"  Trading days     : {results['daily_count']}")
    logger.info(f"  Output directory : {cfg['spark']['output_path']}")
    for name, path in results["output_paths"].items():
        logger.info(f"  {name}: {path}")


if __name__ == "__main__":
    main()
