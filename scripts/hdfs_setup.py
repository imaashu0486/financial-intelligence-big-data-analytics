#!/usr/bin/env python3
"""
HDFS setup script — check availability, create directories, upload data.

Usage:
    python scripts/hdfs_setup.py
    python scripts/hdfs_setup.py --upload
"""

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.bigdata.hdfs_manager import HDFSManager
from src.utils.logger import logger


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Setup HDFS directories and optionally upload data"
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload processed data to HDFS after creating directories",
    )
    args = parser.parse_args()

    with open(ROOT / "config" / "config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    logger.info("=" * 60)
    logger.info("HDFS Setup")
    logger.info("=" * 60)

    mgr = HDFSManager(cfg["hdfs"])

    # Check availability
    if not mgr.is_available():
        logger.info("")
        logger.info("HDFS is not available. The project works in local mode.")
        logger.info("")
        logger.info("To enable HDFS, you can:")
        logger.info("  1. Install Hadoop locally and start HDFS:")
        logger.info("     $HADOOP_HOME/sbin/start-dfs.sh")
        logger.info("")
        logger.info("  2. Or use Docker:")
        logger.info("     docker-compose up -d")
        logger.info("")
        logger.info("The Spark pipeline and MapReduce jobs work without HDFS.")
        logger.info("HDFS adds distributed storage for cluster deployments.")
        logger.info("=" * 60)
        return

    # Create project directories
    logger.info("\n▸ Creating HDFS project directories...")
    mgr.create_project_dirs()

    # Optionally upload data
    if args.upload:
        logger.info("\n▸ Uploading data to HDFS...")

        files_to_upload = [
            (ROOT / cfg["paths"]["processed_market"], cfg["hdfs"]["raw_path"] + "/market_features.csv"),
            (ROOT / cfg["paths"]["processed_news"], cfg["hdfs"]["raw_path"] + "/news_cleaned.csv"),
        ]

        for local, hdfs_dest in files_to_upload:
            if local.exists():
                mgr.upload_file(str(local), hdfs_dest)
            else:
                logger.warning(f"  Skipping (not found): {local}")

    # Summary
    summary = mgr.get_summary()
    logger.info(f"\n── HDFS Summary ──")
    logger.info(f"  Namenode : {summary['namenode']}")
    logger.info(f"  Available: {summary['available']}")
    if summary["available"]:
        logger.info(f"  Raw files: {len(summary.get('raw_files', []))}")
        logger.info(f"  Processed: {len(summary.get('processed_files', []))}")

    logger.info("=" * 60)


if __name__ == "__main__":
    main()
