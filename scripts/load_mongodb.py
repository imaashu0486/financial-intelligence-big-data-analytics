#!/usr/bin/env python3
"""
Data Ingestion Script: Populate MongoDB from Processed CSV & Parquet Datasets.

Usage:
    python scripts/load_mongodb.py
    python scripts/load_mongodb.py --drop-existing
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from pymongo import UpdateOne
from pymongo.database import Database

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.database.aggregations import (
    aggregate_avg_return_by_stock,
    aggregate_avg_volatility_by_stock,
    aggregate_total_volume_by_stock,
)
from src.database.indexes import create_indexes
from src.database.mongodb import check_connection, get_client, get_database, load_mongo_config
from src.database.queries import (
    get_highest_volatility_stocks,
    get_latest_stock_data,
    get_market_summary,
    get_top_stocks_by_return,
)
from src.database.schemas import (
    market_record_to_doc,
    news_record_to_doc,
    stock_feature_record_to_doc,
)
from src.utils.logger import logger


def _execute_bulk_ops(coll, operations: List[UpdateOne], stats: Dict[str, int]) -> None:
    """Execute bulk operations with fallback for mongomock environments."""
    if not operations:
        return
    try:
        result = coll.bulk_write(operations, ordered=False)
        stats["upserted"] += len(result.upserted_ids) if result.upserted_ids else 0
        stats["matched"] += result.matched_count
        stats["modified"] += result.modified_count
    except TypeError:
        # Fallback for mock environments (e.g. mongomock compatibility)
        for op in operations:
            filter_q = op._filter
            inner = op._doc.get("") or op._doc.get("$set") or op._doc
            upsert = bool(op._upsert)
            res = coll.update_one(filter_q, {"$set": inner}, upsert=upsert)
            if res.upserted_id is not None:
                stats["upserted"] += 1
            else:
                stats["matched"] += 1
    operations.clear()


def load_market_data_to_mongo(
    db: Database,
    csv_path: Path,
    batch_size: int = 1000,
) -> Dict[str, int]:
    """
    Import market features CSV into 'market_data' collection using bulk upsert.
    Uniqueness key: (symbol, date)
    """
    if not csv_path.exists():
        logger.error(f"Market features file not found: {csv_path}")
        return {"inserted": 0, "upserted": 0, "matched": 0}

    logger.info(f"Loading market data from {csv_path}...")
    df = pd.read_csv(csv_path)
    coll = db["market_data"]

    operations: List[UpdateOne] = []
    stats = {"upserted": 0, "matched": 0, "modified": 0}

    for _, row in df.iterrows():
        doc = market_record_to_doc(row.to_dict())
        filter_spec = {"symbol": doc["symbol"], "date": doc["date"]}
        operations.append(UpdateOne(filter_spec, {"$set": doc}, upsert=True))

        if len(operations) >= batch_size:
            _execute_bulk_ops(coll, operations, stats)

    _execute_bulk_ops(coll, operations, stats)

    total_in_coll = coll.count_documents({})
    logger.info(
        f"  Market data import: {stats['upserted']} newly upserted, "
        f"{stats['matched']} matched existing. Total in collection: {total_in_coll}"
    )
    return stats


def load_news_to_mongo(
    db: Database,
    csv_path: Path,
    batch_size: int = 500,
) -> Dict[str, int]:
    """
    Import cleaned news CSV into 'news' collection with flexible schema.
    Uniqueness key: (symbol, date, title)
    """
    if not csv_path.exists():
        logger.error(f"News data file not found: {csv_path}")
        return {"upserted": 0, "matched": 0}

    logger.info(f"Loading news data from {csv_path}...")
    df = pd.read_csv(csv_path)
    coll = db["news"]

    operations: List[UpdateOne] = []
    stats = {"upserted": 0, "matched": 0, "modified": 0}

    for _, row in df.iterrows():
        doc = news_record_to_doc(row.to_dict())
        filter_spec = {
            "symbol": doc.get("symbol"),
            "date": doc.get("date"),
            "title": doc.get("title"),
        }
        operations.append(UpdateOne(filter_spec, {"$set": doc}, upsert=True))

        if len(operations) >= batch_size:
            _execute_bulk_ops(coll, operations, stats)

    _execute_bulk_ops(coll, operations, stats)

    total_in_coll = coll.count_documents({})
    logger.info(
        f"  News data import: {stats['upserted']} newly upserted, "
        f"{stats['matched']} matched existing. Total in collection: {total_in_coll}"
    )
    return stats


def load_stock_features_to_mongo(
    db: Database,
    spark_dir: Path,
) -> Dict[str, int]:
    """
    Import aggregated stock analytics Parquet from PySpark output into 'stock_features'.
    Uniqueness key: (symbol)
    """
    if not spark_dir.exists():
        logger.error(f"Spark output directory not found: {spark_dir}")
        return {"upserted": 0, "matched": 0}

    # Locate parquet files in directory
    parquet_files = list(spark_dir.glob("*.parquet"))
    if not parquet_files:
        logger.error(f"No parquet files found in {spark_dir}")
        return {"upserted": 0, "matched": 0}

    logger.info(f"Loading Spark stock features from {spark_dir}...")
    df = pd.read_parquet(spark_dir)
    coll = db["stock_features"]

    operations: List[UpdateOne] = []
    stats = {"upserted": 0, "matched": 0, "modified": 0}

    for _, row in df.iterrows():
        doc = stock_feature_record_to_doc(row.to_dict())
        filter_spec = {"symbol": doc["symbol"]}
        operations.append(UpdateOne(filter_spec, {"$set": doc}, upsert=True))

    _execute_bulk_ops(coll, operations, stats)

    total_in_coll = coll.count_documents({})
    logger.info(
        f"  Stock features import: {stats['upserted']} newly upserted, "
        f"{stats['matched']} matched existing. Total in collection: {total_in_coll}"
    )
    return stats


def run_verification(db: Database) -> None:
    """Run verification queries and aggregations to confirm database state."""
    logger.info("\n" + "=" * 60)
    logger.info("MongoDB Verification & Health Check Queries")
    logger.info("=" * 60)

    # 1. Market Summary
    summary = get_market_summary(db=db)
    logger.info(f"Market Summary:")
    logger.info(f"  Total Market Records : {summary['total_market_records']}")
    logger.info(f"  Total News Records   : {summary['total_news_records']}")
    logger.info(f"  Tracked Symbols ({summary['symbol_count']}): {', '.join(summary['symbols'])}")
    logger.info(f"  Date Range           : {summary['date_range']['start']} → {summary['date_range']['end']}")
    logger.info(f"  Total Market Volume  : {summary['total_market_volume']:,}")
    logger.info(f"  Stock Features Count : {summary['stock_features_count']}")

    # 2. Latest Record Example
    tcs_latest = get_latest_stock_data("TCS", db=db)
    if tcs_latest:
        logger.info(f"\nLatest Record for TCS:")
        logger.info(f"  Date: {tcs_latest['date']} | Close: ₹{tcs_latest['close']} | Vol: {tcs_latest['volume']:,} | RSI: {tcs_latest.get('rsi_14')}")

    # 3. Aggregation Example 1: Average return by stock
    logger.info("\nAggregation 1: Top 5 Stocks by Average Daily Return (MongoDB $group pipeline):")
    top_returns = aggregate_avg_return_by_stock(db=db)[:5]
    for item in top_returns:
        logger.info(f"  {item['symbol']:<12} Avg Return: {item['avg_return']:.6f} ({item['record_count']} days)")

    # 4. Aggregation Example 2: Total trading volume by stock
    logger.info("\nAggregation 2: Top 5 Stocks by Total Trading Volume (MongoDB $group pipeline):")
    top_volumes = aggregate_total_volume_by_stock(db=db)[:5]
    for item in top_volumes:
        logger.info(f"  {item['symbol']:<12} Total Vol: {item['total_volume']:,} | Daily Avg: {int(item['avg_daily_volume']):,}")

    # 5. Aggregation Example 3: Average volatility by stock
    logger.info("\nAggregation 3: Top 5 Stocks by Volatility (MongoDB $group pipeline):")
    top_vol = aggregate_avg_volatility_by_stock(db=db)[:5]
    for item in top_vol:
        logger.info(f"  {item['symbol']:<12} Avg Volatility: {item['avg_volatility']:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Populate MongoDB with processed financial and news data")
    parser.add_argument("--drop-existing", action="store_true", help="Drop collections before reloading")
    parser.add_argument("--mock", action="store_true", help="Run with in-memory mock database (for testing when MongoDB is unavailable)")
    args = parser.parse_args()

    mongo_cfg = load_mongo_config()
    logger.info("=" * 60)
    logger.info("Phase 3 — MongoDB Data Layer Ingestion")
    if args.mock:
        logger.info("Mode       : In-memory simulation (--mock)")
    else:
        logger.info(f"Target URI : {mongo_cfg['uri']}")
    logger.info(f"Database   : {mongo_cfg['database']}")
    logger.info("=" * 60)

    if args.mock:
        import mongomock
        mock_client = mongomock.MongoClient()
        db = mock_client[mongo_cfg["database"]]
    else:
        # 1. Check MongoDB connectivity
        health = check_connection()
        if not health["connected"]:
            logger.error(
                f"\n❌ Could not connect to MongoDB at {mongo_cfg['uri']}.\n"
                f"Error details: {health['error']}\n\n"
                f"To start MongoDB on your system:\n"
                f"  Option 1 (Docker):  docker-compose up -d mongodb\n"
                f"  Option 2 (Native):  net start MongoDB  (or start mongod.exe)\n"
                f"  Option 3 (Remote):  Set MONGODB_URI in .env or config.yaml\n"
                f"  Option 4 (Testing): python scripts/load_mongodb.py --mock\n"
            )
            sys.exit(1)

        db = get_database()

    # Drop existing collections if requested
    if args.drop_existing:
        logger.warning("Dropping existing collections as requested...")
        for col_name in ["market_data", "news", "stock_features"]:
            db[col_name].drop()
            logger.info(f"  Dropped collection '{col_name}'")

    # 2. Create Indexes
    logger.info("\nCreating / verifying MongoDB indexes...")
    create_indexes(db=db)

    # 3. Load Datasets
    market_csv = ROOT / "data" / "processed" / "market_features.csv"
    news_csv = ROOT / "data" / "processed" / "news_cleaned.csv"
    spark_parquet = ROOT / "data" / "processed" / "spark" / "stock_statistics"

    load_market_data_to_mongo(db=db, csv_path=market_csv)
    load_news_to_mongo(db=db, csv_path=news_csv)
    load_stock_features_to_mongo(db=db, spark_dir=spark_parquet)

    # 4. Run verification queries
    run_verification(db=db)

    logger.info("\n" + "=" * 60)
    logger.info("Phase 3 — MongoDB Ingestion Complete ✨")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
