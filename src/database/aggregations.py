"""
MongoDB Aggregation Pipelines for Financial Analytics.

Implements server-side data processing pipelines in MongoDB:
1. Average Daily Return by Stock: ($match, $group, $project, $sort)
2. Total Trading Volume by Stock: ($match, $group, $sort)
3. Average Volatility by Stock: ($match, $group, $sort)
4. Comprehensive Multi-Metric Aggregation Pipeline
"""

from typing import Any, Dict, List, Optional
from pymongo import DESCENDING
from pymongo.database import Database

from src.database.mongodb import get_database


def build_avg_return_pipeline() -> List[Dict[str, Any]]:
    """
    Construct the aggregation pipeline for Average Daily Return by Stock.

    Stages:
    1. $match: Exclude null returns
    2. $group: Group by symbol and compute $avg of daily_return
    3. $project: Format fields cleanly
    4. $sort: Order descending by avg_return
    """
    return [
        {"$match": {"daily_return": {"$ne": None}}},
        {
            "$group": {
                "_id": "$symbol",
                "avg_return": {"$avg": "$daily_return"},
                "record_count": {"$sum": 1},
            }
        },
        {
            "$project": {
                "_id": 0,
                "symbol": "$_id",
                "avg_return": 1,
                "record_count": 1,
            }
        },
        {"$sort": {"avg_return": DESCENDING}},
    ]


def build_total_volume_pipeline() -> List[Dict[str, Any]]:
    """
    Construct the aggregation pipeline for Total Trading Volume by Stock.

    Stages:
    1. $match: Filter positive volume records
    2. $group: Group by symbol and compute $sum and $avg of volume
    3. $project: Format fields
    4. $sort: Order descending by total_volume
    """
    return [
        {"$match": {"volume": {"$gt": 0}}},
        {
            "$group": {
                "_id": "$symbol",
                "total_volume": {"$sum": "$volume"},
                "avg_daily_volume": {"$avg": "$volume"},
                "trading_days": {"$sum": 1},
            }
        },
        {
            "$project": {
                "_id": 0,
                "symbol": "$_id",
                "total_volume": 1,
                "avg_daily_volume": 1,
                "trading_days": 1,
            }
        },
        {"$sort": {"total_volume": DESCENDING}},
    ]


def build_avg_volatility_pipeline() -> List[Dict[str, Any]]:
    """
    Construct the aggregation pipeline for Average Volatility by Stock.

    Stages:
    1. $match: Filter records where volatility is calculated
    2. $group: Group by symbol and compute $avg of volatility
    3. $project: Format fields
    4. $sort: Order descending by avg_volatility
    """
    return [
        {"$match": {"volatility": {"$ne": None}}},
        {
            "$group": {
                "_id": "$symbol",
                "avg_volatility": {"$avg": "$volatility"},
                "min_volatility": {"$min": "$volatility"},
                "max_volatility": {"$max": "$volatility"},
            }
        },
        {
            "$project": {
                "_id": 0,
                "symbol": "$_id",
                "avg_volatility": 1,
                "min_volatility": 1,
                "max_volatility": 1,
            }
        },
        {"$sort": {"avg_volatility": DESCENDING}},
    ]


def build_comprehensive_summary_pipeline() -> List[Dict[str, Any]]:
    """
    Multi-stage pipeline computing comprehensive statistics per stock directly in MongoDB:
    - avg_daily_return
    - avg_volatility
    - total_volume
    - max_price
    - min_price
    - trading_days
    """
    return [
        {"$match": {"close": {"$gt": 0}}},
        {
            "$group": {
                "_id": "$symbol",
                "avg_daily_return": {"$avg": "$daily_return"},
                "avg_log_return": {"$avg": "$log_return"},
                "avg_volatility": {"$avg": "$volatility"},
                "total_volume": {"$sum": "$volume"},
                "avg_volume": {"$avg": "$volume"},
                "max_price": {"$max": "$high"},
                "min_price": {"$min": "$low"},
                "trading_days": {"$sum": 1},
            }
        },
        {
            "$project": {
                "_id": 0,
                "symbol": "$_id",
                "avg_daily_return": 1,
                "avg_log_return": 1,
                "avg_volatility": 1,
                "total_volume": 1,
                "avg_volume": 1,
                "max_price": 1,
                "min_price": 1,
                "trading_days": 1,
            }
        },
        {"$sort": {"avg_daily_return": DESCENDING}},
    ]


def aggregate_avg_return_by_stock(
    db: Optional[Database] = None,
) -> List[Dict[str, Any]]:
    """Execute Aggregation Example 1: Average return by stock."""
    active_db = db if db is not None else get_database()
    pipeline = build_avg_return_pipeline()
    return list(active_db["market_data"].aggregate(pipeline))


def aggregate_total_volume_by_stock(
    db: Optional[Database] = None,
) -> List[Dict[str, Any]]:
    """Execute Aggregation Example 2: Total trading volume by stock."""
    active_db = db if db is not None else get_database()
    pipeline = build_total_volume_pipeline()
    return list(active_db["market_data"].aggregate(pipeline))


def aggregate_avg_volatility_by_stock(
    db: Optional[Database] = None,
) -> List[Dict[str, Any]]:
    """Execute Aggregation Example 3: Average volatility by stock."""
    active_db = db if db is not None else get_database()
    pipeline = build_avg_volatility_pipeline()
    return list(active_db["market_data"].aggregate(pipeline))


def aggregate_comprehensive_metrics(
    db: Optional[Database] = None,
) -> List[Dict[str, Any]]:
    """Execute comprehensive multi-stage aggregation pipeline."""
    active_db = db if db is not None else get_database()
    pipeline = build_comprehensive_summary_pipeline()
    return list(active_db["market_data"].aggregate(pipeline))
