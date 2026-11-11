"""
Reusable query functions for MongoDB operational data layer.

Provides optimized query access with projections, index utilization,
and server-side sorting/filtering to minimize memory overhead:
- get_stock_history
- get_latest_stock_data
- get_stock_statistics
- get_top_stocks_by_return
- get_highest_volatility_stocks
- get_news_for_stock
- get_market_summary
"""

from typing import Any, Dict, List, Optional
from pymongo import ASCENDING, DESCENDING
from pymongo.database import Database

from src.database.mongodb import get_database


def _clean_doc(doc: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Remove MongoDB internal _id field from result for clean serialization."""
    if doc is None:
        return None
    d = dict(doc)
    d.pop("_id", None)
    return d


def get_stock_history(
    symbol: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: Optional[int] = None,
    db: Optional[Database] = None,
) -> List[Dict[str, Any]]:
    """
    Retrieve historical price and feature records for a stock.

    Uses index: (symbol, date)

    Parameters
    ----------
    symbol : str
        Stock symbol (e.g. 'TCS').
    start_date : str, optional
        Filter records on or after YYYY-MM-DD.
    end_date : str, optional
        Filter records on or before YYYY-MM-DD.
    limit : int, optional
        Maximum number of records to return.
    db : Database, optional
        Database instance.
    """
    active_db = db if db is not None else get_database()
    coll = active_db["market_data"]

    filter_query: Dict[str, Any] = {"symbol": symbol.upper().strip()}

    if start_date and end_date:
        filter_query["date"] = {"$gte": str(start_date), "$lte": str(end_date)}
    elif start_date:
        filter_query["date"] = {"$gte": str(start_date)}
    elif end_date:
        filter_query["date"] = {"$lte": str(end_date)}

    cursor = coll.find(filter_query, {"_id": 0}).sort("date", ASCENDING)
    if limit is not None and limit > 0:
        cursor = cursor.limit(limit)

    return list(cursor)


def get_latest_stock_data(
    symbol: str,
    db: Optional[Database] = None,
) -> Optional[Dict[str, Any]]:
    """
    Retrieve the most recent market data record for a given stock symbol.

    Uses index: (symbol, date) descending.
    """
    active_db = db if db is not None else get_database()
    coll = active_db["market_data"]

    doc = coll.find_one(
        {"symbol": symbol.upper().strip()},
        {"_id": 0},
        sort=[("date", DESCENDING)],
    )
    return _clean_doc(doc)


def get_stock_statistics(
    symbol: str,
    db: Optional[Database] = None,
) -> Optional[Dict[str, Any]]:
    """
    Retrieve precomputed analytical statistics from the stock_features collection.

    Uses index: (symbol) unique.
    """
    active_db = db if db is not None else get_database()
    coll = active_db["stock_features"]

    doc = coll.find_one({"symbol": symbol.upper().strip()}, {"_id": 0})
    return _clean_doc(doc)


def get_top_stocks_by_return(
    limit: int = 10,
    db: Optional[Database] = None,
) -> List[Dict[str, Any]]:
    """
    Retrieve stocks with the highest average daily returns.

    Queries stock_features collection with server-side sort.
    """
    active_db = db if db is not None else get_database()
    coll = active_db["stock_features"]

    cursor = coll.find(
        {"avg_daily_return": {"$ne": None}},
        {"_id": 0},
    ).sort("avg_daily_return", DESCENDING).limit(limit)

    return list(cursor)


def get_highest_volatility_stocks(
    limit: int = 10,
    db: Optional[Database] = None,
) -> List[Dict[str, Any]]:
    """
    Retrieve stocks with the highest average volatility.

    Queries stock_features collection using volatility index.
    """
    active_db = db if db is not None else get_database()
    coll = active_db["stock_features"]

    cursor = coll.find(
        {"avg_volatility": {"$ne": None}},
        {"_id": 0},
    ).sort("avg_volatility", DESCENDING).limit(limit)

    return list(cursor)


def get_news_for_stock(
    symbol: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: Optional[int] = 20,
    db: Optional[Database] = None,
) -> List[Dict[str, Any]]:
    """
    Retrieve chronological news records for a specific stock ticker.

    Uses index: (symbol, date).
    """
    active_db = db if db is not None else get_database()
    coll = active_db["news"]

    filter_query: Dict[str, Any] = {"symbol": symbol.upper().strip()}

    if start_date and end_date:
        filter_query["date"] = {"$gte": str(start_date), "$lte": str(end_date)}
    elif start_date:
        filter_query["date"] = {"$gte": str(start_date)}
    elif end_date:
        filter_query["date"] = {"$lte": str(end_date)}

    cursor = coll.find(filter_query, {"_id": 0}).sort("date", DESCENDING)
    if limit is not None and limit > 0:
        cursor = cursor.limit(limit)

    return list(cursor)


def get_market_summary(
    db: Optional[Database] = None,
) -> Dict[str, Any]:
    """
    Compute high-level market summary across the database:
    - total market data rows
    - total news articles
    - tracked symbols
    - earliest and latest trading dates
    - total trading volume across market
    """
    active_db = db if db is not None else get_database()
    market_coll = active_db["market_data"]
    news_coll = active_db["news"]
    features_coll = active_db["stock_features"]

    total_market_records = market_coll.count_documents({})
    total_news_records = news_coll.count_documents({})
    symbols = sorted(market_coll.distinct("symbol"))

    earliest_date = None
    latest_date = None

    first_doc = market_coll.find_one({}, {"date": 1, "_id": 0}, sort=[("date", ASCENDING)])
    last_doc = market_coll.find_one({}, {"date": 1, "_id": 0}, sort=[("date", DESCENDING)])

    if first_doc:
        earliest_date = first_doc.get("date")
    if last_doc:
        latest_date = last_doc.get("date")

    # Aggregate total market-wide volume
    volume_pipeline = [
        {"$match": {"volume": {"$gt": 0}}},
        {"$group": {"_id": None, "total_market_volume": {"$sum": "$volume"}}},
    ]
    vol_res = list(market_coll.aggregate(volume_pipeline))
    total_volume = vol_res[0]["total_market_volume"] if vol_res else 0

    return {
        "total_market_records": total_market_records,
        "total_news_records": total_news_records,
        "symbol_count": len(symbols),
        "symbols": symbols,
        "date_range": {
            "start": earliest_date,
            "end": latest_date,
        },
        "total_market_volume": total_volume,
        "stock_features_count": features_coll.count_documents({}),
    }


def get_sentiment_for_stock(
    symbol: str,
    record_type: str = "daily",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: Optional[int] = None,
    db: Optional[Database] = None,
) -> List[Dict[str, Any]]:
    """
    Retrieve sentiment records for a specific stock.

    Parameters
    ----------
    symbol : str
        Stock symbol (e.g. 'TCS').
    record_type : str
        'article' for article-level, 'daily' for daily aggregated.
    start_date : str, optional
        Filter records on or after YYYY-MM-DD.
    end_date : str, optional
        Filter records on or before YYYY-MM-DD.
    limit : int, optional
        Maximum number of records.
    db : Database, optional
        Database instance.
    """
    active_db = db if db is not None else get_database()
    coll = active_db["sentiment"]

    filter_query: Dict[str, Any] = {
        "symbol": symbol.upper().strip(),
        "type": record_type,
    }

    if start_date and end_date:
        filter_query["date"] = {"$gte": str(start_date), "$lte": str(end_date)}
    elif start_date:
        filter_query["date"] = {"$gte": str(start_date)}
    elif end_date:
        filter_query["date"] = {"$lte": str(end_date)}

    cursor = coll.find(filter_query, {"_id": 0}).sort("date", DESCENDING)
    if limit is not None and limit > 0:
        cursor = cursor.limit(limit)

    return list(cursor)


def get_sentiment_summary(
    db: Optional[Database] = None,
) -> Dict[str, Any]:
    """
    Compute summary statistics across the sentiment collection.

    Returns total article count, daily record count, symbols covered,
    and average sentiment per symbol.
    """
    active_db = db if db is not None else get_database()
    coll = active_db["sentiment"]

    article_count = coll.count_documents({"type": "article"})
    daily_count = coll.count_documents({"type": "daily"})
    symbols = sorted(coll.distinct("symbol"))

    # Average sentiment per symbol (daily records)
    pipeline = [
        {"$match": {"type": "daily", "average_sentiment": {"$ne": None}}},
        {"$group": {
            "_id": "$symbol",
            "avg_sentiment": {"$avg": "$average_sentiment"},
            "day_count": {"$sum": 1},
        }},
        {"$sort": {"avg_sentiment": -1}},
    ]
    per_stock = list(coll.aggregate(pipeline))

    return {
        "total_article_records": article_count,
        "total_daily_records": daily_count,
        "symbols": symbols,
        "per_stock_sentiment": [
            {
                "symbol": item["_id"],
                "avg_sentiment": round(item["avg_sentiment"], 4),
                "day_count": item["day_count"],
            }
            for item in per_stock
        ],
    }


def get_top_association_rules(
    limit: int = 10,
    sort_by: str = "lift",
    db: Optional[Database] = None,
) -> List[Dict[str, Any]]:
    """
    Retrieve top association rules sorted by specified metric.

    Parameters
    ----------
    limit : int, default 10
        Maximum number of rules to return.
    sort_by : str, default 'lift'
        Sort metric: 'lift', 'confidence', or 'support'.
    db : Database, optional
        Database instance.
    """
    active_db = db if db is not None else get_database()
    coll = active_db["association_rules"]

    sort_field = sort_by if sort_by in ("lift", "confidence", "support") else "lift"
    cursor = (
        coll.find({"type": "rule"}, {"_id": 0})
        .sort(sort_field, DESCENDING)
        .limit(limit)
    )
    return list(cursor)


def get_rules_for_stock(
    symbol: str,
    limit: int = 20,
    db: Optional[Database] = None,
) -> List[Dict[str, Any]]:
    """
    Retrieve association rules involving a specific stock symbol.

    Searches both antecedent and consequent fields for the stock.

    Parameters
    ----------
    symbol : str
        Stock symbol (e.g. 'TCS'). Automatically suffixed with '_UP'.
    limit : int, default 20
        Maximum number of rules to return.
    db : Database, optional
        Database instance.
    """
    active_db = db if db is not None else get_database()
    coll = active_db["association_rules"]

    item_label = f"{symbol.upper().strip()}_UP"
    cursor = (
        coll.find(
            {
                "type": "rule",
                "$or": [
                    {"antecedent": item_label},
                    {"consequent": item_label},
                ],
            },
            {"_id": 0},
        )
        .sort("lift", DESCENDING)
        .limit(limit)
    )
    return list(cursor)


def get_frequent_itemsets(
    min_size: int = 2,
    limit: int = 20,
    db: Optional[Database] = None,
) -> List[Dict[str, Any]]:
    """
    Retrieve frequent itemsets from MongoDB.

    Parameters
    ----------
    min_size : int, default 2
        Minimum itemset size.
    limit : int, default 20
        Maximum number of itemsets to return.
    db : Database, optional
        Database instance.
    """
    active_db = db if db is not None else get_database()
    coll = active_db["association_rules"]

    cursor = (
        coll.find(
            {"type": "itemset", "size": {"$gte": min_size}},
            {"_id": 0},
        )
        .sort("support", DESCENDING)
        .limit(limit)
    )
    return list(cursor)


# ── RAG Document & Query Operations (Phase 7) ────────────────────────


def store_rag_query(
    query: str,
    answer: str,
    sources: Optional[List[str]] = None,
    citations: Optional[List[Dict[str, Any]]] = None,
    generator_mode: str = "extractive",
    confidence_score: float = 0.0,
    is_supported: bool = True,
    db: Optional[Database] = None,
) -> str:
    """
    Log a RAG query and its grounded answer to MongoDB.

    Parameters
    ----------
    query : str
        The input question.
    answer : str
        The generated answer text.
    sources : List[str], optional
        Human-readable source strings.
    citations : List[Dict[str, Any]], optional
        Structured citation records.
    generator_mode : str, default 'extractive'
        Generator engine used ('extractive', 'mock', 'llm').
    confidence_score : float, default 0.0
        Top similarity / confidence score.
    is_supported : bool, default True
        Whether the query was supported by context.
    db : Database, optional
        Database instance.

    Returns
    -------
    str
        Inserted MongoDB document ID as string.
    """
    from datetime import datetime, timezone

    active_db = db if db is not None else get_database()
    coll = active_db["rag_queries"]

    doc = {
        "query": query,
        "answer": answer,
        "sources": sources or [],
        "citations": citations or [],
        "generator_mode": generator_mode,
        "confidence_score": float(confidence_score),
        "is_supported": bool(is_supported),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    res = coll.insert_one(doc)
    return str(res.inserted_id)


def get_recent_rag_queries(
    limit: int = 10,
    db: Optional[Database] = None,
) -> List[Dict[str, Any]]:
    """
    Retrieve recently executed RAG queries from MongoDB.

    Parameters
    ----------
    limit : int, default 10
        Max queries to return.
    db : Database, optional
        Database instance.
    """
    active_db = db if db is not None else get_database()
    coll = active_db["rag_queries"]

    cursor = (
        coll.find({}, {"_id": 0})
        .sort("timestamp", DESCENDING)
        .limit(limit)
    )
    return list(cursor)


def store_rag_document_manifest(
    manifest: List[Dict[str, Any]],
    db: Optional[Database] = None,
) -> int:
    """
    Store or update indexed document metadata in MongoDB rag_documents collection.

    Parameters
    ----------
    manifest : List[Dict[str, Any]]
        List of document summary dictionaries containing document_id, document_name, etc.
    db : Database, optional
        Database instance.

    Returns
    -------
    int
        Number of documents upserted.
    """
    if not manifest:
        return 0

    active_db = db if db is not None else get_database()
    coll = active_db["rag_documents"]

    count = 0
    for doc in manifest:
        doc_id = doc.get("document_id")
        if doc_id:
            coll.update_one(
                {"document_id": doc_id},
                {"$set": doc},
                upsert=True,
            )
            count += 1
    return count


def get_rag_documents(
    db: Optional[Database] = None,
) -> List[Dict[str, Any]]:
    """
    Retrieve all indexed document manifests from MongoDB.
    """
    active_db = db if db is not None else get_database()
    coll = active_db["rag_documents"]

    return list(coll.find({}, {"_id": 0}).sort("document_name", ASCENDING))
