"""
MongoDB indexing management and rationales for the Financial Intelligence Platform.

Defines compound, unique, and secondary indexes with documented query benefits:
1. market_data:
   - ("symbol", 1), ("date", 1) [UNIQUE]: Guarantees idempotency and speeds up historical lookups.
   - ("date", 1): Accelerates cross-stock daily market queries.
2. news:
   - ("symbol", 1), ("date", 1): Optimizes stock-specific news lookups ordered chronologically.
3. stock_features:
   - ("symbol", 1) [UNIQUE]: Enforces one analytical summary per ticker and speeds up ticker lookups.
"""

from typing import Any, Dict, List, Optional
from pymongo import ASCENDING, DESCENDING, TEXT
from pymongo.database import Database

from src.database.mongodb import get_database
from src.utils.logger import logger


INDEX_DEFINITIONS: Dict[str, List[Dict[str, Any]]] = {
    "market_data": [
        {
            "keys": [("symbol", ASCENDING), ("date", ASCENDING)],
            "unique": True,
            "name": "idx_market_symbol_date_unique",
            "rationale": "Prevents duplicate price bars per ticker on any date; enables O(log N) historical range queries.",
        },
        {
            "keys": [("date", ASCENDING)],
            "unique": False,
            "name": "idx_market_date",
            "rationale": "Enables cross-sectional daily market scans across all stocks without collection scans.",
        },
    ],
    "news": [
        {
            "keys": [("symbol", ASCENDING), ("date", DESCENDING)],
            "unique": False,
            "name": "idx_news_symbol_date",
            "rationale": "Accelerates news filtering by ticker ordered from newest to oldest.",
        },
        {
            "keys": [("title", TEXT), ("description", TEXT)],
            "unique": False,
            "name": "idx_news_text_search",
            "rationale": "Enables keyword and phrase search across headlines and descriptions.",
        },
    ],
    "stock_features": [
        {
            "keys": [("symbol", ASCENDING)],
            "unique": True,
            "name": "idx_features_symbol_unique",
            "rationale": "Enforces single summary document per ticker; provides O(1) aggregate lookups.",
        },
        {
            "keys": [("avg_volatility", DESCENDING)],
            "unique": False,
            "name": "idx_features_volatility_desc",
            "rationale": "Supports fast volatility rank sorting for risk dashboards.",
        },
    ],
    "sentiment": [
        {
            "keys": [("symbol", ASCENDING), ("date", DESCENDING), ("type", ASCENDING)],
            "unique": False,
            "name": "idx_sentiment_symbol_date_type",
            "rationale": "Accelerates stock-specific sentiment lookups, filtered by article vs daily type.",
        },
        {
            "keys": [("type", ASCENDING), ("date", DESCENDING)],
            "unique": False,
            "name": "idx_sentiment_type_date",
            "rationale": "Enables cross-stock daily sentiment scans ordered chronologically.",
        },
    ],
    "association_rules": [
        {
            "keys": [("type", ASCENDING), ("lift", DESCENDING)],
            "unique": False,
            "name": "idx_assoc_type_lift",
            "rationale": "Enables fast retrieval of top rules/itemsets sorted by lift.",
        },
        {
            "keys": [("antecedent_str", ASCENDING)],
            "unique": False,
            "name": "idx_assoc_antecedent",
            "rationale": "Supports lookups of rules by antecedent stock combination.",
        },
    ],
    "rag_documents": [
        {
            "keys": [("document_id", ASCENDING)],
            "unique": True,
            "name": "idx_rag_doc_id_unique",
            "rationale": "Enforces unique document metadata records by document_id.",
        },
        {
            "keys": [("document_name", ASCENDING)],
            "unique": False,
            "name": "idx_rag_doc_name",
            "rationale": "Supports quick searches by document filename.",
        },
    ],
    "rag_queries": [
        {
            "keys": [("timestamp", DESCENDING)],
            "unique": False,
            "name": "idx_rag_query_timestamp",
            "rationale": "Accelerates chronological retrieval of user RAG queries and answers.",
        },
    ],
}


def create_indexes(db: Optional[Database] = None) -> Dict[str, List[str]]:
    """
    Create all defined indexes across collections.

    Parameters
    ----------
    db : Database, optional
        Target MongoDB database. If None, loaded from configuration.

    Returns
    -------
    dict
        Dictionary mapping collection names to lists of created index names.
    """
    active_db = db if db is not None else get_database()
    results: Dict[str, List[str]] = {}

    for coll_name, index_specs in INDEX_DEFINITIONS.items():
        coll = active_db[coll_name]
        results[coll_name] = []

        for spec in index_specs:
            keys = spec["keys"]
            unique = spec.get("unique", False)
            name = spec.get("name")
            rationale = spec.get("rationale", "")

            try:
                created_name = coll.create_index(keys, unique=unique, name=name)
                results[coll_name].append(created_name)
                logger.info(f"Created index '{created_name}' on collection '{coll_name}'. Rationale: {rationale}")
            except Exception as e:
                logger.warning(f"Index creation note for '{name}' on '{coll_name}': {e}")

    # Ensure placeholder collections exist if specified
    placeholder_colls = ["predictions", "sentiment", "association_rules"]
    for coll_name in placeholder_colls:
        if coll_name not in active_db.list_collection_names():
            try:
                active_db.create_collection(coll_name)
                logger.debug(f"Initialized placeholder collection '{coll_name}'.")
            except Exception:
                pass

    return results
