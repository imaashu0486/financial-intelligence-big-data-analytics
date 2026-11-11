"""
Database package for Financial Intelligence & Stock Analytics Platform.

Provides MongoDB connectivity, indexing, document schemas, query functions,
and aggregation pipelines.
"""

from src.database.mongodb import (
    check_connection,
    close_connection,
    get_client,
    get_collection,
    get_database,
    load_mongo_config,
)
from src.database.schemas import (
    market_record_to_doc,
    news_record_to_doc,
    stock_feature_record_to_doc,
)
from src.database.indexes import INDEX_DEFINITIONS, create_indexes
from src.database.queries import (
    get_highest_volatility_stocks,
    get_latest_stock_data,
    get_market_summary,
    get_news_for_stock,
    get_sentiment_for_stock,
    get_sentiment_summary,
    get_stock_history,
    get_stock_statistics,
    get_top_stocks_by_return,
    get_top_association_rules,
    get_rules_for_stock,
    get_frequent_itemsets,
    store_rag_query,
    get_recent_rag_queries,
    store_rag_document_manifest,
    get_rag_documents,
)
from src.database.aggregations import (
    aggregate_avg_return_by_stock,
    aggregate_avg_volatility_by_stock,
    aggregate_comprehensive_metrics,
    aggregate_total_volume_by_stock,
    build_avg_return_pipeline,
    build_avg_volatility_pipeline,
    build_comprehensive_summary_pipeline,
    build_total_volume_pipeline,
)

__all__ = [
    # Connection
    "get_client",
    "get_database",
    "get_collection",
    "check_connection",
    "close_connection",
    "load_mongo_config",
    # Schemas
    "market_record_to_doc",
    "news_record_to_doc",
    "stock_feature_record_to_doc",
    # Indexes
    "create_indexes",
    "INDEX_DEFINITIONS",
    # Queries
    "get_stock_history",
    "get_latest_stock_data",
    "get_stock_statistics",
    "get_top_stocks_by_return",
    "get_highest_volatility_stocks",
    "get_news_for_stock",
    "get_market_summary",
    "get_sentiment_for_stock",
    "get_sentiment_summary",
    "get_top_association_rules",
    "get_rules_for_stock",
    "get_frequent_itemsets",
    "store_rag_query",
    "get_recent_rag_queries",
    "store_rag_document_manifest",
    "get_rag_documents",
    # Aggregations
    "aggregate_avg_return_by_stock",
    "aggregate_total_volume_by_stock",
    "aggregate_avg_volatility_by_stock",
    "aggregate_comprehensive_metrics",
    "build_avg_return_pipeline",
    "build_total_volume_pipeline",
    "build_avg_volatility_pipeline",
    "build_comprehensive_summary_pipeline",
]
