"""
Unit and mock tests for MongoDB NoSQL data layer.

Tests run 100% offline without requiring a running MongoDB server instance:
- Document schemas & type transformations
- Date handling & NaN sanitization
- Index creation & uniqueness constraints
- Upsert logic & duplicate prevention
- Query functions & projections
- Aggregation pipeline structures & execution
- Configuration & health check fallbacks
"""

import math
from datetime import date, datetime
from pathlib import Path
import mongomock
import pytest
from pymongo import UpdateOne
from pymongo.errors import DuplicateKeyError

ROOT = Path(__file__).resolve().parent.parent

from src.database.aggregations import (
    aggregate_avg_return_by_stock,
    aggregate_avg_volatility_by_stock,
    aggregate_total_volume_by_stock,
    build_avg_return_pipeline,
    build_avg_volatility_pipeline,
    build_comprehensive_summary_pipeline,
    build_total_volume_pipeline,
)
from src.database.indexes import INDEX_DEFINITIONS, create_indexes
from src.database.mongodb import check_connection, load_mongo_config
from src.database.queries import (
    get_highest_volatility_stocks,
    get_latest_stock_data,
    get_market_summary,
    get_news_for_stock,
    get_stock_history,
    get_stock_statistics,
    get_top_stocks_by_return,
)
from src.database.schemas import (
    _sanitize_value,
    market_record_to_doc,
    news_record_to_doc,
    stock_feature_record_to_doc,
)


@pytest.fixture
def mock_db():
    """Create an in-memory MongoDB database instance via mongomock."""
    client = mongomock.MongoClient()
    return client["financial_analytics"]


@pytest.fixture
def sample_market_records():
    return [
        {
            "date": "2023-01-02",
            "symbol": "TCS",
            "open": 3500.0,
            "high": 3550.0,
            "low": 3480.0,
            "close": 3530.0,
            "volume": 1000000,
            "outlier_flag": "False",
            "daily_return": 0.01,
            "log_return": 0.00995,
            "volatility": 0.005,
            "rsi_14": 60.0,
        },
        {
            "date": "2023-01-03",
            "symbol": "TCS",
            "open": 3520.0,
            "high": 3560.0,
            "low": 3500.0,
            "close": 3540.0,
            "volume": 1200000,
            "outlier_flag": "False",
            "daily_return": 0.00283,
            "log_return": 0.00283,
            "volatility": 0.006,
            "rsi_14": 62.0,
        },
        {
            "date": "2023-01-02",
            "symbol": "INFY",
            "open": 1500.0,
            "high": 1520.0,
            "low": 1490.0,
            "close": 1510.0,
            "volume": 2000000,
            "outlier_flag": "False",
            "daily_return": 0.02,
            "log_return": 0.0198,
            "volatility": 0.012,
            "rsi_14": 65.0,
        },
    ]


@pytest.fixture
def populated_mock_db(mock_db, sample_market_records):
    """Populate mock database with market, news, and stock_features data."""
    # 1. Market Data
    for rec in sample_market_records:
        doc = market_record_to_doc(rec)
        mock_db["market_data"].insert_one(doc)

    # 2. News Data
    sample_news = [
        {
            "date": "2023-01-02",
            "symbol": "TCS",
            "company": "Tata Consultancy Services",
            "title": "TCS Q3 Revenue Gains on Strong Cloud Demand",
            "description": "IT giant reports solid operational margin growth.",
            "source": "Reuters",
            "sentiment_score": 0.85,  # Flexible extra metadata
        },
        {
            "date": "2023-01-03",
            "symbol": "INFY",
            "company": "Infosys",
            "title": "Infosys Signs Strategic European Contract",
            "description": "Multi-year deal boosts revenue outlook.",
            "source": "Bloomberg",
        },
    ]
    for rec in sample_news:
        mock_db["news"].insert_one(news_record_to_doc(rec))

    # 3. Stock Features
    sample_features = [
        {
            "symbol": "TCS",
            "avg_daily_return": 0.0064,
            "avg_log_return": 0.00639,
            "avg_volatility": 0.0055,
            "avg_volume": 1100000.0,
            "total_volume": 2200000,
            "max_price": 3560.0,
            "min_price": 3480.0,
            "trading_days": 2,
            "volatility_rank": 2,
        },
        {
            "symbol": "INFY",
            "avg_daily_return": 0.0200,
            "avg_log_return": 0.0198,
            "avg_volatility": 0.0120,
            "avg_volume": 2000000.0,
            "total_volume": 2000000,
            "max_price": 1520.0,
            "min_price": 1490.0,
            "trading_days": 1,
            "volatility_rank": 1,
        },
    ]
    for rec in sample_features:
        mock_db["stock_features"].insert_one(stock_feature_record_to_doc(rec))

    return mock_db


class TestDocumentSchemas:
    """Test document transformation, validation, and type sanitization."""

    def test_sanitize_value_nan_and_inf(self):
        assert _sanitize_value(float("nan")) is None
        assert _sanitize_value(float("inf")) is None
        assert _sanitize_value(None) is None
        assert _sanitize_value(42.5) == 42.5

    def test_sanitize_value_dates(self):
        d = date(2023, 1, 15)
        dt = datetime(2023, 1, 15, 10, 30, 0)
        assert _sanitize_value(d) == "2023-01-15"
        assert _sanitize_value(dt) == "2023-01-15"

    def test_market_record_to_doc(self):
        row = {
            "date": "2023-01-05",
            "symbol": "tcs",
            "open": "3500.0",
            "high": 3550.0,
            "low": 3480.0,
            "close": 3530.0,
            "volume": "1500000",
            "outlier_flag": "True",
            "daily_return": float("nan"),
            "rsi_14": 55.4,
        }
        doc = market_record_to_doc(row)
        assert doc["symbol"] == "TCS"
        assert doc["date"] == "2023-01-05"
        assert doc["high"] == 3550.0
        assert doc["volume"] == 1500000
        assert doc["outlier_flag"] is True
        assert doc["daily_return"] is None
        assert doc["rsi_14"] == 55.4

    def test_news_record_flexible_schema(self):
        row = {
            "date": "2023-05-10",
            "symbol": "RELIANCE",
            "title": "Reliance Expands Green Energy",
            "description": "New solar gigafactory announced.",
            "source": "Economic Times",
            "sentiment_score": 0.92,
            "custom_tag": "ESG",
        }
        doc = news_record_to_doc(row)
        assert doc["symbol"] == "RELIANCE"
        assert doc["source"] == "Economic Times"
        # Flexible attributes preserved:
        assert doc["sentiment_score"] == 0.92
        assert doc["custom_tag"] == "ESG"

    def test_stock_feature_record_to_doc(self):
        row = {
            "symbol": "infy",
            "avg_daily_return": 0.0012,
            "total_volume": 50000000,
            "trading_days": 250,
            "volatility_rank": 3,
        }
        doc = stock_feature_record_to_doc(row)
        assert doc["symbol"] == "INFY"
        assert doc["avg_daily_return"] == 0.0012
        assert doc["total_volume"] == 50000000
        assert doc["trading_days"] == 250
        assert doc["volatility_rank"] == 3


class TestIndexingAndUpsert:
    """Test index registration and upsert uniqueness logic."""

    def test_index_definitions_structure(self):
        assert "market_data" in INDEX_DEFINITIONS
        assert "news" in INDEX_DEFINITIONS
        assert "stock_features" in INDEX_DEFINITIONS

        # Verify unique constraints
        market_specs = INDEX_DEFINITIONS["market_data"]
        assert any(spec.get("unique") is True for spec in market_specs)

    def test_create_indexes_on_mock_db(self, mock_db):
        res = create_indexes(db=mock_db)
        assert "market_data" in res
        assert "stock_features" in res

    def test_upsert_prevents_duplicate_market_records(self, mock_db):
        coll = mock_db["market_data"]
        doc1 = market_record_to_doc({"date": "2023-01-02", "symbol": "TCS", "close": 3500.0, "volume": 100})
        doc2_updated = market_record_to_doc({"date": "2023-01-02", "symbol": "TCS", "close": 3525.0, "volume": 150})

        # First upsert -> insert
        filter_spec = {"symbol": doc1["symbol"], "date": doc1["date"]}
        coll.update_one(filter_spec, {"$set": doc1}, upsert=True)
        assert coll.count_documents({}) == 1
        assert coll.find_one(filter_spec)["close"] == 3500.0

        # Second upsert with same (symbol, date) -> update existing doc
        coll.update_one(filter_spec, {"$set": doc2_updated}, upsert=True)
        assert coll.count_documents({}) == 1  # Still 1 document!
        assert coll.find_one(filter_spec)["close"] == 3525.0

        # Also test loader bulk execution helper with mongomock fallback
        from scripts.load_mongodb import _execute_bulk_ops
        stats = {"upserted": 0, "matched": 0, "modified": 0}
        ops = [
            UpdateOne({"symbol": "INFY", "date": "2023-01-02"}, {"$set": {"close": 1500.0}}, upsert=True),
            UpdateOne({"symbol": "INFY", "date": "2023-01-02"}, {"$set": {"close": 1515.0}}, upsert=True),
        ]
        _execute_bulk_ops(coll, ops, stats)
        assert coll.count_documents({"symbol": "INFY"}) == 1
        assert stats["upserted"] == 1
        assert stats["matched"] == 1


class TestQueries:
    """Test query functions using mock database."""

    def test_get_stock_history(self, populated_mock_db):
        history = get_stock_history("TCS", db=populated_mock_db)
        assert len(history) == 2
        assert history[0]["date"] == "2023-01-02"
        assert history[1]["date"] == "2023-01-03"
        assert "_id" not in history[0]

    def test_get_stock_history_date_filter(self, populated_mock_db):
        history = get_stock_history("TCS", start_date="2023-01-03", db=populated_mock_db)
        assert len(history) == 1
        assert history[0]["date"] == "2023-01-03"

    def test_get_latest_stock_data(self, populated_mock_db):
        latest = get_latest_stock_data("TCS", db=populated_mock_db)
        assert latest is not None
        assert latest["date"] == "2023-01-03"
        assert latest["close"] == 3540.0

    def test_get_stock_statistics(self, populated_mock_db):
        stats = get_stock_statistics("INFY", db=populated_mock_db)
        assert stats is not None
        assert stats["symbol"] == "INFY"
        assert stats["avg_daily_return"] == 0.02
        assert stats["total_volume"] == 2000000

    def test_get_top_stocks_by_return(self, populated_mock_db):
        top_stocks = get_top_stocks_by_return(limit=5, db=populated_mock_db)
        assert len(top_stocks) == 2
        # INFY (0.02) should rank higher than TCS (0.0064)
        assert top_stocks[0]["symbol"] == "INFY"
        assert top_stocks[1]["symbol"] == "TCS"

    def test_get_highest_volatility_stocks(self, populated_mock_db):
        high_vol = get_highest_volatility_stocks(limit=5, db=populated_mock_db)
        assert len(high_vol) == 2
        assert high_vol[0]["symbol"] == "INFY"

    def test_get_news_for_stock(self, populated_mock_db):
        tcs_news = get_news_for_stock("TCS", db=populated_mock_db)
        assert len(tcs_news) == 1
        assert "TCS Q3 Revenue" in tcs_news[0]["title"]
        assert tcs_news[0]["source"] == "Reuters"

    def test_get_market_summary(self, populated_mock_db):
        summary = get_market_summary(db=populated_mock_db)
        assert summary["total_market_records"] == 3
        assert summary["total_news_records"] == 2
        assert summary["symbol_count"] == 2
        assert "TCS" in summary["symbols"]
        assert "INFY" in summary["symbols"]
        assert summary["date_range"]["start"] == "2023-01-02"
        assert summary["date_range"]["end"] == "2023-01-03"
        assert summary["total_market_volume"] == 4200000


class TestAggregationPipelines:
    """Test aggregation pipeline construction and mock execution."""

    def test_pipeline_structures(self):
        # 1. Average return pipeline
        ret_pipe = build_avg_return_pipeline()
        stages = [list(stage.keys())[0] for stage in ret_pipe]
        assert "$match" in stages
        assert "$group" in stages
        assert "$sort" in stages

        # 2. Volume pipeline
        vol_pipe = build_total_volume_pipeline()
        vol_stages = [list(stage.keys())[0] for stage in vol_pipe]
        assert "$match" in vol_stages
        assert "$group" in vol_stages

        # 3. Volatility pipeline
        vola_pipe = build_avg_volatility_pipeline()
        vola_stages = [list(stage.keys())[0] for stage in vola_pipe]
        assert "$match" in vola_stages
        assert "$group" in vola_stages

    def test_aggregate_avg_return_execution(self, populated_mock_db):
        results = aggregate_avg_return_by_stock(db=populated_mock_db)
        assert len(results) == 2
        # Verify aggregation results computed inside MongoDB
        symbols = {r["symbol"] for r in results}
        assert "TCS" in symbols
        assert "INFY" in symbols

    def test_aggregate_total_volume_execution(self, populated_mock_db):
        results = aggregate_total_volume_by_stock(db=populated_mock_db)
        assert len(results) == 2
        tcs_res = next(r for r in results if r["symbol"] == "TCS")
        assert tcs_res["total_volume"] == 2200000  # 1,000,000 + 1,200,000

    def test_aggregate_avg_volatility_execution(self, populated_mock_db):
        results = aggregate_avg_volatility_by_stock(db=populated_mock_db)
        assert len(results) == 2
        symbols = [r["symbol"] for r in results]
        assert "INFY" in symbols


class TestConfigAndHealthCheck:
    """Test config loading and health check failure handling."""

    def test_load_mongo_config(self):
        cfg = load_mongo_config()
        assert "uri" in cfg
        assert "database" in cfg
        assert "collections" in cfg
        assert cfg["database"] == "financial_analytics"

class TestDataLoaders:
    """Test data loading functions with mock database using sample datasets."""

    def test_load_market_data(self, mock_db, tmp_path):
        from scripts.load_mongodb import load_market_data_to_mongo
        import pandas as pd
        # Create a 20-row sample CSV to test upsert logic quickly in mongomock
        sample_rows = [
            {"date": f"2023-01-{i:02d}", "symbol": "TCS", "open": 3500.0, "high": 3550.0, "low": 3450.0, "close": 3520.0, "volume": 1000000}
            for i in range(1, 21)
        ]
        sample_csv = tmp_path / "sample_market.csv"
        pd.DataFrame(sample_rows).to_csv(sample_csv, index=False)

        stats = load_market_data_to_mongo(mock_db, sample_csv, batch_size=10)
        assert stats["upserted"] == 20
        assert mock_db["market_data"].count_documents({}) == 20

        # Running again should match existing (0 newly upserted)
        stats2 = load_market_data_to_mongo(mock_db, sample_csv, batch_size=10)
        assert stats2["upserted"] == 0
        assert stats2["matched"] == 20
        assert mock_db["market_data"].count_documents({}) == 20

    def test_load_news_data(self, mock_db, tmp_path):
        from scripts.load_mongodb import load_news_to_mongo
        import pandas as pd
        sample_rows = [
            {"date": f"2023-01-{i:02d}", "symbol": "INFY", "title": f"News Headline {i}", "description": "Desc", "source": "Reuters"}
            for i in range(1, 11)
        ]
        sample_csv = tmp_path / "sample_news.csv"
        pd.DataFrame(sample_rows).to_csv(sample_csv, index=False)

        stats = load_news_to_mongo(mock_db, sample_csv, batch_size=5)
        assert stats["upserted"] == 10
        assert mock_db["news"].count_documents({}) == 10

    def test_load_stock_features(self, mock_db):
        from scripts.load_mongodb import load_stock_features_to_mongo
        spark_dir = ROOT / "data" / "processed" / "spark" / "stock_statistics"
        stats = load_stock_features_to_mongo(mock_db, spark_dir)
        assert stats["upserted"] == 10
        assert mock_db["stock_features"].count_documents({}) == 10
