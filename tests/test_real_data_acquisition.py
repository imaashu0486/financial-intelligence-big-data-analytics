"""
Unit and integration tests for Real Financial Data Acquisition & Validation.

All external network operations (Yahoo Finance, GDELT) are mocked to ensure
deterministic, high-speed, and offline-reliable execution.
"""

import json
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.ingestion.real_market_collector import (
    DEFAULT_CONSTITUENTS_PATH,
    compute_sha256,
    download_ticker_data,
    load_constituents,
    normalize_yfinance_dataframe,
)
from src.ingestion.real_news_collector import (
    collect_corporate_documents,
    compute_article_id,
    fetch_gdelt_news_for_company,
    get_curated_real_news_corpus,
)
from src.preprocessing.real_data_validator import (
    validate_corporate_documents,
    validate_news_dataset,
    validate_single_market_file,
)


class TestConstituentsMetadata:
    """Validate NIFTY 50 constituents master metadata."""

    def test_constituents_file_exists_and_loads(self):
        assert DEFAULT_CONSTITUENTS_PATH.exists(), "Constituents file does not exist"
        data = load_constituents(DEFAULT_CONSTITUENTS_PATH)
        assert isinstance(data, dict)
        assert data.get("index_name") == "NIFTY 50"
        assert data.get("constituent_count") == 50
        assert data.get("benchmark_symbol") == "^NSEI"

    def test_constituents_count_and_fields(self):
        data = load_constituents(DEFAULT_CONSTITUENTS_PATH)
        constituents = data.get("constituents", [])
        assert len(constituents) == 50, f"Expected 50 constituents, got {len(constituents)}"

        symbols = set()
        for comp in constituents:
            assert "symbol" in comp and comp["symbol"]
            assert "company_name" in comp and comp["company_name"]
            assert "sector" in comp and comp["sector"]
            assert "yfinance_symbol" in comp
            assert comp["yfinance_symbol"].endswith(".NS"), f"{comp['yfinance_symbol']} must end with .NS"
            assert "isin" in comp and comp["isin"].startswith("INE")
            symbols.add(comp["symbol"])

        assert len(symbols) == 50, "Constituent symbols must be unique"

    def test_survivorship_bias_documentation(self):
        data = load_constituents(DEFAULT_CONSTITUENTS_PATH)
        note = data.get("survivorship_bias_note", "")
        assert len(note) > 20, "Survivorship bias note must be documented"
        assert "survivorship" in note.lower()


class TestMarketDataNormalization:
    """Verify DataFrame normalization and schema enforcement."""

    def test_normalize_single_level_dataframe(self):
        raw_df = pd.DataFrame({
            "Date": pd.date_range("2024-01-01", periods=3, freq="D"),
            "Open": [100.0, 102.0, 101.0],
            "High": [105.0, 106.0, 104.0],
            "Low": [98.0, 100.0, 99.0],
            "Close": [103.0, 105.0, 102.0],
            "Adj Close": [103.0, 105.0, 102.0],
            "Volume": [10000, 15000, 12000],
        })

        norm_df = normalize_yfinance_dataframe(raw_df, "TCS")
        expected_cols = ["date", "symbol", "open", "high", "low", "close", "adj_close", "volume"]
        assert list(norm_df.columns) == expected_cols
        assert (norm_df["symbol"] == "TCS").all()
        assert len(norm_df) == 3
        assert norm_df["date"].iloc[0] == "2024-01-01"

    def test_normalize_multiindex_dataframe(self):
        # yfinance often returns MultiIndex columns e.g. ('Close', 'RELIANCE.NS')
        tuples = [
            ("Date", ""),
            ("Open", "RELIANCE.NS"),
            ("High", "RELIANCE.NS"),
            ("Low", "RELIANCE.NS"),
            ("Close", "RELIANCE.NS"),
            ("Adj Close", "RELIANCE.NS"),
            ("Volume", "RELIANCE.NS"),
        ]
        columns = pd.MultiIndex.from_tuples(tuples)
        raw_df = pd.DataFrame([
            ["2024-01-01", 2500.0, 2550.0, 2480.0, 2520.0, 2520.0, 500000],
        ], columns=columns)

        norm_df = normalize_yfinance_dataframe(raw_df, "RELIANCE")
        assert list(norm_df.columns) == ["date", "symbol", "open", "high", "low", "close", "adj_close", "volume"]
        assert norm_df["symbol"].iloc[0] == "RELIANCE"
        assert norm_df["close"].iloc[0] == 2520.0

    def test_normalize_empty_dataframe(self):
        empty_df = pd.DataFrame()
        norm_df = normalize_yfinance_dataframe(empty_df, "EMPTY")
        assert norm_df.empty
        assert "date" in norm_df.columns


class TestMarketValidationInvariants:
    """Verify quantitative financial invariants and violation detection."""

    @pytest.fixture
    def valid_market_csv(self, tmp_path):
        df = pd.DataFrame({
            "date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
            "symbol": ["INFY"] * 4,
            "open": [1500.0, 1510.0, 1520.0, 1515.0],
            "high": [1530.0, 1540.0, 1535.0, 1525.0],
            "low": [1490.0, 1500.0, 1510.0, 1505.0],
            "close": [1515.0, 1525.0, 1515.0, 1520.0],
            "adj_close": [1515.0, 1525.0, 1515.0, 1520.0],
            "volume": [100000, 120000, 110000, 95000],
        })
        p = tmp_path / "INFY.csv"
        df.to_csv(p, index=False)
        return p

    def test_valid_file_passes(self, valid_market_csv):
        res = validate_single_market_file(valid_market_csv)
        assert res["is_valid"] is True
        assert res["row_count"] == 4
        assert res["violations"]["negative_or_zero_prices"] == 0
        assert res["violations"]["high_less_than_low"] == 0
        assert res["violations"]["chronological_order_errors"] == 0

    def test_chronological_order_violation(self, tmp_path):
        df = pd.DataFrame({
            "date": ["2024-01-04", "2024-01-02"],
            "symbol": ["INFY", "INFY"],
            "open": [1500.0, 1510.0],
            "high": [1530.0, 1540.0],
            "low": [1490.0, 1500.0],
            "close": [1515.0, 1525.0],
            "volume": [100000, 120000],
        })
        p = tmp_path / "BAD_ORDER.csv"
        df.to_csv(p, index=False)
        res = validate_single_market_file(p)
        assert res["is_valid"] is False
        assert res["violations"]["chronological_order_errors"] == 1

    def test_duplicate_dates_violation(self, tmp_path):
        df = pd.DataFrame({
            "date": ["2024-01-02", "2024-01-02"],
            "symbol": ["INFY", "INFY"],
            "open": [1500.0, 1510.0],
            "high": [1530.0, 1540.0],
            "low": [1490.0, 1500.0],
            "close": [1515.0, 1525.0],
            "volume": [100000, 120000],
        })
        p = tmp_path / "DUPE_DATES.csv"
        df.to_csv(p, index=False)
        res = validate_single_market_file(p)
        assert res["is_valid"] is False
        assert res["violations"]["duplicate_dates"] == 1

    def test_high_less_than_low_violation(self, tmp_path):
        df = pd.DataFrame({
            "date": ["2024-01-01"],
            "symbol": ["INFY"],
            "open": [1500.0],
            "high": [1480.0],  # Invalid: High < Low
            "low": [1490.0],
            "close": [1485.0],
            "volume": [100000],
        })
        p = tmp_path / "HIGH_LOW_ERR.csv"
        df.to_csv(p, index=False)
        res = validate_single_market_file(p)
        assert res["is_valid"] is False
        assert res["violations"]["high_less_than_low"] == 1

    def test_extreme_jump_detection(self, tmp_path):
        df = pd.DataFrame({
            "date": ["2024-01-01", "2024-01-02"],
            "symbol": ["SPLIT_CORP", "SPLIT_CORP"],
            "open": [1000.0, 500.0],
            "high": [1020.0, 510.0],
            "low": [980.0, 490.0],
            "close": [1000.0, 500.0],  # -50% jump (e.g. 1:2 split)
            "volume": [100000, 200000],
        })
        p = tmp_path / "SPLIT_TEST.csv"
        df.to_csv(p, index=False)
        res = validate_single_market_file(p)
        assert res["violations"]["extreme_single_day_jumps"] == 1
        assert len(res["jump_dates"]) == 1
        assert res["jump_dates"][0]["return_pct"] == -50.0


class TestNewsAndDeduplication:
    """Validate news schema, hashing, and deduplication engine."""

    def test_deterministic_article_id_generation(self):
        id1 = compute_article_id("TCS Announces Wage Hikes", "2024-04-12T16:00:00Z", "TCS")
        id2 = compute_article_id("tcs announces wage hikes!", "2024-04-12", "TCS")
        assert id1 == id2, "Normalized headlines and dates must produce identical article_id"

        id3 = compute_article_id("TCS Announces Record Profit", "2024-04-12T16:00:00Z", "TCS")
        assert id1 != id3, "Different headlines must produce different IDs"

    def test_curated_news_corpus_integrity(self):
        corpus = get_curated_real_news_corpus()
        assert len(corpus) >= 40, f"Curated corpus should have comprehensive articles, got {len(corpus)}"

        tickers = {item["ticker"] for item in corpus}
        assert "RELIANCE" in tickers
        assert "TCS" in tickers
        assert "HDFCBANK" in tickers

        # Check required fields
        for art in corpus:
            assert art["article_id"]
            assert art["company_name"]
            assert art["ticker"]
            assert art["headline"]
            assert art["article_url"].startswith("http")
            assert art["source_name"]
            assert art["language"] == "en"

    def test_news_dataset_validator(self, tmp_path):
        csv_file = tmp_path / "valid_news.csv"
        df = pd.DataFrame([{
            "article_id": "art001",
            "company_name": "Infosys Limited",
            "ticker": "INFY",
            "headline": "Infosys signs digital transformation framework",
            "description_or_snippet": "Deal details announced.",
            "article_url": "https://example.com/infosys-deal",
            "source_name": "Reuters",
            "published_at": "2024-01-15T10:00:00Z",
            "language": "en",
            "article_type": "news",
            "collection_timestamp": "2024-01-15T12:00:00Z",
            "content_available": True,
            "content_provenance": "gdelt",
            "sentiment_status": "unprocessed",
        }])
        df.to_csv(csv_file, index=False)

        res = validate_news_dataset(csv_file)
        assert res["is_valid"] is True
        assert res["total_articles"] == 1
        assert res["unique_tickers"] == 1


class TestCorporateDocuments:
    """Validate corporate reports and regulatory filings acquisition."""

    def test_corporate_documents_collection_and_validation(self, tmp_path):
        docs_dir = tmp_path / "documents"
        manifest_path = tmp_path / "docs_manifest.json"

        manifest = collect_corporate_documents(
            output_dir=docs_dir,
            manifest_path=manifest_path,
        )

        assert manifest["total_documents"] >= 4
        assert manifest_path.exists()
        assert "RELIANCE" in manifest["companies_covered"]
        assert "TCS" in manifest["companies_covered"]

        # Validate with document validator
        val_res = validate_corporate_documents(docs_dir)
        assert val_res["is_valid"] is True
        assert val_res["total_documents"] >= 4
        assert val_res["violations"]["empty_files"] == 0
        assert val_res["violations"]["undersized_documents"] == 0


class TestMockedDownloadAndErrorIsolation:
    """Verify market downloader retry logic and error isolation with mocked yfinance."""

    @patch("yfinance.download")
    def test_successful_ticker_download(self, mock_download, tmp_path):
        mock_df = pd.DataFrame({
            "Date": ["2024-01-01", "2024-01-02"],
            "Open": [100.0, 102.0],
            "High": [105.0, 106.0],
            "Low": [98.0, 100.0],
            "Close": [103.0, 105.0],
            "Adj Close": [103.0, 105.0],
            "Volume": [10000, 15000],
        })
        mock_download.return_value = mock_df

        rec = download_ticker_data(
            symbol="TEST_SYM",
            yfinance_symbol="TEST_SYM.NS",
            output_dir=tmp_path,
        )

        assert rec["status"] == "SUCCESS"
        assert rec["records_count"] == 2
        assert rec["sha256"] is not None
        assert Path(rec["target_file"]).exists()

    @patch("yfinance.download")
    def test_retry_and_graceful_failure_handling(self, mock_download, tmp_path):
        # Simulate network error
        mock_download.side_effect = Exception("Simulated connection timeout")

        rec = download_ticker_data(
            symbol="FAIL_SYM",
            yfinance_symbol="FAIL_SYM.NS",
            output_dir=tmp_path,
            max_retries=2,
            retry_delay_seconds=0.01,
        )

        assert rec["status"] == "FAILED"
        assert rec["attempts"] == 2
        assert "Simulated connection timeout" in rec["error_message"]
        assert rec["records_count"] == 0


class TestSyntheticDataPreservation:
    """Verify that existing synthetic data assets remain completely unmodified."""

    def test_synthetic_market_data_intact(self):
        p = Path("data/sample/market_data.csv")
        assert p.exists(), "data/sample/market_data.csv must not be deleted"
        df = pd.read_csv(p)
        assert len(df) >= 5000, f"Expected >= 5000 rows, found {len(df)}"
        assert "date" in df.columns and "symbol" in df.columns

    def test_synthetic_news_data_intact(self):
        p = Path("data/sample/financial_news.csv")
        assert p.exists(), "data/sample/financial_news.csv must not be deleted"
        df = pd.read_csv(p)
        assert len(df) >= 450, f"Expected >= 450 rows, found {len(df)}"

    def test_processed_market_features_intact(self):
        p = Path("data/processed/market_features.csv")
        assert p.exists(), "data/processed/market_features.csv must not be deleted"
        assert p.stat().st_size > 500000

    def test_synthetic_documents_intact(self):
        docs = list(Path("data/documents").glob("*.txt"))
        assert len(docs) == 3, f"Expected 3 synthetic documents, found {len(docs)}"
        doc_names = {d.name for d in docs}
        assert "acme_corp_annual_report_2024.txt" in doc_names
        assert "globex_investor_presentation_2024.txt" in doc_names
        assert "initech_risk_disclosure_2024.txt" in doc_names

    def test_vectorstore_and_models_intact(self):
        assert Path("data/vectorstore/embeddings.npy").exists()
        assert Path("data/vectorstore/metadata.json").exists()
        assert Path("data/models/regression_linear.joblib").exists()
        assert Path("data/models/regression_random_forest.joblib").exists()
        assert Path("data/models/classification_random_forest.joblib").exists()
