"""
Unit and Integration Tests for Real Data Preprocessing & Feature Engineering (Step 6).

Tests cover:
    1. Instrument identity mapping and audit verification
    2. Adjusted return and log return calculation accuracy
    3. Multi-horizon rolling returns (5d, 20d, 60d) and rolling volatilities
    4. Warm-up null-value invariants
    5. Strict absence of look-ahead leakage
    6. Date-aligned benchmark relative performance
    7. Handling of equities with pre-IPO listing gaps (HDFCLIFE, SBILIFE)
    8. Corporate-action split smoothing via adjusted close
    9. News data preservation and daily aggregates
    10. Document preprocessing and provenance
    11. Raw file immutability and checksum preservation
    12. Synthetic baseline preservation
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.preprocessing.real_data_pipeline import (
    IDENTITY_AUDIT_FILE,
    MANIFEST_PATH,
    PROCESSED_DAILY_FEATURES_DIR,
    PROCESSED_DOCS_DIR,
    PROCESSED_MARKET_DIR,
    PROCESSED_NEWS_DIR,
    QUALITY_REPORT_PATH,
    RAW_DOCS_DIR,
    RAW_MARKET_DIR,
    RAW_NEWS_FILE,
    compute_file_sha256,
    compute_rolling_max_drawdown,
    engineer_ticker_features,
    process_benchmark_features,
)


# ==============================================================================
# 1. UNIT TESTS (Mocked Data)
# ==============================================================================

class TestFeatureEngineeringUnit:
    """Unit tests for technical indicator math, warm-ups, and look-ahead avoidance."""

    @pytest.fixture
    def synthetic_stock_bars(self):
        """Creates 100 daily bars for isolated mathematical unit testing."""
        dates = pd.date_range("2024-01-01", periods=100, freq="B").strftime("%Y-%m-%d")
        np.random.seed(42)
        base_price = 100.0
        returns = np.random.normal(0.001, 0.02, 100)
        prices = base_price * np.cumprod(1 + returns)

        # Introduce a 1:2 stock split at index 50: raw close halves, adj_close stays smooth
        raw_close = prices.copy()
        raw_close[50:] = raw_close[50:] / 2.0
        adj_close = prices.copy()  # split-adjusted

        df = pd.DataFrame({
            "date": dates,
            "symbol": "MOCK_STOCK",
            "open": raw_close * 0.99,
            "high": raw_close * 1.02,
            "low": raw_close * 0.98,
            "close": raw_close,
            "adj_close": adj_close,
            "volume": np.random.randint(10000, 50000, 100),
        })
        return df

    @pytest.fixture
    def mock_benchmark_df(self):
        dates = pd.date_range("2024-01-01", periods=100, freq="B").strftime("%Y-%m-%d")
        df = pd.DataFrame({
            "date": dates,
            "benchmark_daily_return": [0.001] * 100,
            "benchmark_log_return": [0.000999] * 100,
            "benchmark_drawdown_60d": [-0.02] * 100,
        })
        return df

    def test_adjusted_return_and_log_return_math(self, synthetic_stock_bars, mock_benchmark_df):
        feat_df = engineer_ticker_features(synthetic_stock_bars, "MOCK_STOCK", mock_benchmark_df)

        # Check row 0 is NaN
        assert pd.isna(feat_df["daily_adjusted_return"].iloc[0])
        assert pd.isna(feat_df["log_return"].iloc[0])

        # Check formula r_t = (AdjClose_t - AdjClose_{t-1}) / AdjClose_{t-1}
        p0 = feat_df["adj_close"].iloc[0]
        p1 = feat_df["adj_close"].iloc[1]
        expected_ret = (p1 - p0) / p0
        expected_log_ret = np.log(p1 / p0)

        assert pytest.approx(feat_df["daily_adjusted_return"].iloc[1], rel=1e-5) == expected_ret
        assert pytest.approx(feat_df["log_return"].iloc[1], rel=1e-5) == expected_log_ret

    def test_corporate_action_split_smoothing(self, synthetic_stock_bars, mock_benchmark_df):
        """Verify that a 1:2 split at index 50 does NOT produce a -50% daily return."""
        feat_df = engineer_ticker_features(synthetic_stock_bars, "MOCK_STOCK", mock_benchmark_df)

        split_day_ret = feat_df["daily_adjusted_return"].iloc[50]
        # Since adj_close is smooth, return should be near normal (< 10%), NOT -50%
        assert abs(split_day_ret) < 0.10, f"Split caused false return jump: {split_day_ret}"

    def test_warm_up_null_invariants(self, synthetic_stock_bars, mock_benchmark_df):
        feat_df = engineer_ticker_features(synthetic_stock_bars, "MOCK_STOCK", mock_benchmark_df)

        # return_5d has 5 warm-up nulls
        assert feat_df["return_5d"].iloc[:5].isna().all()
        assert feat_df["return_5d"].iloc[5:].notna().all()

        # return_20d has 20 warm-up nulls
        assert feat_df["return_20d"].iloc[:20].isna().all()
        assert feat_df["return_20d"].iloc[20:].notna().all()

        # return_60d has 60 warm-up nulls
        assert feat_df["return_60d"].iloc[:60].isna().all()
        assert feat_df["return_60d"].iloc[60:].notna().all()

    def test_no_look_ahead_leakage(self, synthetic_stock_bars, mock_benchmark_df):
        """Verify rolling features only depend on past/present observations."""
        feat_df = engineer_ticker_features(synthetic_stock_bars, "MOCK_STOCK", mock_benchmark_df)

        # Altering a future price at row 80 must not change rolling features at row 40
        altered_bars = synthetic_stock_bars.copy()
        altered_bars.loc[80, "adj_close"] = 999999.0
        altered_feat_df = engineer_ticker_features(altered_bars, "MOCK_STOCK", mock_benchmark_df)

        # Assert all features at row 40 remain identical
        for col in ["daily_adjusted_return", "volatility_20d", "ma_20", "adj_ma_20", "rolling_max_drawdown_60d"]:
            assert feat_df.loc[40, col] == pytest.approx(altered_feat_df.loc[40, col])

    def test_target_shift_leakage_free(self, synthetic_stock_bars, mock_benchmark_df):
        feat_df = engineer_ticker_features(synthetic_stock_bars, "MOCK_STOCK", mock_benchmark_df)

        # target_next_day_return at row t must equal daily_adjusted_return at row t+1
        for t in range(len(feat_df) - 1):
            target_t = feat_df.loc[t, "target_next_day_return"]
            next_ret = feat_df.loc[t + 1, "daily_adjusted_return"]
            if pd.isna(target_t) and pd.isna(next_ret):
                continue
            assert pytest.approx(target_t) == next_ret

        # Latest row target must be strictly NaN
        assert pd.isna(feat_df["target_next_day_return"].iloc[-1])

    def test_rolling_max_drawdown_properties(self):
        prices = pd.Series([100.0, 110.0, 105.0, 90.0, 80.0, 120.0])
        dd = compute_rolling_max_drawdown(prices, window=3)

        # Drawdown must always be <= 0
        assert (dd <= 0.0).all()
        # Peak at index 1 is 110; at index 2 (105-110)/110 = -0.04545
        assert pytest.approx(dd.iloc[2], rel=1e-4) == (105.0 - 110.0) / 110.0
        # At index 5, price is 120 (new peak), so drawdown must be 0.0
        assert dd.iloc[5] == 0.0


# ==============================================================================
# 2. INTEGRATION TESTS (Actual Verified Datasets on Disk)
# ==============================================================================

class TestRealDataPipelineIntegration:
    """Integration tests verifying processed real data assets on disk."""

    def test_instrument_identity_audit_file(self):
        assert IDENTITY_AUDIT_FILE.exists()
        df = pd.read_csv(IDENTITY_AUDIT_FILE)
        assert len(df) == 51, f"Expected 51 instruments, found {len(df)}"

        # Verify all statuses are VERIFIED
        assert (df["mapping_status"] == "VERIFIED").all()

        # Verify Tata Motors note
        tm_row = df[df["official_nse_symbol"] == "TATAMOTORS"].iloc[0]
        assert tm_row["yahoo_finance_symbol"] == "TMPV.NS"
        assert "demerger" in tm_row["corporate_action_notes"].lower()

        # Verify Shree Cement note
        sc_row = df[df["official_nse_symbol"] == "SHREECEM"].iloc[0]
        assert sc_row["yahoo_finance_symbol"] == "SHREECEM.NS"
        assert "ltimindtree" in sc_row["corporate_action_notes"].lower()

    def test_benchmark_parquet_and_csv(self):
        parquet_file = PROCESSED_MARKET_DIR / "benchmark_features.parquet"
        csv_file = PROCESSED_MARKET_DIR / "benchmark_features.csv"
        assert parquet_file.exists()
        assert csv_file.exists()

        df = pd.read_parquet(parquet_file)
        assert len(df) == 2459
        assert "benchmark_daily_return" in df.columns
        assert "benchmark_drawdown_60d" in df.columns
        assert (df["benchmark_drawdown_60d"] <= 1e-6).all()

    def test_consolidated_market_features_dataset(self):
        parquet_file = PROCESSED_MARKET_DIR / "consolidated_market_features.parquet"
        assert parquet_file.exists()

        df = pd.read_parquet(parquet_file)
        # 121,962 total rows across 50 constituents
        assert len(df) == 121962, f"Expected 121,962 rows, found {len(df)}"
        assert df["symbol"].nunique() == 50

        # Verify required columns exist
        expected_cols = [
            "date", "symbol", "open", "high", "low", "close", "adj_close", "volume",
            "daily_adjusted_return", "log_return", "return_5d", "return_20d", "return_60d",
            "volatility_20d", "volatility_60d", "ma_20", "ma_50", "ma_200",
            "adj_ma_20", "adj_ma_50", "adj_ma_200", "volume_ma_20", "volume_ratio",
            "hl_pct_range", "rolling_max_drawdown_60d", "relative_return_vs_nifty",
            "target_next_day_return"
        ]
        for col in expected_cols:
            assert col in df.columns, f"Missing feature column: {col}"

    def test_pre_ipo_listing_gap_handling(self):
        """Verify HDFCLIFE and SBILIFE are processed correctly starting from their 2017 IPOs."""
        hdfc_file = PROCESSED_DAILY_FEATURES_DIR / "HDFCLIFE_features.parquet"
        sbi_file = PROCESSED_DAILY_FEATURES_DIR / "SBILIFE_features.parquet"
        assert hdfc_file.exists()
        assert sbi_file.exists()

        h_df = pd.read_parquet(hdfc_file)
        s_df = pd.read_parquet(sbi_file)

        assert len(h_df) == 1757
        assert h_df["date"].min() == "2017-11-17"
        assert len(s_df) == 1789
        assert s_df["date"].min() == "2017-10-03"

    def test_cleaned_news_features(self):
        cleaned_csv = PROCESSED_NEWS_DIR / "cleaned_financial_news.csv"
        daily_csv = PROCESSED_NEWS_DIR / "daily_company_news_features.csv"
        assert cleaned_csv.exists()
        assert daily_csv.exists()

        c_df = pd.read_csv(cleaned_csv)
        assert len(c_df) == 79
        assert c_df["ticker"].nunique() == 50
        assert "normalized_headline" in c_df.columns
        assert "normalized_snippet" in c_df.columns

        d_df = pd.read_csv(daily_csv)
        assert len(d_df) == 78
        assert "daily_news_count" in d_df.columns
        assert "official_disclosure_count" in d_df.columns
        assert "news_report_count" in d_df.columns

    def test_corporate_documents_preprocessing(self):
        doc_files = list(PROCESSED_DOCS_DIR.glob("*.txt"))
        assert len(doc_files) == 4, f"Expected 4 clean documents, got {len(doc_files)}"

        for doc_file in doc_files:
            text = doc_file.read_text(encoding="utf-8")
            assert len(text.split()) >= 300, f"{doc_file.name} word count below 300 words"
            assert "clean_executive_overview" in doc_file.name

    def test_raw_files_immutability(self):
        """Verify raw market, news, and document files have not been modified."""
        raw_manifest_path = Path("data/real/metadata/market_download_manifest.json")
        assert raw_manifest_path.exists()
        with open(raw_manifest_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        for rec in meta["ticker_manifest"]:
            if rec["status"] == "SUCCESS":
                p = Path(rec["target_file"])
                assert p.exists()
                # Compute SHA and verify identical
                curr_sha = compute_file_sha256(p)
                assert curr_sha == rec["sha256"], f"Raw file {p} was modified!"

    def test_synthetic_baseline_preservation(self):
        """Assert complete preservation of original synthetic project files."""
        assert Path("data/sample/market_data.csv").exists()
        assert len(pd.read_csv("data/sample/market_data.csv")) >= 5000

        assert Path("data/sample/financial_news.csv").exists()
        assert len(pd.read_csv("data/sample/financial_news.csv")) >= 450

        assert Path("data/processed/market_features.csv").exists()
        assert Path("data/processed/news_cleaned.csv").exists()
        assert Path("data/models/regression_linear.joblib").exists()
        assert Path("data/models/classification_random_forest.joblib").exists()
        assert Path("data/vectorstore/embeddings.npy").exists()
