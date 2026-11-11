"""Tests for src.preprocessing.cleaning"""

import numpy as np
import pandas as pd
import pytest

from src.preprocessing.cleaning import clean_market_data, clean_news_data


@pytest.fixture
def cleaning_cfg() -> dict:
    """Standard cleaning config."""
    return {
        "max_price": 500_000,
        "min_price": 0.01,
        "max_volume": 5_000_000_000,
        "outlier_zscore": 3.5,
        "forward_fill_limit": 5,
    }


@pytest.fixture
def sample_market_df() -> pd.DataFrame:
    """Basic market DataFrame for testing."""
    return pd.DataFrame({
        "date": pd.to_datetime(["2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05"]),
        "symbol": ["TCS", "TCS", "TCS", "TCS"],
        "open": [3500.0, 3520.0, 3510.0, 3530.0],
        "high": [3550.0, 3560.0, 3540.0, 3570.0],
        "low": [3480.0, 3500.0, 3490.0, 3510.0],
        "close": [3530.0, 3540.0, 3520.0, 3550.0],
        "volume": [1_000_000, 1_200_000, 900_000, 1_100_000],
    })


class TestCleanMarketData:
    """Tests for clean_market_data."""

    def test_removes_exact_duplicates(self, cleaning_cfg):
        """Exact duplicate rows should be removed."""
        df = pd.DataFrame({
            "date": pd.to_datetime(["2023-01-02", "2023-01-02"]),
            "symbol": ["TCS", "TCS"],
            "open": [100.0, 100.0],
            "high": [110.0, 110.0],
            "low": [90.0, 90.0],
            "close": [105.0, 105.0],
            "volume": [1000, 1000],
        })
        result = clean_market_data(df, cleaning_cfg)
        assert len(result) == 1

    def test_removes_date_symbol_duplicates(self, cleaning_cfg):
        """Duplicate (date, symbol) rows should keep last."""
        df = pd.DataFrame({
            "date": pd.to_datetime(["2023-01-02", "2023-01-02"]),
            "symbol": ["TCS", "TCS"],
            "open": [100.0, 200.0],
            "high": [110.0, 210.0],
            "low": [90.0, 190.0],
            "close": [105.0, 205.0],
            "volume": [1000, 2000],
        })
        result = clean_market_data(df, cleaning_cfg)
        assert len(result) == 1
        assert result["close"].iloc[0] == 205.0  # kept last

    def test_forward_fills_missing_prices(self, cleaning_cfg):
        """NaN prices should be forward-filled within limit."""
        df = pd.DataFrame({
            "date": pd.to_datetime(["2023-01-02", "2023-01-03", "2023-01-04"]),
            "symbol": ["TCS", "TCS", "TCS"],
            "open": [100.0, np.nan, 102.0],
            "high": [110.0, np.nan, 112.0],
            "low": [90.0, np.nan, 92.0],
            "close": [105.0, np.nan, 107.0],
            "volume": [1000, 1100, 1200],
        })
        result = clean_market_data(df, cleaning_cfg)
        assert len(result) == 3
        assert result["close"].iloc[1] == 105.0  # forward-filled

    def test_drops_insane_prices(self, cleaning_cfg):
        """Prices outside [min_price, max_price] should be dropped."""
        df = pd.DataFrame({
            "date": pd.to_datetime(["2023-01-02", "2023-01-03"]),
            "symbol": ["TCS", "TCS"],
            "open": [100.0, 999_999.0],
            "high": [110.0, 999_999.0],
            "low": [90.0, 999_999.0],
            "close": [105.0, 999_999.0],
            "volume": [1000, 2000],
        })
        result = clean_market_data(df, cleaning_cfg)
        assert len(result) == 1

    def test_outlier_flag_column_exists(self, sample_market_df, cleaning_cfg):
        """Output should contain an outlier_flag column."""
        result = clean_market_data(sample_market_df, cleaning_cfg)
        assert "outlier_flag" in result.columns

    def test_volume_capped(self, cleaning_cfg):
        """Volumes exceeding max_volume should be dropped."""
        df = pd.DataFrame({
            "date": pd.to_datetime(["2023-01-02", "2023-01-03"]),
            "symbol": ["TCS", "TCS"],
            "open": [100.0, 102.0],
            "high": [110.0, 112.0],
            "low": [90.0, 92.0],
            "close": [105.0, 107.0],
            "volume": [1000, 99_999_999_999],  # second exceeds max
        })
        result = clean_market_data(df, cleaning_cfg)
        assert len(result) == 1


class TestCleanNewsData:
    """Tests for clean_news_data."""

    def test_removes_duplicates(self):
        """Duplicate (date, title) rows should be removed."""
        df = pd.DataFrame({
            "date": pd.to_datetime(["2023-01-02", "2023-01-02"]),
            "title": ["Same title", "Same title"],
            "source": ["ET", "MC"],
            "symbol": ["TCS", "TCS"],
        })
        result = clean_news_data(df)
        assert len(result) == 1

    def test_drops_empty_titles(self):
        """Rows with empty or NaN titles should be dropped."""
        df = pd.DataFrame({
            "date": pd.to_datetime(["2023-01-02", "2023-01-03", "2023-01-04"]),
            "title": ["Good title", "", None],
            "source": ["ET", "MC", "BS"],
            "symbol": ["TCS", "TCS", "TCS"],
        })
        result = clean_news_data(df)
        assert len(result) == 1
        assert result["title"].iloc[0] == "Good title"

    def test_strips_whitespace(self):
        """Text fields should have whitespace stripped."""
        df = pd.DataFrame({
            "date": pd.to_datetime(["2023-01-02"]),
            "title": ["  padded title  "],
            "source": [" ET "],
            "symbol": [" tcs "],
        })
        result = clean_news_data(df)
        assert result["title"].iloc[0] == "padded title"
        assert result["source"].iloc[0] == "ET"
