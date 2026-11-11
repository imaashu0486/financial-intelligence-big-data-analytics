"""Tests for src.ingestion.market_loader"""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.ingestion.market_loader import MarketDataError, load_market_data


@pytest.fixture
def valid_csv(tmp_path: Path) -> Path:
    """Create a minimal valid market CSV."""
    df = pd.DataFrame({
        "date": ["2023-01-02", "2023-01-03", "2023-01-04"],
        "symbol": ["TCS", "TCS", "TCS"],
        "open": [3500.0, 3520.0, 3510.0],
        "high": [3550.0, 3560.0, 3540.0],
        "low": [3480.0, 3500.0, 3490.0],
        "close": [3530.0, 3540.0, 3520.0],
        "volume": [1000000, 1200000, 900000],
    })
    path = tmp_path / "market.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def csv_with_bad_dates(tmp_path: Path) -> Path:
    """CSV with some unparseable dates."""
    df = pd.DataFrame({
        "date": ["2023-01-02", "NOT_A_DATE", "2023-01-04"],
        "symbol": ["TCS", "TCS", "TCS"],
        "open": [3500.0, 3520.0, 3510.0],
        "high": [3550.0, 3560.0, 3540.0],
        "low": [3480.0, 3500.0, 3490.0],
        "close": [3530.0, 3540.0, 3520.0],
        "volume": [1000000, 1200000, 900000],
    })
    path = tmp_path / "bad_dates.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def csv_with_negative_prices(tmp_path: Path) -> Path:
    """CSV with negative price values."""
    df = pd.DataFrame({
        "date": ["2023-01-02", "2023-01-03"],
        "symbol": ["TCS", "TCS"],
        "open": [3500.0, -100.0],
        "high": [3550.0, 3560.0],
        "low": [3480.0, 3500.0],
        "close": [3530.0, 3540.0],
        "volume": [1000000, 1200000],
    })
    path = tmp_path / "neg_prices.csv"
    df.to_csv(path, index=False)
    return path


class TestLoadMarketData:
    """Tests for load_market_data."""

    def test_valid_csv_loads(self, valid_csv):
        """Valid CSV should load with correct shape and columns."""
        df = load_market_data(valid_csv)
        assert len(df) == 3
        assert set(df.columns) >= {"date", "symbol", "open", "high", "low", "close", "volume"}
        assert df["symbol"].unique().tolist() == ["TCS"]

    def test_dates_parsed_correctly(self, valid_csv):
        """Dates should be parsed as datetime."""
        df = load_market_data(valid_csv)
        assert pd.api.types.is_datetime64_any_dtype(df["date"])

    def test_sorted_by_date_symbol(self, valid_csv):
        """Output should be sorted by (date, symbol)."""
        df = load_market_data(valid_csv)
        assert df["date"].is_monotonic_increasing

    def test_missing_columns_raises(self, tmp_path):
        """CSV missing required columns should raise MarketDataError."""
        df = pd.DataFrame({"date": ["2023-01-02"], "symbol": ["TCS"]})
        path = tmp_path / "missing_cols.csv"
        df.to_csv(path, index=False)
        with pytest.raises(MarketDataError, match="Missing required columns"):
            load_market_data(path)

    def test_file_not_found_raises(self):
        """Non-existent file should raise MarketDataError."""
        with pytest.raises(MarketDataError, match="File not found"):
            load_market_data("/nonexistent/path.csv")

    def test_bad_dates_dropped(self, csv_with_bad_dates):
        """Rows with unparseable dates should be dropped."""
        df = load_market_data(csv_with_bad_dates)
        assert len(df) == 2  # one bad date row dropped

    def test_negative_prices_dropped(self, csv_with_negative_prices):
        """Rows with negative prices should be dropped."""
        df = load_market_data(csv_with_negative_prices)
        assert len(df) == 1  # row with negative open dropped
        assert (df["open"] > 0).all()

    def test_volume_clamped_to_zero(self, tmp_path):
        """Negative volumes should be clamped to 0."""
        df = pd.DataFrame({
            "date": ["2023-01-02"],
            "symbol": ["TCS"],
            "open": [100.0], "high": [110.0], "low": [90.0], "close": [105.0],
            "volume": [-500],
        })
        path = tmp_path / "neg_vol.csv"
        df.to_csv(path, index=False)
        result = load_market_data(path)
        assert result["volume"].iloc[0] == 0

    def test_symbol_uppercased(self, tmp_path):
        """Symbols should be normalised to uppercase."""
        df = pd.DataFrame({
            "date": ["2023-01-02"],
            "symbol": ["  tcs  "],
            "open": [100.0], "high": [110.0], "low": [90.0], "close": [105.0],
            "volume": [1000],
        })
        path = tmp_path / "lower_sym.csv"
        df.to_csv(path, index=False)
        result = load_market_data(path)
        assert result["symbol"].iloc[0] == "TCS"
