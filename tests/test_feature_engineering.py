"""Tests for src.preprocessing.feature_engineering"""

import numpy as np
import pandas as pd
import pytest

from src.preprocessing.feature_engineering import engineer_features


@pytest.fixture
def sample_market_df() -> pd.DataFrame:
    """Create a simple market DataFrame with known values for verification."""
    dates = pd.bdate_range("2023-01-02", periods=60)
    # Linear-ish close for easy mental verification
    close = 100.0 + np.arange(60) * 0.5  # 100, 100.5, 101, ...
    return pd.DataFrame({
        "date": dates,
        "symbol": "TEST",
        "open": close - 0.2,
        "high": close + 1.0,
        "low": close - 1.0,
        "close": close,
        "volume": np.arange(1_000_000, 1_060_000, 1000),
    })


@pytest.fixture
def feature_cfg() -> dict:
    """Standard feature config."""
    return {
        "moving_averages": [7, 20],
        "rsi_period": 14,
        "macd": {"fast": 12, "slow": 26, "signal": 9},
        "volatility_window": 20,
        "min_periods": 1,
    }


class TestEngineerFeatures:
    """Tests for engineer_features."""

    def test_all_feature_columns_present(self, sample_market_df, feature_cfg):
        """All expected feature columns should be added."""
        df = engineer_features(sample_market_df, feature_cfg)
        expected = {
            "daily_return", "log_return",
            "ma_7", "ma_20",
            "volatility", "volume_change",
            "rsi_14", "macd", "macd_signal",
        }
        assert expected.issubset(set(df.columns))

    def test_no_lookahead(self, sample_market_df, feature_cfg):
        """Features at row i should not depend on future rows.

        Verify by computing features on first 30 rows and checking they
        match the first 30 rows of the full computation.
        """
        full = engineer_features(sample_market_df, feature_cfg)
        partial = engineer_features(sample_market_df.iloc[:30].copy(), feature_cfg)

        # Compare MA_7 for the overlapping rows
        full_ma7 = full["ma_7"].iloc[:30].values
        partial_ma7 = partial["ma_7"].values
        np.testing.assert_allclose(full_ma7, partial_ma7, rtol=1e-10)

    def test_daily_return_correct(self, sample_market_df, feature_cfg):
        """daily_return should equal (close - prev_close) / prev_close."""
        df = engineer_features(sample_market_df, feature_cfg)
        expected = sample_market_df["close"].pct_change().values
        actual = df["daily_return"].values
        # First value is NaN
        assert np.isnan(actual[0])
        np.testing.assert_allclose(actual[1:], expected[1:], rtol=1e-10)

    def test_log_return_correct(self, sample_market_df, feature_cfg):
        """log_return should equal ln(close / prev_close)."""
        df = engineer_features(sample_market_df, feature_cfg)
        expected = np.log(
            sample_market_df["close"].values[1:]
            / sample_market_df["close"].values[:-1]
        )
        actual = df["log_return"].values[1:]
        np.testing.assert_allclose(actual, expected, rtol=1e-10)

    def test_rsi_bounded(self, sample_market_df, feature_cfg):
        """RSI should be in [0, 100] where not NaN."""
        df = engineer_features(sample_market_df, feature_cfg)
        rsi = df["rsi_14"].dropna()
        assert (rsi >= 0).all() and (rsi <= 100).all()

    def test_moving_average_correct(self, sample_market_df, feature_cfg):
        """MA_7 at row 10 should be mean of close[4:11]."""
        df = engineer_features(sample_market_df, feature_cfg)
        # For a single stock sorted by date, MA_7 at index 10 = mean(close[4..10])
        expected_ma7 = sample_market_df["close"].iloc[4:11].mean()
        actual_ma7 = df["ma_7"].iloc[10]
        np.testing.assert_allclose(actual_ma7, expected_ma7, rtol=1e-10)

    def test_handles_multiple_symbols(self, feature_cfg):
        """Features should be computed independently per symbol."""
        dates = pd.bdate_range("2023-01-02", periods=60)
        df = pd.concat([
            pd.DataFrame({
                "date": dates, "symbol": "A",
                "open": 100.0, "high": 110.0, "low": 90.0,
                "close": np.linspace(100, 160, 60),
                "volume": 1_000_000,
            }),
            pd.DataFrame({
                "date": dates, "symbol": "B",
                "open": 200.0, "high": 210.0, "low": 190.0,
                "close": np.linspace(200, 140, 60),  # declining
                "volume": 2_000_000,
            }),
        ], ignore_index=True)

        result = engineer_features(df, feature_cfg)
        a_data = result[result["symbol"] == "A"]
        b_data = result[result["symbol"] == "B"]

        # Stock A is rising → last non-NaN RSI should be high
        # Stock B is falling → last non-NaN RSI should be low
        a_rsi = a_data["rsi_14"].dropna()
        b_rsi = b_data["rsi_14"].dropna()
        assert len(a_rsi) > 0 and a_rsi.iloc[-1] > 50
        assert len(b_rsi) > 0 and b_rsi.iloc[-1] < 50

    def test_output_shape(self, sample_market_df, feature_cfg):
        """Row count should not change; only new columns added."""
        df = engineer_features(sample_market_df, feature_cfg)
        assert len(df) == len(sample_market_df)
        assert len(df.columns) > len(sample_market_df.columns)
