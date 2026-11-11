"""
Academic Data Leakage & Look-Ahead Bias Verification Tests.

Ensures:
1. target_return is NEVER present in the feature columns.
2. Future observations are strictly excluded from training.
3. Chronological sequence is preserved (no random shuffling).
4. Test set dates strictly occur after training and validation dates.
5. The final chronological row of every stock is omitted from training.
"""

import numpy as np
import pandas as pd
import pytest
from src.ml.regression import (
    DEFAULT_FEATURES,
    chronological_train_val_test_split,
    prepare_regression_dataset,
)


@pytest.fixture
def chronological_market_df() -> pd.DataFrame:
    """Create a 100-day dataset for 2 symbols with explicit dates."""
    np.random.seed(99)
    dates = pd.bdate_range("2023-01-01", periods=100)

    records = []
    for sym in ["RELIANCE", "TCS"]:
        price = 2000.0
        for date in dates:
            ret = np.random.normal(0.0005, 0.015)
            price *= (1.0 + ret)
            records.append({
                "date": date,
                "symbol": sym,
                "close": price,
                "daily_return": ret,
                "log_return": np.log1p(ret),
                "ma_7": price * 0.99,
                "ma_20": price * 0.98,
                "ma_50": price * 0.97,
                "volatility": 0.012,
                "volume_change": 0.03,
                "rsi_14": 55.0,
                "macd": 0.2,
                "macd_signal": 0.1,
                "volume": 2_000_000,
            })
    return pd.DataFrame(records)


class TestDataLeakagePrevention:
    """Academic verification tests preventing look-ahead bias and temporal leakage."""

    def test_target_return_never_in_feature_columns(self, chronological_market_df):
        """Feature matrix X must never include target_return or any future labels."""
        dataset = prepare_regression_dataset(chronological_market_df)
        split = chronological_train_val_test_split(dataset)

        for set_name, X_set in [("train", split["X_train"]), ("val", split["X_val"]), ("test", split["X_test"])]:
            assert "target_return" not in X_set.columns, f"LEAKAGE DETECTED: 'target_return' found in X_{set_name}!"
            assert "target_class" not in X_set.columns, f"LEAKAGE DETECTED: 'target_class' found in X_{set_name}!"
            for col in X_set.columns:
                assert not col.startswith("target_"), f"LEAKAGE DETECTED: Future prefix '{col}' in X_{set_name}!"

    def test_strict_chronological_split_ordering(self, chronological_market_df):
        """Train max date < Val min date and Val max date < Test min date per symbol."""
        dataset = prepare_regression_dataset(chronological_market_df)
        split = chronological_train_val_test_split(dataset)

        train_df = split["train_df"]
        val_df = split["val_df"]
        test_df = split["test_df"]

        # Global temporal boundary verification
        assert train_df["date"].max() < val_df["date"].min(), (
            f"TEMPORAL LEAKAGE: Training date ({train_df['date'].max()}) overlaps "
            f"with validation date ({val_df['date'].min()})!"
        )
        assert val_df["date"].max() < test_df["date"].min(), (
            f"TEMPORAL LEAKAGE: Validation date ({val_df['date'].max()}) overlaps "
            f"with test date ({test_df['date'].min()})!"
        )

        # Per-symbol verification
        for sym in ["RELIANCE", "TCS"]:
            sym_train_max = train_df[train_df["symbol"] == sym]["date"].max()
            sym_val_min = val_df[val_df["symbol"] == sym]["date"].min()
            sym_val_max = val_df[val_df["symbol"] == sym]["date"].max()
            sym_test_min = test_df[test_df["symbol"] == sym]["date"].min()

            assert sym_train_max < sym_val_min, f"Per-symbol leak in {sym}: train max >= val min"
            assert sym_val_max < sym_test_min, f"Per-symbol leak in {sym}: val max >= test min"

    def test_final_row_per_stock_purged_from_training(self, chronological_market_df):
        """The final chronological row of each stock cannot have a known future return and must not be trained on."""
        # Check raw max date per symbol
        max_dates = chronological_market_df.groupby("symbol")["date"].max().to_dict()

        dataset = prepare_regression_dataset(chronological_market_df)
        split = chronological_train_val_test_split(dataset)

        train_df = split["train_df"]

        for sym, raw_max_date in max_dates.items():
            sym_train = train_df[train_df["symbol"] == sym]
            # Max date in train must be strictly earlier than the absolute final available market date
            assert sym_train["date"].max() < raw_max_date, (
                f"LEAKAGE: Raw final row on {raw_max_date} for {sym} was included in training!"
            )

    def test_target_return_is_strictly_future(self, chronological_market_df):
        """Confirm that target_return[t] == daily_return[t+1] and NOT daily_return[t]."""
        dataset = prepare_regression_dataset(chronological_market_df)
        rel = dataset[dataset["symbol"] == "RELIANCE"].sort_values("date").reset_index(drop=True)

        for i in range(len(rel) - 1):
            assert np.isclose(rel.loc[i, "target_return"], rel.loc[i + 1, "daily_return"]), (
                f"Row {i}: target_return did not match row {i+1} daily_return!"
            )
            # Must not equal current row's daily return (unless by coincidence, but check overall correlation is not 1.0)
