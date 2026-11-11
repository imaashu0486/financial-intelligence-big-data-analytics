"""Tests for src.ml.regression pipeline."""

import numpy as np
import pandas as pd
import pytest
from src.ml.regression import (
    DEFAULT_FEATURES,
    chronological_train_val_test_split,
    get_feature_importance,
    prepare_regression_dataset,
    train_linear_regression,
    train_random_forest_regressor,
)


@pytest.fixture
def synthetic_market_data() -> pd.DataFrame:
    """Create a 100-row synthetic market dataset across 2 symbols."""
    np.random.seed(42)
    dates = pd.bdate_range("2023-01-01", periods=50)

    records = []
    for symbol in ["AAPL", "GOOG"]:
        price = 100.0
        for i, date in enumerate(dates):
            ret = np.random.normal(0.001, 0.02)
            price = price * (1.0 + ret)
            records.append({
                "date": date,
                "symbol": symbol,
                "close": price,
                "daily_return": ret,
                "log_return": np.log1p(ret),
                "ma_7": price * 0.99,
                "ma_20": price * 0.98,
                "ma_50": price * 0.97,
                "volatility": 0.015,
                "volume_change": 0.05,
                "rsi_14": 52.0,
                "macd": 0.5,
                "macd_signal": 0.4,
                "volume": 1_000_000,
            })

    return pd.DataFrame(records)


class TestRegressionPipeline:
    """Unit tests for regression components."""

    def test_prepare_regression_dataset_target_creation(self, synthetic_market_data):
        """Verify target_return is next trading day return per symbol."""
        df = prepare_regression_dataset(synthetic_market_data)

        # Confirm target_return exists
        assert "target_return" in df.columns

        # For AAPL, row 0 target_return should equal row 1 daily_return
        aapl = df[df["symbol"] == "AAPL"].sort_values("date").reset_index(drop=True)
        assert np.isclose(aapl.loc[0, "target_return"], aapl.loc[1, "daily_return"])

    def test_prepare_regression_dataset_drops_last_row(self, synthetic_market_data):
        """The last chronological row for each symbol has no future return and must be dropped."""
        df = prepare_regression_dataset(synthetic_market_data)

        # 50 rows each -> 1 dropped per symbol -> 49 rows each -> 98 total
        assert len(df) == 98
        assert df.groupby("symbol").size().to_dict() == {"AAPL": 49, "GOOG": 49}

    def test_chronological_split_shapes_and_dates(self, synthetic_market_data):
        """Verify chronological split splits 70% train, 15% val, 15% test in order."""
        dataset = prepare_regression_dataset(synthetic_market_data)
        split = chronological_train_val_test_split(
            dataset,
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
        )

        X_train, y_train = split["X_train"], split["y_train"]
        X_val, y_val = split["X_val"], split["y_val"]
        X_test, y_test = split["X_test"], split["y_test"]

        # 49 rows per symbol:
        # train: int(49 * 0.7) = 34 * 2 = 68
        # val: int(49 * 0.15) = 7 * 2 = 14
        # test: (49 - 34 - 7) = 8 * 2 = 16
        assert len(X_train) == 68
        assert len(X_val) == 14
        assert len(X_test) == 16
        assert len(X_train) + len(X_val) + len(X_test) == 98

        # Date ordering: Max train date < Min val date; Max val date < Min test date
        train_df = split["train_df"]
        val_df = split["val_df"]
        test_df = split["test_df"]

        assert train_df["date"].max() < val_df["date"].min()
        assert val_df["date"].max() < test_df["date"].min()

    def test_train_linear_regression(self, synthetic_market_data):
        dataset = prepare_regression_dataset(synthetic_market_data)
        split = chronological_train_val_test_split(dataset)

        model = train_linear_regression(split["X_train"], split["y_train"])
        preds = model.predict(split["X_test"])

        assert len(preds) == len(split["X_test"])
        assert not np.isnan(preds).any()

    def test_train_random_forest_regressor_and_feature_importance(self, synthetic_market_data):
        dataset = prepare_regression_dataset(synthetic_market_data)
        split = chronological_train_val_test_split(dataset)

        model = train_random_forest_regressor(
            split["X_train"],
            split["y_train"],
            n_estimators=10,
            max_depth=3,
        )
        preds = model.predict(split["X_test"])
        assert len(preds) == len(split["X_test"])

        # Feature importance
        fi = get_feature_importance(model, split["feature_cols"])
        assert list(fi.columns) == ["feature", "importance"]
        assert len(fi) == len(DEFAULT_FEATURES)
        assert np.isclose(fi["importance"].sum(), 1.0, atol=1e-3)
