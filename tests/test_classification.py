"""Tests for src.ml.classification pipeline."""

import numpy as np
import pandas as pd
import pytest
from src.ml.classification import (
    CLASS_ORDER,
    chronological_classification_split,
    prepare_classification_dataset,
    train_logistic_regression,
    train_random_forest_classifier,
)
from src.ml.evaluation import calculate_classification_metrics


@pytest.fixture
def classification_sample_data() -> pd.DataFrame:
    """Create a synthetic dataset with distinct positive, neutral, and negative returns."""
    np.random.seed(123)
    dates = pd.bdate_range("2023-01-01", periods=60)

    records = []
    for symbol in ["INFY", "TCS"]:
        price = 1000.0
        # Alternating returns: positive (>0.01), near zero (0.001), negative (<-0.01)
        for i, date in enumerate(dates):
            if i % 3 == 0:
                ret = 0.015  # Bullish
            elif i % 3 == 1:
                ret = -0.012  # Bearish
            else:
                ret = 0.001  # Neutral

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
                "volume_change": 0.02,
                "rsi_14": 50.0,
                "macd": 0.1,
                "macd_signal": 0.05,
                "volume": 500_000,
            })

    return pd.DataFrame(records)


class TestClassificationPipeline:
    """Unit tests for classification pipeline."""

    def test_prepare_classification_dataset_labels(self, classification_sample_data):
        df = prepare_classification_dataset(classification_sample_data, threshold=0.005)

        assert "target_class" in df.columns
        unique_classes = set(df["target_class"].unique())
        # All three classes should be present
        assert unique_classes == {"Bullish", "Bearish", "Neutral"}

        # Verify threshold mapping
        bullish_sample = df[df["target_return"] > 0.005]
        assert (bullish_sample["target_class"] == "Bullish").all()

        bearish_sample = df[df["target_return"] < -0.005]
        assert (bearish_sample["target_class"] == "Bearish").all()

        neutral_sample = df[df["target_return"].abs() <= 0.005]
        assert (neutral_sample["target_class"] == "Neutral").all()

    def test_train_logistic_regression(self, classification_sample_data):
        df = prepare_classification_dataset(classification_sample_data, threshold=0.005)
        split = chronological_classification_split(df)

        model = train_logistic_regression(split["X_train"], split["y_train"])
        preds = model.predict(split["X_test"])

        assert len(preds) == len(split["X_test"])
        metrics = calculate_classification_metrics(split["y_test"], preds, labels=CLASS_ORDER)
        assert "accuracy" in metrics
        assert "f1_macro" in metrics

    def test_train_random_forest_classifier(self, classification_sample_data):
        df = prepare_classification_dataset(classification_sample_data, threshold=0.005)
        split = chronological_classification_split(df)

        model = train_random_forest_classifier(
            split["X_train"],
            split["y_train"],
            n_estimators=10,
            max_depth=3,
        )
        preds = model.predict(split["X_test"])

        assert len(preds) == len(split["X_test"])
        metrics = calculate_classification_metrics(split["y_test"], preds, labels=CLASS_ORDER)
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert len(metrics["confusion_matrix"]) == 3
