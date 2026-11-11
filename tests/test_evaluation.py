"""Tests for src.ml.evaluation metrics and baselines."""

import numpy as np
import pytest
from src.ml.evaluation import (
    calculate_classification_metrics,
    calculate_clustering_metrics,
    calculate_regression_metrics,
    evaluate_majority_baseline,
    evaluate_zero_return_baseline,
)


class TestRegressionEvaluation:
    """Unit tests for regression evaluation metrics."""

    def test_calculate_regression_metrics_perfect_prediction(self):
        y_true = np.array([0.01, -0.02, 0.015, 0.0])
        metrics = calculate_regression_metrics(y_true, y_true)
        assert metrics["mae"] == 0.0
        assert metrics["rmse"] == 0.0
        assert metrics["r2"] == 1.0

    def test_calculate_regression_metrics_known_values(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 1.9, 3.1])
        metrics = calculate_regression_metrics(y_true, y_pred)
        assert np.isclose(metrics["mae"], 0.1, atol=1e-4)
        assert np.isclose(metrics["rmse"], 0.1, atol=1e-4)
        assert metrics["r2"] > 0.95

    def test_empty_input_raises(self):
        with pytest.raises(ValueError, match="y_true cannot be empty"):
            calculate_regression_metrics([], [])

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="Shape mismatch"):
            calculate_regression_metrics([1.0, 2.0], [1.0])

    def test_evaluate_zero_return_baseline(self):
        y_true = np.array([0.02, -0.01, 0.03, -0.02])
        metrics = evaluate_zero_return_baseline(y_true)
        assert metrics["model"] == "zero_return_baseline"
        assert metrics["mae"] > 0
        assert metrics["rmse"] > 0


class TestClassificationEvaluation:
    """Unit tests for classification evaluation metrics."""

    def test_calculate_classification_metrics_perfect(self):
        y_true = ["Bullish", "Bearish", "Neutral", "Bullish"]
        metrics = calculate_classification_metrics(y_true, y_true)
        assert metrics["accuracy"] == 1.0
        assert metrics["f1_macro"] == 1.0
        assert len(metrics["confusion_matrix"]) == 3

    def test_calculate_classification_metrics_partial(self):
        y_true = ["Bullish", "Neutral", "Bearish"]
        y_pred = ["Bullish", "Bearish", "Neutral"]
        metrics = calculate_classification_metrics(y_true, y_pred)
        assert metrics["accuracy"] == pytest.approx(1 / 3, abs=0.01)
        assert "classification_report" in metrics

    def test_evaluate_majority_baseline(self):
        y_train = ["Bullish", "Bullish", "Neutral", "Bearish"]  # Bullish is majority
        y_test = ["Bullish", "Neutral", "Bearish", "Bullish"]
        metrics = evaluate_majority_baseline(y_train, y_test)
        assert metrics["model"] == "majority_class_baseline"
        assert metrics["majority_class"] == "Bullish"
        assert metrics["accuracy"] == 0.5  # 2 out of 4 are Bullish


class TestClusteringEvaluation:
    """Unit tests for clustering evaluation metrics."""

    def test_calculate_clustering_metrics_valid(self):
        # Two clearly separated synthetic clusters
        X = np.array([
            [1.0, 1.0],
            [1.1, 0.9],
            [5.0, 5.0],
            [5.1, 4.9],
        ])
        labels = np.array([0, 0, 1, 1])
        metrics = calculate_clustering_metrics(X, labels, inertia=0.04)
        assert metrics["k"] == 2
        assert metrics["inertia"] == 0.04
        assert metrics["silhouette_score"] is not None
        assert metrics["silhouette_score"] > 0.8
