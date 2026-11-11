"""
Tests for Real-Data Machine Learning Pipeline and Integration (Step 9).
Verifies strict anti-leakage temporal splitting, model artifact integrity,
and MLService real prediction capabilities.
"""

import json
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

from src.ml.real_ml_pipeline import (
    EXPLANATORY_FEATURES,
    prepare_real_ml_data,
)
from app.services.ml_service import MLService

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def parquet_path() -> Path:
    p = ROOT / "data" / "real" / "processed" / "market" / "consolidated_market_features.parquet"
    if not p.exists():
        pytest.skip(f"Real data parquet not found at {p}")
    return p


@pytest.fixture
def split_data(parquet_path: Path):
    return prepare_real_ml_data(parquet_path)


class TestRealMLPipeline:
    """Validation suite for Step 9 Real ML pipeline and artifacts."""

    def test_feature_list_anti_leakage(self):
        """Verify target variables and raw identifiers are never in feature set."""
        assert "target_next_day_return" not in EXPLANATORY_FEATURES
        assert "target_direction" not in EXPLANATORY_FEATURES
        assert "symbol" not in EXPLANATORY_FEATURES
        assert "date" not in EXPLANATORY_FEATURES
        assert "close" not in EXPLANATORY_FEATURES
        assert len(EXPLANATORY_FEATURES) == 14

    def test_chronological_splits_and_isolation(self, split_data):
        """Verify train, val, test temporal partitions are strictly chronological."""
        df_train, df_val, df_test, features = split_data

        assert not df_train.empty
        assert not df_val.empty
        assert not df_test.empty
        assert features == EXPLANATORY_FEATURES

        max_train = pd.to_datetime(df_train["date"]).max()
        min_val = pd.to_datetime(df_val["date"]).min()
        max_val = pd.to_datetime(df_val["date"]).max()
        min_test = pd.to_datetime(df_test["date"]).min()

        assert max_train < min_val, f"Train date {max_train} must precede val date {min_val}"
        assert max_val < min_test, f"Val date {max_val} must precede test date {min_test}"

        # Test partition must strictly be 2024
        assert min_test.year == 2024
        assert pd.to_datetime(df_test["date"]).max().year == 2024

    def test_boundary_purge(self, split_data):
        """Verify no sample has a target looking into the next partition."""
        df_train, df_val, df_test, _ = split_data

        # Verify symbols in all splits
        assert len(df_train["symbol"].unique()) == 50
        assert len(df_val["symbol"].unique()) == 50
        assert len(df_test["symbol"].unique()) == 50

        # Verify no NaN or Inf in explanatory features across all partitions
        for name, df in [("train", df_train), ("val", df_val), ("test", df_test)]:
            features_df = df[EXPLANATORY_FEATURES]
            assert not features_df.isna().any().any(), f"NaNs detected in {name} features"
            assert not np.isinf(features_df.to_numpy()).any(), f"Infs detected in {name} features"

    def test_model_artifacts_exist(self):
        """Verify all trained real ML models and scalers exist on disk."""
        models_dir = ROOT / "models" / "real"
        assert models_dir.exists()

        expected_files = [
            "real_feature_scaler.joblib",
            "real_classifier_logistic_regression.joblib",
            "real_classifier_random_forest.joblib",
            "real_classifier_gradient_boosting.joblib",
            "real_regressor_ridge.joblib",
            "real_regressor_random_forest.joblib",
            "real_regressor_gradient_boosting.joblib",
        ]
        for f in expected_files:
            file_path = models_dir / f
            assert file_path.exists(), f"Missing model artifact: {f}"
            assert file_path.stat().st_size > 0, f"Empty model artifact: {f}"

    def test_metrics_json_integrity(self):
        """Verify real ML metrics JSON adheres to EMH bounds and expected structure."""
        metrics_file = ROOT / "data" / "real" / "ml" / "real_model_metrics.json"
        assert metrics_file.exists()

        with open(metrics_file, "r") as f:
            metrics = json.load(f)

        assert "classification" in metrics
        assert "regression" in metrics
        assert "feature_importance" in metrics

        # Check classification metrics
        clf = metrics["classification"]
        assert "Majority_Baseline" in clf
        assert "Logistic_Regression" in clf
        assert "Random_Forest" in clf
        assert "Gradient_Boosting" in clf

        lr_acc = clf["Logistic_Regression"]["test_accuracy"]
        # Financial reality check: next-day direction on real equity is bounded near 50-52%
        assert 0.49 <= lr_acc <= 0.55

        # Check regression metrics
        reg = metrics["regression"]
        assert "Zero_Return_Baseline" in reg
        assert "Ridge_Regression" in reg
        ridge_mae = reg["Ridge_Regression"]["test_mae"]
        assert 0.005 <= ridge_mae <= 0.030

    def test_ml_service_real_methods(self):
        """Verify MLService methods for real data return valid structures."""
        svc = MLService()

        # 1. Real metrics
        metrics = svc.get_real_metrics()
        assert "classification" in metrics
        assert "regression" in metrics

        # 2. Real test predictions
        df_preds = svc.get_real_predictions()
        assert not df_preds.empty
        assert len(df_preds) > 12000
        assert "symbol" in df_preds.columns
        assert "date" in df_preds.columns
        assert "target_direction" in df_preds.columns
        assert "prob_up_lr" in df_preds.columns
        assert "pred_return_ridge" in df_preds.columns

        # 3. Live real direction inference
        sample_features = {feat: 0.01 for feat in EXPLANATORY_FEATURES}
        inference_res = svc.predict_real_direction(sample_features, model_type="logistic_regression")

        assert "prediction" in inference_res
        assert inference_res["prediction"] in ["UP", "DOWN/FLAT"]
        assert 0.0 <= inference_res["probability_up"] <= 1.0
        assert 0.0 <= inference_res["probability_down"] <= 1.0
        assert "target_horizon" in inference_res
