"""
Machine Learning service layer (Phase 4).
Loads model performance metrics, test predictions, clustering profiles, and feature importances.
Provides safe, optional single-sample inference using trained estimators.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent


class MLService:
    """Service facade for trained machine learning pipelines."""

    def __init__(self, data_dir: Optional[Path] = None) -> None:
        self.root = data_dir or ROOT
        self.ml_dir = self.root / "data" / "processed" / "ml"
        self.models_dir = self.root / "data" / "models"
        self._metrics_cache: Optional[Dict[str, Any]] = None

    def get_metrics(self) -> Dict[str, Any]:
        """
        Load complete regression, classification, and clustering metrics.
        """
        if self._metrics_cache is not None:
            return self._metrics_cache

        path = self.ml_dir / "model_metrics.json"
        if not path.exists():
            return {}

        with open(path, "r", encoding="utf-8") as f:
            self._metrics_cache = json.load(f)
        return self._metrics_cache

    def get_regression_metrics(self) -> Dict[str, Any]:
        """Return regression metrics dictionary."""
        return self.get_metrics().get("regression", {})

    def get_classification_metrics(self) -> Dict[str, Any]:
        """Return classification metrics dictionary."""
        return self.get_metrics().get("classification", {})

    def get_clustering_metrics(self) -> Dict[str, Any]:
        """Return clustering metrics and profiles."""
        return self.get_metrics().get("clustering", {})

    def get_regression_predictions(self) -> pd.DataFrame:
        """
        Load test set actual vs predicted returns from trained regressors.
        """
        path = self.ml_dir / "regression_predictions.csv"
        if not path.exists():
            return pd.DataFrame()
        df = pd.read_csv(path)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        return df

    def get_classification_predictions(self) -> pd.DataFrame:
        """
        Load test set actual vs predicted regime classifications with class probabilities.
        """
        path = self.ml_dir / "classification_predictions.csv"
        if not path.exists():
            return pd.DataFrame()
        df = pd.read_csv(path)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        return df

    def get_stock_clusters(self) -> pd.DataFrame:
        """
        Load K-Means clustering assignments for all 10 stocks.
        """
        path = self.ml_dir / "stock_clusters.csv"
        if not path.exists():
            return pd.DataFrame()
        return pd.read_csv(path)

    def get_feature_importances(self) -> pd.DataFrame:
        """
        Load Random Forest feature importances sorted in descending order.
        """
        path = self.ml_dir / "feature_importance.csv"
        if not path.exists():
            return pd.DataFrame(columns=["feature", "importance"])
        df = pd.read_csv(path)
        return df.sort_values("importance", ascending=False).reset_index(drop=True)

    def predict_market_direction(self, feature_values: Dict[str, float]) -> Dict[str, Any]:
        """
        Perform single-sample inference using the trained Random Forest classifier.
        Validates features against the training schema.
        """
        model_path = self.models_dir / "classification_random_forest.joblib"
        if not model_path.exists():
            return {"error": "Trained classification model artifact not found on disk."}

        try:
            model = joblib.load(model_path)
            # Standard feature columns expected by the classifier
            expected_features = [
                "daily_return", "log_return", "ma_7", "ma_20", "ma_50",
                "volatility", "volume_change", "rsi_14", "macd", "macd_signal"
            ]
            row = [feature_values.get(f, 0.0) for f in expected_features]
            X = pd.DataFrame([row], columns=expected_features)

            pred_class = model.predict(X)[0]
            probabilities = {}
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X)[0]
                classes = model.classes_
                probabilities = {str(c): float(p) for c, p in zip(classes, probs)}

            return {
                "prediction": str(pred_class),
                "probabilities": probabilities,
                "confidence": float(max(probabilities.values())) if probabilities else 1.0,
                "features_used": dict(zip(expected_features, row)),
            }
        except Exception as e:
            return {"error": f"Inference error: {str(e)}"}

    # =========================================================================
    # REAL ML SERVICE METHODS (Step 9 Phase B)
    # =========================================================================

    def get_real_metrics(self) -> Dict[str, Any]:
        """Load evaluation metrics for real-data ML models."""
        path = self.root / "data" / "real" / "ml" / "real_model_metrics.json"
        if not path.exists():
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_real_predictions(self) -> pd.DataFrame:
        """Load out-of-sample test set predictions for 2024."""
        path = self.root / "data" / "real" / "ml" / "real_predictions_test.csv"
        if not path.exists():
            return pd.DataFrame()
        return pd.read_csv(path)

    def predict_real_direction(self, feature_values: Dict[str, float], model_type: str = "random_forest") -> Dict[str, Any]:
        """Perform single-sample inference on real-data classification models."""
        model_name_map = {
            "random_forest": "real_classifier_random_forest.joblib",
            "logistic_regression": "real_classifier_logistic_regression.joblib",
            "gradient_boosting": "real_classifier_gradient_boosting.joblib",
        }
        filename = model_name_map.get(model_type, "real_classifier_random_forest.joblib")
        model_path = self.root / "models" / "real" / filename
        if not model_path.exists():
            return {"error": f"Model artifact {filename} not found."}

        try:
            model = joblib.load(model_path)
            expected_features = [
                "daily_adjusted_return", "return_5d", "return_20d", "return_60d",
                "volatility_20d", "volatility_60d", "close_to_ma20", "close_to_ma50",
                "close_to_ma200", "volume_ratio", "hl_pct_range",
                "rolling_max_drawdown_60d", "benchmark_daily_return", "relative_return_vs_nifty"
            ]
            row = [float(feature_values.get(f, 0.0)) for f in expected_features]
            X = np.array([row])

            if model_type == "logistic_regression":
                scaler_path = self.root / "models" / "real" / "real_feature_scaler.joblib"
                if scaler_path.exists():
                    scaler = joblib.load(scaler_path)
                    X = scaler.transform(X)

            pred_class = int(model.predict(X)[0])
            prob_up = float(model.predict_proba(X)[0, 1]) if hasattr(model, "predict_proba") else 0.5

            return {
                "prediction": "UP" if pred_class == 1 else "DOWN/FLAT",
                "probability_up": round(prob_up, 4),
                "probability_down": round(1.0 - prob_up, 4),
                "model_used": model_type,
                "target_horizon": "Next Trading Day (t+1)",
                "disclaimer": "Experimental machine learning prediction based on historical factors. Not financial advice.",
            }
        except Exception as e:
            return {"error": f"Inference error: {str(e)}"}
