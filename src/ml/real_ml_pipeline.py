"""
Real-Data Machine Learning Pipeline for NIFTY 50 Equities (Step 9 Phase B).

Implements strictly chronological date-based panel evaluation for:
1. Next-Day Return Direction Classification (Binary: Up vs Down/Flat).
2. Next-Day Return Magnitude Regression (Continuous: r_{t+1}).

Enforces strict anti-leakage guards:
- Feature scaling fit strictly on Training partition.
- Panel boundary purging to eliminate label-transition leakage.
- Target variable target_next_day_return strictly isolated from input features.
- No raw identifiers or scale-dependent absolute price levels in feature set.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score
)

logger = logging.getLogger(__name__)

# Explicit list of explanatory feature columns
EXPLANATORY_FEATURES = [
    "daily_adjusted_return",
    "return_5d",
    "return_20d",
    "return_60d",
    "volatility_20d",
    "volatility_60d",
    "close_to_ma20",
    "close_to_ma50",
    "close_to_ma200",
    "volume_ratio",
    "hl_pct_range",
    "rolling_max_drawdown_60d",
    "benchmark_daily_return",
    "relative_return_vs_nifty",
]


def prepare_real_ml_data(
    parquet_path: Path,
    train_end_date: str = "2021-12-31",
    val_end_date: str = "2023-12-31",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Load real consolidated features, compute clean relative features,
    and partition into chronological Train, Validation, and Test sets with boundary purging.
    """
    df = pd.read_parquet(parquet_path)
    logger.info(f"Loaded {len(df)} rows from {parquet_path}")

    # 1. Feature Engineering: Relative and stationary transformations
    df_clean = df.copy()
    df_clean["close_to_ma20"] = (df_clean["close"] / df_clean["ma_20"]) - 1.0
    df_clean["close_to_ma50"] = (df_clean["close"] / df_clean["ma_50"]) - 1.0
    df_clean["close_to_ma200"] = (df_clean["close"] / df_clean["ma_200"]) - 1.0

    # 2. Target Variable Isolation
    # Drop rows where target_next_day_return is NaN (last trading day of universe)
    df_clean = df_clean.dropna(subset=["target_next_day_return"] + EXPLANATORY_FEATURES)
    df_clean["target_direction"] = (df_clean["target_next_day_return"] > 0).astype(int)

    # 3. Chronological Date-Based Splitting
    df_clean = df_clean.sort_values(["date", "symbol"]).reset_index(drop=True)

    # Apply Purging: Drop the final boundary day of Train and Val
    # to guarantee target_{t+1} from train doesn't cross into val features
    train_dates = df_clean[df_clean["date"] <= train_end_date]["date"].unique()
    val_dates = df_clean[(df_clean["date"] > train_end_date) & (df_clean["date"] <= val_end_date)]["date"].unique()
    test_dates = df_clean[df_clean["date"] > val_end_date]["date"].unique()

    train_dates_purged = train_dates[:-1] if len(train_dates) > 1 else train_dates
    val_dates_purged = val_dates[:-1] if len(val_dates) > 1 else val_dates

    df_train = df_clean[df_clean["date"].isin(train_dates_purged)].copy()
    df_val = df_clean[df_clean["date"].isin(val_dates_purged)].copy()
    df_test = df_clean[df_clean["date"].isin(test_dates)].copy()

    logger.info(
        f"Chronological Splitting: Train={len(df_train)} rows ({df_train['date'].min()} to {df_train['date'].max()}), "
        f"Val={len(df_val)} rows ({df_val['date'].min()} to {df_val['date'].max()}), "
        f"Test={len(df_test)} rows ({df_test['date'].min()} to {df_test['date'].max()})"
    )

    return df_train, df_val, df_test, EXPLANATORY_FEATURES


def train_and_evaluate_real_ml(
    parquet_path: Path,
    output_model_dir: Path,
    output_metrics_dir: Path,
) -> Dict[str, Any]:
    """
    Execute end-to-end training, chronological evaluation, and artifact saving.
    """
    output_model_dir.mkdir(parents=True, exist_ok=True)
    output_metrics_dir.mkdir(parents=True, exist_ok=True)

    df_train, df_val, df_test, features = prepare_real_ml_data(parquet_path)

    X_train = df_train[features].values
    y_class_train = df_train["target_direction"].values
    y_reg_train = df_train["target_next_day_return"].values

    X_val = df_val[features].values
    y_class_val = df_val["target_direction"].values
    y_reg_val = df_val["target_next_day_return"].values

    X_test = df_test[features].values
    y_class_test = df_test["target_direction"].values
    y_reg_test = df_test["target_next_day_return"].values

    # Fit Scaler strictly on Training Partition
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Save Scaler
    scaler_path = output_model_dir / "real_feature_scaler.joblib"
    joblib.dump(scaler, scaler_path)

    metrics_report: Dict[str, Any] = {
        "metadata": {
            "pipeline_name": "NIFTY_50_Real_Market_ML_Pipeline",
            "feature_set": features,
            "feature_count": len(features),
            "train_period": f"{df_train['date'].min()} to {df_train['date'].max()}",
            "train_samples": len(df_train),
            "val_period": f"{df_val['date'].min()} to {df_val['date'].max()}",
            "val_samples": len(df_val),
            "test_period": f"{df_test['date'].min()} to {df_test['date'].max()}",
            "test_samples": len(df_test),
            "target_variable": "target_next_day_return (isolated)",
            "leakage_controls": [
                "Strict date-based panel split (no random row shuffling)",
                "Boundary day purging at split transitions",
                "StandardScaler fit exclusively on train set",
                "Zero inclusion of future target returns or price levels",
            ],
        },
        "classification": {},
        "regression": {},
        "feature_importance": {},
    }

    # =========================================================================
    # TASK 1: CLASSIFICATION (Next-Day Direction)
    # =========================================================================
    logger.info("Training Classification Models...")

    # 1. Baseline: Majority Class Classifier
    majority_class = int(np.bincount(y_class_train).argmax())
    y_pred_base_val = np.full_like(y_class_val, majority_class)
    y_pred_base_test = np.full_like(y_class_test, majority_class)

    metrics_report["classification"]["Majority_Baseline"] = {
        "val_accuracy": round(float(accuracy_score(y_class_val, y_pred_base_val)), 4),
        "test_accuracy": round(float(accuracy_score(y_class_test, y_pred_base_test)), 4),
        "test_balanced_accuracy": round(float(balanced_accuracy_score(y_class_test, y_pred_base_test)), 4),
        "test_precision": round(float(precision_score(y_class_test, y_pred_base_test, zero_division=0)), 4),
        "test_recall": round(float(recall_score(y_class_test, y_pred_base_test, zero_division=0)), 4),
        "test_f1": round(float(f1_score(y_class_test, y_pred_base_test, zero_division=0)), 4),
        "test_roc_auc": 0.5000,
    }

    # 2. Logistic Regression
    clf_lr = LogisticRegression(max_iter=1000, C=0.1, random_state=42)
    clf_lr.fit(X_train_scaled, y_class_train)
    y_pred_lr_val = clf_lr.predict(X_val_scaled)
    y_pred_lr_test = clf_lr.predict(X_test_scaled)
    y_prob_lr_test = clf_lr.predict_proba(X_test_scaled)[:, 1]

    metrics_report["classification"]["Logistic_Regression"] = {
        "val_accuracy": round(float(accuracy_score(y_class_val, y_pred_lr_val)), 4),
        "test_accuracy": round(float(accuracy_score(y_class_test, y_pred_lr_test)), 4),
        "test_balanced_accuracy": round(float(balanced_accuracy_score(y_class_test, y_pred_lr_test)), 4),
        "test_precision": round(float(precision_score(y_class_test, y_pred_lr_test, zero_division=0)), 4),
        "test_recall": round(float(recall_score(y_class_test, y_pred_lr_test, zero_division=0)), 4),
        "test_f1": round(float(f1_score(y_class_test, y_pred_lr_test, zero_division=0)), 4),
        "test_roc_auc": round(float(roc_auc_score(y_class_test, y_prob_lr_test)), 4),
        "confusion_matrix": confusion_matrix(y_class_test, y_pred_lr_test).tolist(),
    }
    joblib.dump(clf_lr, output_model_dir / "real_classifier_logistic_regression.joblib")

    # 3. Random Forest Classifier
    clf_rf = RandomForestClassifier(n_estimators=100, max_depth=8, min_samples_leaf=20, random_state=42, n_jobs=-1)
    clf_rf.fit(X_train, y_class_train)
    y_pred_rf_clf_val = clf_rf.predict(X_val)
    y_pred_rf_clf_test = clf_rf.predict(X_test)
    y_prob_rf_clf_test = clf_rf.predict_proba(X_test)[:, 1]

    metrics_report["classification"]["Random_Forest"] = {
        "val_accuracy": round(float(accuracy_score(y_class_val, y_pred_rf_clf_val)), 4),
        "test_accuracy": round(float(accuracy_score(y_class_test, y_pred_rf_clf_test)), 4),
        "test_balanced_accuracy": round(float(balanced_accuracy_score(y_class_test, y_pred_rf_clf_test)), 4),
        "test_precision": round(float(precision_score(y_class_test, y_pred_rf_clf_test, zero_division=0)), 4),
        "test_recall": round(float(recall_score(y_class_test, y_pred_rf_clf_test, zero_division=0)), 4),
        "test_f1": round(float(f1_score(y_class_test, y_pred_rf_clf_test, zero_division=0)), 4),
        "test_roc_auc": round(float(roc_auc_score(y_class_test, y_prob_rf_clf_test)), 4),
        "confusion_matrix": confusion_matrix(y_class_test, y_pred_rf_clf_test).tolist(),
    }
    joblib.dump(clf_rf, output_model_dir / "real_classifier_random_forest.joblib")

    # 4. HistGradientBoosting Classifier
    clf_gb = HistGradientBoostingClassifier(max_iter=100, max_depth=6, min_samples_leaf=30, random_state=42)
    clf_gb.fit(X_train, y_class_train)
    y_pred_gb_clf_val = clf_gb.predict(X_val)
    y_pred_gb_clf_test = clf_gb.predict(X_test)
    y_prob_gb_clf_test = clf_gb.predict_proba(X_test)[:, 1]

    metrics_report["classification"]["Gradient_Boosting"] = {
        "val_accuracy": round(float(accuracy_score(y_class_val, y_pred_gb_clf_val)), 4),
        "test_accuracy": round(float(accuracy_score(y_class_test, y_pred_gb_clf_test)), 4),
        "test_balanced_accuracy": round(float(balanced_accuracy_score(y_class_test, y_pred_gb_clf_test)), 4),
        "test_precision": round(float(precision_score(y_class_test, y_pred_gb_clf_test, zero_division=0)), 4),
        "test_recall": round(float(recall_score(y_class_test, y_pred_gb_clf_test, zero_division=0)), 4),
        "test_f1": round(float(f1_score(y_class_test, y_pred_gb_clf_test, zero_division=0)), 4),
        "test_roc_auc": round(float(roc_auc_score(y_class_test, y_prob_gb_clf_test)), 4),
        "confusion_matrix": confusion_matrix(y_class_test, y_pred_gb_clf_test).tolist(),
    }
    joblib.dump(clf_gb, output_model_dir / "real_classifier_gradient_boosting.joblib")

    # =========================================================================
    # TASK 2: REGRESSION (Next-Day Return Magnitude)
    # =========================================================================
    logger.info("Training Regression Models...")

    # 1. Zero-Return Baseline
    y_pred_zero_val = np.zeros_like(y_reg_val)
    y_pred_zero_test = np.zeros_like(y_reg_test)
    conv_r2_zero = float(r2_score(y_reg_test, y_pred_zero_test))

    metrics_report["regression"]["Zero_Return_Baseline"] = {
        "val_mae": round(float(mean_absolute_error(y_reg_val, y_pred_zero_val)), 6),
        "val_rmse": round(float(math.sqrt(mean_squared_error(y_reg_val, y_pred_zero_val))), 6),
        "test_mae": round(float(mean_absolute_error(y_reg_test, y_pred_zero_test)), 6),
        "test_rmse": round(float(math.sqrt(mean_squared_error(y_reg_test, y_pred_zero_test))), 6),
        "test_r2_conventional": round(conv_r2_zero, 6),
        "test_r2_relative_vs_zero": 0.0,
        "test_r2": round(conv_r2_zero, 6),
        "test_directional_accuracy": 0.5000,
    }

    # 2. Ridge Regression
    reg_ridge = Ridge(alpha=10.0, random_state=42)
    reg_ridge.fit(X_train_scaled, y_reg_train)
    y_pred_ridge_val = reg_ridge.predict(X_val_scaled)
    y_pred_ridge_test = reg_ridge.predict(X_test_scaled)
    dir_acc_ridge = float(np.mean((y_pred_ridge_test > 0) == (y_reg_test > 0)))
    conv_r2_ridge = float(r2_score(y_reg_test, y_pred_ridge_test))
    rel_r2_ridge = float(1.0 - np.sum((y_reg_test - y_pred_ridge_test)**2) / np.sum(y_reg_test**2))

    metrics_report["regression"]["Ridge_Regression"] = {
        "val_mae": round(float(mean_absolute_error(y_reg_val, y_pred_ridge_val)), 6),
        "val_rmse": round(float(math.sqrt(mean_squared_error(y_reg_val, y_pred_ridge_val))), 6),
        "test_mae": round(float(mean_absolute_error(y_reg_test, y_pred_ridge_test)), 6),
        "test_rmse": round(float(math.sqrt(mean_squared_error(y_reg_test, y_pred_ridge_test))), 6),
        "test_r2_conventional": round(conv_r2_ridge, 6),
        "test_r2_relative_vs_zero": round(rel_r2_ridge, 6),
        "test_r2": round(conv_r2_ridge, 6),
        "test_directional_accuracy": round(dir_acc_ridge, 4),
    }
    joblib.dump(reg_ridge, output_model_dir / "real_regressor_ridge.joblib")

    # 3. Random Forest Regressor
    reg_rf = RandomForestRegressor(n_estimators=100, max_depth=8, min_samples_leaf=20, random_state=42, n_jobs=-1)
    reg_rf.fit(X_train, y_reg_train)
    y_pred_rf_reg_val = reg_rf.predict(X_val)
    y_pred_rf_reg_test = reg_rf.predict(X_test)
    dir_acc_rf = float(np.mean((y_pred_rf_reg_test > 0) == (y_reg_test > 0)))
    conv_r2_rf = float(r2_score(y_reg_test, y_pred_rf_reg_test))
    rel_r2_rf = float(1.0 - np.sum((y_reg_test - y_pred_rf_reg_test)**2) / np.sum(y_reg_test**2))

    metrics_report["regression"]["Random_Forest_Regressor"] = {
        "val_mae": round(float(mean_absolute_error(y_reg_val, y_pred_rf_reg_val)), 6),
        "val_rmse": round(float(math.sqrt(mean_squared_error(y_reg_val, y_pred_rf_reg_val))), 6),
        "test_mae": round(float(mean_absolute_error(y_reg_test, y_pred_rf_reg_test)), 6),
        "test_rmse": round(float(math.sqrt(mean_squared_error(y_reg_test, y_pred_rf_reg_test))), 6),
        "test_r2_conventional": round(conv_r2_rf, 6),
        "test_r2_relative_vs_zero": round(rel_r2_rf, 6),
        "test_r2": round(conv_r2_rf, 6),
        "test_directional_accuracy": round(dir_acc_rf, 4),
    }
    joblib.dump(reg_rf, output_model_dir / "real_regressor_random_forest.joblib")

    # 4. HistGradientBoosting Regressor
    reg_gb = HistGradientBoostingRegressor(max_iter=100, max_depth=6, min_samples_leaf=30, random_state=42)
    reg_gb.fit(X_train, y_reg_train)
    y_pred_gb_reg_val = reg_gb.predict(X_val)
    y_pred_gb_reg_test = reg_gb.predict(X_test)
    dir_acc_gb = float(np.mean((y_pred_gb_reg_test > 0) == (y_reg_test > 0)))
    conv_r2_gb = float(r2_score(y_reg_test, y_pred_gb_reg_test))
    rel_r2_gb = float(1.0 - np.sum((y_reg_test - y_pred_gb_reg_test)**2) / np.sum(y_reg_test**2))

    metrics_report["regression"]["Gradient_Boosting_Regressor"] = {
        "val_mae": round(float(mean_absolute_error(y_reg_val, y_pred_gb_reg_val)), 6),
        "val_rmse": round(float(math.sqrt(mean_squared_error(y_reg_val, y_pred_gb_reg_val))), 6),
        "test_mae": round(float(mean_absolute_error(y_reg_test, y_pred_gb_reg_test)), 6),
        "test_rmse": round(float(math.sqrt(mean_squared_error(y_reg_test, y_pred_gb_reg_test))), 6),
        "test_r2_conventional": round(conv_r2_gb, 6),
        "test_r2_relative_vs_zero": round(rel_r2_gb, 6),
        "test_r2": round(conv_r2_gb, 6),
        "test_directional_accuracy": round(dir_acc_gb, 4),
    }
    joblib.dump(reg_gb, output_model_dir / "real_regressor_gradient_boosting.joblib")

    # Feature Importance (Random Forest)
    importances = clf_rf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    metrics_report["feature_importance"] = {
        features[i]: round(float(importances[i]), 4) for i in sorted_idx
    }

    # Save Test Set Predictions for Interactive Dashboard Exploration
    df_preds_test = df_test[["date", "symbol", "target_next_day_return", "target_direction"]].copy()
    df_preds_test["pred_direction_lr"] = y_pred_lr_test
    df_preds_test["prob_up_lr"] = np.round(y_prob_lr_test, 4)
    df_preds_test["pred_direction_rf"] = y_pred_rf_clf_test
    df_preds_test["prob_up_rf"] = np.round(y_prob_rf_clf_test, 4)
    df_preds_test["pred_return_ridge"] = np.round(y_pred_ridge_test, 6)
    df_preds_test["pred_return_rf"] = np.round(y_pred_rf_reg_test, 6)
    df_preds_test["pred_return_gb"] = np.round(y_pred_gb_reg_test, 6)

    preds_path = output_metrics_dir / "real_predictions_test.csv"
    df_preds_test.to_csv(preds_path, index=False)
    logger.info(f"Saved test predictions ({len(df_preds_test)} rows) to {preds_path}")

    # Save Metrics JSON
    metrics_path = output_metrics_dir / "real_model_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_report, f, indent=2)
    logger.info(f"Saved real ML metrics to {metrics_path}")

    return metrics_report

