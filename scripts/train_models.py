"""
End-to-End Machine Learning Pipeline Execution Script.

Tasks:
1. Load market_features.csv (or from MongoDB if requested/available).
2. Validate required columns and schema.
3. Construct future targets (target_return = daily_return.shift(-1), target_class).
4. Remove rows with NaN (rolling indicator warmup & last row per stock).
5. Perform strict chronological splitting (70% train / 15% validation / 15% test).
6. Train and evaluate Regression models (LinearRegression, RandomForestRegressor, Zero-Return baseline).
7. Train and evaluate Classification models (LogisticRegression, RandomForestClassifier, Majority-Class baseline).
8. Perform stock behavioral clustering with KMeans (Elbow Method & Silhouette Score).
9. Persist trained models and scalers into data/models/ using joblib.
10. Save prediction CSVs, feature importances, and model metrics into data/processed/ml/.
11. Store predictions into MongoDB (predictions collection) if connected or in --mock mode.
12. Output detailed performance summaries and academic disclaimers.

Usage:
    python scripts/train_models.py
    python scripts/train_models.py --mock
    python scripts/train_models.py --source-mongo
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
import yaml

from src.ml.classification import (
    CLASS_ORDER,
    chronological_classification_split,
    prepare_classification_dataset,
    train_logistic_regression,
    train_random_forest_classifier,
)
from src.ml.clustering import (
    extract_stock_cluster_features,
    find_optimal_clusters,
    fit_stock_clusters,
)
from src.ml.evaluation import (
    calculate_classification_metrics,
    calculate_regression_metrics,
    evaluate_majority_baseline,
    evaluate_zero_return_baseline,
)
from src.ml.regression import (
    DEFAULT_FEATURES,
    chronological_train_val_test_split,
    get_feature_importance,
    prepare_regression_dataset,
    train_linear_regression,
    train_random_forest_regressor,
)
from src.utils.logger import logger

ROOT = Path(__file__).resolve().parent.parent


def load_ml_config() -> Dict[str, Any]:
    """Load ML parameters from config/config.yaml."""
    cfg_path = ROOT / "config" / "config.yaml"
    cfg = {}
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            full = yaml.safe_load(f) or {}
            cfg = full.get("ml", {})

    return {
        "train_size": cfg.get("train_size", 0.70),
        "validation_size": cfg.get("validation_size", 0.15),
        "test_size": cfg.get("test_size", 0.15),
        "classification_threshold": cfg.get("classification_threshold", 0.005),
        "random_state": cfg.get("random_state", 42),
        "models_path": cfg.get("models_path", "data/models"),
        "output_path": cfg.get("output_path", "data/processed/ml"),
        "clustering": cfg.get("clustering", {"min_k": 2, "max_k": 6, "default_k": 3}),
        "features": cfg.get("features", DEFAULT_FEATURES),
    }


def load_market_data(source_mongo: bool = False, mock_mongo: bool = False) -> pd.DataFrame:
    """Load market dataset from MongoDB if requested, else from CSV."""
    if source_mongo:
        try:
            from src.database.mongodb import check_connection, get_collection
            health = check_connection()
            if health["connected"]:
                logger.info("Loading market data from live MongoDB collection 'market_data'...")
                coll = get_collection("market_data")
                docs = list(coll.find({}, {"_id": 0}))
                if docs:
                    df = pd.DataFrame(docs)
                    logger.info(f"Loaded {len(df)} records from MongoDB.")
                    return df
            else:
                logger.warning(f"MongoDB not connected ({health.get('error')}). Falling back to CSV.")
        except Exception as e:
            logger.warning(f"Failed to load from MongoDB ({e}). Falling back to CSV.")

    csv_path = ROOT / "data" / "processed" / "market_features.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Processed market data not found at {csv_path}")

    logger.info(f"Loading market data from {csv_path}...")
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} records with columns: {list(df.columns)}")
    return df


def store_predictions_in_mongo(
    predictions: List[Dict[str, Any]],
    use_mock: bool = False,
) -> bool:
    """
    Store model predictions in MongoDB collection 'predictions'.

    Returns True if stored successfully, False if skipped.
    """
    try:
        from pymongo import UpdateOne
        from src.database.mongodb import check_connection, get_client, get_collection

        if use_mock:
            import mongomock
            client = mongomock.MongoClient()
            coll = client["financial_analytics"]["predictions"]
            logger.info("Using mock MongoDB client for prediction storage verification.")
        else:
            health = check_connection()
            if not health["connected"]:
                logger.info(
                    "MongoDB is currently offline. Skipped database prediction storage. "
                    "(Predictions were safely written to local CSV files.)"
                )
                return False
            coll = get_collection("predictions")

        if use_mock:
            # Under mongomock, insert_many is instantaneous for verification
            coll.insert_many(predictions)
            logger.info(f"Successfully stored {len(predictions)} prediction documents in mock MongoDB collection 'predictions'.")
            return True

        # Real MongoDB: upsert to prevent duplicate entries
        ops = []
        for p in predictions:
            filter_doc = {
                "symbol": p["symbol"],
                "date": str(p["date"]),
                "model": p["model"],
                "prediction_type": p["prediction_type"],
            }
            update_doc = {"$set": p}
            ops.append(UpdateOne(filter_doc, update_doc, upsert=True))

        if ops:
            coll.bulk_write(ops, ordered=False)
            logger.info(f"Successfully upserted {len(ops)} prediction documents into MongoDB collection 'predictions'.")
            return True


    except Exception as e:
        logger.warning(f"Could not store predictions in MongoDB: {e}")
        return False

    return False


def run_ml_pipeline(
    mock_mongo: bool = False,
    source_mongo: bool = False,
) -> Dict[str, Any]:
    """Execute the full end-to-end Machine Learning pipeline."""
    cfg = load_ml_config()
    models_dir = ROOT / cfg["models_path"]
    output_dir = ROOT / cfg["output_path"]
    models_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("STARTING PHASE 4: MACHINE LEARNING PIPELINE")
    logger.info("=" * 60)

    # 1. Load Data
    raw_df = load_market_data(source_mongo=source_mongo, mock_mongo=mock_mongo)
    feature_cols = cfg["features"]

    # 2. Regression Dataset Preparation (Zero Look-Ahead Bias)
    logger.info("--- Step 1: Regression Dataset & Chronological Split ---")
    reg_df = prepare_regression_dataset(raw_df, feature_cols=feature_cols, drop_na=True)
    reg_split = chronological_train_val_test_split(
        reg_df,
        feature_cols=feature_cols,
        target_col="target_return",
        train_ratio=cfg["train_size"],
        val_ratio=cfg["validation_size"],
        test_ratio=cfg["test_size"],
    )

    X_train_r, y_train_r = reg_split["X_train"], reg_split["y_train"]
    X_val_r, y_val_r = reg_split["X_val"], reg_split["y_val"]
    X_test_r, y_test_r = reg_split["X_test"], reg_split["y_test"]
    test_df_r = reg_split["test_df"]

    # 3. Train Regression Models
    logger.info("--- Step 2: Training Regression Models ---")
    # A. Linear Regression
    lr_model = train_linear_regression(X_train_r, y_train_r)
    lr_preds = lr_model.predict(X_test_r)
    lr_metrics = calculate_regression_metrics(y_test_r, lr_preds)
    lr_metrics["model"] = "linear_regression"

    # B. Random Forest Regressor
    rf_reg_model = train_random_forest_regressor(
        X_train_r,
        y_train_r,
        n_estimators=100,
        max_depth=6,
        random_state=cfg["random_state"],
    )
    rf_reg_preds = rf_reg_model.predict(X_test_r)
    rf_reg_metrics = calculate_regression_metrics(y_test_r, rf_reg_preds)
    rf_reg_metrics["model"] = "random_forest_regressor"

    # C. Zero-Return Baseline
    zero_baseline_metrics = evaluate_zero_return_baseline(y_test_r)

    # Feature Importance for RF Regressor
    rf_reg_fi = get_feature_importance(rf_reg_model, feature_cols)
    rf_reg_fi["model"] = "random_forest_regressor"

    # 4. Classification Dataset Preparation
    logger.info("--- Step 3: Classification Dataset & Chronological Split ---")
    cls_df = prepare_classification_dataset(
        raw_df,
        feature_cols=feature_cols,
        threshold=cfg["classification_threshold"],
        drop_na=True,
    )
    cls_split = chronological_classification_split(
        cls_df,
        feature_cols=feature_cols,
        target_col="target_class",
        train_ratio=cfg["train_size"],
        val_ratio=cfg["validation_size"],
        test_ratio=cfg["test_size"],
    )

    X_train_c, y_train_c = cls_split["X_train"], cls_split["y_train"]
    X_val_c, y_val_c = cls_split["X_val"], cls_split["y_val"]
    X_test_c, y_test_c = cls_split["X_test"], cls_split["y_test"]
    test_df_c = cls_split["test_df"]

    # 5. Train Classification Models
    logger.info("--- Step 4: Training Classification Models ---")
    # A. Logistic Regression
    logit_model = train_logistic_regression(
        X_train_c,
        y_train_c,
        class_weight="balanced",
        random_state=cfg["random_state"],
    )
    logit_preds = logit_model.predict(X_test_c)
    logit_metrics = calculate_classification_metrics(y_test_c, logit_preds, labels=CLASS_ORDER)
    logit_metrics["model"] = "logistic_regression"

    # B. Random Forest Classifier
    rf_cls_model = train_random_forest_classifier(
        X_train_c,
        y_train_c,
        n_estimators=100,
        max_depth=6,
        class_weight="balanced",
        random_state=cfg["random_state"],
    )
    rf_cls_preds = rf_cls_model.predict(X_test_c)
    rf_cls_metrics = calculate_classification_metrics(y_test_c, rf_cls_preds, labels=CLASS_ORDER)
    rf_cls_metrics["model"] = "random_forest_classifier"

    # C. Majority Class Baseline
    majority_baseline_metrics = evaluate_majority_baseline(y_train_c, y_test_c, labels=CLASS_ORDER)

    # Feature Importance for RF Classifier
    rf_cls_fi = get_feature_importance(rf_cls_model, feature_cols)
    rf_cls_fi["model"] = "random_forest_classifier"

    # Combine feature importances
    all_fi = pd.concat([rf_reg_fi, rf_cls_fi], ignore_index=True)
    fi_path = output_dir / "feature_importance.csv"
    all_fi.to_csv(fi_path, index=False)
    logger.info(f"Saved combined feature importance to {fi_path}")

    # 6. Unsupervised Stock Clustering
    logger.info("--- Step 5: Unsupervised Stock Clustering (KMeans) ---")
    cluster_features_df = extract_stock_cluster_features(raw_df)
    clustering_cfg = cfg["clustering"]

    # Evaluate Elbow & Silhouette
    elbow_results = find_optimal_clusters(
        cluster_features_df,
        min_k=clustering_cfg.get("min_k", 2),
        max_k=clustering_cfg.get("max_k", 6),
        random_state=cfg["random_state"],
    )

    # Fit Chosen k (default 3)
    k_choice = clustering_cfg.get("default_k", 3)
    kmeans_model, cluster_scaler, clustered_stocks_df, cluster_profiles_df = fit_stock_clusters(
        cluster_features_df,
        n_clusters=k_choice,
        random_state=cfg["random_state"],
    )

    stock_clusters_path = output_dir / "stock_clusters.csv"
    clustered_stocks_df.to_csv(stock_clusters_path, index=False)
    logger.info(f"Saved stock cluster assignments to {stock_clusters_path}")

    # 7. Model Persistence
    logger.info("--- Step 6: Model Persistence (joblib) ---")
    persisted_files = {
        "regression_linear": models_dir / "regression_linear.joblib",
        "regression_random_forest": models_dir / "regression_random_forest.joblib",
        "classification_logistic": models_dir / "classification_logistic.joblib",
        "classification_random_forest": models_dir / "classification_random_forest.joblib",
        "kmeans_stock_clusters": models_dir / "kmeans_stock_clusters.joblib",
        "scaler_clustering": models_dir / "scaler_clustering.joblib",
    }
    joblib.dump(lr_model, persisted_files["regression_linear"])
    joblib.dump(rf_reg_model, persisted_files["regression_random_forest"])
    joblib.dump(logit_model, persisted_files["classification_logistic"])
    joblib.dump(rf_cls_model, persisted_files["classification_random_forest"])
    joblib.dump(kmeans_model, persisted_files["kmeans_stock_clusters"])
    joblib.dump(cluster_scaler, persisted_files["scaler_clustering"])
    logger.info(f"Saved 6 trained model and scaler artifacts to {models_dir}")

    # 8. Save Predictions to CSV
    logger.info("--- Step 7: Prediction Output Generation ---")
    # Regression Predictions
    reg_preds_df = pd.concat([
        pd.DataFrame({
            "date": test_df_r["date"].astype(str),
            "symbol": test_df_r["symbol"],
            "actual_return": test_df_r["target_return"].round(6),
            "predicted_return": pd.Series(lr_preds, index=test_df_r.index).round(6),
            "model": "linear_regression",
        }),
        pd.DataFrame({
            "date": test_df_r["date"].astype(str),
            "symbol": test_df_r["symbol"],
            "actual_return": test_df_r["target_return"].round(6),
            "predicted_return": pd.Series(rf_reg_preds, index=test_df_r.index).round(6),
            "model": "random_forest_regressor",
        }),
    ], ignore_index=True)
    reg_preds_path = output_dir / "regression_predictions.csv"
    reg_preds_df.to_csv(reg_preds_path, index=False)
    logger.info(f"Saved {len(reg_preds_df)} regression predictions to {reg_preds_path}")

    # Classification Predictions
    cls_preds_df = pd.concat([
        pd.DataFrame({
            "date": test_df_c["date"].astype(str),
            "symbol": test_df_c["symbol"],
            "actual_class": test_df_c["target_class"],
            "predicted_class": logit_preds,
            "model": "logistic_regression",
        }),
        pd.DataFrame({
            "date": test_df_c["date"].astype(str),
            "symbol": test_df_c["symbol"],
            "actual_class": test_df_c["target_class"],
            "predicted_class": rf_cls_preds,
            "model": "random_forest_classifier",
        }),
    ], ignore_index=True)
    cls_preds_path = output_dir / "classification_predictions.csv"
    cls_preds_df.to_csv(cls_preds_path, index=False)
    logger.info(f"Saved {len(cls_preds_df)} classification predictions to {cls_preds_path}")

    # 9. Save Evaluation Metrics JSON
    metrics_summary = {
        "regression": {
            "zero_return_baseline": zero_baseline_metrics,
            "linear_regression": lr_metrics,
            "random_forest_regressor": rf_reg_metrics,
        },
        "classification": {
            "majority_class_baseline": majority_baseline_metrics,
            "logistic_regression": logit_metrics,
            "random_forest_classifier": rf_cls_metrics,
        },
        "clustering": {
            "k_evaluated": elbow_results["k_values"],
            "inertias": elbow_results["inertias"],
            "silhouette_scores": elbow_results["silhouette_scores"],
            "chosen_k": k_choice,
            "cluster_profiles": cluster_profiles_df.to_dict(orient="records"),
        },
    }
    metrics_path = output_dir / "model_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)
    logger.info(f"Saved evaluation metrics summary to {metrics_path}")

    # 10. MongoDB Integration
    logger.info("--- Step 8: MongoDB Predictions Ingestion ---")
    mongo_records = []
    # Add classification predictions
    for _, row in cls_preds_df.iterrows():
        mongo_records.append({
            "symbol": row["symbol"],
            "date": str(row["date"]),
            "model": row["model"],
            "prediction": row["predicted_class"],
            "actual": row["actual_class"],
            "prediction_type": "classification",
        })
    # Add regression predictions
    for _, row in reg_preds_df.iterrows():
        mongo_records.append({
            "symbol": row["symbol"],
            "date": str(row["date"]),
            "model": row["model"],
            "prediction": float(row["predicted_return"]),
            "actual": float(row["actual_return"]),
            "prediction_type": "regression",
        })

    mongo_stored = store_predictions_in_mongo(mongo_records, use_mock=mock_mongo)

    # 11. Print Formatted Summary
    print("\n" + "=" * 80)
    print("PHASE 4: MACHINE LEARNING RESULTS SUMMARY")
    print("=" * 80)

    print("\n[A] REGRESSION BENCHMARK (Target: Next-Day Return):")
    print(f"{'Model':<26} | {'MAE':<10} | {'RMSE':<10} | {'R²':<10}")
    print("-" * 65)
    for m in [zero_baseline_metrics, lr_metrics, rf_reg_metrics]:
        print(f"{m['model']:<26} | {m['mae']:<10.6f} | {m['rmse']:<10.6f} | {m['r2']:<10.6f}")

    print("\n[B] CLASSIFICATION BENCHMARK (Target: Next-Day Direction):")
    print(f"{'Model':<26} | {'Accuracy':<10} | {'F1 Macro':<10} | {'F1 Weighted':<12}")
    print("-" * 65)
    for m in [majority_baseline_metrics, logit_metrics, rf_cls_metrics]:
        print(f"{m['model']:<26} | {m['accuracy']:<10.4f} | {m['f1_macro']:<10.4f} | {m['f1_weighted']:<12.4f}")

    print("\n[C] CLUSTERING SUMMARY (KMeans k=3):")
    for _, prof in cluster_profiles_df.iterrows():
        print(f"  Cluster {prof['cluster']}: {prof['symbols']} (Count: {prof['stock_count']})")
        print(f"    Avg Return: {prof['avg_daily_return']:.6f}, Volatility: {prof['avg_volatility']:.6f}, Max DD: {prof['max_drawdown']:.2%}")

    print("\n[D] TOP FEATURE IMPORTANCE (Random Forest Classifier):")
    for _, row in rf_cls_fi.head(5).iterrows():
        print(f"  {row['feature']:<16} : {row['importance']:.4f}")

    print("\n[E] INTEGRATION & PERSISTENCE:")
    print(f"  Models Saved:      {len(persisted_files)} files in {models_dir}")
    print(f"  Outputs Saved:     {output_dir}")
    print(f"  MongoDB Ingestion: {'Completed' if mongo_stored else 'Skipped (MongoDB offline)'}")
    print("=" * 80 + "\n")

    return metrics_summary


def main():
    parser = argparse.ArgumentParser(description="Run Phase 4 Machine Learning Pipeline.")
    parser.add_argument("--mock", action="store_true", help="Use mongomock for database storage.")
    parser.add_argument("--source-mongo", action="store_true", help="Load market data from MongoDB if available.")
    args = parser.parse_args()

    run_ml_pipeline(mock_mongo=args.mock, source_mongo=args.source_mongo)


if __name__ == "__main__":
    main()
