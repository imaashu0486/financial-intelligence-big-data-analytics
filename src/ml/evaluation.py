"""
Reusable evaluation metrics and baseline comparisons for Phase 4 ML.

Supports:
- Regression: MAE, RMSE, R2, and zero-return baseline comparison.
- Classification: Accuracy, Precision, Recall, F1, Confusion Matrix, and majority-class baseline.
- Clustering: Inertia and Silhouette Score.
"""

from typing import Any, Dict, List, Optional, Union
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    silhouette_score,
)

from src.utils.logger import logger


# =====================================================================
# 1. Regression Metrics & Baselines
# =====================================================================

def calculate_regression_metrics(
    y_true: Union[np.ndarray, List[float]],
    y_pred: Union[np.ndarray, List[float]],
) -> Dict[str, float]:
    """
    Calculate MAE, RMSE, and R2 for continuous predictions.

    Parameters
    ----------
    y_true : array-like
        Actual target returns.
    y_pred : array-like
        Predicted target returns.

    Returns
    -------
    dict
        {"mae": float, "rmse": float, "r2": float}
    """
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)

    if len(y_t) == 0:
        raise ValueError("y_true cannot be empty.")
    if len(y_t) != len(y_p):
        raise ValueError(f"Shape mismatch: y_true ({len(y_t)}) vs y_pred ({len(y_p)}).")

    mae = float(mean_absolute_error(y_t, y_p))
    rmse = float(np.sqrt(mean_squared_error(y_t, y_p)))
    r2 = float(r2_score(y_t, y_p))

    return {
        "mae": round(mae, 6),
        "rmse": round(rmse, 6),
        "r2": round(r2, 6),
    }


def evaluate_zero_return_baseline(
    y_true: Union[np.ndarray, List[float]],
) -> Dict[str, float]:
    """
    Evaluate the naive financial benchmark: predicting next-day return = 0.

    In financial asset pricing, predicting 0 return is a standard baseline
    because daily stock returns are zero-centered and noisy.

    Parameters
    ----------
    y_true : array-like
        Actual test returns.

    Returns
    -------
    dict
        Evaluation metrics for the zero-return baseline.
    """
    y_t = np.asarray(y_true, dtype=float)
    y_zero = np.zeros_like(y_t)
    metrics = calculate_regression_metrics(y_t, y_zero)
    metrics["model"] = "zero_return_baseline"
    return metrics


# =====================================================================
# 2. Classification Metrics & Baselines
# =====================================================================

DEFAULT_CLASSES = ["Bearish", "Neutral", "Bullish"]


def calculate_classification_metrics(
    y_true: Union[np.ndarray, List[str], List[int]],
    y_pred: Union[np.ndarray, List[str], List[int]],
    labels: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Calculate Accuracy, Precision, Recall, F1-scores, Confusion Matrix, and report.

    Parameters
    ----------
    y_true : array-like
        Ground truth class labels.
    y_pred : array-like
        Predicted class labels.
    labels : list of str, optional
        Class order for confusion matrix. Defaults to ['Bearish', 'Neutral', 'Bullish'].

    Returns
    -------
    dict
        Evaluation dictionary containing:
        - accuracy, precision_macro, recall_macro, f1_macro
        - precision_weighted, recall_weighted, f1_weighted
        - confusion_matrix: list of lists
        - labels: list of str
        - classification_report: dict per-class metrics
    """
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)

    if len(y_t) == 0:
        raise ValueError("y_true cannot be empty.")
    if len(y_t) != len(y_p):
        raise ValueError(f"Shape mismatch: y_true ({len(y_t)}) vs y_pred ({len(y_p)}).")

    class_labels = labels or DEFAULT_CLASSES

    acc = float(accuracy_score(y_t, y_p))
    prec_macro = float(precision_score(y_t, y_p, labels=class_labels, average="macro", zero_division=0))
    rec_macro = float(recall_score(y_t, y_p, labels=class_labels, average="macro", zero_division=0))
    f1_m = float(f1_score(y_t, y_p, labels=class_labels, average="macro", zero_division=0))

    prec_weighted = float(precision_score(y_t, y_p, labels=class_labels, average="weighted", zero_division=0))
    rec_weighted = float(recall_score(y_t, y_p, labels=class_labels, average="weighted", zero_division=0))
    f1_w = float(f1_score(y_t, y_p, labels=class_labels, average="weighted", zero_division=0))

    cm = confusion_matrix(y_t, y_p, labels=class_labels)
    cm_list = cm.tolist()

    report = classification_report(
        y_t,
        y_p,
        labels=class_labels,
        output_dict=True,
        zero_division=0,
    )

    return {
        "accuracy": round(acc, 4),
        "precision_macro": round(prec_macro, 4),
        "recall_macro": round(rec_macro, 4),
        "f1_macro": round(f1_m, 4),
        "precision_weighted": round(prec_weighted, 4),
        "recall_weighted": round(rec_weighted, 4),
        "f1_weighted": round(f1_w, 4),
        "confusion_matrix": cm_list,
        "labels": class_labels,
        "classification_report": report,
    }


def evaluate_majority_baseline(
    y_train: Union[np.ndarray, List[str]],
    y_test: Union[np.ndarray, List[str]],
    labels: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Evaluate the naive majority-class benchmark.

    Predicts the most frequent class observed in the training set for all test rows.

    Parameters
    ----------
    y_train : array-like
        Training class labels to derive majority class.
    y_test : array-like
        Test ground truth labels.
    labels : list of str, optional
        Class labels order.

    Returns
    -------
    dict
        Evaluation metrics for majority baseline.
    """
    y_tr = np.asarray(y_train)
    y_te = np.asarray(y_test)

    unique, counts = np.unique(y_tr, return_counts=True)
    majority_class = unique[np.argmax(counts)]

    y_pred_majority = np.full_like(y_te, fill_value=majority_class)
    metrics = calculate_classification_metrics(y_te, y_pred_majority, labels=labels)
    metrics["model"] = "majority_class_baseline"
    metrics["majority_class"] = str(majority_class)
    return metrics


# =====================================================================
# 3. Clustering Metrics
# =====================================================================

def calculate_clustering_metrics(
    X: Union[np.ndarray, Any],
    labels: Union[np.ndarray, List[int]],
    inertia: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Compute Inertia and Silhouette Score for cluster assignments.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Standardized feature matrix.
    labels : array-like of shape (n_samples,)
        Cluster assignments.
    inertia : float, optional
        Model inertia (within-cluster sum-of-squares).

    Returns
    -------
    dict
        {"inertia": float or None, "silhouette_score": float or None, "k": int}
    """
    X_arr = np.asarray(X)
    labels_arr = np.asarray(labels)

    unique_labels = np.unique(labels_arr)
    n_clusters = len(unique_labels)

    sil_score: Optional[float] = None
    # Silhouette score is only defined for 2 <= n_clusters < n_samples
    if 2 <= n_clusters < len(X_arr):
        try:
            sil_score = float(silhouette_score(X_arr, labels_arr))
            sil_score = round(sil_score, 4)
        except Exception as e:
            logger.warning(f"Could not compute silhouette score: {e}")
            sil_score = None

    return {
        "k": int(n_clusters),
        "inertia": round(float(inertia), 4) if inertia is not None else None,
        "silhouette_score": sil_score,
    }
