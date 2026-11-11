"""
Supervised Classification pipeline for market direction forecasting.

Predicts next-day market regime:
- Bullish (target_return > +threshold)
- Bearish (target_return < -threshold)
- Neutral (otherwise)

Uses chronological splitting, balanced class weighting, LogisticRegression,
and RandomForestClassifier.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.ml.evaluation import calculate_classification_metrics, evaluate_majority_baseline

from src.ml.regression import (
    DEFAULT_FEATURES,
    chronological_train_val_test_split,
    prepare_regression_dataset,
)
from src.utils.logger import logger

CLASS_ORDER = ["Bearish", "Neutral", "Bullish"]


# =====================================================================
# 1. Dataset Preparation & Directional Target Labeling
# =====================================================================

def prepare_classification_dataset(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    threshold: float = 0.005,
    drop_na: bool = True,
) -> pd.DataFrame:
    """
    Construct 3-class market direction targets from next-day returns:
    - Bullish: target_return > +threshold
    - Bearish: target_return < -threshold
    - Neutral: -threshold <= target_return <= +threshold

    Parameters
    ----------
    df : pd.DataFrame
        Market features DataFrame.
    feature_cols : list of str, optional
        Features to retain.
    threshold : float, default 0.005 (0.5% return)
        Return boundary separating Bullish and Bearish regimes.
    drop_na : bool, default True
        If True, drops rows with missing features or missing targets.

    Returns
    -------
    pd.DataFrame
        DataFrame with 'target_return' and 'target_class' columns.
    """
    # First build the clean, leak-free regression target
    data = prepare_regression_dataset(df, feature_cols=feature_cols, drop_na=drop_na)

    # Class assignment conditions
    conditions = [
        data["target_return"] > threshold,
        data["target_return"] < -threshold,
    ]
    choices = ["Bullish", "Bearish"]

    data["target_class"] = np.select(conditions, choices, default="Neutral")

    # Log class distribution
    counts = data["target_class"].value_counts().to_dict()
    pcts = {k: f"{v / len(data) * 100:.1f}%" for k, v in counts.items()}
    logger.info(
        f"Classification dataset prepared with threshold={threshold}: "
        f"Counts={counts}, Distribution={pcts}"
    )

    return data


# =====================================================================
# 2. Chronological Split for Classification
# =====================================================================

def chronological_classification_split(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    target_col: str = "target_class",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> Dict[str, Any]:
    """
    Chronological train/validation/test split for classification targets.

    Re-uses the strict per-symbol chronological split mechanism.
    """
    return chronological_train_val_test_split(
        df=df,
        feature_cols=feature_cols,
        target_col=target_col,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )


# =====================================================================
# 3. Model Training
# =====================================================================

def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    class_weight: Optional[str] = "balanced",
    max_iter: int = 1000,
    random_state: int = 42,
    use_scaler: bool = True,
    **kwargs: Any,
) -> Any:
    """
    Train a multinomial Logistic Regression classifier.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training class labels.
    class_weight : str or dict, default 'balanced'
        Handles class imbalance in financial regimes.
    max_iter : int, default 1000
        Maximum iterations for convergence.
    random_state : int, default 42
        Seed for reproducibility.
    use_scaler : bool, default True
        Standardizes features before fitting to prevent convergence warnings.
    **kwargs
        Additional arguments passed to LogisticRegression.

    Returns
    -------
    LogisticRegression or Pipeline
        Fitted model or pipeline.
    """
    clf = LogisticRegression(
        class_weight=class_weight,
        max_iter=max_iter,
        random_state=random_state,
        **kwargs,
    )
    if use_scaler:
        model = make_pipeline(StandardScaler(), clf)
    else:
        model = clf
    model.fit(X_train, y_train)
    logger.info(f"Trained LogisticRegression model (use_scaler={use_scaler}) on {len(X_train)} samples.")
    return model



def train_random_forest_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 100,
    max_depth: Optional[int] = 6,
    class_weight: Optional[str] = "balanced",
    random_state: int = 42,
    n_jobs: int = -1,
    **kwargs: Any,
) -> RandomForestClassifier:
    """
    Train a Random Forest Classifier for directional market regimes.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training class labels.
    n_estimators : int, default 100
        Number of trees.
    max_depth : int, default 6
        Max tree depth.
    class_weight : str or dict, default 'balanced'
        Handles class imbalance.
    random_state : int, default 42
        Seed.
    n_jobs : int, default -1
        Parallel CPU jobs.

    Returns
    -------
    RandomForestClassifier
        Fitted model.
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=n_jobs,
        **kwargs,
    )
    model.fit(X_train, y_train)
    logger.info(f"Trained RandomForestClassifier (trees={n_estimators}, max_depth={max_depth}) on {len(X_train)} samples.")
    return model
