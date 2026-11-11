"""
Supervised Regression pipeline for stock return forecasting.

Predicts next-day return (target_return = daily_return.shift(-1))
using strictly historical technical indicators with zero look-ahead bias
and chronological train/validation/test splitting.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.ml.evaluation import calculate_regression_metrics, evaluate_zero_return_baseline
from src.utils.logger import logger


DEFAULT_FEATURES = [
    "daily_return",
    "log_return",
    "ma_7",
    "ma_20",
    "ma_50",
    "volatility",
    "volume_change",
    "rsi_14",
    "macd",
    "macd_signal",
]


# =====================================================================
# 1. Dataset Preparation & Target Construction (No Look-Ahead)
# =====================================================================

def prepare_regression_dataset(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    drop_na: bool = True,
) -> pd.DataFrame:
    """
    Construct target_return = daily_return.shift(-1) per symbol with zero lookahead bias.

    CRITICAL DATA LEAKAGE PREVENTION:
    - target_return is the NEXT trading day's return: daily_return.shift(-1) within each symbol.
    - The final row of each stock will have target_return = NaN because its future return
      is unknown at prediction time.
    - All features strictly represent information known at the close of trading day t.
    - Rows with NaN in features (warmup period for rolling indicators like ma_50)
      or in target_return (the final observation per stock) are dropped when drop_na=True.

    Parameters
    ----------
    df : pd.DataFrame
        Market DataFrame containing 'date', 'symbol', and technical features.
    feature_cols : list of str, optional
        Feature column names to validate. Defaults to DEFAULT_FEATURES.
    drop_na : bool, default True
        If True, drops rows with missing features or missing future targets.

    Returns
    -------
    pd.DataFrame
        Prepared DataFrame sorted by ['symbol', 'date'] with 'target_return' column.
    """
    features = feature_cols or DEFAULT_FEATURES

    # Validate required columns
    required_cols = {"date", "symbol", "daily_return"}.union(features)
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for regression: {sorted(missing)}")

    # Work on a copy and sort deterministically
    data = df.copy()
    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values(["symbol", "date"]).reset_index(drop=True)

    # Shift future return by -1 per stock symbol
    data["target_return"] = (
        data.groupby("symbol")["daily_return"].shift(-1)
    )

    # Validate that features do not accidentally contain target_return
    if "target_return" in features:
        raise ValueError("FATAL: target_return cannot be an input feature (look-ahead leakage)!")

    if drop_na:
        # Drop rows where target or features are NaN
        initial_rows = len(data)
        subset_cols = ["target_return"] + features
        data = data.dropna(subset=subset_cols).reset_index(drop=True)
        dropped_rows = initial_rows - len(data)
        logger.info(
            f"Prepared regression dataset: {len(data)} valid rows retained ({dropped_rows} dropped due to indicator warmup or last-row future target boundary)."
        )

    return data


# =====================================================================
# 2. Chronological Train / Validation / Test Splitting
# =====================================================================

def chronological_train_val_test_split(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    target_col: str = "target_return",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> Dict[str, Any]:
    """
    Perform strict chronological train/validation/test split per stock symbol.

    Financial market data exhibits strong temporal autocorrelation. Random splitting
    would cause severe look-ahead bias where past dates are predicted using future data.
    Chronological splitting preserves the arrow of time:
    - Train set: first 70% of chronological dates
    - Validation set: middle 15% of chronological dates
    - Test set: final 15% of chronological dates (strictly in the future relative to train & val)

    Parameters
    ----------
    df : pd.DataFrame
        Dataset sorted by date/symbol with features and target.
    feature_cols : list of str, optional
        List of predictor columns.
    target_col : str, default 'target_return'
        Name of target column.
    train_ratio : float
        Fraction for training.
    val_ratio : float
        Fraction for validation.
    test_ratio : float
        Fraction for test.

    Returns
    -------
    dict
        {
            "X_train": DataFrame, "y_train": Series, "train_df": DataFrame,
            "X_val": DataFrame, "y_val": Series, "val_df": DataFrame,
            "X_test": DataFrame, "y_test": Series, "test_df": DataFrame,
            "feature_cols": list
        }
    """
    features = feature_cols or DEFAULT_FEATURES
    total_ratio = train_ratio + val_ratio + test_ratio
    if not np.isclose(total_ratio, 1.0, atol=1e-3):
        raise ValueError(f"Split ratios must sum to 1.0 (got {total_ratio:.3f})")

    # Ensure chronological order
    df_sorted = df.sort_values(["symbol", "date"]).reset_index(drop=True)

    train_parts = []
    val_parts = []
    test_parts = []

    for symbol, group in df_sorted.groupby("symbol", sort=False):
        n = len(group)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        # Remainder goes to test to ensure exact row coverage
        n_test = n - n_train - n_val

        g_train = group.iloc[:n_train]
        g_val = group.iloc[n_train : n_train + n_val]
        g_test = group.iloc[n_train + n_val :]

        train_parts.append(g_train)
        val_parts.append(g_val)
        test_parts.append(g_test)

    train_df = pd.concat(train_parts, ignore_index=True).sort_values(["date", "symbol"]).reset_index(drop=True)
    val_df = pd.concat(val_parts, ignore_index=True).sort_values(["date", "symbol"]).reset_index(drop=True)
    test_df = pd.concat(test_parts, ignore_index=True).sort_values(["date", "symbol"]).reset_index(drop=True)

    # Verification of chronological boundary
    if not train_df.empty and not test_df.empty:
        logger.debug(
            f"Chronological split dates: Train [{train_df['date'].min().date()} to {train_df['date'].max().date()}], "
            f"Val [{val_df['date'].min().date()} to {val_df['date'].max().date()}], "
            f"Test [{test_df['date'].min().date()} to {test_df['date'].max().date()}]"
        )

    return {
        "X_train": train_df[features],
        "y_train": train_df[target_col],
        "train_df": train_df,
        "X_val": val_df[features],
        "y_val": val_df[target_col],
        "val_df": val_df,
        "X_test": test_df[features],
        "y_test": test_df[target_col],
        "test_df": test_df,
        "feature_cols": features,
    }


# =====================================================================
# 3. Model Training & Feature Importance
# =====================================================================

def train_linear_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    use_scaler: bool = True,
    **kwargs: Any,
) -> Any:
    """
    Train an Ordinary Least Squares Linear Regression baseline.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix.
    y_train : pd.Series
        Training target returns.
    use_scaler : bool, default True
        Standardizes features before fitting to prevent numerical instability.
    **kwargs
        Additional arguments passed to LinearRegression.

    Returns
    -------
    LinearRegression or Pipeline
        Fitted linear model or pipeline.
    """
    lr = LinearRegression(**kwargs)
    if use_scaler:
        model = make_pipeline(StandardScaler(), lr)
    else:
        model = lr
    model.fit(X_train, y_train)
    logger.info(f"Trained LinearRegression model (use_scaler={use_scaler}) on {len(X_train)} samples.")
    return model



def train_random_forest_regressor(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 100,
    max_depth: Optional[int] = 6,
    random_state: int = 42,
    n_jobs: int = -1,
    **kwargs: Any,
) -> RandomForestRegressor:
    """
    Train a Random Forest Regressor for non-linear return prediction.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix.
    y_train : pd.Series
        Training target returns.
    n_estimators : int, default 100
        Number of decision trees.
    max_depth : int, default 6
        Maximum tree depth to prevent overfitting noisy financial returns.
    random_state : int, default 42
        Seed for reproducibility.
    n_jobs : int, default -1
        Parallel CPU jobs.

    Returns
    -------
    RandomForestRegressor
        Fitted random forest model.
    """
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=n_jobs,
        **kwargs,
    )
    model.fit(X_train, y_train)
    logger.info(f"Trained RandomForestRegressor (trees={n_estimators}, max_depth={max_depth}) on {len(X_train)} samples.")
    return model


def get_feature_importance(
    model: Any,
    feature_names: List[str],
) -> pd.DataFrame:
    """
    Extract feature importances from a tree-based model.

    Parameters
    ----------
    model : fitted estimator
        Model exposing feature_importances_.
    feature_names : list of str
        Names of the input features.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['feature', 'importance'] sorted descending.
    """
    if not hasattr(model, "feature_importances_"):
        raise AttributeError(f"Model {type(model).__name__} does not have feature_importances_ attribute.")

    importances = model.feature_importances_
    fi_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values(by="importance", ascending=False).reset_index(drop=True)
    fi_df["importance"] = fi_df["importance"].round(6)
    return fi_df
