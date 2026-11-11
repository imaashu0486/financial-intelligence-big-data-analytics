"""
Unsupervised Stock Clustering pipeline using KMeans.

Segments stock symbols based on historical behavioral metrics:
- Average Daily Return
- Average Volatility
- Average Volume
- Maximum Drawdown (peak-to-trough decline)
- Return / Risk Ratio (mean return / volatility)

Features are standardized using StandardScaler to ensure volume does not dominate.
Provides Elbow Method (inertia) and Silhouette Score evaluation across candidate k.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.ml.evaluation import calculate_clustering_metrics
from src.utils.logger import logger

CLUSTER_FEATURE_NAMES = [
    "avg_daily_return",
    "avg_volatility",
    "avg_volume",
    "max_drawdown",
    "return_risk_ratio",
]


# =====================================================================
# 1. Historical Stock Behavioral Feature Extraction
# =====================================================================

def calculate_max_drawdown(prices: pd.Series) -> float:
    """
    Calculate the historical maximum peak-to-trough drawdown of a price series.

    Formula:
    Peak_t = max_{tau <= t}(Price_tau)
    Drawdown_t = (Price_t - Peak_t) / Peak_t
    Max Drawdown = min_t(Drawdown_t)

    Returns
    -------
    float
        Maximum drawdown as a negative decimal (e.g., -0.18 means 18% max decline).
        Returns 0.0 if prices are strictly monotonically increasing or empty.
    """
    if prices.empty:
        return 0.0

    # Running cumulative maximum
    running_max = prices.cummax()
    # Avoid zero division
    drawdowns = (prices - running_max) / running_max.replace(0, np.nan)
    max_dd = float(drawdowns.min())
    return round(max_dd if not np.isnan(max_dd) else 0.0, 6)


def extract_stock_cluster_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract stock-level summary behavioral features across the entire timeline.

    Parameters
    ----------
    df : pd.DataFrame
        Market features dataset containing 'date', 'symbol', 'close', 'daily_return',
        'volatility', 'volume'.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        ['symbol', 'avg_daily_return', 'avg_volatility', 'avg_volume',
         'max_drawdown', 'return_risk_ratio']
    """
    required = {"symbol", "close", "daily_return", "volatility", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for clustering: {sorted(missing)}")

    records = []
    # Ensure chronological sort for drawdown
    df_sorted = df.sort_values(["symbol", "date"]).reset_index(drop=True)

    for symbol, group in df_sorted.groupby("symbol"):
        avg_ret = float(group["daily_return"].dropna().mean())
        avg_vol = float(group["volatility"].dropna().mean())
        avg_volm = float(group["volume"].dropna().mean())
        max_dd = calculate_max_drawdown(group["close"])

        # Return / risk ratio (annualized or daily ratio)
        if avg_vol > 1e-7:
            rr_ratio = avg_ret / avg_vol
        else:
            rr_ratio = 0.0

        records.append({
            "symbol": symbol,
            "avg_daily_return": round(avg_ret, 6),
            "avg_volatility": round(avg_vol, 6),
            "avg_volume": round(avg_volm, 2),
            "max_drawdown": round(max_dd, 6),
            "return_risk_ratio": round(rr_ratio, 6),
        })

    features_df = pd.DataFrame(records).sort_values("symbol").reset_index(drop=True)
    logger.info(f"Extracted clustering features for {len(features_df)} stock symbols.")
    return features_df


# =====================================================================
# 2. Elbow & Silhouette Evaluation
# =====================================================================

def find_optimal_clusters(
    features_df: pd.DataFrame,
    min_k: int = 2,
    max_k: int = 6,
    feature_cols: Optional[List[str]] = None,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Evaluate candidate cluster counts k using the Elbow Method (inertia)
    and Silhouette Score.

    Parameters
    ----------
    features_df : pd.DataFrame
        Stock-level features DataFrame.
    min_k : int, default 2
        Minimum k to evaluate.
    max_k : int, default 6
        Maximum k to evaluate (capped at len(features_df) - 1).
    feature_cols : list of str, optional
        Features to standardize and cluster on.
    random_state : int, default 42
        KMeans seed.

    Returns
    -------
    dict
        {
            "k_values": list of int,
            "inertias": list of float,
            "silhouette_scores": list of float,
            "scaler": StandardScaler,
            "X_scaled": ndarray
        }
    """
    cols = feature_cols or CLUSTER_FEATURE_NAMES
    n_samples = len(features_df)
    # Ensure max_k is valid for sample size
    effective_max_k = min(max_k, n_samples - 1)
    if effective_max_k < min_k:
        effective_max_k = min_k

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features_df[cols])

    k_values: List[int] = []
    inertias: List[float] = []
    silhouette_scores: List[float] = []

    for k in range(min_k, effective_max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(X_scaled)

        k_values.append(k)
        inertias.append(round(float(kmeans.inertia_), 4))

        metrics = calculate_clustering_metrics(X_scaled, labels, inertia=kmeans.inertia_)
        silhouette_scores.append(metrics["silhouette_score"] or 0.0)

    logger.info(
        f"Elbow & Silhouette analysis complete for k={k_values}. "
        f"Inertias={inertias}, Silhouettes={silhouette_scores}"
    )

    return {
        "k_values": k_values,
        "inertias": inertias,
        "silhouette_scores": silhouette_scores,
        "scaler": scaler,
        "X_scaled": X_scaled,
    }


# =====================================================================
# 3. Model Fitting & Cluster Profiling
# =====================================================================

def fit_stock_clusters(
    features_df: pd.DataFrame,
    n_clusters: int = 3,
    feature_cols: Optional[List[str]] = None,
    random_state: int = 42,
) -> Tuple[KMeans, StandardScaler, pd.DataFrame, pd.DataFrame]:
    """
    Fit final KMeans clustering model on standardized stock behavioral features.

    Parameters
    ----------
    features_df : pd.DataFrame
        Stock-level behavioral DataFrame.
    n_clusters : int, default 3
        Number of clusters to partition symbols into.
    feature_cols : list of str, optional
        Feature subset to cluster on.
    random_state : int, default 42
        KMeans seed.

    Returns
    -------
    tuple
        (fitted_kmeans, fitted_scaler, clustered_stocks_df, cluster_profiles_df)
    """
    cols = feature_cols or CLUSTER_FEATURE_NAMES
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features_df[cols])

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)

    # Attach cluster assignments
    clustered_df = features_df.copy()
    clustered_df["cluster"] = cluster_labels

    # Compute descriptive cluster profiles (mean of unscaled features per cluster)
    profiles = (
        clustered_df.groupby("cluster")[cols]
        .mean()
        .reset_index()
    )
    # Add count of stocks in cluster
    counts = clustered_df.groupby("cluster")["symbol"].count().reset_index(name="stock_count")
    symbols_list = clustered_df.groupby("cluster")["symbol"].apply(lambda s: ", ".join(s)).reset_index(name="symbols")
    profiles = profiles.merge(counts, on="cluster").merge(symbols_list, on="cluster")

    # Generate academic behavioral characterization
    def describe_cluster(row: pd.Series) -> str:
        ret = row["avg_daily_return"]
        vol = row["avg_volatility"]
        dd = row["max_drawdown"]
        return (
            f"Cluster {int(row['cluster'])}: "
            f"{'High' if vol > clustered_df['avg_volatility'].mean() else 'Moderate'} Volatility, "
            f"{'Positive' if ret > 0 else 'Negative'} Mean Daily Return ({ret:.4f}), "
            f"Max DD ({dd:.2%})"
        )

    profiles["characterization"] = profiles.apply(describe_cluster, axis=1)

    logger.info(f"Fitted KMeans(k={n_clusters}). Profiles:\n{profiles[['cluster', 'stock_count', 'characterization']]}")

    return kmeans, scaler, clustered_df, profiles
