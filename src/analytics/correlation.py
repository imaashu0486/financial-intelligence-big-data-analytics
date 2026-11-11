"""
Correlation and Diversification Analytics across NIFTY 50 Equities.

Computes pairwise return correlation matrices, identifies highly correlated
asset clusters, and discovers diversification candidates.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd


def compute_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot daily returns to create wide format (date x symbol) and compute correlation matrix.
    """
    pivot = df.pivot(index="date", columns="symbol", values="daily_adjusted_return")
    corr_matrix = pivot.corr()
    return corr_matrix


def extract_correlation_pairs(corr_matrix: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract sorted list of unique pairs:
    1. Highest positive correlation pairs.
    2. Lowest / least correlation pairs.
    """
    pairs = []
    symbols = list(corr_matrix.columns)
    n = len(symbols)

    for i in range(n):
        for j in range(i + 1, n):
            s1 = symbols[i]
            s2 = symbols[j]
            r = corr_matrix.loc[s1, s2]
            if pd.notna(r):
                pairs.append({
                    "stock_1": s1,
                    "stock_2": s2,
                    "correlation": round(float(r), 4),
                })

    df_pairs = pd.DataFrame(pairs)
    high_corr = df_pairs.sort_values("correlation", ascending=False).reset_index(drop=True)
    low_corr = df_pairs.sort_values("correlation", ascending=True).reset_index(drop=True)

    return high_corr, low_corr

