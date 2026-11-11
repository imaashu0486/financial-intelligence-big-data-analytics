"""
Multi-Factor Stock Ranking and Screening Engine.

Provides transparent, configurable multi-factor scoring:
- Risk-Adjusted Return (Sharpe / Sortino)
- Volatility Mitigation (Lower is better)
- Drawdown Defense (Lower maximum drawdown is better)
- Benchmark Alpha (Jensen's Alpha)
- Consistency (Positive Trading Days %)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


def compute_multi_factor_rankings(
    risk_df: pd.DataFrame,
    bench_df: pd.DataFrame,
    weights: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    Compute multi-factor stock scores and rankings from risk-return and benchmark metrics.

    Parameters
    ----------
    risk_df : pd.DataFrame
        Output of calculate_equity_risk_return().
    bench_df : pd.DataFrame
        Output of calculate_benchmark_metrics().
    weights : dict, optional
        Custom factor weights (must sum to 1.0). Default:
        - sharpe: 0.25
        - volatility: 0.20
        - drawdown: 0.20
        - alpha: 0.20
        - consistency: 0.15

    Returns
    -------
    pd.DataFrame
        Ranked equities with factor scores and final composite scores.
    """
    if weights is None:
        weights = {
            "sharpe": 0.25,
            "volatility": 0.20,
            "drawdown": 0.20,
            "alpha": 0.20,
            "consistency": 0.15,
        }

    # Normalize weights if sum != 1.0
    w_sum = sum(weights.values())
    w = {k: v / w_sum for k, v in weights.items()}

    # Merge risk and benchmark metrics
    merged = pd.merge(risk_df, bench_df, on="symbol", suffixes=("_risk", "_bench"))

    if merged.empty:
        return pd.DataFrame()

    # Disambiguate return column if collision occurred
    if "annualized_return" not in merged.columns and "annualized_return_risk" in merged.columns:
        merged["annualized_return"] = merged["annualized_return_risk"]
    elif "annualized_return" not in merged.columns and "annualized_stock_return" in merged.columns:
        merged["annualized_return"] = merged["annualized_stock_return"]

    # Compute Percentile Scores (0 to 100)
    # Higher Sharpe is better
    sharpe_col = "sharpe_ratio_rf6" if "sharpe_ratio_rf6" in merged.columns else ("sharpe_ratio" if "sharpe_ratio" in merged.columns else ("sharpe_ratio_rf0" if "sharpe_ratio_rf0" in merged.columns else None))
    if sharpe_col and merged[sharpe_col].notna().any():
        merged["score_sharpe"] = merged[sharpe_col].rank(pct=True) * 100.0
    else:
        merged["score_sharpe"] = 50.0

    # Lower Volatility is better (invert rank)
    vol_col = "annualized_volatility" if "annualized_volatility" in merged.columns else ("daily_volatility" if "daily_volatility" in merged.columns else None)
    if vol_col and merged[vol_col].notna().any():
        merged["score_volatility"] = (1.0 - merged[vol_col].rank(pct=True)) * 100.0
    else:
        merged["score_volatility"] = 50.0

    # Lower Drawdown magnitude is better (max_drawdown is negative, so higher algebraic value is better).
    # If historical peak-to-trough drawdown is unavailable, assign neutral median rank (50.0).
    # Note: worst_day_return is a single-session tail return and must NOT be substituted for cumulative drawdown.
    if "max_drawdown" in merged.columns and merged["max_drawdown"].notna().any():
        merged["score_drawdown"] = merged["max_drawdown"].rank(pct=True) * 100.0
    elif "rolling_max_drawdown_60d" in merged.columns and merged["rolling_max_drawdown_60d"].notna().any():
        merged["max_drawdown"] = merged["rolling_max_drawdown_60d"]
        merged["score_drawdown"] = merged["max_drawdown"].rank(pct=True) * 100.0
    else:
        merged["max_drawdown"] = np.nan
        merged["score_drawdown"] = 50.0

    # Higher Jensen's Alpha is better
    alpha_col = "jensen_alpha_rf6" if "jensen_alpha_rf6" in merged.columns else ("jensen_alpha" if "jensen_alpha" in merged.columns else ("annualized_alpha" if "annualized_alpha" in merged.columns else None))
    if alpha_col and merged[alpha_col].notna().any():
        merged["score_alpha"] = merged[alpha_col].rank(pct=True) * 100.0
    else:
        merged["score_alpha"] = 50.0

    # Higher Consistency (% positive days) is better
    if "positive_days_pct" in merged.columns and merged["positive_days_pct"].notna().any():
        merged["score_consistency"] = merged["positive_days_pct"].rank(pct=True) * 100.0
    else:
        merged["score_consistency"] = 50.0

    # Composite Score
    merged["composite_score"] = (
        merged["score_sharpe"] * w["sharpe"]
        + merged["score_volatility"] * w["volatility"]
        + merged["score_drawdown"] * w["drawdown"]
        + merged["score_alpha"] * w["alpha"]
        + merged["score_consistency"] * w["consistency"]
    ).round(2)

    # Rank by composite score
    merged["rank"] = merged["composite_score"].rank(ascending=False, method="min").astype(int)

    output_cols = [
        "rank",
        "symbol",
        "composite_score",
        "score_sharpe",
        "score_volatility",
        "score_drawdown",
        "score_alpha",
        "score_consistency",
        "annualized_return",
        "annualized_volatility",
        "sharpe_ratio_rf6",
        "max_drawdown",
        "jensen_alpha_rf6",
        "beta",
        "positive_days_pct",
    ]

    for col in output_cols:
        if col not in merged.columns:
            merged[col] = np.nan

    return merged[output_cols].sort_values("rank").reset_index(drop=True)

