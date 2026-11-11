"""
Advanced Risk and Return Analytics for Real Financial Markets.

Calculates comprehensive risk, return, drawdown, and tail-risk metrics
across historical daily equity observations.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


def calculate_equity_risk_return(
    df: pd.DataFrame,
    symbol_col: str = "symbol",
    date_col: str = "date",
    price_col: str = "adj_close",
    return_col: str = "daily_adjusted_return",
    rf_rate: float = 0.06,  # Baseline 6% annualized risk-free rate (approx. 10Y Indian G-Sec)
) -> pd.DataFrame:
    """
    Compute comprehensive risk and return analytics for all stocks in the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Market dataset containing date, symbol, adj_close, and daily_adjusted_return.
    rf_rate : float, default 0.06
        Annualized risk-free rate assumption.

    Returns
    -------
    pd.DataFrame
        Summary table with one row per stock.
    """
    results = []

    for sym, group in df.groupby(symbol_col):
        grp = group.sort_values(date_col).copy()
        returns = grp[return_col].dropna().values
        prices = grp[price_col].dropna().values

        n_days = len(returns)
        if n_days < 20:
            continue

        # 1. Basic Return & Volatility
        mean_daily = float(np.mean(returns))
        std_daily = float(np.std(returns, ddof=1))
        ann_return = mean_daily * 252.0
        ann_vol = std_daily * math.sqrt(252.0)

        # 2. Sharpe Ratios (Both Rf=0 and Rf=configured)
        sharpe_rf0 = ann_return / ann_vol if ann_vol > 0 else 0.0
        sharpe_rf = (ann_return - rf_rate) / ann_vol if ann_vol > 0 else 0.0

        # 3. Maximum Drawdown Calculation
        cum_max = np.maximum.accumulate(prices)
        drawdowns = (prices - cum_max) / cum_max
        max_dd = float(np.min(drawdowns))
        max_dd_idx = int(np.argmin(drawdowns))

        # Peak date preceding the max drawdown trough
        trough_date = str(grp[date_col].iloc[max_dd_idx])
        peak_idx = int(np.argmax(prices[: max_dd_idx + 1]))
        peak_date = str(grp[date_col].iloc[peak_idx])

        # 4. Downside Deviation & Sortino Ratio
        # Target daily return is rf_rate / 252
        daily_target = rf_rate / 252.0
        downside_diff = np.minimum(0.0, returns - daily_target)
        downside_dev = float(math.sqrt(np.mean(downside_diff**2)) * math.sqrt(252.0))
        sortino_rf = (ann_return - rf_rate) / downside_dev if downside_dev > 0 else 0.0

        # Downside deviation relative to 0
        downside_diff_0 = np.minimum(0.0, returns)
        downside_dev_0 = float(math.sqrt(np.mean(downside_diff_0**2)) * math.sqrt(252.0))
        sortino_rf0 = ann_return / downside_dev_0 if downside_dev_0 > 0 else 0.0

        # 5. Calmar Ratio
        calmar = ann_return / abs(max_dd) if abs(max_dd) > 0 else 0.0

        # 6. Breadth and Extremes
        pos_days = int(np.sum(returns > 0))
        pos_days_pct = float(pos_days / n_days * 100.0)

        best_day_ret = float(np.max(returns))
        best_day_idx = int(np.argmax(returns))
        best_day_date = str(grp[date_col].iloc[best_day_idx + (1 if len(grp) > len(returns) else 0)])

        worst_day_ret = float(np.min(returns))
        worst_day_idx = int(np.argmin(returns))
        worst_day_date = str(grp[date_col].iloc[worst_day_idx + (1 if len(grp) > len(returns) else 0)])

        # 7. Tail Risk: Value at Risk (Historical 95% & 99%) and Expected Shortfall
        var_95 = float(-np.percentile(returns, 5.0))
        var_99 = float(-np.percentile(returns, 1.0))
        tail_returns_95 = returns[returns <= -var_95]
        cvar_95 = float(-np.mean(tail_returns_95)) if len(tail_returns_95) > 0 else var_95

        results.append({
            "symbol": sym,
            "observation_count": n_days,
            "mean_daily_return": round(mean_daily, 6),
            "annualized_return": round(ann_return, 4),
            "annualized_volatility": round(ann_vol, 4),
            "sharpe_ratio_rf0": round(sharpe_rf0, 4),
            "sharpe_ratio_rf6": round(sharpe_rf, 4),
            "max_drawdown": round(max_dd, 4),
            "max_dd_peak_date": peak_date,
            "max_dd_trough_date": trough_date,
            "downside_deviation": round(downside_dev, 4),
            "sortino_ratio_rf0": round(sortino_rf0, 4),
            "sortino_ratio_rf6": round(sortino_rf, 4),
            "calmar_ratio": round(calmar, 4),
            "positive_days_pct": round(pos_days_pct, 4),
            "best_day_return": round(best_day_ret, 4),
            "best_day_date": best_day_date,
            "worst_day_return": round(worst_day_ret, 4),
            "worst_day_date": worst_day_date,
            "var_95_daily": round(var_95, 4),
            "var_99_daily": round(var_99, 4),
            "cvar_95_daily": round(cvar_95, 4),
        })

    return pd.DataFrame(results).sort_values("annualized_return", ascending=False).reset_index(drop=True)
