"""
Benchmark-Relative Analytics for Real Financial Equities vs NIFTY 50.

Calculates CAPM Beta, Jensen's Alpha, Tracking Error, Information Ratio,
and Benchmark Correlation with strict trading-date alignment.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


def calculate_benchmark_metrics(
    df: pd.DataFrame,
    symbol_col: str = "symbol",
    date_col: str = "date",
    stock_ret_col: str = "daily_adjusted_return",
    bench_ret_col: str = "benchmark_daily_return",
    rf_rate: float = 0.06,
) -> pd.DataFrame:
    """
    Compute benchmark-relative metrics for each equity against NIFTY 50.

    Parameters
    ----------
    df : pd.DataFrame
        Market dataset containing date, symbol, daily_adjusted_return, and benchmark_daily_return.
    rf_rate : float, default 0.06
        Annualized risk-free rate assumption.

    Returns
    -------
    pd.DataFrame
        Table with benchmark metrics per stock.
    """
    results = []

    for sym, group in df.groupby(symbol_col):
        # Drop rows missing either stock return or benchmark return
        valid = group.dropna(subset=[stock_ret_col, bench_ret_col]).sort_values(date_col)
        n = len(valid)
        if n < 30:
            continue

        r_stock = valid[stock_ret_col].values
        r_bench = valid[bench_ret_col].values

        mean_stock = float(np.mean(r_stock))
        mean_bench = float(np.mean(r_bench))
        var_bench = float(np.var(r_bench, ddof=1))
        cov_matrix = np.cov(r_stock, r_bench)
        cov = float(cov_matrix[0, 1])

        # Beta
        beta = cov / var_bench if var_bench > 0 else 1.0

        # Correlation & R-squared
        std_stock = float(np.std(r_stock, ddof=1))
        std_bench = float(np.std(r_bench, ddof=1))
        corr = cov / (std_stock * std_bench) if (std_stock * std_bench) > 0 else 0.0
        r_squared = corr**2

        # Annualized Returns
        ann_stock_ret = mean_stock * 252.0
        ann_bench_ret = mean_bench * 252.0
        ann_excess_ret = ann_stock_ret - ann_bench_ret

        # Jensen's Alpha (Annualized)
        # alpha = (R_i - R_f) - beta * (R_m - R_f)
        jensen_alpha_rf6 = (ann_stock_ret - rf_rate) - beta * (ann_bench_ret - rf_rate)
        jensen_alpha_rf0 = ann_stock_ret - beta * ann_bench_ret

        # Tracking Error (Annualized standard deviation of return differences)
        diff = r_stock - r_bench
        te = float(np.std(diff, ddof=1) * math.sqrt(252.0))

        # Information Ratio
        info_ratio = ann_excess_ret / te if te > 0 else 0.0

        # Up-market & Down-market Beta
        up_mask = r_bench > 0
        down_mask = r_bench < 0

        if np.sum(up_mask) > 10:
            up_cov = np.cov(r_stock[up_mask], r_bench[up_mask])[0, 1]
            up_var = np.var(r_bench[up_mask], ddof=1)
            up_beta = float(up_cov / up_var) if up_var > 0 else beta
        else:
            up_beta = beta

        if np.sum(down_mask) > 10:
            down_cov = np.cov(r_stock[down_mask], r_bench[down_mask])[0, 1]
            down_var = np.var(r_bench[down_mask], ddof=1)
            down_beta = float(down_cov / down_var) if down_var > 0 else beta
        else:
            down_beta = beta

        results.append({
            "symbol": sym,
            "aligned_sessions": n,
            "beta": round(beta, 4),
            "up_market_beta": round(up_beta, 4),
            "down_market_beta": round(down_beta, 4),
            "jensen_alpha_rf6": round(jensen_alpha_rf6, 4),
            "jensen_alpha_rf0": round(jensen_alpha_rf0, 4),
            "benchmark_correlation": round(corr, 4),
            "r_squared": round(r_squared, 4),
            "annualized_excess_return": round(ann_excess_ret, 4),
            "tracking_error": round(te, 4),
            "information_ratio": round(info_ratio, 4),
            "benchmark_annualized_return": round(ann_bench_ret, 4),
        })

    return pd.DataFrame(results).sort_values("jensen_alpha_rf6", ascending=False).reset_index(drop=True)

