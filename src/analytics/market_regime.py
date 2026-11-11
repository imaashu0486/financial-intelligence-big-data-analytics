"""
Market Regime and Breadth Analytics for NIFTY 50 Equities.

Provides mathematical classification of:
1. Daily Market Breadth (% advancing, % above moving averages).
2. Macro Benchmark Regimes (Bull, Bear, Transition).
3. Volatility Regimes (Low, Normal, High).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


def calculate_daily_market_breadth(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily market-wide breadth metrics across all reporting constituents.
    """
    records = []
    for dt, group in df.groupby("date"):
        total = len(group)
        if total == 0:
            continue

        valid_ret = group["daily_adjusted_return"].dropna()
        advances = int((valid_ret > 0).sum())
        declines = int((valid_ret < 0).sum())
        unchanged = int((valid_ret == 0).sum())
        advance_ratio = advances / total if total > 0 else 0.0

        # Moving average breadth
        valid_ma20 = group.dropna(subset=["close", "ma_20"])
        pct_above_ma20 = (valid_ma20["close"] > valid_ma20["ma_20"]).sum() / len(valid_ma20) if len(valid_ma20) > 0 else 0.5

        valid_ma50 = group.dropna(subset=["close", "ma_50"])
        pct_above_ma50 = (valid_ma50["close"] > valid_ma50["ma_50"]).sum() / len(valid_ma50) if len(valid_ma50) > 0 else 0.5

        valid_ma200 = group.dropna(subset=["close", "ma_200"])
        pct_above_ma200 = (valid_ma200["close"] > valid_ma200["ma_200"]).sum() / len(valid_ma200) if len(valid_ma200) > 0 else 0.5

        records.append({
            "date": dt,
            "total_constituents": total,
            "advances": advances,
            "declines": declines,
            "unchanged": unchanged,
            "advance_ratio": round(advance_ratio, 4),
            "pct_above_ma20": round(pct_above_ma20, 4),
            "pct_above_ma50": round(pct_above_ma50, 4),
            "pct_above_ma200": round(pct_above_ma200, 4),
        })

    return pd.DataFrame(records).sort_values("date").reset_index(drop=True)


def classify_benchmark_regimes(benchmark_df: pd.DataFrame) -> pd.DataFrame:
    """
    Classify macro trend and volatility regimes for NIFTY 50 benchmark index.

    Trend Regimes:
    - Bull Market: close > ma_200 AND ma_50 > ma_200
    - Bear Market: close < ma_200 AND ma_50 < ma_200
    - Transition / Sideways: otherwise

    Volatility Regimes:
    - Low Volatility: ann_vol < 12%
    - Normal Volatility: 12% <= ann_vol <= 20%
    - High Volatility: ann_vol > 20%
    """
    df = benchmark_df.sort_values("date").copy()

    trend_regimes = []
    vol_regimes = []

    for _, row in df.iterrows():
        c = row.get("close")
        ma50 = row.get("benchmark_ma_50")
        ma200 = row.get("benchmark_ma_200")
        vol60 = row.get("benchmark_volatility_60d")

        # Trend Regime
        if pd.notna(c) and pd.notna(ma50) and pd.notna(ma200):
            if c > ma200 and ma50 > ma200:
                trend = "Bull"
            elif c < ma200 and ma50 < ma200:
                trend = "Bear"
            else:
                trend = "Transition"
        else:
            trend = "Unclassified"
        trend_regimes.append(trend)

        # Volatility Regime
        if pd.notna(vol60):
            ann_vol = vol60 * np.sqrt(252)
            if ann_vol < 0.12:
                vol_reg = "Low_Vol"
            elif ann_vol <= 0.20:
                vol_reg = "Normal_Vol"
            else:
                vol_reg = "High_Vol"
        else:
            vol_reg = "Normal_Vol"
        vol_regimes.append(vol_reg)

    df["trend_regime"] = trend_regimes
    df["volatility_regime"] = vol_regimes
    return df
