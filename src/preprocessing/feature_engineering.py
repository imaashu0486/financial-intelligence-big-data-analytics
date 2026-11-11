"""
Feature engineering module — per-stock technical indicators.

All calculations are **look-ahead-free**: feature at row i uses only data ≤ i.

Usage:
    from src.preprocessing.feature_engineering import engineer_features
    df_feat = engineer_features(df_clean, cfg["features"])
"""

import numpy as np
import pandas as pd

from src.utils.logger import logger


def _compute_rsi(series: pd.Series, period: int) -> pd.Series:
    """
    Compute Relative Strength Index (RSI).

    RSI = 100 - 100 / (1 + RS)
    RS  = avg_gain / avg_loss  (exponential moving average)

    Edge cases:
        - avg_loss == 0 → RSI = 100 (all gains)
        - avg_gain == 0 → RSI = 0   (all losses)
    """
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Use EMA (Wilder's smoothing = alpha=1/period)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    # Handle edge cases: where both are 0 → RSI=50, loss=0 → RSI=100, gain=0 → RSI=0
    rsi = pd.Series(np.nan, index=series.index)
    valid = avg_gain.notna() & avg_loss.notna()

    both_zero = valid & (avg_gain == 0) & (avg_loss == 0)
    loss_zero = valid & (avg_loss == 0) & (avg_gain > 0)
    gain_zero = valid & (avg_gain == 0) & (avg_loss > 0)
    normal = valid & (avg_loss > 0) & (avg_gain >= 0)

    rsi[both_zero] = 50.0
    rsi[loss_zero] = 100.0
    rsi[gain_zero] = 0.0
    rs = avg_gain[normal] / avg_loss[normal]
    rsi[normal] = 100 - (100 / (1 + rs))

    return rsi


def _compute_macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series]:
    """
    Compute MACD line and signal line.

    MACD   = EMA(fast) - EMA(slow)
    Signal = EMA(MACD, signal)
    """
    ema_fast = series.ewm(span=fast, min_periods=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, min_periods=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, min_periods=signal, adjust=False).mean()
    return macd_line, signal_line


def engineer_features(df: pd.DataFrame, feature_cfg: dict) -> pd.DataFrame:
    """
    Add technical features to market data, computed per-stock.

    Features added:
        - daily_return, log_return
        - ma_{w} for each window in config
        - volatility (rolling std of daily_return)
        - volume_change
        - rsi_{period}
        - macd, macd_signal

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned market data with columns: date, symbol, open, high, low, close, volume.
    feature_cfg : dict
        Feature parameters from config.yaml.

    Returns
    -------
    pd.DataFrame
        Market data with additional feature columns.
    """
    df = df.copy()
    df.sort_values(["symbol", "date"], inplace=True)

    ma_windows = feature_cfg.get("moving_averages", [7, 20, 50])
    rsi_period = feature_cfg.get("rsi_period", 14)
    macd_cfg = feature_cfg.get("macd", {"fast": 12, "slow": 26, "signal": 9})
    vol_window = feature_cfg.get("volatility_window", 20)
    min_periods = feature_cfg.get("min_periods", 1)

    logger.info(
        f"Engineering features: MAs={ma_windows}, RSI={rsi_period}, "
        f"MACD=({macd_cfg['fast']},{macd_cfg['slow']},{macd_cfg['signal']}), "
        f"vol_window={vol_window}"
    )

    result_frames = []

    for symbol, group in df.groupby("symbol"):
        g = group.copy()

        # ── Returns ──────────────────────────────────────────
        g["daily_return"] = g["close"].pct_change()
        g["log_return"] = np.log(g["close"] / g["close"].shift(1))

        # ── Moving Averages ──────────────────────────────────
        for w in ma_windows:
            g[f"ma_{w}"] = g["close"].rolling(window=w, min_periods=min_periods).mean()

        # ── Volatility (rolling std of daily return) ─────────
        g["volatility"] = g["daily_return"].rolling(
            window=vol_window, min_periods=min_periods
        ).std()

        # ── Volume Change ────────────────────────────────────
        g["volume_change"] = g["volume"].pct_change()

        # ── RSI ──────────────────────────────────────────────
        g[f"rsi_{rsi_period}"] = _compute_rsi(g["close"], rsi_period)

        # ── MACD ─────────────────────────────────────────────
        macd_line, signal_line = _compute_macd(
            g["close"],
            fast=macd_cfg["fast"],
            slow=macd_cfg["slow"],
            signal=macd_cfg["signal"],
        )
        g["macd"] = macd_line
        g["macd_signal"] = signal_line

        result_frames.append(g)

    result = pd.concat(result_frames, ignore_index=True)
    result.sort_values(["date", "symbol"], inplace=True)
    result.reset_index(drop=True, inplace=True)

    new_cols = [
        c for c in result.columns
        if c not in {"date", "symbol", "open", "high", "low", "close", "volume", "outlier_flag"}
    ]
    logger.info(f"Features added: {new_cols}")
    logger.info(f"Final shape: {result.shape}")

    return result
