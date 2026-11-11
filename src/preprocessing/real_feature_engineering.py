"""
Real Data Feature Engineering & Preprocessing Pipeline.

Processes raw historical market OHLCV data (124,421 bars across 51 instruments)
and real financial news/announcements into machine learning-ready technical features.

Key Principles:
    - Look-ahead-free: Feature at row t strictly uses data <= t.
    - Split-safe returns: Uses `adj_close` for return, RSI, and MACD calculations
      to prevent artificial -90% split distortions.
    - Complete non-destructive isolation: Outputs are stored in data/real/processed/,
      leaving data/sample/ and data/processed/ untouched.
"""

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.utils.logger import logger

DEFAULT_RAW_MARKET_DIR = Path("data/real/raw/market")
DEFAULT_RAW_NEWS_FILE = Path("data/real/raw/news/financial_news.csv")
DEFAULT_PROCESSED_DIR = Path("data/real/processed")
DEFAULT_MANIFEST_PATH = Path("data/real/metadata/real_features_manifest.json")


def compute_file_sha256(filepath: Union[str, Path]) -> str:
    """Compute SHA-256 hash of a file."""
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Compute Wilder's Relative Strength Index (RSI).

    RSI = 100 - (100 / (1 + RS))
    where RS = EMA(gain, period) / EMA(loss, period)
    """
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Wilder's exponential smoothing: alpha = 1 / period
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

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
    rsi[normal] = 100.0 - (100.0 / (1.0 + rs))

    return rsi


def compute_macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> Tuple[pd.Series, pd.Series]:
    """
    Compute Moving Average Convergence Divergence (MACD) and Signal Line.
    """
    ema_fast = series.ewm(span=fast, min_periods=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, min_periods=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, min_periods=signal, adjust=False).mean()
    return macd_line, signal_line


def engineer_real_market_features(
    market_df: pd.DataFrame,
    ma_windows: Optional[List[int]] = None,
    rsi_period: int = 14,
    vol_window: int = 20,
    zscore_threshold: float = 3.5,
) -> pd.DataFrame:
    """
    Compute look-ahead-free technical indicators per stock on real market data.
    """
    if ma_windows is None:
        ma_windows = [7, 20, 50, 200]

    df = market_df.copy()
    df.sort_values(by=["symbol", "date"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    result_groups = []

    for sym, group in df.groupby("symbol"):
        g = group.copy()
        price_base = g["adj_close"] if "adj_close" in g.columns else g["close"]

        # 1. Returns (computed on split-adjusted prices)
        g["daily_return"] = price_base.pct_change()
        g["log_return"] = np.log(price_base / price_base.shift(1))

        # 2. Moving Averages
        for w in ma_windows:
            g[f"ma_{w}"] = g["close"].rolling(window=w, min_periods=1).mean()
            if "adj_close" in g.columns:
                g[f"adj_ma_{w}"] = g["adj_close"].rolling(window=w, min_periods=1).mean()

        # 3. Volatility (rolling std of daily return)
        g["volatility"] = g["daily_return"].rolling(window=vol_window, min_periods=2).std()

        # 4. Volume change
        g["volume_change"] = g["volume"].pct_change().clip(lower=-1.0, upper=10.0)

        # 5. High-Low Spread
        g["hl_spread"] = (g["high"] - g["low"]) / g["close"].replace(0, np.nan)

        # 6. RSI (on adjusted close)
        g[f"rsi_{rsi_period}"] = compute_rsi(price_base, period=rsi_period)

        # 7. MACD & Signal Line (on adjusted close)
        macd, signal = compute_macd(price_base, fast=12, slow=26, signal=9)
        g["macd"] = macd
        g["macd_signal"] = signal

        # 8. Bollinger Bands (20-day on close)
        rolling_mean_20 = g["close"].rolling(window=20, min_periods=1).mean()
        rolling_std_20 = g["close"].rolling(window=20, min_periods=2).std().fillna(0)
        g["bollinger_mid"] = rolling_mean_20
        g["bollinger_upper"] = rolling_mean_20 + (2.0 * rolling_std_20)
        g["bollinger_lower"] = rolling_mean_20 - (2.0 * rolling_std_20)

        # 9. Outlier Flagging (z-score on return)
        mean_ret = g["daily_return"].mean()
        std_ret = g["daily_return"].std()
        if std_ret and std_ret > 0:
            g["outlier_flag"] = (
                np.abs((g["daily_return"] - mean_ret) / std_ret) > zscore_threshold
            )
        else:
            g["outlier_flag"] = False

        result_groups.append(g)

    out_df = pd.concat(result_groups, ignore_index=True)
    out_df.sort_values(by=["date", "symbol"], inplace=True)
    out_df.reset_index(drop=True, inplace=True)
    return out_df


def process_real_market_data(
    raw_dir: Union[str, Path] = DEFAULT_RAW_MARKET_DIR,
    output_dir: Union[str, Path] = DEFAULT_PROCESSED_DIR,
) -> pd.DataFrame:
    """
    Ingest, combine, and feature-engineer all individual real market CSV files.
    """
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(list(raw_dir.glob("*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No market CSV files found in {raw_dir}")

    logger.info(f"Loading {len(csv_files)} market instrument files from {raw_dir}...")
    dfs = []
    for f in csv_files:
        temp_df = pd.read_csv(f)
        dfs.append(temp_df)

    combined_df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Loaded {len(combined_df):,} total raw bars across {len(csv_files)} instruments.")

    # Engineer features
    processed_df = engineer_real_market_features(combined_df)

    out_file = output_dir / "market_features.csv"
    processed_df.to_csv(out_file, index=False)
    logger.info(
        f"Saved real market features: {len(processed_df):,} rows, "
        f"{len(processed_df.columns)} columns to {out_file}"
    )
    return processed_df


def process_real_news_data(
    raw_news_file: Union[str, Path] = DEFAULT_RAW_NEWS_FILE,
    output_dir: Union[str, Path] = DEFAULT_PROCESSED_DIR,
) -> pd.DataFrame:
    """
    Clean and normalize real financial news dataset.
    """
    raw_news_file = Path(raw_news_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not raw_news_file.exists():
        raise FileNotFoundError(f"Raw news file not found: {raw_news_file}")

    df = pd.read_csv(raw_news_file)
    logger.info(f"Cleaning {len(df)} real news articles from {raw_news_file}...")

    # Standardize columns to match downstream NLP expectations:
    # [date, symbol, title, description, source, ...]
    cleaned = pd.DataFrame()
    cleaned["date"] = pd.to_datetime(df["published_at"], errors="coerce").dt.strftime("%Y-%m-%d")
    cleaned["symbol"] = df["ticker"].astype(str).str.strip().str.upper()
    cleaned["title"] = df["headline"].astype(str).str.strip()
    cleaned["description"] = df["description_or_snippet"].astype(str).str.strip()
    cleaned["source"] = df["source_name"].astype(str).str.strip()
    cleaned["article_id"] = df["article_id"]
    cleaned["article_type"] = df.get("article_type", "news")
    cleaned["content_provenance"] = df.get("content_provenance", "gdelt")
    cleaned["article_url"] = df.get("article_url", "")

    # Drop any null dates
    cleaned.dropna(subset=["date", "symbol", "title"], inplace=True)
    cleaned.sort_values(by=["date", "symbol"], inplace=True)
    cleaned.reset_index(drop=True, inplace=True)

    out_file = output_dir / "news_cleaned.csv"
    cleaned.to_csv(out_file, index=False)
    logger.info(f"Saved cleaned real news: {len(cleaned)} rows to {out_file}")
    return cleaned


def run_real_feature_engineering_pipeline(
    raw_market_dir: Union[str, Path] = DEFAULT_RAW_MARKET_DIR,
    raw_news_file: Union[str, Path] = DEFAULT_RAW_NEWS_FILE,
    output_dir: Union[str, Path] = DEFAULT_PROCESSED_DIR,
    manifest_path: Union[str, Path] = DEFAULT_MANIFEST_PATH,
) -> Dict[str, Any]:
    """
    Execute full feature engineering and preprocessing for real datasets.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    m_df = process_real_market_data(raw_market_dir, output_dir)
    n_df = process_real_news_data(raw_news_file, output_dir)

    market_file = output_dir / "market_features.csv"
    news_file = output_dir / "news_cleaned.csv"

    manifest = {
        "pipeline_execution_timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": "SUCCESS",
        "market_features": {
            "output_path": str(market_file),
            "rows_count": len(m_df),
            "columns_count": len(m_df.columns),
            "columns": list(m_df.columns),
            "instruments_count": int(m_df["symbol"].nunique()),
            "earliest_date": str(m_df["date"].min()),
            "latest_date": str(m_df["date"].max()),
            "file_size_bytes": market_file.stat().st_size,
            "sha256": compute_file_sha256(market_file),
        },
        "news_cleaned": {
            "output_path": str(news_file),
            "rows_count": len(n_df),
            "columns_count": len(n_df.columns),
            "columns": list(n_df.columns),
            "instruments_count": int(n_df["symbol"].nunique()),
            "earliest_date": str(n_df["date"].min()),
            "latest_date": str(n_df["date"].max()),
            "file_size_bytes": news_file.stat().st_size,
            "sha256": compute_file_sha256(news_file),
        },
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Manifest written to {manifest_path}")
    return manifest


if __name__ == "__main__":
    run_real_feature_engineering_pipeline()

