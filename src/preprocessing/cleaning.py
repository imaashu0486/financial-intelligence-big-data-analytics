"""
Data cleaning module — deduplication, missing-value handling, outlier flagging.

Usage:
    from src.preprocessing.cleaning import clean_market_data
    df_clean = clean_market_data(df_raw, cfg["cleaning"])
"""

import numpy as np
import pandas as pd

from src.utils.logger import logger


def clean_market_data(df: pd.DataFrame, cleaning_cfg: dict) -> pd.DataFrame:
    """
    Clean and validate market OHLCV data.

    Steps:
        1. Remove exact duplicate rows.
        2. Remove duplicate (date, symbol) rows — keep last.
        3. Forward-fill missing prices (per stock, with limit).
        4. Drop rows still missing critical columns.
        5. Validate price sanity (min/max bounds).
        6. Validate volume sanity.
        7. Flag return-based outliers (z-score).

    Parameters
    ----------
    df : pd.DataFrame
        Raw market data (must have: date, symbol, open, high, low, close, volume).
    cleaning_cfg : dict
        Cleaning parameters from config.yaml.

    Returns
    -------
    pd.DataFrame
        Cleaned data with an ``outlier_flag`` column.
    """
    df = df.copy()
    initial_rows = len(df)
    logger.info(f"Cleaning market data: {initial_rows:,} rows")

    # 1. Exact duplicates
    df.drop_duplicates(inplace=True)
    logger.debug(f"  After drop_duplicates: {len(df):,}")

    # 2. (date, symbol) duplicates — keep last
    before = len(df)
    df.drop_duplicates(subset=["date", "symbol"], keep="last", inplace=True)
    if before - len(df):
        logger.info(f"  Removed {before - len(df)} duplicate (date, symbol) rows")

    # 3. Forward-fill missing prices per stock
    price_cols = ["open", "high", "low", "close"]
    ff_limit = cleaning_cfg.get("forward_fill_limit", 5)
    df.sort_values(["symbol", "date"], inplace=True)
    df[price_cols] = df.groupby("symbol")[price_cols].transform(
        lambda s: s.ffill(limit=ff_limit)
    )

    # 4. Drop rows still missing critical values
    before = len(df)
    df.dropna(subset=price_cols, inplace=True)
    if before - len(df):
        logger.warning(f"  Dropped {before - len(df)} rows with unfillable missing prices")

    # 5. Price sanity
    min_p = cleaning_cfg.get("min_price", 0.01)
    max_p = cleaning_cfg.get("max_price", 500_000)
    before = len(df)
    for col in price_cols:
        df = df[(df[col] >= min_p) & (df[col] <= max_p)]
    if before - len(df):
        logger.warning(f"  Dropped {before - len(df)} rows with prices outside [{min_p}, {max_p}]")

    # 6. Volume sanity
    max_vol = cleaning_cfg.get("max_volume", 5_000_000_000)
    df["volume"] = df["volume"].fillna(0).clip(lower=0)
    before = len(df)
    df = df[df["volume"] <= max_vol]
    if before - len(df):
        logger.warning(f"  Dropped {before - len(df)} rows with volume > {max_vol:,}")

    # 7. Outlier flagging (z-score on simple return)
    z_threshold = cleaning_cfg.get("outlier_zscore", 3.5)
    df["_daily_return"] = df.groupby("symbol")["close"].pct_change()
    mean_ret = df["_daily_return"].mean()
    std_ret = df["_daily_return"].std()

    if std_ret and std_ret > 0:
        df["outlier_flag"] = (
            np.abs((df["_daily_return"] - mean_ret) / std_ret) > z_threshold
        )
    else:
        df["outlier_flag"] = False

    n_outliers = df["outlier_flag"].sum()
    if n_outliers:
        logger.info(f"  Flagged {n_outliers} outlier rows (z > {z_threshold})")

    df.drop(columns=["_daily_return"], inplace=True)

    # Final sort and reset
    df.sort_values(["date", "symbol"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    logger.info(
        f"Cleaning complete: {initial_rows:,} → {len(df):,} rows "
        f"({initial_rows - len(df)} removed)"
    )
    return df


def clean_news_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean news data — trim whitespace, drop missing titles, deduplicate.

    Parameters
    ----------
    df : pd.DataFrame
        Raw news data.

    Returns
    -------
    pd.DataFrame
        Cleaned news data.
    """
    df = df.copy()
    initial = len(df)

    # Drop rows without a title
    df.dropna(subset=["title"], inplace=True)

    # Trim whitespace
    str_cols = df.select_dtypes(include=["object", "string"]).columns
    for col in str_cols:
        df[col] = df[col].str.strip()

    # Drop empty titles after stripping
    df = df[df["title"].str.len() > 0]

    # Deduplicate
    df.drop_duplicates(subset=["date", "title"], inplace=True)

    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)

    logger.info(f"News cleaning: {initial:,} → {len(df):,} rows")
    return df
