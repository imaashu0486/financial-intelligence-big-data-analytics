"""
Market data loader — loads and validates OHLCV CSV files.

Usage:
    from src.ingestion.market_loader import load_market_data
    df = load_market_data("data/sample/market_data.csv")
"""

from pathlib import Path
from typing import Union

import pandas as pd

from src.utils.logger import logger

# Required columns and their expected dtypes after parsing
_REQUIRED_COLS = {"date", "symbol", "open", "high", "low", "close", "volume"}


class MarketDataError(Exception):
    """Raised when market data fails validation."""


def load_market_data(filepath: Union[str, Path]) -> pd.DataFrame:
    """
    Load market OHLCV data from a CSV file.

    Steps:
        1. Read CSV and normalise column names to lowercase.
        2. Validate required columns exist.
        3. Parse dates, coerce numeric columns.
        4. Drop rows with invalid dates or negative/zero prices.
        5. Sort by (date, symbol).

    Parameters
    ----------
    filepath : str | Path
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Validated market data.

    Raises
    ------
    MarketDataError
        If the file is missing or required columns are absent.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise MarketDataError(f"File not found: {filepath}")

    logger.info(f"Loading market data from {filepath}")
    df = pd.read_csv(filepath)

    # Normalise column names
    df.columns = df.columns.str.strip().str.lower()

    # Check required columns
    missing = _REQUIRED_COLS - set(df.columns)
    if missing:
        raise MarketDataError(
            f"Missing required columns: {sorted(missing)}. "
            f"Found: {sorted(df.columns.tolist())}"
        )

    # Parse dates
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    bad_dates = df["date"].isna().sum()
    if bad_dates:
        logger.warning(f"Dropping {bad_dates} rows with unparseable dates")
        df.dropna(subset=["date"], inplace=True)

    # Coerce numeric columns
    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows where price columns are NaN or non-positive
    price_cols = ["open", "high", "low", "close"]
    before = len(df)
    df.dropna(subset=price_cols, inplace=True)
    for col in price_cols:
        df = df[df[col] > 0]
    after = len(df)
    if before - after:
        logger.warning(f"Dropped {before - after} rows with invalid/negative prices")

    # Volume: fill NaN with 0, clamp negatives
    df["volume"] = df["volume"].fillna(0).clip(lower=0).astype(int)

    # Symbol to uppercase string
    df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()

    # Sort
    df.sort_values(["date", "symbol"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    logger.info(
        f"Loaded {len(df):,} market rows | "
        f"{df['symbol'].nunique()} symbols | "
        f"{df['date'].min():%Y-%m-%d} → {df['date'].max():%Y-%m-%d}"
    )
    return df
