"""
News data loader — loads and validates financial news CSV files.

Usage:
    from src.ingestion.news_loader import load_news_data
    df = load_news_data("data/sample/financial_news.csv")
"""

from pathlib import Path
from typing import Union

import pandas as pd

from src.utils.logger import logger

_REQUIRED_COLS = {"date", "title", "source", "symbol"}


class NewsDataError(Exception):
    """Raised when news data fails validation."""


def load_news_data(filepath: Union[str, Path]) -> pd.DataFrame:
    """
    Load financial news from a CSV file.

    Steps:
        1. Read CSV and normalise column names.
        2. Validate required columns.
        3. Parse dates; drop unparseable.
        4. Strip whitespace from text fields.
        5. Deduplicate by (date, title).
        6. Sort by date.

    Parameters
    ----------
    filepath : str | Path
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Validated news data.

    Raises
    ------
    NewsDataError
        If the file is missing or required columns are absent.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise NewsDataError(f"File not found: {filepath}")

    logger.info(f"Loading news data from {filepath}")
    df = pd.read_csv(filepath)

    df.columns = df.columns.str.strip().str.lower()

    missing = _REQUIRED_COLS - set(df.columns)
    if missing:
        raise NewsDataError(
            f"Missing required columns: {sorted(missing)}. "
            f"Found: {sorted(df.columns.tolist())}"
        )

    # Parse dates
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    bad_dates = df["date"].isna().sum()
    if bad_dates:
        logger.warning(f"Dropping {bad_dates} rows with unparseable dates")
        df.dropna(subset=["date"], inplace=True)

    # Clean text fields
    for col in ["title", "source", "symbol"]:
        df[col] = df[col].astype(str).str.strip()
    if "description" in df.columns:
        df["description"] = df["description"].astype(str).str.strip()

    df["symbol"] = df["symbol"].str.upper()

    # Deduplicate
    before = len(df)
    df.drop_duplicates(subset=["date", "title"], inplace=True)
    dupes = before - len(df)
    if dupes:
        logger.info(f"Removed {dupes} duplicate news entries")

    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)

    logger.info(
        f"Loaded {len(df):,} news rows | "
        f"{df['symbol'].nunique()} symbols | "
        f"{df['date'].min():%Y-%m-%d} → {df['date'].max():%Y-%m-%d}"
    )
    return df
