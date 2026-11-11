"""
Transaction Construction from Daily Stock Returns.

Converts daily market return data into a transactional format suitable for
frequent itemset mining. Each trading date becomes one transaction containing
the stock symbols whose daily return exceeds a configurable threshold.

Example:
    If RELIANCE returned +1.5% and TCS returned +2.0% on 2023-06-15,
    and the threshold is 1%, then the transaction for that date is:

        {RELIANCE_UP, TCS_UP}
"""

from typing import Any, Dict, List, Optional, Set

import pandas as pd

from src.utils.logger import logger


def build_transactions(
    market_df: pd.DataFrame,
    return_threshold: float = 0.01,
    return_column: str = "daily_return",
    symbol_column: str = "symbol",
    date_column: str = "date",
    suffix: str = "_UP",
) -> List[Set[str]]:
    """
    Convert daily stock returns into a list of transactions.

    Each trading date becomes one transaction containing item labels for
    stocks whose return meets or exceeds the threshold on that date.

    Parameters
    ----------
    market_df : pd.DataFrame
        Market features DataFrame with at least date, symbol, and return columns.
    return_threshold : float, default 0.01
        Minimum daily return for a stock to be included in the transaction.
    return_column : str, default 'daily_return'
        Column name containing the return values.
    symbol_column : str, default 'symbol'
        Column name containing stock ticker symbols.
    date_column : str, default 'date'
        Column name containing dates.
    suffix : str, default '_UP'
        Suffix appended to stock symbol to form item label.

    Returns
    -------
    list of set of str
        Each element is a set of item labels for one trading date.
        Empty transactions (no stock met the threshold) are excluded.
    """
    # Validate required columns
    required = {date_column, symbol_column, return_column}
    missing = required - set(market_df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = market_df.copy()

    # Drop rows with missing/invalid returns
    df = df.dropna(subset=[return_column, date_column, symbol_column])
    df = df[pd.to_numeric(df[return_column], errors="coerce").notna()]
    df[return_column] = df[return_column].astype(float)

    # Filter to stocks meeting the threshold
    strong = df[df[return_column] >= return_threshold]

    if strong.empty:
        logger.warning(
            f"No stocks met the return threshold {return_threshold:.4f}. "
            f"Returning empty transaction list."
        )
        return []

    # Group by date to form transactions
    transactions: List[Set[str]] = []
    dates_seen: List[str] = []

    for date_val, group in strong.groupby(date_column):
        items: Set[str] = set()
        for symbol in group[symbol_column].unique():
            item_label = f"{str(symbol).strip().upper()}{suffix}"
            items.add(item_label)
        if items:  # Only include non-empty transactions
            transactions.append(items)
            dates_seen.append(str(date_val))

    logger.info(
        f"Built {len(transactions)} transactions from {len(df[date_column].unique())} "
        f"trading dates (threshold={return_threshold:.4f}, "
        f"items per txn: min={min(len(t) for t in transactions) if transactions else 0}, "
        f"max={max(len(t) for t in transactions) if transactions else 0}, "
        f"avg={sum(len(t) for t in transactions) / len(transactions):.1f})"
    )

    return transactions


def compute_transaction_summary(
    transactions: List[Set[str]],
    return_threshold: float = 0.01,
    min_support: float = 0.05,
    min_confidence: float = 0.30,
    min_lift: float = 1.0,
) -> Dict[str, Any]:
    """
    Compute descriptive statistics about the transaction dataset.

    Parameters
    ----------
    transactions : list of set of str
        Transaction list from build_transactions().
    return_threshold : float
        Return threshold used to construct transactions.
    min_support, min_confidence, min_lift : float
        Mining parameters for documentation.

    Returns
    -------
    dict
        Summary statistics including counts, sizes, and configuration.
    """
    if not transactions:
        return {
            "num_transactions": 0,
            "num_unique_items": 0,
            "avg_transaction_size": 0.0,
            "min_transaction_size": 0,
            "max_transaction_size": 0,
            "return_threshold": return_threshold,
            "min_support": min_support,
            "min_confidence": min_confidence,
            "min_lift": min_lift,
        }

    all_items: Set[str] = set()
    sizes = []
    for txn in transactions:
        all_items.update(txn)
        sizes.append(len(txn))

    return {
        "num_transactions": len(transactions),
        "num_unique_items": len(all_items),
        "unique_items": sorted(all_items),
        "avg_transaction_size": round(sum(sizes) / len(sizes), 2),
        "min_transaction_size": min(sizes),
        "max_transaction_size": max(sizes),
        "median_transaction_size": sorted(sizes)[len(sizes) // 2],
        "return_threshold": return_threshold,
        "min_support": min_support,
        "min_confidence": min_confidence,
        "min_lift": min_lift,
    }
