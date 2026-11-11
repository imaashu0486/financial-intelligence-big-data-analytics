"""
Real Market Data Collector — Historical OHLCV Ingestion via Yahoo Finance.

Acquires real-world daily stock market data for the NIFTY 50 constituents
and benchmark index (^NSEI) from Yahoo Finance using yfinance.

Features:
    - 10-year historical daily OHLCV acquisition (2015-01-01 to 2025-01-01).
    - Unadjusted bar preservation (auto_adjust=False) with Adj Close retained.
    - Pacing, exponential backoff retries, and per-ticker error isolation.
    - Schema validation & normalization strictly compatible with Phase 1.
    - Deterministic SHA-256 fingerprinting and manifest generation.

Usage:
    python -m src.ingestion.real_market_collector --start 2015-01-01 --end 2025-01-01
"""

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

from src.utils.logger import logger

# Path defaults
DEFAULT_CONSTITUENTS_PATH = Path("data/real/metadata/nifty50_constituents.json")
DEFAULT_OUTPUT_DIR = Path("data/real/raw/market")
DEFAULT_MANIFEST_PATH = Path("data/real/metadata/market_download_manifest.json")

REQUIRED_MARKET_COLUMNS = ["date", "symbol", "open", "high", "low", "close", "adj_close", "volume"]


def compute_sha256(filepath: Union[str, Path]) -> str:
    """Compute SHA-256 cryptographic hash of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            sha256.update(chunk)
    return sha256.hexdigest()


def load_constituents(filepath: Union[str, Path] = DEFAULT_CONSTITUENTS_PATH) -> Dict[str, Any]:
    """Load NIFTY 50 constituent master metadata."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Constituents file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_yfinance_dataframe(raw_df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Standardize raw yfinance output DataFrame to the platform's market schema.

    Parameters
    ----------
    raw_df : pd.DataFrame
        DataFrame directly returned by yfinance.
    symbol : str
        Clean constituent ticker (e.g. 'RELIANCE').

    Returns
    -------
    pd.DataFrame
        Standardized DataFrame with columns [date, symbol, open, high, low, close, adj_close, volume].
    """
    if raw_df is None or raw_df.empty:
        return pd.DataFrame(columns=REQUIRED_MARKET_COLUMNS)

    df = raw_df.copy()

    # Handle MultiIndex columns that yfinance can return for multiple tickers or single ticker
    if isinstance(df.columns, pd.MultiIndex):
        # Flatten MultiIndex e.g. ('Close', 'RELIANCE.NS') -> 'Close'
        df.columns = [col[0] for col in df.columns]

    # Normalize column names to lowercase
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

    # Reset index to extract Date
    if "date" not in df.columns:
        df.reset_index(inplace=True)
        # Check newly created column name
        first_col = df.columns[0].lower()
        if first_col in ["date", "datetime", "index"]:
            df.rename(columns={df.columns[0]: "date"}, inplace=True)

    # Standardize column mapping
    col_mapping = {
        "adj_close": "adj_close",
        "adj close": "adj_close",
        "close": "close",
        "open": "open",
        "high": "high",
        "low": "low",
        "volume": "volume",
        "date": "date",
    }
    df.rename(columns=col_mapping, inplace=True)

    # Format Date to YYYY-MM-DD string
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

    # Add symbol
    df["symbol"] = symbol.upper()

    # If adj_close is missing, fallback to close
    if "adj_close" not in df.columns and "close" in df.columns:
        df["adj_close"] = df["close"]

    # Ensure required columns
    for col in ["open", "high", "low", "close", "adj_close"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").round(4)

    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype("int64")

    # Filter and reorder
    available_cols = [c for c in REQUIRED_MARKET_COLUMNS if c in df.columns]
    df = df[available_cols]

    # Sort strictly chronologically
    df.sort_values(by="date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def download_ticker_data(
    symbol: str,
    yfinance_symbol: str,
    start_date: str = "2015-01-01",
    end_date: str = "2025-01-01",
    output_dir: Union[str, Path] = DEFAULT_OUTPUT_DIR,
    max_retries: int = 3,
    retry_delay_seconds: float = 2.0,
) -> Dict[str, Any]:
    """
    Download historical daily OHLCV for a single ticker with retry and error isolation.

    Returns manifest record dict.
    """
    import yfinance as yf

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"{symbol}.csv"

    manifest_record: Dict[str, Any] = {
        "symbol": symbol,
        "yfinance_symbol": yfinance_symbol,
        "start_requested": start_date,
        "end_requested": end_date,
        "target_file": str(out_file),
        "status": "FAILED",
        "records_count": 0,
        "start_actual": None,
        "end_actual": None,
        "sha256": None,
        "file_size_bytes": 0,
        "error_message": None,
        "attempts": 0,
    }

    for attempt in range(1, max_retries + 1):
        manifest_record["attempts"] = attempt
        try:
            logger.info(
                f"[{symbol}] Fetching historical data ({yfinance_symbol}) "
                f"from {start_date} to {end_date} (Attempt {attempt}/{max_retries})..."
            )
            # Call yfinance download with auto_adjust=False
            raw_data = yf.download(
                tickers=yfinance_symbol,
                start=start_date,
                end=end_date,
                auto_adjust=False,
                progress=False,
                timeout=30,
            )

            if raw_data is None or raw_data.empty:
                logger.warning(f"[{symbol}] Empty response received from yfinance.")
                if attempt < max_retries:
                    time.sleep(retry_delay_seconds * (2 ** (attempt - 1)))
                    continue
                else:
                    manifest_record["error_message"] = "Empty dataset returned by Yahoo Finance"
                    return manifest_record

            cleaned_df = normalize_yfinance_dataframe(raw_data, symbol)
            if cleaned_df.empty:
                manifest_record["error_message"] = "Normalization resulted in empty dataframe"
                return manifest_record

            # Save to CSV
            cleaned_df.to_csv(out_file, index=False)

            # Update manifest record
            manifest_record["status"] = "SUCCESS"
            manifest_record["records_count"] = len(cleaned_df)
            manifest_record["start_actual"] = str(cleaned_df["date"].min())
            manifest_record["end_actual"] = str(cleaned_df["date"].max())
            manifest_record["file_size_bytes"] = out_file.stat().st_size
            manifest_record["sha256"] = compute_sha256(out_file)
            manifest_record["error_message"] = None

            logger.info(
                f"[{symbol}] Successfully saved {len(cleaned_df)} bars "
                f"({manifest_record['start_actual']} to {manifest_record['end_actual']}) to {out_file}"
            )
            return manifest_record

        except Exception as exc:
            logger.error(f"[{symbol}] Attempt {attempt} failed: {exc}")
            manifest_record["error_message"] = str(exc)
            if attempt < max_retries:
                time.sleep(retry_delay_seconds * (2 ** (attempt - 1)))

    return manifest_record


def download_all_nifty50(
    constituents_path: Union[str, Path] = DEFAULT_CONSTITUENTS_PATH,
    output_dir: Union[str, Path] = DEFAULT_OUTPUT_DIR,
    manifest_path: Union[str, Path] = DEFAULT_MANIFEST_PATH,
    start_date: str = "2015-01-01",
    end_date: str = "2025-01-01",
    inter_ticker_delay: float = 1.0,
    include_benchmark: bool = True,
    selected_tickers: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Download historical market data for all NIFTY 50 constituents and benchmark index.

    Generates and persists a comprehensive download manifest.
    """
    constituents_data = load_constituents(constituents_path)
    constituents = constituents_data.get("constituents", [])
    benchmark_symbol = constituents_data.get("benchmark_symbol", "^NSEI")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    # Filter if selected_tickers provided
    if selected_tickers:
        selected_set = {s.upper() for s in selected_tickers}
        items_to_download = [c for c in constituents if c["symbol"].upper() in selected_set]
    else:
        items_to_download = list(constituents)

    records: List[Dict[str, Any]] = []
    total_items = len(items_to_download) + (1 if include_benchmark and not selected_tickers else 0)
    logger.info(f"Starting acquisition for {total_items} market instruments ({start_date} to {end_date})...")

    # Download benchmark if requested
    if include_benchmark and not selected_tickers:
        benchmark_record = download_ticker_data(
            symbol="NIFTY50_INDEX",
            yfinance_symbol=benchmark_symbol,
            start_date=start_date,
            end_date=end_date,
            output_dir=output_dir,
        )
        records.append(benchmark_record)
        time.sleep(inter_ticker_delay)

    # Download constituents
    for idx, item in enumerate(items_to_download, start=1):
        sym = item["symbol"]
        yf_sym = item["yfinance_symbol"]
        logger.info(f"[{idx}/{len(items_to_download)}] Downloading {sym} ({yf_sym})...")

        rec = download_ticker_data(
            symbol=sym,
            yfinance_symbol=yf_sym,
            start_date=start_date,
            end_date=end_date,
            output_dir=output_dir,
        )
        records.append(rec)

        if idx < len(items_to_download):
            time.sleep(inter_ticker_delay)

    # Compile manifest
    successful = [r for r in records if r["status"] == "SUCCESS"]
    failed = [r for r in records if r["status"] == "FAILED"]
    total_records = sum(r["records_count"] for r in successful)
    total_bytes = sum(r["file_size_bytes"] for r in successful)

    manifest_data = {
        "acquisition_timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "index_name": constituents_data.get("index_name", "NIFTY 50"),
        "date_range_requested": {"start": start_date, "end": end_date},
        "summary": {
            "total_requested": len(records),
            "successful": len(successful),
            "failed": len(failed),
            "total_market_bars": total_records,
            "total_bytes": total_bytes,
        },
        "failed_symbols": [r["symbol"] for r in failed],
        "ticker_manifest": records,
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    logger.info(
        f"Acquisition complete: {len(successful)}/{len(records)} succeeded, "
        f"{total_records:,} total daily bars saved. Manifest written to {manifest_path}"
    )
    return manifest_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Acquire Real Historical NIFTY 50 Market Data")
    parser.add_argument("--start", default="2015-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2025-01-01", help="End date (YYYY-MM-DD)")
    parser.add_argument("--delay", type=float, default=0.5, help="Inter-ticker delay in seconds")
    parser.add_argument("--tickers", nargs="*", default=None, help="Optional subset of symbols")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory")
    args = parser.parse_args()

    download_all_nifty50(
        output_dir=args.output_dir,
        start_date=args.start,
        end_date=args.end,
        inter_ticker_delay=args.delay,
        selected_tickers=args.tickers,
    )
