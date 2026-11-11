"""
Document schemas and transformation utilities for MongoDB collections.

Provides type coercion, NaN sanitization, and flexible schema builders:
- market_data: Time-series technical analysis documents
- news: Flexible unstructured financial news documents
- stock_features: Aggregated stock-level analytical metrics
"""

import math
from datetime import date, datetime
from typing import Any, Dict, Optional


def _sanitize_value(val: Any) -> Any:
    """
    Convert pandas / numpy / Python values to MongoDB-compatible types.
    Converts float NaN/Inf to None, timestamps to ISO strings, numpy types to int/float.
    """
    if val is None:
        return None

    # Handle float NaN or infinite
    if isinstance(val, float):
        if math.isnan(val) or math.isinf(val):
            return None
        return float(val)

    # Handle datetime / date
    if isinstance(val, (datetime, date)):
        return val.strftime("%Y-%m-%d")

    # Handle boolean strings
    if isinstance(val, str):
        val_strip = val.strip()
        if val_strip.lower() == "nan":
            return None
        return val_strip

    # Handle numpy or 64-bit ints
    if hasattr(val, "item"):
        converted = val.item()
        if isinstance(converted, float) and (math.isnan(converted) or math.isinf(converted)):
            return None
        return converted

    return val


def market_record_to_doc(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a market feature record into a standardized market_data document.

    Document Schema:
    ----------------
    {
        "symbol": "TCS",
        "date": "2023-01-02",
        "open": 3500.0,
        "high": 3550.0,
        "low": 3480.0,
        "close": 3530.0,
        "volume": 1000000,
        "outlier_flag": False,
        "daily_return": 0.01,
        "log_return": 0.00995,
        "ma_7": 3530.0,
        "ma_20": 3530.0,
        "ma_50": 3530.0,
        "volatility": 0.005,
        "volume_change": 0.0,
        "rsi_14": 60.0,
        "macd": 1.0,
        "macd_signal": 0.5
    }
    """
    raw_date = row.get("date")
    if isinstance(raw_date, (datetime, date)):
        date_str = raw_date.strftime("%Y-%m-%d")
    else:
        date_str = str(raw_date).split("T")[0].split(" ")[0].strip()

    symbol = str(row.get("symbol", "")).upper().strip()

    # Outlier flag normalization
    outlier_val = row.get("outlier_flag")
    if isinstance(outlier_val, str):
        outlier_flag = outlier_val.strip().lower() in ("true", "1", "yes")
    elif isinstance(outlier_val, (bool, int)):
        outlier_flag = bool(outlier_val)
    else:
        outlier_flag = False

    doc: Dict[str, Any] = {
        "symbol": symbol,
        "date": date_str,
        "open": _sanitize_value(row.get("open")),
        "high": _sanitize_value(row.get("high")),
        "low": _sanitize_value(row.get("low")),
        "close": _sanitize_value(row.get("close")),
        "volume": int(row.get("volume", 0)) if row.get("volume") is not None and not math.isnan(float(row.get("volume", 0))) else 0,
        "outlier_flag": outlier_flag,
        "daily_return": _sanitize_value(row.get("daily_return")),
        "log_return": _sanitize_value(row.get("log_return")),
        "ma_7": _sanitize_value(row.get("ma_7")),
        "ma_20": _sanitize_value(row.get("ma_20")),
        "ma_50": _sanitize_value(row.get("ma_50")),
        "volatility": _sanitize_value(row.get("volatility")),
        "volume_change": _sanitize_value(row.get("volume_change")),
        "rsi_14": _sanitize_value(row.get("rsi_14")),
        "macd": _sanitize_value(row.get("macd")),
        "macd_signal": _sanitize_value(row.get("macd_signal")),
    }
    return doc


def news_record_to_doc(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a news record into a flexible MongoDB document.

    Preserves standard fields (date, symbol, company, title, description, source)
    and dynamically includes any extra metadata without strict schema restrictions.
    """
    raw_date = row.get("date")
    if isinstance(raw_date, (datetime, date)):
        date_str = raw_date.strftime("%Y-%m-%d")
    elif raw_date is not None:
        date_str = str(raw_date).split("T")[0].split(" ")[0].strip()
    else:
        date_str = None

    symbol = str(row.get("symbol", "")).upper().strip() if row.get("symbol") else None

    doc: Dict[str, Any] = {
        "date": date_str,
        "symbol": symbol,
        "company": str(row.get("company", "")).strip() if row.get("company") else None,
        "title": str(row.get("title", "")).strip() if row.get("title") else "",
        "description": str(row.get("description", "")).strip() if row.get("description") else "",
        "source": str(row.get("source", "unknown")).strip() if row.get("source") else "unknown",
    }

    # Dynamically capture additional custom metadata columns (Schema Flexibility)
    standard_keys = {"date", "symbol", "company", "title", "description", "source"}
    for k, v in row.items():
        if k not in standard_keys:
            cleaned = _sanitize_value(v)
            if cleaned is not None:
                doc[k] = cleaned

    return doc


def stock_feature_record_to_doc(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert an aggregated Spark output record into a stock_features document.

    Document Schema:
    ----------------
    {
        "symbol": "TCS",
        "avg_daily_return": 0.00045,
        "avg_log_return": 0.00041,
        "avg_volatility": 0.0126,
        "avg_volume": 3200000.0,
        "total_volume": 1650000000,
        "max_price": 3800.0,
        "min_price": 3100.0,
        "trading_days": 519,
        "volatility_rank": 8 (optional)
    }
    """
    symbol = str(row.get("symbol", "")).upper().strip()

    doc: Dict[str, Any] = {
        "symbol": symbol,
        "avg_daily_return": _sanitize_value(row.get("avg_daily_return")),
        "avg_log_return": _sanitize_value(row.get("avg_log_return")),
        "avg_volatility": _sanitize_value(row.get("avg_volatility")),
        "avg_volume": _sanitize_value(row.get("avg_volume")),
        "total_volume": int(row.get("total_volume", 0)) if row.get("total_volume") is not None else 0,
        "max_price": _sanitize_value(row.get("max_price")),
        "min_price": _sanitize_value(row.get("min_price")),
        "trading_days": int(row.get("trading_days", 0)) if row.get("trading_days") is not None else 0,
    }

    if "volatility_rank" in row and row["volatility_rank"] is not None:
        doc["volatility_rank"] = int(row["volatility_rank"])

    return doc
