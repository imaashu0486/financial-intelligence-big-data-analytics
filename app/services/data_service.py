"""
Data service layer for Market Overview (Phase 1).
Loads and caches market data, engineered features, and summary KPIs.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent

# Company mapping for Indian stock market symbols used in dataset
COMPANY_NAMES = {
    "BAJFINANCE": "Bajaj Finance Ltd.",
    "BHARTIARTL": "Bharti Airtel Ltd.",
    "HDFCBANK": "HDFC Bank Ltd.",
    "ICICIBANK": "ICICI Bank Ltd.",
    "INFY": "Infosys Ltd.",
    "LT": "Larsen & Toubro Ltd.",
    "RELIANCE": "Reliance Industries Ltd.",
    "SBIN": "State Bank of India",
    "TCS": "Tata Consultancy Services",
    "WIPRO": "Wipro Ltd.",
}

SECTORS = {
    "BAJFINANCE": "Financial Services / NBFC",
    "BHARTIARTL": "Telecommunications",
    "HDFCBANK": "Banking (Private)",
    "ICICIBANK": "Banking (Private)",
    "INFY": "Information Technology",
    "LT": "Infrastructure & Engineering",
    "RELIANCE": "Oil, Gas & Conglomerate",
    "SBIN": "Banking (Public)",
    "TCS": "Information Technology",
    "WIPRO": "Information Technology",
}


class DataService:
    """Service facade for market data and technical features."""

    def __init__(self, data_dir: Optional[Path] = None) -> None:
        self.root = data_dir or ROOT
        self.processed_path = self.root / "data" / "processed" / "market_features.csv"
        self.raw_path = self.root / "data" / "sample" / "market_data.csv"
        self.real_parquet_path = self.root / "data" / "real" / "processed" / "market" / "consolidated_market_features.parquet"
        self.real_csv_path = self.root / "data" / "real" / "processed" / "market" / "consolidated_market_features.csv"
        self.constituents_path = self.root / "data" / "real" / "metadata" / "nifty50_constituents.json"
        self._constituents_meta: Optional[Dict[str, Dict[str, str]]] = None

    def _load_real_metadata(self) -> Dict[str, Dict[str, str]]:
        """Load constituent company names and sectors from metadata JSON."""
        if self._constituents_meta is not None:
            return self._constituents_meta

        meta: Dict[str, Dict[str, str]] = {}
        if self.constituents_path.exists():
            try:
                with open(self.constituents_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                constituents = data.get("constituents", []) if isinstance(data, dict) else []
                for item in constituents:
                    sym = item.get("symbol", "")
                    if sym:
                        meta[sym] = {
                            "company_name": item.get("company_name", sym),
                            "sector": item.get("sector", "Equities"),
                        }
            except Exception:
                pass
        self._constituents_meta = meta
        return meta

    def load_market_features(self, scope: Optional[str] = None) -> pd.DataFrame:
        """
        Load market features according to active dataset scope (Real NIFTY 50 or Synthetic Baseline).
        Never silently falls back to synthetic data if real data is requested.
        """
        if scope is None:
            try:
                import streamlit as st
                scope = st.session_state.get("dataset_scope", "Real Market History (NIFTY 50)")
            except Exception:
                scope = "Real Market History (NIFTY 50)"

        is_real = "real" in str(scope).lower()

        if is_real:
            target = self.real_parquet_path if self.real_parquet_path.exists() else (
                self.real_csv_path if self.real_csv_path.exists() else None
            )
            if target is None:
                raise FileNotFoundError(
                    f"Real market dataset not found at {self.real_parquet_path}. "
                    "Real dataset scope must not silently fall back to synthetic baseline."
                )

            if target.suffix == ".parquet":
                df = pd.read_parquet(target)
            else:
                df = pd.read_csv(target)

            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                df = df.sort_values(by=["date", "symbol"]).reset_index(drop=True)

            # Ensure standard column aliases expected across frontend
            if "daily_return" not in df.columns and "daily_adjusted_return" in df.columns:
                df["daily_return"] = df["daily_adjusted_return"]
            if "volatility" not in df.columns and "volatility_20d" in df.columns:
                df["volatility"] = df["volatility_20d"]

            # Enrich with real metadata
            meta = self._load_real_metadata()
            if meta and "symbol" in df.columns:
                df["company_name"] = df["symbol"].map(lambda s: meta.get(s, {}).get("company_name", s))
                df["sector"] = df["symbol"].map(lambda s: meta.get(s, {}).get("sector", "Equities"))
            elif "symbol" in df.columns:
                df["company_name"] = df["symbol"].map(COMPANY_NAMES).fillna(df["symbol"])
                df["sector"] = df["symbol"].map(SECTORS).fillna("Equities")

            return df
        else:
            # Synthetic baseline mode
            target = self.processed_path if self.processed_path.exists() else self.raw_path
            if not target.exists():
                return pd.DataFrame()

            df = pd.read_csv(target)
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                df = df.sort_values(by=["date", "symbol"]).reset_index(drop=True)

            if "symbol" in df.columns:
                df["company_name"] = df["symbol"].map(COMPANY_NAMES).fillna(df["symbol"])
                df["sector"] = df["symbol"].map(SECTORS).fillna("Unknown")

            return df

    def get_market_kpis(self, df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Compute high-level market KPI statistics.
        """
        if df is None:
            df = self.load_market_features()

        if df.empty:
            return {
                "total_records": 0,
                "symbols_count": 0,
                "symbols": [],
                "min_date": None,
                "max_date": None,
                "trading_days": 0,
                "avg_daily_volume": 0.0,
                "mean_daily_return": 0.0,
            }

        unique_symbols = sorted(df["symbol"].unique().tolist()) if "symbol" in df.columns else []
        dates = df["date"] if "date" in df.columns else pd.Series(dtype=object)

        kpis = {
            "total_records": len(df),
            "symbols_count": len(unique_symbols),
            "symbols": unique_symbols,
            "min_date": dates.min().strftime("%Y-%m-%d") if not dates.empty else "N/A",
            "max_date": dates.max().strftime("%Y-%m-%d") if not dates.empty else "N/A",
            "trading_days": int(df["date"].nunique()) if not dates.empty else 0,
            "avg_daily_volume": float(df["volume"].mean()) if "volume" in df.columns else 0.0,
            "mean_daily_return": float(df["daily_return"].mean()) if "daily_return" in df.columns else 0.0,
        }
        return kpis

    def get_ticker_data(
        self,
        symbol: str,
        df: Optional[pd.DataFrame] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Filter dataset for a single stock symbol and optional date bounds,
        ensuring technical indicators (RSI-14, MACD) are populated.
        """
        if df is None:
            df = self.load_market_features()

        if df.empty or "symbol" not in df.columns:
            return pd.DataFrame()

        filtered = df[df["symbol"].str.upper() == symbol.upper()].copy()

        if start_date and "date" in filtered.columns:
            filtered = filtered[filtered["date"] >= pd.to_datetime(start_date)]
        if end_date and "date" in filtered.columns:
            filtered = filtered[filtered["date"] <= pd.to_datetime(end_date)]

        filtered = filtered.sort_values(by="date").reset_index(drop=True)

        if not filtered.empty and "close" in filtered.columns:
            if "rsi_14" not in filtered.columns:
                delta = filtered["close"].diff()
                gain = delta.clip(lower=0)
                loss = -delta.clip(upper=0)
                avg_gain = gain.ewm(com=13, adjust=False).mean()
                avg_loss = loss.ewm(com=13, adjust=False).mean()
                rs = avg_gain / avg_loss.replace(0, np.nan)
                filtered["rsi_14"] = (100 - (100 / (1 + rs))).fillna(50.0)

            if "macd" not in filtered.columns or "macd_signal" not in filtered.columns:
                ema_12 = filtered["close"].ewm(span=12, adjust=False).mean()
                ema_26 = filtered["close"].ewm(span=26, adjust=False).mean()
                filtered["macd"] = ema_12 - ema_26
                filtered["macd_signal"] = filtered["macd"].ewm(span=9, adjust=False).mean()

        return filtered

    def get_latest_snapshot(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Get the latest trading record for each symbol.
        """
        if df is None:
            df = self.load_market_features()

        if df.empty or "symbol" not in df.columns or "date" not in df.columns:
            return pd.DataFrame()

        latest_records = (
            df.sort_values("date")
            .groupby("symbol")
            .last()
            .reset_index()
        )

        if "rsi_14" not in latest_records.columns and "close" in df.columns:
            rsi_map = {}
            for sym, grp in df.groupby("symbol"):
                prices = grp.sort_values("date")["close"]
                if len(prices) >= 15:
                    delta = prices.diff()
                    gain = delta.clip(lower=0)
                    loss = -delta.clip(upper=0)
                    avg_gain = gain.ewm(com=13, adjust=False).mean()
                    avg_loss = loss.ewm(com=13, adjust=False).mean()
                    rs = avg_gain / avg_loss.replace(0, np.nan)
                    rsi = 100 - (100 / (1 + rs))
                    rsi_map[sym] = round(float(rsi.iloc[-1]), 2)
            if rsi_map:
                latest_records["rsi_14"] = latest_records["symbol"].map(rsi_map)

        return latest_records

