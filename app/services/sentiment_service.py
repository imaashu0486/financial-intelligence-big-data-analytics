"""
Sentiment Intelligence service layer (Phase 5).
Loads financial news sentiment classifications, daily aggregations,
sentiment regimes, and statistical correlation results with market returns.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent


class SentimentService:
    """Service facade for Financial News NLP and Sentiment Analytics."""

    def __init__(self, data_dir: Optional[Path] = None) -> None:
        self.root = data_dir or ROOT
        self.sentiment_dir = self.root / "data" / "processed" / "sentiment"
        self._analytics_cache: Optional[Dict[str, Any]] = None

    def get_sentiment_analytics(self) -> Dict[str, Any]:
        """
        Load summary statistics, correlations, regime returns, and stock breakdowns.
        """
        if self._analytics_cache is not None:
            return self._analytics_cache

        path = self.sentiment_dir / "sentiment_analytics.json"
        if not path.exists():
            return {}

        with open(path, "r", encoding="utf-8") as f:
            self._analytics_cache = json.load(f)
        return self._analytics_cache

    def get_correlations(self) -> Dict[str, Any]:
        """Return Pearson and Spearman correlation statistics."""
        return self.get_sentiment_analytics().get("correlations", {})

    def get_regime_statistics(self) -> Dict[str, Any]:
        """Return return statistics partitioned by sentiment regime (Positive/Neutral/Negative)."""
        return self.get_sentiment_analytics().get("regime_statistics", {})

    def get_sentiment_distribution(self) -> Dict[str, int]:
        """Return global count of Positive, Neutral, and Negative news articles."""
        return self.get_sentiment_analytics().get("sentiment_distribution", {})

    def get_article_sentiments(self, symbol: Optional[str] = None) -> pd.DataFrame:
        """
        Load article-level sentiment classifications, confidence scores, and headlines.
        """
        path = self.sentiment_dir / "article_sentiment.csv"
        if not path.exists():
            return pd.DataFrame()

        df = pd.read_csv(path)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])

        if symbol and "symbol" in df.columns:
            df = df[df["symbol"].str.upper() == symbol.upper()]

        return df.sort_values("date", ascending=False).reset_index(drop=True)

    def get_daily_sentiments(self) -> pd.DataFrame:
        """
        Load aggregated daily sentiment scores across all equities.
        """
        path = self.sentiment_dir / "daily_sentiment.csv"
        if not path.exists():
            return pd.DataFrame()

        df = pd.read_csv(path)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").reset_index(drop=True)
        return df

    def get_stock_sentiment_breakdown(self) -> pd.DataFrame:
        """
        Load sentiment distribution breakdown per ticker.
        """
        path = self.sentiment_dir / "stock_sentiment_breakdown.csv"
        if not path.exists():
            return pd.DataFrame()
        return pd.read_csv(path)

    def get_sentiment_market_joined(self) -> pd.DataFrame:
        """
        Load time-aligned sentiment and market returns for scatter & regression analysis.
        """
        path = self.sentiment_dir / "sentiment_market_joined.csv"
        if not path.exists():
            return pd.DataFrame()

        df = pd.read_csv(path)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").reset_index(drop=True)
        return df

