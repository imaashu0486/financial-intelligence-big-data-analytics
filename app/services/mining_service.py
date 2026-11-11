"""
Association Rule Mining service layer (Phase 6).
Loads market basket co-movement transactions, frequent itemsets,
and mined association rules with support, confidence, and lift metrics.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent


class MiningService:
    """Service facade for Apriori association rule mining."""

    def __init__(self, data_dir: Optional[Path] = None) -> None:
        self.root = data_dir or ROOT
        self.mining_dir = self.root / "data" / "processed" / "association"
        self._stats_cache: Optional[Dict[str, Any]] = None

    def get_association_rules(
        self,
        min_support: Optional[float] = None,
        min_confidence: Optional[float] = None,
        min_lift: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Load association rules with optional dynamic filtering.
        """
        path = self.mining_dir / "association_rules.csv"
        if not path.exists():
            return pd.DataFrame()

        df = pd.read_csv(path)

        if min_support is not None and "support" in df.columns:
            df = df[df["support"] >= min_support]
        if min_confidence is not None and "confidence" in df.columns:
            df = df[df["confidence"] >= min_confidence]
        if min_lift is not None and "lift" in df.columns:
            df = df[df["lift"] >= min_lift]

        return df.sort_values("lift", ascending=False).reset_index(drop=True)

    def get_frequent_itemsets(self, min_support: Optional[float] = None) -> pd.DataFrame:
        """
        Load discovered frequent itemsets.
        """
        path = self.mining_dir / "frequent_itemsets.csv"
        if not path.exists():
            return pd.DataFrame()

        df = pd.read_csv(path)
        if min_support is not None and "support" in df.columns:
            df = df[df["support"] >= min_support]

        return df.sort_values("support", ascending=False).reset_index(drop=True)

    def get_mining_statistics(self) -> Dict[str, Any]:
        """
        Load summary mining statistics.
        """
        if self._stats_cache is not None:
            return self._stats_cache

        path = self.mining_dir / "association_statistics.json"
        if not path.exists():
            return {}

        with open(path, "r", encoding="utf-8") as f:
            self._stats_cache = json.load(f)
        return self._stats_cache

    def get_transaction_summary(self) -> Dict[str, Any]:
        """
        Load transaction summary details.
        """
        path = self.mining_dir / "transaction_summary.json"
        if not path.exists():
            return {}

        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

