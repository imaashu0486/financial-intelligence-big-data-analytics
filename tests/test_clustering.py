"""Tests for src.ml.clustering pipeline."""

import numpy as np
import pandas as pd
import pytest
from src.ml.clustering import (
    CLUSTER_FEATURE_NAMES,
    calculate_max_drawdown,
    extract_stock_cluster_features,
    find_optimal_clusters,
    fit_stock_clusters,
)


class TestMaxDrawdown:
    """Unit tests for max drawdown calculation."""

    def test_strictly_increasing_prices(self):
        prices = pd.Series([100.0, 105.0, 110.0, 120.0])
        assert calculate_max_drawdown(prices) == 0.0

    def test_known_drawdown(self):
        # 100 -> 120 -> 60 (drop of 50% from peak of 120) -> 80
        prices = pd.Series([100.0, 120.0, 60.0, 80.0])
        dd = calculate_max_drawdown(prices)
        assert np.isclose(dd, -0.50, atol=1e-4)

    def test_empty_prices(self):
        assert calculate_max_drawdown(pd.Series([], dtype=float)) == 0.0


class TestClusteringPipeline:
    """Unit tests for feature extraction and KMeans clustering."""

    @pytest.fixture
    def multi_stock_history(self) -> pd.DataFrame:
        """Create 4 distinct synthetic stocks."""
        dates = pd.bdate_range("2023-01-01", periods=30)
        records = []
        symbols_config = {
            "STOCK_A": {"ret": 0.005, "vol": 0.01, "volm": 1_000_000},
            "STOCK_B": {"ret": 0.000, "vol": 0.03, "volm": 500_000},
            "STOCK_C": {"ret": -0.003, "vol": 0.02, "volm": 2_000_000},
            "STOCK_D": {"ret": 0.002, "vol": 0.015, "volm": 800_000},
        }

        for sym, cfg in symbols_config.items():
            price = 100.0
            for date in dates:
                ret = cfg["ret"]
                price *= (1.0 + ret)
                records.append({
                    "date": date,
                    "symbol": sym,
                    "close": price,
                    "daily_return": ret,
                    "volatility": cfg["vol"],
                    "volume": cfg["volm"],
                })

        return pd.DataFrame(records)

    def test_extract_stock_cluster_features(self, multi_stock_history):
        feats = extract_stock_cluster_features(multi_stock_history)

        assert len(feats) == 4
        assert "symbol" in feats.columns
        for col in CLUSTER_FEATURE_NAMES:
            assert col in feats.columns
            assert not feats[col].isna().any()

    def test_find_optimal_clusters(self, multi_stock_history):
        feats = extract_stock_cluster_features(multi_stock_history)
        result = find_optimal_clusters(feats, min_k=2, max_k=3, random_state=42)

        assert result["k_values"] == [2, 3]
        assert len(result["inertias"]) == 2
        # Inertia decreases or stays equal with higher k
        assert result["inertias"][1] <= result["inertias"][0]

    def test_fit_stock_clusters(self, multi_stock_history):
        feats = extract_stock_cluster_features(multi_stock_history)
        kmeans, scaler, clustered_df, profiles = fit_stock_clusters(
            feats,
            n_clusters=2,
            random_state=42,
        )

        assert "cluster" in clustered_df.columns
        assert set(clustered_df["cluster"].unique()) == {0, 1}
        assert len(profiles) == 2
        assert "characterization" in profiles.columns
        assert "stock_count" in profiles.columns
