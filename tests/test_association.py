"""Tests for Phase 6 — Frequent Itemset Mining & Association Rule Mining."""

import json
import math
from pathlib import Path

import pandas as pd
import pytest

from src.mining.transactions import build_transactions, compute_transaction_summary
from src.mining.apriori import (
    apriori,
    count_support,
    generate_candidates,
    get_frequent_1_itemsets,
)
from src.mining.rules import generate_association_rules
from src.mining.analytics import (
    compute_association_statistics,
    compute_item_frequency,
    top_itemsets_by_support,
    top_rules_by_lift,
)


# =====================================================================
# Fixtures: deterministic transaction datasets
# =====================================================================

@pytest.fixture
def simple_transactions():
    """
    5 transactions with known support values.

    Item frequencies:
        A: 5/5 = 1.00
        B: 3/5 = 0.60
        C: 3/5 = 0.60
        D: 2/5 = 0.40
        E: 1/5 = 0.20

    Pair frequencies:
        {A,B}: 3/5 = 0.60
        {A,C}: 3/5 = 0.60
        {B,C}: 2/5 = 0.40
        {A,D}: 2/5 = 0.40
        {B,D}: 2/5 = 0.40
        {C,D}: 1/5 = 0.20
        {A,E}: 1/5 = 0.20
    """
    return [
        {"A", "B", "C"},
        {"A", "B", "D"},
        {"A", "B", "C", "D"},
        {"A", "C"},
        {"A", "E"},
    ]


@pytest.fixture
def market_df_for_transactions():
    """Small market DataFrame for transaction construction tests."""
    return pd.DataFrame([
        {"date": "2023-01-02", "symbol": "TCS", "daily_return": 0.015},
        {"date": "2023-01-02", "symbol": "INFY", "daily_return": 0.020},
        {"date": "2023-01-02", "symbol": "SBIN", "daily_return": 0.005},
        {"date": "2023-01-03", "symbol": "TCS", "daily_return": -0.010},
        {"date": "2023-01-03", "symbol": "INFY", "daily_return": 0.012},
        {"date": "2023-01-03", "symbol": "SBIN", "daily_return": 0.018},
        {"date": "2023-01-04", "symbol": "TCS", "daily_return": 0.003},
        {"date": "2023-01-04", "symbol": "INFY", "daily_return": 0.001},
        {"date": "2023-01-04", "symbol": "SBIN", "daily_return": -0.005},
    ])


# =====================================================================
# 1. Transaction Construction Tests
# =====================================================================

class TestTransactionConstruction:
    """Tests for converting market data into transactions."""

    def test_basic_transaction_construction(self, market_df_for_transactions):
        txns = build_transactions(
            market_df_for_transactions, return_threshold=0.01
        )
        # Day 1: TCS (0.015) + INFY (0.020) >= 0.01; SBIN (0.005) < 0.01
        # Day 2: INFY (0.012) + SBIN (0.018) >= 0.01; TCS (-0.01) < 0.01
        # Day 3: nothing >= 0.01
        assert len(txns) == 2
        assert {"TCS_UP", "INFY_UP"} in txns
        assert {"INFY_UP", "SBIN_UP"} in txns

    def test_threshold_zero(self, market_df_for_transactions):
        """All positive returns should be included with threshold 0."""
        txns = build_transactions(
            market_df_for_transactions, return_threshold=0.0
        )
        # Day 1: TCS, INFY, SBIN all positive
        # Day 2: INFY, SBIN positive
        # Day 3: TCS, INFY positive
        assert len(txns) == 3

    def test_high_threshold_empty(self, market_df_for_transactions):
        """Very high threshold should produce no transactions."""
        txns = build_transactions(
            market_df_for_transactions, return_threshold=0.50
        )
        assert len(txns) == 0

    def test_missing_return_values(self):
        """Rows with NaN returns should be skipped."""
        df = pd.DataFrame([
            {"date": "2023-01-02", "symbol": "TCS", "daily_return": 0.02},
            {"date": "2023-01-02", "symbol": "INFY", "daily_return": float("nan")},
        ])
        txns = build_transactions(df, return_threshold=0.01)
        assert len(txns) == 1
        assert txns[0] == {"TCS_UP"}

    def test_missing_columns_raises(self):
        """Missing required columns should raise ValueError."""
        df = pd.DataFrame([{"date": "2023-01-02", "symbol": "TCS"}])
        with pytest.raises(ValueError, match="Missing required columns"):
            build_transactions(df)

    def test_duplicate_symbols_same_date(self):
        """Duplicate symbols on same date should produce unique items."""
        df = pd.DataFrame([
            {"date": "2023-01-02", "symbol": "TCS", "daily_return": 0.02},
            {"date": "2023-01-02", "symbol": "TCS", "daily_return": 0.03},
        ])
        txns = build_transactions(df, return_threshold=0.01)
        assert len(txns) == 1
        assert txns[0] == {"TCS_UP"}

    def test_empty_dataframe(self):
        """Empty DataFrame should return empty list."""
        df = pd.DataFrame(columns=["date", "symbol", "daily_return"])
        txns = build_transactions(df, return_threshold=0.01)
        assert txns == []


class TestTransactionSummary:
    """Tests for transaction summary statistics."""

    def test_summary_basic(self, simple_transactions):
        summary = compute_transaction_summary(simple_transactions)
        assert summary["num_transactions"] == 5
        assert summary["num_unique_items"] == 5
        assert summary["min_transaction_size"] == 2
        assert summary["max_transaction_size"] == 4
        assert summary["avg_transaction_size"] == pytest.approx(2.8, abs=0.1)

    def test_summary_empty(self):
        summary = compute_transaction_summary([])
        assert summary["num_transactions"] == 0
        assert summary["num_unique_items"] == 0


# =====================================================================
# 2. Apriori Algorithm Tests
# =====================================================================

class TestFrequent1Itemsets:
    """Tests for Level 1 of Apriori."""

    def test_frequent_1_itemsets_basic(self, simple_transactions):
        freq_1, all_items = get_frequent_1_itemsets(
            simple_transactions, min_support=0.30
        )
        # A=0.80, B=0.60, C=0.60, D=0.40, E=0.20
        # With min_support=0.30: A, B, C, D are frequent; E is not
        assert len(freq_1) == 4
        assert frozenset(["A"]) in freq_1
        assert frozenset(["B"]) in freq_1
        assert frozenset(["C"]) in freq_1
        assert frozenset(["D"]) in freq_1
        assert frozenset(["E"]) not in freq_1

    def test_support_values(self, simple_transactions):
        freq_1, _ = get_frequent_1_itemsets(
            simple_transactions, min_support=0.0
        )
        assert freq_1[frozenset(["A"])] == pytest.approx(1.00)
        assert freq_1[frozenset(["B"])] == pytest.approx(0.60)
        assert freq_1[frozenset(["E"])] == pytest.approx(0.20)

    def test_empty_transactions(self):
        freq_1, all_items = get_frequent_1_itemsets([], min_support=0.05)
        assert freq_1 == {}
        assert all_items == set()


class TestCandidateGeneration:
    """Tests for Apriori candidate generation and pruning."""

    def test_generate_2_candidates(self):
        """From 3 frequent 1-itemsets, should get C(3,2) = 3 candidates."""
        freq_1 = [frozenset(["A"]), frozenset(["B"]), frozenset(["C"])]
        candidates = generate_candidates(freq_1, k=2)
        assert len(candidates) == 3
        assert frozenset(["A", "B"]) in candidates
        assert frozenset(["A", "C"]) in candidates
        assert frozenset(["B", "C"]) in candidates

    def test_pruning_removes_invalid_3_candidates(self):
        """
        If frequent 2-itemsets are {A,B}, {A,C}, {B,C}, then {A,B,C}
        is a valid 3-candidate (all 2-subsets are frequent).

        If frequent 2-itemsets are {A,B}, {A,C} only (no {B,C}),
        then {A,B,C} should be pruned (subset {B,C} not frequent).
        """
        freq_2_full = [
            frozenset(["A", "B"]),
            frozenset(["A", "C"]),
            frozenset(["B", "C"]),
        ]
        candidates_full = generate_candidates(freq_2_full, k=3)
        assert frozenset(["A", "B", "C"]) in candidates_full

        freq_2_partial = [
            frozenset(["A", "B"]),
            frozenset(["A", "C"]),
        ]
        candidates_partial = generate_candidates(freq_2_partial, k=3)
        assert frozenset(["A", "B", "C"]) not in candidates_partial


class TestSupportCounting:
    """Tests for support counting."""

    def test_count_support(self, simple_transactions):
        candidates = [frozenset(["A", "B"]), frozenset(["A", "E"])]
        support = count_support(candidates, simple_transactions, 5)
        assert support[frozenset(["A", "B"])] == pytest.approx(0.60)
        assert support[frozenset(["A", "E"])] == pytest.approx(0.20)


class TestAprioriFullAlgorithm:
    """Integration tests for the complete Apriori algorithm."""

    def test_apriori_known_dataset(self, simple_transactions):
        """Full Apriori on deterministic dataset with known answers."""
        freq = apriori(simple_transactions, min_support=0.30)

        # 1-itemsets: A(1.0), B(0.6), C(0.6), D(0.4) = 4
        assert frozenset(["A"]) in freq
        assert frozenset(["B"]) in freq
        assert frozenset(["C"]) in freq
        assert frozenset(["D"]) in freq
        assert frozenset(["E"]) not in freq

        # 2-itemsets with support >= 0.30:
        # {A,B}=0.6, {A,C}=0.6, {B,C}=0.4, {A,D}=0.4, {B,D}=0.4
        # {C,D}=0.2 (pruned < 0.30)
        assert frozenset(["A", "B"]) in freq
        assert freq[frozenset(["A", "B"])] == pytest.approx(0.60)
        assert frozenset(["A", "C"]) in freq
        assert frozenset(["B", "C"]) in freq
        assert frozenset(["A", "D"]) in freq
        assert frozenset(["B", "D"]) in freq
        assert frozenset(["C", "D"]) not in freq  # 0.20 < 0.30

    def test_apriori_high_support_no_results(self, simple_transactions):
        """Very high min_support should return empty."""
        freq = apriori(simple_transactions, min_support=1.01)
        assert freq == {}

    def test_apriori_max_itemset_size(self, simple_transactions):
        """Max itemset size should limit depth."""
        freq = apriori(
            simple_transactions, min_support=0.30, max_itemset_size=1
        )
        # Only 1-itemsets
        for itemset in freq:
            assert len(itemset) == 1

    def test_apriori_empty_transactions(self):
        freq = apriori([], min_support=0.05)
        assert freq == {}


# =====================================================================
# 3. Association Rule Tests
# =====================================================================

class TestAssociationRules:
    """Tests for rule generation, confidence, and lift."""

    def test_rule_generation_basic(self, simple_transactions):
        freq = apriori(simple_transactions, min_support=0.30)
        rules = generate_association_rules(
            freq, min_confidence=0.0, min_lift=0.0, num_transactions=5
        )
        assert len(rules) > 0
        # Every rule must have non-empty antecedent and consequent
        for r in rules:
            assert len(r["antecedent"]) >= 1
            assert len(r["consequent"]) >= 1

    def test_confidence_calculation(self, simple_transactions):
        """Verify confidence = support(A U B) / support(A)."""
        freq = apriori(simple_transactions, min_support=0.30)
        rules = generate_association_rules(
            freq, min_confidence=0.0, min_lift=0.0
        )

        # Find rule {B} -> {A}: conf = support({A,B}) / support({B}) = 0.6/0.6 = 1.0
        rule_b_to_a = [
            r for r in rules
            if r["antecedent"] == ["B"] and r["consequent"] == ["A"]
        ]
        assert len(rule_b_to_a) == 1
        assert rule_b_to_a[0]["confidence"] == pytest.approx(1.0)

    def test_lift_calculation(self, simple_transactions):
        """Verify lift = confidence / support(B)."""
        freq = apriori(simple_transactions, min_support=0.30)
        rules = generate_association_rules(
            freq, min_confidence=0.0, min_lift=0.0
        )

        # Rule {B} -> {A}: conf=1.0, support(A)=1.0, lift=1.0/1.0=1.00
        rule_b_to_a = [
            r for r in rules
            if r["antecedent"] == ["B"] and r["consequent"] == ["A"]
        ]
        assert len(rule_b_to_a) == 1
        assert rule_b_to_a[0]["lift"] == pytest.approx(1.00, abs=0.01)

        # Rule {C} -> {B}: conf=0.4/0.6=0.667, support(B)=0.6, lift=0.667/0.6=1.111
        rule_c_to_b = [
            r for r in rules
            if r["antecedent"] == ["C"] and r["consequent"] == ["B"]
        ]
        assert len(rule_c_to_b) == 1
        assert rule_c_to_b[0]["lift"] == pytest.approx((0.4 / 0.6) / 0.6, abs=0.01)

    def test_min_confidence_filtering(self, simple_transactions):
        """High min_confidence should filter out weak rules."""
        freq = apriori(simple_transactions, min_support=0.30)
        all_rules = generate_association_rules(
            freq, min_confidence=0.0, min_lift=0.0
        )
        filtered = generate_association_rules(
            freq, min_confidence=0.90, min_lift=0.0
        )
        assert len(filtered) <= len(all_rules)
        for r in filtered:
            assert r["confidence"] >= 0.90

    def test_min_lift_filtering(self, simple_transactions):
        """Rules should respect min_lift threshold."""
        freq = apriori(simple_transactions, min_support=0.30)
        rules = generate_association_rules(
            freq, min_confidence=0.0, min_lift=1.2
        )
        for r in rules:
            assert r["lift"] >= 1.2

    def test_antecedent_consequent_disjoint(self, simple_transactions):
        """Antecedent and consequent must be disjoint."""
        freq = apriori(simple_transactions, min_support=0.30)
        rules = generate_association_rules(
            freq, min_confidence=0.0, min_lift=0.0
        )
        for r in rules:
            ant_set = set(r["antecedent"])
            con_set = set(r["consequent"])
            assert ant_set.isdisjoint(con_set)

    def test_no_rules_from_1_itemsets(self):
        """Single-item frequent itemsets cannot produce rules."""
        freq = {frozenset(["A"]): 0.5, frozenset(["B"]): 0.5}
        rules = generate_association_rules(freq, min_confidence=0.0, min_lift=0.0)
        assert rules == []

    def test_occurrence_count(self, simple_transactions):
        freq = apriori(simple_transactions, min_support=0.30)
        rules = generate_association_rules(
            freq, min_confidence=0.0, min_lift=0.0, num_transactions=5
        )
        for r in rules:
            assert "occurrence_count" in r
            assert isinstance(r["occurrence_count"], int)
            assert r["occurrence_count"] >= 1


# =====================================================================
# 4. Analytics Tests
# =====================================================================

class TestAnalytics:
    """Tests for analytics and summary utilities."""

    def test_item_frequency(self, simple_transactions):
        freq = apriori(simple_transactions, min_support=0.30)
        item_freq = compute_item_frequency(freq)
        # A participates in the most itemsets
        assert len(item_freq) > 0
        items = {f["item"] for f in item_freq}
        assert "A" in items

    def test_top_itemsets_by_support(self, simple_transactions):
        freq = apriori(simple_transactions, min_support=0.30)
        top = top_itemsets_by_support(freq, top_n=3, min_size=2)
        assert len(top) <= 3
        for t in top:
            assert t["size"] >= 2
        # First should have highest support
        if len(top) >= 2:
            assert top[0]["support"] >= top[1]["support"]

    def test_association_statistics(self, simple_transactions):
        freq = apriori(simple_transactions, min_support=0.30)
        rules = generate_association_rules(
            freq, min_confidence=0.30, min_lift=1.0
        )
        txn_summary = compute_transaction_summary(simple_transactions)
        stats = compute_association_statistics(freq, rules, txn_summary)

        assert "num_frequent_itemsets" in stats
        assert "num_rules" in stats
        assert "itemset_size_distribution" in stats
        assert "rule_metrics" in stats
        assert stats["num_frequent_itemsets"] == len(freq)
        assert stats["num_rules"] == len(rules)


# =====================================================================
# 5. Integration Tests
# =====================================================================

class TestIntegration:
    """Integration tests using the actual project data."""

    def test_full_pipeline_with_market_data(self):
        """Run the full pipeline on actual project data."""
        csv_path = Path("data/processed/market_features.csv")
        if not csv_path.exists():
            pytest.skip("market_features.csv not found")

        df = pd.read_csv(csv_path)
        txns = build_transactions(df, return_threshold=0.01)
        assert len(txns) > 0

        freq = apriori(txns, min_support=0.05)
        assert len(freq) > 0

        rules = generate_association_rules(
            freq, min_confidence=0.30, min_lift=1.0, num_transactions=len(txns)
        )
        # Rules may or may not exist depending on data

        summary = compute_transaction_summary(txns)
        assert summary["num_transactions"] > 0
        assert summary["num_unique_items"] > 0

    def test_mongodb_mock_storage(self, simple_transactions):
        """Verify mock MongoDB storage works."""
        import sys
        from pathlib import Path
        ROOT = Path(__file__).resolve().parent.parent
        sys.path.insert(0, str(ROOT))

        freq = apriori(simple_transactions, min_support=0.30)
        rules = generate_association_rules(
            freq, min_confidence=0.30, min_lift=1.0, num_transactions=5
        )

        itemsets_list = [
            {"itemset": sorted(itemset), "size": len(itemset),
             "support": round(sup, 6), "occurrence_count": int(round(sup * 5))}
            for itemset, sup in freq.items() if len(itemset) >= 2
        ]

        try:
            import mongomock
            client = mongomock.MongoClient()
            db = client["test_db"]
            coll = db["association_rules"]

            # Store rules
            docs = []
            for r in rules:
                doc = dict(r)
                doc["type"] = "rule"
                docs.append(doc)
            for i in itemsets_list:
                doc = dict(i)
                doc["type"] = "itemset"
                docs.append(doc)

            if docs:
                coll.insert_many(docs)

            stored_rules = list(coll.find({"type": "rule"}, {"_id": 0}))
            stored_itemsets = list(coll.find({"type": "itemset"}, {"_id": 0}))

            assert len(stored_rules) == len(rules)
            assert len(stored_itemsets) == len(itemsets_list)
        except ImportError:
            pytest.skip("mongomock not installed")
