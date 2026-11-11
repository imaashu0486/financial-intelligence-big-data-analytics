"""
Analytics and Summary Utilities for Association Rule Mining.

Provides aggregate statistics, item frequency analysis, and data preparation
for downstream dashboard visualization and reporting.
"""

from collections import Counter
from typing import Any, Dict, FrozenSet, List

from src.mining.apriori import SupportMap
from src.utils.logger import logger


def compute_item_frequency(
    frequent_itemsets: SupportMap,
) -> List[Dict[str, Any]]:
    """
    Compute how often each individual item appears across all frequent itemsets.

    Parameters
    ----------
    frequent_itemsets : SupportMap
        Frequent itemsets from Apriori.

    Returns
    -------
    list of dict
        Sorted by frequency descending: [{"item": str, "frequency": int, "support": float}]
    """
    # Get 1-itemset supports
    item_supports: Dict[str, float] = {}
    for itemset, sup in frequent_itemsets.items():
        if len(itemset) == 1:
            item = next(iter(itemset))
            item_supports[item] = sup

    # Count how many frequent itemsets each item participates in
    participation: Counter = Counter()
    for itemset in frequent_itemsets:
        for item in itemset:
            participation[item] += 1

    result = []
    for item, count in participation.most_common():
        result.append({
            "item": item,
            "frequency_in_itemsets": count,
            "support": round(item_supports.get(item, 0.0), 6),
        })

    return result


def top_itemsets_by_support(
    frequent_itemsets: SupportMap,
    top_n: int = 10,
    min_size: int = 1,
) -> List[Dict[str, Any]]:
    """
    Return the top-N frequent itemsets sorted by support descending.

    Parameters
    ----------
    frequent_itemsets : SupportMap
        Frequent itemsets from Apriori.
    top_n : int, default 10
        Number of itemsets to return.
    min_size : int, default 1
        Minimum itemset size to include.

    Returns
    -------
    list of dict
        Each dict: {"itemset": list, "size": int, "support": float}
    """
    filtered = [
        {"itemset": sorted(itemset), "size": len(itemset), "support": round(sup, 6)}
        for itemset, sup in frequent_itemsets.items()
        if len(itemset) >= min_size
    ]
    filtered.sort(key=lambda x: (-x["support"], -x["size"]))
    return filtered[:top_n]


def top_rules_by_lift(
    rules: List[Dict[str, Any]],
    top_n: int = 10,
) -> List[Dict[str, Any]]:
    """Return top-N rules sorted by lift descending."""
    sorted_rules = sorted(rules, key=lambda r: (-r["lift"], -r["confidence"]))
    return sorted_rules[:top_n]


def top_rules_by_confidence(
    rules: List[Dict[str, Any]],
    top_n: int = 10,
) -> List[Dict[str, Any]]:
    """Return top-N rules sorted by confidence descending."""
    sorted_rules = sorted(rules, key=lambda r: (-r["confidence"], -r["lift"]))
    return sorted_rules[:top_n]


def compute_association_statistics(
    frequent_itemsets: SupportMap,
    rules: List[Dict[str, Any]],
    transaction_summary: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compute aggregate statistics about the mining results.

    Parameters
    ----------
    frequent_itemsets : SupportMap
        All frequent itemsets.
    rules : list of dict
        Generated association rules.
    transaction_summary : dict
        Summary from compute_transaction_summary().

    Returns
    -------
    dict
        Comprehensive statistics dictionary.
    """
    # Itemset size distribution
    size_dist: Dict[int, int] = {}
    for itemset in frequent_itemsets:
        size = len(itemset)
        size_dist[size] = size_dist.get(size, 0) + 1

    # Rule metrics
    lifts = [r["lift"] for r in rules] if rules else []
    confidences = [r["confidence"] for r in rules] if rules else []
    supports = [r["support"] for r in rules] if rules else []

    # Antecedent/consequent size distribution
    ant_sizes: Dict[int, int] = {}
    con_sizes: Dict[int, int] = {}
    for r in rules:
        a_size = len(r["antecedent"])
        c_size = len(r["consequent"])
        ant_sizes[a_size] = ant_sizes.get(a_size, 0) + 1
        con_sizes[c_size] = con_sizes.get(c_size, 0) + 1

    stats = {
        "num_frequent_itemsets": len(frequent_itemsets),
        "itemset_size_distribution": {
            str(k): v for k, v in sorted(size_dist.items())
        },
        "num_rules": len(rules),
        "num_rules_lift_gt_1": sum(1 for r in rules if r["lift"] > 1.0),
        "num_rules_lift_gt_1_5": sum(1 for r in rules if r["lift"] > 1.5),
        "rule_antecedent_size_distribution": {
            str(k): v for k, v in sorted(ant_sizes.items())
        },
        "rule_consequent_size_distribution": {
            str(k): v for k, v in sorted(con_sizes.items())
        },
        "rule_metrics": {
            "lift": {
                "min": round(min(lifts), 4) if lifts else 0.0,
                "max": round(max(lifts), 4) if lifts else 0.0,
                "mean": round(sum(lifts) / len(lifts), 4) if lifts else 0.0,
            },
            "confidence": {
                "min": round(min(confidences), 4) if confidences else 0.0,
                "max": round(max(confidences), 4) if confidences else 0.0,
                "mean": round(sum(confidences) / len(confidences), 4) if confidences else 0.0,
            },
            "support": {
                "min": round(min(supports), 6) if supports else 0.0,
                "max": round(max(supports), 6) if supports else 0.0,
                "mean": round(sum(supports) / len(supports), 6) if supports else 0.0,
            },
        },
        "top_itemsets_by_support": top_itemsets_by_support(
            frequent_itemsets, top_n=10, min_size=2
        ),
        "top_rules_by_lift": [
            {
                "rule": f"{' + '.join(r['antecedent'])} -> {' + '.join(r['consequent'])}",
                "lift": r["lift"],
                "confidence": r["confidence"],
                "support": r["support"],
            }
            for r in top_rules_by_lift(rules, top_n=10)
        ],
        "top_rules_by_confidence": [
            {
                "rule": f"{' + '.join(r['antecedent'])} -> {' + '.join(r['consequent'])}",
                "confidence": r["confidence"],
                "lift": r["lift"],
                "support": r["support"],
            }
            for r in top_rules_by_confidence(rules, top_n=10)
        ],
        "item_frequency": compute_item_frequency(frequent_itemsets),
        "transaction_summary": transaction_summary,
    }

    return stats
