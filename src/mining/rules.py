"""
Association Rule Generation from Frequent Itemsets.

Generates rules of the form A -> B where:
- A (antecedent) and B (consequent) are disjoint non-empty subsets
  of a frequent itemset.

Metrics:
- support(A U B) = frequency of the full itemset
- confidence(A -> B) = support(A U B) / support(A)
- lift(A -> B) = confidence(A -> B) / support(B)
  - lift > 1: A and B appear together MORE often than expected under independence
  - lift = 1: A and B are independent
  - lift < 1: A and B appear together LESS often than expected
"""

from itertools import combinations
from typing import Any, Dict, FrozenSet, List, Optional

from src.mining.apriori import SupportMap
from src.utils.logger import logger


def generate_association_rules(
    frequent_itemsets: SupportMap,
    min_confidence: float = 0.30,
    min_lift: float = 1.0,
    num_transactions: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Generate association rules from frequent itemsets.

    For each frequent itemset of size >= 2, generates all valid rules
    A -> B where |A| >= 1 and |B| >= 1 and A ∩ B = ∅ and A ∪ B = itemset.

    Parameters
    ----------
    frequent_itemsets : SupportMap
        Dictionary mapping frequent itemsets (frozenset) to support values.
    min_confidence : float, default 0.30
        Minimum confidence threshold for rules.
    min_lift : float, default 1.0
        Minimum lift threshold for rules.
    num_transactions : int, optional
        Total transaction count (used for occurrence count calculation).

    Returns
    -------
    list of dict
        Each rule dict contains:
        - antecedent: sorted list of items
        - consequent: sorted list of items
        - support: support(A ∪ B)
        - confidence: support(A ∪ B) / support(A)
        - lift: confidence / support(B)
        - antecedent_support: support(A)
        - consequent_support: support(B)
        - occurrence_count: int (if num_transactions provided)

    Rules are sorted by lift (desc), confidence (desc), support (desc).
    """
    rules: List[Dict[str, Any]] = []

    for itemset, itemset_support in frequent_itemsets.items():
        if len(itemset) < 2:
            continue

        items = sorted(itemset)

        # Generate all non-empty proper subsets as antecedents
        for ant_size in range(1, len(items)):
            for antecedent_tuple in combinations(items, ant_size):
                antecedent = frozenset(antecedent_tuple)
                consequent = itemset - antecedent

                if not consequent:
                    continue

                # Look up supports
                antecedent_support = frequent_itemsets.get(antecedent)
                consequent_support = frequent_itemsets.get(consequent)

                # Both subsets must be frequent (guaranteed by Apriori property,
                # but we check defensively)
                if antecedent_support is None or antecedent_support == 0:
                    continue
                if consequent_support is None or consequent_support == 0:
                    continue

                # Calculate metrics
                confidence = itemset_support / antecedent_support
                lift = confidence / consequent_support

                # Apply thresholds
                if confidence < min_confidence:
                    continue
                if lift < min_lift:
                    continue

                rule = {
                    "antecedent": sorted(antecedent),
                    "consequent": sorted(consequent),
                    "support": round(itemset_support, 6),
                    "confidence": round(confidence, 6),
                    "lift": round(lift, 4),
                    "antecedent_support": round(antecedent_support, 6),
                    "consequent_support": round(consequent_support, 6),
                }

                if num_transactions is not None:
                    rule["occurrence_count"] = int(
                        round(itemset_support * num_transactions)
                    )

                rules.append(rule)

    # Sort: lift desc, confidence desc, support desc
    rules.sort(key=lambda r: (-r["lift"], -r["confidence"], -r["support"]))

    logger.info(
        f"Generated {len(rules)} association rules "
        f"(min_confidence={min_confidence}, min_lift={min_lift})"
    )

    return rules
