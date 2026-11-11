"""
Association Rule Mining package for Financial Intelligence Platform.

Exports:
- Transaction construction from stock return data
- Transparent Apriori frequent itemset mining
- Association rule generation with support/confidence/lift
- Analytics and visualization helpers
"""

from src.mining.transactions import (
    build_transactions,
    compute_transaction_summary,
)
from src.mining.apriori import (
    apriori,
    count_support,
    generate_candidates,
    get_frequent_1_itemsets,
)
from src.mining.rules import (
    generate_association_rules,
)
from src.mining.analytics import (
    compute_association_statistics,
    compute_item_frequency,
    top_itemsets_by_support,
    top_rules_by_lift,
    top_rules_by_confidence,
)

__all__ = [
    "build_transactions",
    "compute_transaction_summary",
    "apriori",
    "count_support",
    "generate_candidates",
    "get_frequent_1_itemsets",
    "generate_association_rules",
    "compute_association_statistics",
    "compute_item_frequency",
    "top_itemsets_by_support",
    "top_rules_by_lift",
    "top_rules_by_confidence",
]
