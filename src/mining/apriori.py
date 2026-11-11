"""
Transparent Apriori Algorithm for Frequent Itemset Mining.

Implements the classic Apriori algorithm with clear educational structure:

1. Scan transactions to find frequent 1-itemsets (items meeting min_support).
2. Generate candidate k-itemsets from frequent (k-1)-itemsets.
3. Apply the Apriori pruning property: if ANY (k-1)-subset of a candidate
   is not frequent, prune that candidate.
4. Count support across all transactions.
5. Keep only candidates meeting min_support.
6. Repeat until no new frequent itemsets are found.

Apriori Property (Downward Closure):
    "If an itemset is infrequent, all of its supersets must also be infrequent."
    Equivalently: "If an itemset is frequent, all of its subsets must also be frequent."

This property enables aggressive pruning of the candidate space, avoiding
the need to enumerate all 2^n possible subsets.
"""

from itertools import combinations
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

from src.utils.logger import logger


# Type alias for clarity
ItemSet = FrozenSet[str]
SupportMap = Dict[ItemSet, float]


def count_support(
    candidates: List[ItemSet],
    transactions: List[Set[str]],
    num_transactions: int,
) -> SupportMap:
    """
    Count the support of each candidate itemset across all transactions.

    support(X) = (number of transactions containing X) / (total transactions)

    Parameters
    ----------
    candidates : list of frozenset
        Candidate itemsets to evaluate.
    transactions : list of set
        Transaction database.
    num_transactions : int
        Total number of transactions (denominator for support).

    Returns
    -------
    dict
        Mapping from itemset to its support value.
    """
    if num_transactions == 0:
        return {}

    counts: Dict[ItemSet, int] = {c: 0 for c in candidates}

    for txn in transactions:
        for candidate in candidates:
            if candidate.issubset(txn):
                counts[candidate] += 1

    return {
        itemset: count / num_transactions
        for itemset, count in counts.items()
    }


def get_frequent_1_itemsets(
    transactions: List[Set[str]],
    min_support: float,
) -> Tuple[SupportMap, Set[str]]:
    """
    Find all items that individually meet minimum support.

    This is the first pass of Apriori: scanning all transactions
    to identify frequent individual items.

    Parameters
    ----------
    transactions : list of set
        Transaction database.
    min_support : float
        Minimum support threshold in [0.0, 1.0].

    Returns
    -------
    tuple of (SupportMap, set of str)
        - SupportMap: frozenset({item}) -> support for each frequent item.
        - set: all unique items found (for reference).
    """
    if not transactions:
        return {}, set()

    num_txn = len(transactions)

    # Count occurrences of each item
    item_counts: Dict[str, int] = {}
    for txn in transactions:
        for item in txn:
            item_counts[item] = item_counts.get(item, 0) + 1

    all_items = set(item_counts.keys())

    # Filter by minimum support
    frequent: SupportMap = {}
    for item, count in item_counts.items():
        support = count / num_txn
        if support >= min_support:
            frequent[frozenset([item])] = support

    logger.debug(
        f"Frequent 1-itemsets: {len(frequent)} of {len(all_items)} items "
        f"(min_support={min_support})"
    )

    return frequent, all_items


def generate_candidates(
    frequent_prev: List[ItemSet],
    k: int,
) -> List[ItemSet]:
    """
    Generate candidate k-itemsets from frequent (k-1)-itemsets using the
    Apriori candidate generation strategy.

    Two (k-1)-itemsets can be joined if they share exactly (k-2) items.
    The resulting candidate is then pruned if any of its (k-1)-subsets
    is not in the frequent set (Apriori pruning property).

    Parameters
    ----------
    frequent_prev : list of frozenset
        Frequent (k-1)-itemsets from the previous level.
    k : int
        Target itemset size.

    Returns
    -------
    list of frozenset
        Candidate k-itemsets that survived Apriori pruning.
    """
    frequent_set = set(frequent_prev)
    candidates: Set[ItemSet] = set()

    # Join step: combine pairs sharing (k-2) items
    sorted_prev = sorted(frequent_prev, key=lambda x: sorted(x))
    n = len(sorted_prev)

    for i in range(n):
        for j in range(i + 1, n):
            union = sorted_prev[i] | sorted_prev[j]
            if len(union) == k:
                # Prune step: check all (k-1)-subsets are frequent
                # This is the core Apriori property application
                all_subsets_frequent = True
                for item in union:
                    subset = union - frozenset([item])
                    if subset not in frequent_set:
                        all_subsets_frequent = False
                        break

                if all_subsets_frequent:
                    candidates.add(union)

    return list(candidates)


def apriori(
    transactions: List[Set[str]],
    min_support: float = 0.05,
    max_itemset_size: Optional[int] = None,
) -> SupportMap:
    """
    Run the complete Apriori algorithm to find all frequent itemsets.

    Parameters
    ----------
    transactions : list of set
        Transaction database where each element is a set of item labels.
    min_support : float, default 0.05
        Minimum support threshold in [0.0, 1.0].
    max_itemset_size : int, optional
        Maximum size of itemsets to mine. If None, mines until exhaustion.

    Returns
    -------
    SupportMap
        Dictionary mapping each frequent itemset (frozenset) to its support.

    Algorithm
    ---------
    1. L1 = {frequent 1-itemsets}
    2. for k = 2, 3, ...:
        a. Ck = generate_candidates(L_{k-1}, k)   # Join + Prune
        b. Count support of each candidate in Ck
        c. Lk = {c in Ck : support(c) >= min_support}
        d. if Lk is empty: stop
    3. Return union of all L1, L2, ..., Lk
    """
    if not transactions:
        logger.warning("Empty transaction list. No frequent itemsets found.")
        return {}

    num_txn = len(transactions)

    # Step 1: Frequent 1-itemsets
    freq_1, all_items = get_frequent_1_itemsets(transactions, min_support)

    if not freq_1:
        logger.info("No frequent 1-itemsets found. Stopping Apriori.")
        return {}

    # Collect all frequent itemsets across all levels
    all_frequent: SupportMap = dict(freq_1)

    logger.info(
        f"Apriori Level 1: {len(freq_1)} frequent 1-itemsets "
        f"(of {len(all_items)} unique items)"
    )

    # Step 2: Iteratively find frequent k-itemsets
    prev_frequent = list(freq_1.keys())
    k = 2

    while True:
        if max_itemset_size is not None and k > max_itemset_size:
            logger.info(f"Reached max itemset size {max_itemset_size}. Stopping.")
            break

        # Generate candidates
        candidates = generate_candidates(prev_frequent, k)
        if not candidates:
            logger.info(f"No candidates at level {k}. Apriori complete.")
            break

        # Count support
        candidate_support = count_support(candidates, transactions, num_txn)

        # Filter by min_support
        freq_k = {
            itemset: sup
            for itemset, sup in candidate_support.items()
            if sup >= min_support
        }

        if not freq_k:
            logger.info(
                f"Apriori Level {k}: 0 frequent {k}-itemsets "
                f"(from {len(candidates)} candidates). Stopping."
            )
            break

        logger.info(
            f"Apriori Level {k}: {len(freq_k)} frequent {k}-itemsets "
            f"(from {len(candidates)} candidates)"
        )

        all_frequent.update(freq_k)
        prev_frequent = list(freq_k.keys())
        k += 1

    logger.info(
        f"Apriori complete: {len(all_frequent)} total frequent itemsets found "
        f"(min_support={min_support})"
    )

    return all_frequent
