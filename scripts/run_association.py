#!/usr/bin/env python3
"""
End-to-End Association Rule Mining Pipeline.

Tasks:
1. Load processed market features.
2. Construct transactions from daily stock returns above threshold.
3. Run Apriori algorithm to find frequent itemsets.
4. Generate association rules with confidence and lift.
5. Compute analytics and summary statistics.
6. Save CSV/JSON outputs to data/processed/association/.
7. Store rules in MongoDB when available.
8. Print concise execution summary.

Usage:
    python scripts/run_association.py
    python scripts/run_association.py --mock
    python scripts/run_association.py --threshold 0.005 --min-support 0.03
    python scripts/run_association.py --min-confidence 0.25 --min-lift 1.2
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import yaml

from src.mining.transactions import build_transactions, compute_transaction_summary
from src.mining.apriori import apriori
from src.mining.rules import generate_association_rules
from src.mining.analytics import (
    compute_association_statistics,
    top_rules_by_lift,
    top_rules_by_confidence,
)
from src.utils.logger import logger


# =====================================================================
# Configuration
# =====================================================================

def load_association_config() -> Dict[str, Any]:
    """Load association mining parameters from config/config.yaml."""
    cfg_path = ROOT / "config" / "config.yaml"
    cfg: Dict[str, Any] = {}
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            full = yaml.safe_load(f) or {}
            cfg = full.get("association", {})

    return {
        "return_threshold": cfg.get("return_threshold", 0.01),
        "min_support": cfg.get("min_support", 0.05),
        "min_confidence": cfg.get("min_confidence", 0.30),
        "min_lift": cfg.get("min_lift", 1.0),
        "max_itemset_size": cfg.get("max_itemset_size", 4),
        "output_path": cfg.get("output_path", "data/processed/association"),
    }


# =====================================================================
# Data Loading
# =====================================================================

def load_market_data() -> pd.DataFrame:
    """Load processed market features with daily returns."""
    csv_path = ROOT / "data" / "processed" / "market_features.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Market features not found at {csv_path}. "
            f"Run 'python scripts/ingest_data.py --source sample' first."
        )
    logger.info(f"Loading market data from {csv_path}...")
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} records for {df['symbol'].nunique()} symbols.")
    return df


# =====================================================================
# MongoDB Integration
# =====================================================================

def store_rules_in_mongo(
    rules: List[Dict[str, Any]],
    frequent_itemsets_list: List[Dict[str, Any]],
    use_mock: bool = False,
) -> bool:
    """
    Store association rules and frequent itemsets in MongoDB.

    Collection: 'association_rules'

    Returns True if stored successfully, False if skipped.
    """
    try:
        from pymongo import UpdateOne

        if use_mock:
            import mongomock
            client = mongomock.MongoClient()
            db = client["financial_analytics"]
            logger.info("Using mock MongoDB client for association rule storage.")
        else:
            from src.database.mongodb import check_connection, get_database
            health = check_connection()
            if not health["connected"]:
                logger.info(
                    "MongoDB offline. Skipping association rule storage. "
                    "(Results saved to local CSV/JSON files.)"
                )
                return False
            db = get_database()

        coll = db["association_rules"]

        # Store rules
        rule_ops = []
        for r in rules:
            doc = dict(r)
            doc["type"] = "rule"
            doc["antecedent_str"] = " + ".join(r["antecedent"])
            doc["consequent_str"] = " + ".join(r["consequent"])
            filter_spec = {
                "type": "rule",
                "antecedent_str": doc["antecedent_str"],
                "consequent_str": doc["consequent_str"],
            }
            rule_ops.append(UpdateOne(filter_spec, {"$set": doc}, upsert=True))

        # Store top frequent itemsets (size >= 2)
        itemset_ops = []
        for item in frequent_itemsets_list:
            doc = dict(item)
            doc["type"] = "itemset"
            doc["itemset_str"] = " + ".join(item["itemset"])
            filter_spec = {
                "type": "itemset",
                "itemset_str": doc["itemset_str"],
            }
            itemset_ops.append(UpdateOne(filter_spec, {"$set": doc}, upsert=True))

        all_ops = rule_ops + itemset_ops

        if use_mock:
            all_docs = []
            for r in rules:
                doc = dict(r)
                doc["type"] = "rule"
                all_docs.append(doc)
            for item in frequent_itemsets_list:
                doc = dict(item)
                doc["type"] = "itemset"
                all_docs.append(doc)
            if all_docs:
                coll.insert_many(all_docs)
            logger.info(
                f"Stored {len(rule_ops)} rules + {len(itemset_ops)} itemsets "
                f"in mock MongoDB."
            )
            return True

        if all_ops:
            coll.bulk_write(all_ops, ordered=False)
            logger.info(
                f"Upserted {len(rule_ops)} rules + {len(itemset_ops)} itemsets "
                f"into MongoDB collection 'association_rules'."
            )
        return True

    except Exception as e:
        logger.warning(f"Could not store association rules in MongoDB: {e}")
        return False


# =====================================================================
# Main Pipeline
# =====================================================================

def run_association_pipeline(
    return_threshold: float = None,
    min_support: float = None,
    min_confidence: float = None,
    min_lift: float = None,
    use_mock_mongo: bool = False,
) -> Dict[str, Any]:
    """Execute the full association rule mining pipeline."""
    cfg = load_association_config()

    # CLI overrides
    threshold = return_threshold if return_threshold is not None else cfg["return_threshold"]
    m_support = min_support if min_support is not None else cfg["min_support"]
    m_confidence = min_confidence if min_confidence is not None else cfg["min_confidence"]
    m_lift = min_lift if min_lift is not None else cfg["min_lift"]
    max_size = cfg["max_itemset_size"]
    output_dir = ROOT / cfg["output_path"]
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("STARTING PHASE 6: ASSOCIATION RULE MINING PIPELINE")
    logger.info("=" * 60)
    logger.info(f"  Return threshold : {threshold}")
    logger.info(f"  Min support      : {m_support}")
    logger.info(f"  Min confidence   : {m_confidence}")
    logger.info(f"  Min lift         : {m_lift}")
    logger.info(f"  Max itemset size : {max_size}")

    # ── Step 1: Load Data ──────────────────────────────────────
    logger.info("--- Step 1: Loading Market Data ---")
    market_df = load_market_data()

    # ── Step 2: Build Transactions ─────────────────────────────
    logger.info("--- Step 2: Building Transactions ---")
    transactions = build_transactions(
        market_df, return_threshold=threshold
    )
    txn_summary = compute_transaction_summary(
        transactions,
        return_threshold=threshold,
        min_support=m_support,
        min_confidence=m_confidence,
        min_lift=m_lift,
    )

    txn_summary_path = output_dir / "transaction_summary.json"
    with open(txn_summary_path, "w", encoding="utf-8") as f:
        json.dump(txn_summary, f, indent=2)
    logger.info(f"Saved transaction summary to {txn_summary_path}")

    # ── Step 3: Apriori ────────────────────────────────────────
    logger.info("--- Step 3: Running Apriori Algorithm ---")
    frequent_itemsets = apriori(
        transactions,
        min_support=m_support,
        max_itemset_size=max_size,
    )

    # Save frequent itemsets CSV
    itemsets_list = []
    for itemset, sup in frequent_itemsets.items():
        itemsets_list.append({
            "itemset": ", ".join(sorted(itemset)),
            "size": len(itemset),
            "support": round(sup, 6),
            "occurrence_count": int(round(sup * len(transactions))),
        })
    itemsets_list.sort(key=lambda x: (-x["support"], -x["size"]))

    itemsets_df = pd.DataFrame(itemsets_list)
    itemsets_path = output_dir / "frequent_itemsets.csv"
    itemsets_df.to_csv(itemsets_path, index=False)
    logger.info(f"Saved {len(itemsets_list)} frequent itemsets to {itemsets_path}")

    # ── Step 4: Association Rules ──────────────────────────────
    logger.info("--- Step 4: Generating Association Rules ---")
    rules = generate_association_rules(
        frequent_itemsets,
        min_confidence=m_confidence,
        min_lift=m_lift,
        num_transactions=len(transactions),
    )

    rules_df = pd.DataFrame(rules)
    if not rules_df.empty:
        # Convert list columns to strings for CSV
        rules_df["antecedent"] = rules_df["antecedent"].apply(
            lambda x: ", ".join(x) if isinstance(x, list) else str(x)
        )
        rules_df["consequent"] = rules_df["consequent"].apply(
            lambda x: ", ".join(x) if isinstance(x, list) else str(x)
        )
    rules_path = output_dir / "association_rules.csv"
    rules_df.to_csv(rules_path, index=False)
    logger.info(f"Saved {len(rules)} association rules to {rules_path}")

    # ── Step 5: Analytics ──────────────────────────────────────
    logger.info("--- Step 5: Computing Analytics ---")
    stats = compute_association_statistics(
        frequent_itemsets, rules, txn_summary
    )

    stats_path = output_dir / "association_statistics.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, default=str)
    logger.info(f"Saved association statistics to {stats_path}")

    # ── Step 6: MongoDB ────────────────────────────────────────
    logger.info("--- Step 6: MongoDB Storage ---")
    # Prepare itemsets for MongoDB (size >= 2)
    mongo_itemsets = [
        i for i in itemsets_list if i["size"] >= 2
    ]
    mongo_stored = store_rules_in_mongo(rules, mongo_itemsets, use_mock=use_mock_mongo)

    # ── Step 7: Print Summary ──────────────────────────────────
    _print_summary(txn_summary, itemsets_list, rules, stats, mongo_stored)

    return stats


def _print_summary(
    txn_summary: Dict[str, Any],
    itemsets_list: List[Dict[str, Any]],
    rules: List[Dict[str, Any]],
    stats: Dict[str, Any],
    mongo_stored: bool,
) -> None:
    """Print formatted pipeline results."""
    print("\n" + "=" * 80)
    print("PHASE 6: ASSOCIATION RULE MINING RESULTS SUMMARY")
    print("=" * 80)

    # A. Transactions
    print(f"\n[A] TRANSACTION DATASET:")
    print(f"  Transactions         : {txn_summary['num_transactions']}")
    print(f"  Unique items         : {txn_summary['num_unique_items']}")
    print(f"  Avg transaction size : {txn_summary['avg_transaction_size']:.2f}")
    print(f"  Min/Max size         : {txn_summary['min_transaction_size']} / {txn_summary['max_transaction_size']}")
    print(f"  Return threshold     : {txn_summary['return_threshold']}")

    # B. Frequent Itemsets
    print(f"\n[B] FREQUENT ITEMSETS:")
    print(f"  Total found : {len(itemsets_list)}")
    size_dist = stats.get("itemset_size_distribution", {})
    for size, count in sorted(size_dist.items()):
        print(f"    Size {size}: {count} itemsets")

    # Top itemsets (size >= 2)
    top_multi = [i for i in itemsets_list if i["size"] >= 2][:5]
    if top_multi:
        print(f"\n  Top multi-item frequent itemsets:")
        for i in top_multi:
            print(f"    {i['itemset']:<45} support={i['support']:.4f}  count={i['occurrence_count']}")

    # C. Association Rules
    print(f"\n[C] ASSOCIATION RULES:")
    print(f"  Total rules          : {len(rules)}")
    print(f"  Rules with lift > 1  : {stats.get('num_rules_lift_gt_1', 0)}")
    print(f"  Rules with lift > 1.5: {stats.get('num_rules_lift_gt_1_5', 0)}")

    if rules:
        rm = stats.get("rule_metrics", {})
        lift_m = rm.get("lift", {})
        conf_m = rm.get("confidence", {})
        print(f"  Lift range           : {lift_m.get('min', 0):.4f} - {lift_m.get('max', 0):.4f}")
        print(f"  Confidence range     : {conf_m.get('min', 0):.4f} - {conf_m.get('max', 0):.4f}")

        # Top rules
        top = rules[:5]
        print(f"\n  Top rules by lift:")
        for r in top:
            ant = ", ".join(r["antecedent"]) if isinstance(r["antecedent"], list) else r["antecedent"]
            con = ", ".join(r["consequent"]) if isinstance(r["consequent"], list) else r["consequent"]
            print(
                f"    {ant} -> {con}\n"
                f"      lift={r['lift']:.4f}  conf={r['confidence']:.4f}  "
                f"sup={r['support']:.4f}"
            )
    else:
        print("  (No rules generated with current thresholds)")

    # D. Integration
    print(f"\n[D] OUTPUTS & INTEGRATION:")
    print(f"  CSV/JSON saved to  : data/processed/association/")
    print(f"  MongoDB ingestion  : {'Completed' if mongo_stored else 'Skipped (MongoDB offline)'}")

    # E. Disclaimer
    print(f"\n[E] DISCLAIMER:")
    print(f"  Association rules identify historical co-movement patterns in stock returns.")
    print(f"  They do NOT imply causation, prediction ability, or trading signals.")
    print(f"  Results are based on synthetic sample data for educational purposes only.")
    print("=" * 80 + "\n")


# =====================================================================
# Entry Point
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run Phase 6 Association Rule Mining Pipeline."
    )
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Return threshold for transaction construction (default: 0.01).",
    )
    parser.add_argument(
        "--min-support", type=float, default=None,
        help="Minimum support threshold (default: 0.05).",
    )
    parser.add_argument(
        "--min-confidence", type=float, default=None,
        help="Minimum confidence threshold (default: 0.30).",
    )
    parser.add_argument(
        "--min-lift", type=float, default=None,
        help="Minimum lift threshold (default: 1.0).",
    )
    parser.add_argument(
        "--mock", action="store_true",
        help="Use mongomock for MongoDB storage (no real MongoDB needed).",
    )
    args = parser.parse_args()

    run_association_pipeline(
        return_threshold=args.threshold,
        min_support=args.min_support,
        min_confidence=args.min_confidence,
        min_lift=args.min_lift,
        use_mock_mongo=args.mock,
    )


if __name__ == "__main__":
    main()
