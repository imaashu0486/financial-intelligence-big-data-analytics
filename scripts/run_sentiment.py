#!/usr/bin/env python3
"""
End-to-End Sentiment Analysis Pipeline Execution Script.

Tasks:
1. Load cleaned financial news (from processed CSV or from sample data).
2. Run sentiment analysis (FinancialLexiconAnalyzer or FinBERT if available).
3. Aggregate daily sentiment per (symbol, date).
4. Join with market features (same-day + next-day return alignment, no leakage).
5. Compute correlation, regime, and breakdown analytics.
6. Save all outputs to data/processed/sentiment/.
7. Optionally store sentiment results in MongoDB.
8. Print formatted summary with academic disclaimers.

Usage:
    python scripts/run_sentiment.py
    python scripts/run_sentiment.py --use-finbert
    python scripts/run_sentiment.py --mock
    python scripts/run_sentiment.py --source sample
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import yaml

from src.nlp.sentiment import (
    FinancialLexiconAnalyzer,
    analyze_news_dataframe,
    get_sentiment_analyzer,
)
from src.nlp.aggregation import (
    aggregate_daily_sentiment,
    join_sentiment_with_market,
)
from src.nlp.analytics import (
    calculate_regime_return_statistics,
    calculate_sentiment_correlations,
    calculate_stock_sentiment_breakdown,
    generate_visualization_payload,
)
from src.utils.logger import logger


# =====================================================================
# Configuration
# =====================================================================

def load_sentiment_config() -> Dict[str, Any]:
    """Load sentiment parameters from config/config.yaml."""
    cfg_path = ROOT / "config" / "config.yaml"
    cfg: Dict[str, Any] = {}
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            full = yaml.safe_load(f) or {}
            cfg = full.get("sentiment", {})

    return {
        "use_finbert": cfg.get("use_finbert", False),
        "model_name": cfg.get("model_name", "ProsusAI/finbert"),
        "neutral_threshold": cfg.get("neutral_threshold", 0.15),
        "batch_size": cfg.get("batch_size", 32),
        "output_path": cfg.get("output_path", "data/processed/sentiment"),
    }


# =====================================================================
# Data Loading
# =====================================================================

def load_news_data(source: str = "processed") -> pd.DataFrame:
    """
    Load financial news from processed or sample CSV.

    Parameters
    ----------
    source : str
        'processed' uses data/processed/news_cleaned.csv,
        'sample' uses data/sample/financial_news.csv.
    """
    if source == "sample":
        csv_path = ROOT / "data" / "sample" / "financial_news.csv"
    else:
        csv_path = ROOT / "data" / "processed" / "news_cleaned.csv"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"News data file not found at {csv_path}. "
            f"Run 'python scripts/generate_sample_data.py' and "
            f"'python scripts/ingest_data.py --source sample' first."
        )

    logger.info(f"Loading news data from {csv_path}...")
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} news articles with columns: {list(df.columns)}")
    return df


def load_market_features() -> pd.DataFrame:
    """Load processed market features with returns and technical indicators."""
    csv_path = ROOT / "data" / "processed" / "market_features.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Market features not found at {csv_path}. "
            f"Run 'python scripts/ingest_data.py --source sample' first."
        )
    logger.info(f"Loading market features from {csv_path}...")
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} market records.")
    return df


# =====================================================================
# MongoDB Integration
# =====================================================================

def store_sentiment_in_mongo(
    article_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    use_mock: bool = False,
) -> bool:
    """
    Store article-level and daily-aggregated sentiment in MongoDB.

    Collection: 'sentiment'
    Document schema:
        - type: 'article' | 'daily'
        - symbol, date, sentiment fields...

    Returns True if stored successfully, False if skipped.
    """
    try:
        from pymongo import UpdateOne

        if use_mock:
            import mongomock
            client = mongomock.MongoClient()
            coll = client["financial_analytics"]["sentiment"]
            logger.info("Using mock MongoDB client for sentiment storage.")
        else:
            from src.database.mongodb import check_connection, get_collection
            health = check_connection()
            if not health["connected"]:
                logger.info(
                    "MongoDB offline. Skipping sentiment storage. "
                    "(Sentiment results saved to local CSV files.)"
                )
                return False
            coll = get_collection("sentiment")

        # Store article-level sentiment
        article_ops = []
        article_cols = [
            "date", "symbol", "title", "sentiment_score",
            "sentiment_label", "confidence", "sentiment_model",
        ]
        for _, row in article_df.iterrows():
            doc = {col: row.get(col) for col in article_cols if col in row.index}
            doc["type"] = "article"
            # Sanitize NaN
            doc = {k: (None if pd.isna(v) else v) for k, v in doc.items()}
            filter_spec = {
                "type": "article",
                "symbol": doc.get("symbol"),
                "date": str(doc.get("date")),
                "title": doc.get("title"),
            }
            article_ops.append(UpdateOne(filter_spec, {"$set": doc}, upsert=True))

        # Store daily aggregated sentiment
        daily_ops = []
        for _, row in daily_df.iterrows():
            doc = row.to_dict()
            doc["type"] = "daily"
            doc = {k: (None if isinstance(v, float) and pd.isna(v) else v) for k, v in doc.items()}
            filter_spec = {
                "type": "daily",
                "symbol": doc.get("symbol"),
                "date": str(doc.get("date")),
            }
            daily_ops.append(UpdateOne(filter_spec, {"$set": doc}, upsert=True))

        all_ops = article_ops + daily_ops

        if use_mock:
            # mongomock: use insert_many for simplicity
            all_docs = []
            for _, row in article_df.iterrows():
                doc = {col: row.get(col) for col in article_cols if col in row.index}
                doc["type"] = "article"
                all_docs.append(doc)
            for _, row in daily_df.iterrows():
                doc = row.to_dict()
                doc["type"] = "daily"
                all_docs.append(doc)
            coll.insert_many(all_docs)
            logger.info(
                f"Stored {len(article_ops)} article + {len(daily_ops)} daily "
                f"sentiment documents in mock MongoDB."
            )
            return True

        if all_ops:
            coll.bulk_write(all_ops, ordered=False)
            logger.info(
                f"Upserted {len(article_ops)} article + {len(daily_ops)} daily "
                f"sentiment documents into MongoDB collection 'sentiment'."
            )
        return True

    except Exception as e:
        logger.warning(f"Could not store sentiment in MongoDB: {e}")
        return False


# =====================================================================
# Main Pipeline
# =====================================================================

def run_sentiment_pipeline(
    source: str = "processed",
    use_finbert: bool = False,
    use_mock_mongo: bool = False,
) -> Dict[str, Any]:
    """
    Execute the full sentiment analysis pipeline.

    Parameters
    ----------
    source : str
        Data source: 'processed' or 'sample'.
    use_finbert : bool
        If True, attempt to use FinBERT transformer model.
    use_mock_mongo : bool
        If True, use mongomock for MongoDB operations.

    Returns
    -------
    dict
        Summary of pipeline results.
    """
    cfg = load_sentiment_config()
    output_dir = ROOT / cfg["output_path"]
    output_dir.mkdir(parents=True, exist_ok=True)

    # Override config with CLI flags
    if use_finbert:
        cfg["use_finbert"] = True

    logger.info("=" * 60)
    logger.info("STARTING PHASE 5: SENTIMENT ANALYSIS PIPELINE")
    logger.info("=" * 60)

    # ── Step 1: Load Data ──────────────────────────────────────
    logger.info("--- Step 1: Loading News & Market Data ---")
    news_df = load_news_data(source=source)
    market_df = load_market_features()

    # ── Step 2: Sentiment Analysis ─────────────────────────────
    logger.info("--- Step 2: Running Sentiment Analysis ---")
    analyzer = get_sentiment_analyzer(
        use_finbert=cfg["use_finbert"],
        model_name=cfg["model_name"],
        neutral_threshold=cfg["neutral_threshold"],
    )
    logger.info(f"Using sentiment model: {analyzer.model_name}")

    analyzed_df = analyze_news_dataframe(
        news_df,
        analyzer=analyzer,
        neutral_threshold=cfg["neutral_threshold"],
    )

    # Save article-level sentiment
    article_path = output_dir / "article_sentiment.csv"
    analyzed_df.to_csv(article_path, index=False)
    logger.info(f"Saved {len(analyzed_df)} article sentiment records to {article_path}")

    # ── Step 3: Daily Aggregation ──────────────────────────────
    logger.info("--- Step 3: Daily Sentiment Aggregation ---")
    daily_df = aggregate_daily_sentiment(analyzed_df)

    daily_path = output_dir / "daily_sentiment.csv"
    daily_df.to_csv(daily_path, index=False)
    logger.info(f"Saved {len(daily_df)} daily sentiment records to {daily_path}")

    # ── Step 4: Market Join (Lead-Lag, No Leakage) ─────────────
    logger.info("--- Step 4: Sentiment-Market Join ---")
    joined_df = join_sentiment_with_market(daily_df, market_df, how="inner")

    joined_path = output_dir / "sentiment_market_joined.csv"
    joined_df.to_csv(joined_path, index=False)
    logger.info(f"Saved {len(joined_df)} joined sentiment-market records to {joined_path}")

    # ── Step 5: Statistical Analytics ──────────────────────────
    logger.info("--- Step 5: Correlation & Regime Analytics ---")
    correlations = calculate_sentiment_correlations(joined_df)
    regime_stats = calculate_regime_return_statistics(
        joined_df, threshold=cfg["neutral_threshold"]
    )
    stock_breakdown = calculate_stock_sentiment_breakdown(analyzed_df)

    breakdown_path = output_dir / "stock_sentiment_breakdown.csv"
    stock_breakdown.to_csv(breakdown_path, index=False)
    logger.info(f"Saved stock sentiment breakdown to {breakdown_path}")

    # ── Step 6: Visualization Payload ──────────────────────────
    viz_payload = generate_visualization_payload(analyzed_df, daily_df, joined_df)

    # Save analytics JSON
    analytics_summary = {
        "correlations": correlations,
        "regime_statistics": regime_stats,
        "stock_breakdown": stock_breakdown.to_dict(orient="records"),
        "sentiment_distribution": viz_payload["sentiment_distribution"],
        "pipeline_config": {
            "model": analyzer.model_name,
            "neutral_threshold": cfg["neutral_threshold"],
            "source": source,
            "total_articles": len(analyzed_df),
            "total_daily_records": len(daily_df),
            "total_joined_records": len(joined_df),
        },
    }

    analytics_path = output_dir / "sentiment_analytics.json"
    with open(analytics_path, "w", encoding="utf-8") as f:
        json.dump(analytics_summary, f, indent=2, default=str)
    logger.info(f"Saved sentiment analytics summary to {analytics_path}")

    # ── Step 7: MongoDB Storage ────────────────────────────────
    logger.info("--- Step 6: MongoDB Sentiment Storage ---")
    mongo_stored = store_sentiment_in_mongo(
        analyzed_df, daily_df, use_mock=use_mock_mongo
    )

    # ── Step 8: Print Summary ──────────────────────────────────
    _print_summary(analyzed_df, daily_df, joined_df, correlations, regime_stats,
                   stock_breakdown, analyzer, mongo_stored)

    return analytics_summary


def _print_summary(
    analyzed_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    joined_df: pd.DataFrame,
    correlations: Dict[str, Any],
    regime_stats: Dict[str, Any],
    stock_breakdown: pd.DataFrame,
    analyzer,
    mongo_stored: bool,
) -> None:
    """Print formatted pipeline results."""
    print("\n" + "=" * 80)
    print("PHASE 5: SENTIMENT ANALYSIS RESULTS SUMMARY")
    print("=" * 80)

    # A. Overview
    dist = analyzed_df["sentiment_label"].value_counts()
    print(f"\n[A] SENTIMENT OVERVIEW (Model: {analyzer.model_name}):")
    print(f"  Total articles analyzed  : {len(analyzed_df)}")
    print(f"  Daily aggregated records : {len(daily_df)}")
    print(f"  Market-joined records    : {len(joined_df)}")
    print(f"  {'Label':<12} | {'Count':>6} | {'Percent':>8}")
    print(f"  {'-' * 35}")
    for label in ["Positive", "Neutral", "Negative"]:
        cnt = dist.get(label, 0)
        pct = cnt / len(analyzed_df) * 100 if len(analyzed_df) > 0 else 0
        print(f"  {label:<12} | {cnt:>6} | {pct:>7.1f}%")

    # B. Stock Breakdown
    print(f"\n[B] SENTIMENT BY STOCK:")
    print(f"  {'Symbol':<12} | {'Articles':>8} | {'Mean Score':>10} | {'Dominant':>10}")
    print(f"  {'-' * 50}")
    for _, row in stock_breakdown.head(10).iterrows():
        print(
            f"  {row['symbol']:<12} | {row['total_articles']:>8} | "
            f"{row['mean_sentiment']:>10.4f} | {row['dominant_regime']:>10}"
        )

    # C. Correlations
    same = correlations["same_day_correlation"]
    nxt = correlations["next_day_correlation"]
    print(f"\n[C] SENTIMENT-RETURN CORRELATIONS:")
    sig_same = "[Significant]" if same.get("statistically_significant") else "[Not significant]"
    sig_nxt = "[Significant]" if nxt.get("statistically_significant") else "[Not significant]"
    print(f"  Same-Day  : Pearson={same['pearson']:+.4f} (p={same.get('p_value', 'N/A')})"
          f"  n={same['sample_size']}  {sig_same}")
    print(f"  Next-Day  : Pearson={nxt['pearson']:+.4f} (p={nxt.get('p_value', 'N/A')})"
          f"  n={nxt['sample_size']}  {sig_nxt}")

    # D. Regime Statistics
    if regime_stats:
        print(f"\n[D] RETURN DISTRIBUTION BY SENTIMENT REGIME:")
        print(f"  {'Regime':<10} | {'Days':>5} | {'Same-Day Mean':>14} | {'Next-Day Mean':>14} | {'Win Rate':>9}")
        print(f"  {'-' * 65}")
        for regime in ["Positive", "Neutral", "Negative"]:
            if regime in regime_stats:
                r = regime_stats[regime]
                sd = r["same_day_return"]
                nd = r["next_day_return"]
                print(
                    f"  {regime:<10} | {r['day_count']:>5} | "
                    f"{sd['mean']:>+14.6f} | {nd['mean']:>+14.6f} | "
                    f"{sd['win_rate']:>8.1%}"
                )

    # E. Integration
    print(f"\n[E] OUTPUTS & INTEGRATION:")
    print(f"  CSV files saved to   : data/processed/sentiment/")
    print(f"  Analytics JSON       : data/processed/sentiment/sentiment_analytics.json")
    print(f"  MongoDB Ingestion    : {'Completed' if mongo_stored else 'Skipped (MongoDB offline)'}")

    # F. Disclaimer
    print(f"\n[F] ACADEMIC DISCLAIMER:")
    print(f"  {correlations.get('disclaimer', 'Correlation does not imply causation.')}")
    print(f"  This analysis is for educational purposes only and does not constitute")
    print(f"  financial advice or a reliable prediction system.")
    print("=" * 80 + "\n")


# =====================================================================
# Entry Point
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run Phase 5 Sentiment Analysis Pipeline."
    )
    parser.add_argument(
        "--source",
        choices=["processed", "sample"],
        default="processed",
        help="Data source: 'processed' (default) or 'sample'.",
    )
    parser.add_argument(
        "--use-finbert",
        action="store_true",
        help="Use FinBERT transformer model (requires transformers + torch).",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mongomock for MongoDB storage (no real MongoDB needed).",
    )
    args = parser.parse_args()

    run_sentiment_pipeline(
        source=args.source,
        use_finbert=args.use_finbert,
        use_mock_mongo=args.mock,
    )


if __name__ == "__main__":
    main()
