"""
News and Corporate Document Intelligence Engine (Step 8 Phase D).

Extracts defensible intelligence from 2024 official news disclosures and
corporate annual report executive extracts:
1. Company-wise disclosure frequency and category distribution.
2. Lexicon-based financial sentiment scoring with confidence bounds.
3. Temporal association between disclosures and observed same-day/next-day returns.
4. Corporate filing structural summary across available FY24 extracts.
5. Explicit academic limitations and non-causality disclosures.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from src.nlp.lexicon import score_financial_text

logger = logging.getLogger(__name__)

EVENT_CATEGORIES = {
    "Capital_Raising_and_Capex": ["qip", "capex", "capital", "raise", "raises", "funding", "placement", "borrowing", "issuance"],
    "Earnings_and_Results": ["profit", "net profit", "quarterly", "earnings", "revenue", "ebitda", "pat", "margin", "results"],
    "Corporate_Action_and_Regulatory": ["sebi", "cci", "merger", "demerger", "dividend", "split", "drhp", "ipo", "regulatory", "filing"],
    "Contracts_and_Expansion": ["order", "contract", "deal", "agreement", "partnership", "client", "expansion", "commissioned", "solar", "wind"],
}


def classify_event_category(text: str) -> str:
    lower_t = text.lower()
    for cat, kws in EVENT_CATEGORIES.items():
        if any(kw in lower_t for kw in kws):
            return cat
    return "General_Corporate_Announcement"


def analyze_news_and_documents(
    news_csv: Path,
    market_parquet: Path,
    docs_dir: Path,
) -> Dict[str, Any]:
    """
    Execute comprehensive news and document intelligence analytics.
    """
    # 1. Load and process news
    df_news = pd.read_csv(news_csv)
    logger.info(f"Loaded {len(df_news)} news records from {news_csv}")

    scored_records = []
    for _, row in df_news.iterrows():
        title = str(row.get("headline", ""))
        snippet = str(row.get("description_or_snippet", ""))
        combined = f"{title}. {snippet}" if snippet else title

        sentiment = score_financial_text(combined)
        category = classify_event_category(combined)

        scored_records.append({
            "article_id": row.get("article_id"),
            "ticker": row.get("ticker"),
            "company_name": row.get("company_name"),
            "publication_date": str(row.get("publication_date")),
            "headline": title,
            "category": category,
            "sentiment_score": sentiment["sentiment_score"],
            "sentiment_label": sentiment["sentiment_label"],
            "confidence": sentiment["confidence"],
            "positive_score": sentiment.get("positive_score", 0.0),
            "negative_score": sentiment.get("negative_score", 0.0),
            "tokens_matched": sentiment.get("tokens_matched", 0),
        })

    df_scored = pd.DataFrame(scored_records)

    # 2. Company disclosure frequency
    disclosure_counts = (
        df_scored.groupby("ticker")
        .agg(
            total_disclosures=("article_id", "count"),
            positive_count=("sentiment_label", lambda s: (s == "Positive").sum()),
            neutral_count=("sentiment_label", lambda s: (s == "Neutral").sum()),
            negative_count=("sentiment_label", lambda s: (s == "Negative").sum()),
            mean_sentiment=("sentiment_score", "mean"),
        )
        .reset_index()
        .sort_values("total_disclosures", ascending=False)
    )

    # 3. Category distribution
    cat_dist = df_scored["category"].value_counts().to_dict()

    # 4. Temporal association with market returns
    df_market = pd.read_parquet(market_parquet)
    merged = pd.merge(
        df_scored,
        df_market[["date", "symbol", "daily_adjusted_return", "target_next_day_return"]],
        left_on=["publication_date", "ticker"],
        right_on=["date", "symbol"],
        how="inner",
    )

    same_day_corr = float(merged["sentiment_score"].corr(merged["daily_adjusted_return"])) if len(merged) > 3 else 0.0
    next_day_corr = float(merged["sentiment_score"].corr(merged["target_next_day_return"])) if len(merged) > 3 else 0.0

    # 5. Corporate document overview
    doc_summaries = []
    if docs_dir.exists():
        for doc_file in sorted(docs_dir.glob("*.txt")):
            content = doc_file.read_text(encoding="utf-8", errors="ignore")
            words = content.split()
            doc_summaries.append({
                "filename": doc_file.name,
                "word_count": len(words),
                "character_count": len(content),
                "key_themes": [
                    w for w in ["transformation", "growth", "digital", "governance", "capital", "strategy", "renewable"]
                    if w in content.lower()
                ],
            })

    report = {
        "dataset_scope": {
            "total_news_records": len(df_news),
            "date_range": f"{df_scored['publication_date'].min()} to {df_scored['publication_date'].max()}",
            "instrument_coverage_count": int(df_scored['ticker'].nunique()),
            "corporate_documents_analyzed": len(doc_summaries),
        },
        "event_category_distribution": cat_dist,
        "company_disclosure_summary": disclosure_counts.to_dict(orient="records"),
        "sentiment_distribution": {
            "Positive": int((df_scored["sentiment_label"] == "Positive").sum()),
            "Neutral": int((df_scored["sentiment_label"] == "Neutral").sum()),
            "Negative": int((df_scored["sentiment_label"] == "Negative").sum()),
            "mean_score": round(float(df_scored["sentiment_score"].mean()), 4),
        },
        "market_return_temporal_association": {
            "aligned_observation_count": len(merged),
            "same_day_correlation": round(same_day_corr, 4) if not np.isnan(same_day_corr) else 0.0,
            "next_day_correlation": round(next_day_corr, 4) if not np.isnan(next_day_corr) else 0.0,
            "statistical_caveat": "Correlation reflects temporal co-movement, not causality. Sample is restricted to 2024 disclosures.",
        },
        "corporate_filings_extract_summary": doc_summaries,
        "academic_limitations": [
            "News dataset is cross-sectional across calendar year 2024 and does not span the full 10-year market history.",
            "Text inputs consist of official headlines and lead snippets rather than full-length journalistic articles.",
            "Corporate filings are executive MD&A and strategic overview extracts, not full 400-page annual reports.",
            "Observed return-sentiment correlations must not be interpreted as predictive or causal signals.",
        ],
    }

    return report
