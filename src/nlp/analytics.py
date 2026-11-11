"""
Statistical Analytics and Visualization Preparation for Financial Sentiment.

Provides:
- Pearson & Spearman correlations between sentiment and same-day / next-day returns
- Return distribution profiles across sentiment regimes (Positive, Neutral, Negative)
- Sentiment coverage and frequency breakdowns by stock symbol
- Reusable visualization data generators ready for Streamlit (Phase 8)
- Explicit non-causality disclaimers and academic caveats
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from scipy import stats

from src.utils.logger import logger


# =====================================================================
# 1. Correlation Analysis
# =====================================================================

def calculate_sentiment_correlations(joined_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate Pearson and Spearman correlations between average sentiment and returns.

    Computes:
    - Contemporaneous correlation: average_sentiment[t] vs daily_return[t]
    - Predictive (lead-lag) correlation: average_sentiment[t] vs next_day_return[t]

    ACADEMIC DISCLAIMER:
    Correlation does NOT imply causation. Financial news sentiment and asset returns
    often co-move due to unobserved macroeconomic factors, earnings releases,
    or noise without news directly causing the price change.
    """
    if joined_df.empty:
        return {
            "same_day_correlation": {"pearson": 0.0, "p_value": 1.0, "spearman": 0.0, "sample_size": 0},
            "next_day_correlation": {"pearson": 0.0, "p_value": 1.0, "spearman": 0.0, "sample_size": 0},
            "disclaimer": "Correlation does not imply causation.",
        }

    # Contemporaneous (same-day)
    same_day_valid = joined_df.dropna(subset=["average_sentiment", "daily_return"])
    if len(same_day_valid) >= 3:
        r_same, p_same = stats.pearsonr(same_day_valid["average_sentiment"], same_day_valid["daily_return"])
        s_same, sp_same = stats.spearmanr(same_day_valid["average_sentiment"], same_day_valid["daily_return"])
        same_day_res = {
            "pearson": round(float(r_same), 4),
            "p_value": round(float(p_same), 6),
            "spearman": round(float(s_same), 4),
            "spearman_p_value": round(float(sp_same), 6),
            "sample_size": len(same_day_valid),
            "statistically_significant": bool(p_same < 0.05),
        }
    else:
        same_day_res = {"pearson": 0.0, "p_value": 1.0, "spearman": 0.0, "sample_size": len(same_day_valid), "statistically_significant": False}

    # Predictive lead-lag (next-day)
    next_day_valid = joined_df.dropna(subset=["average_sentiment", "next_day_return"])
    if len(next_day_valid) >= 3:
        r_next, p_next = stats.pearsonr(next_day_valid["average_sentiment"], next_day_valid["next_day_return"])
        s_next, sp_next = stats.spearmanr(next_day_valid["average_sentiment"], next_day_valid["next_day_return"])
        next_day_res = {
            "pearson": round(float(r_next), 4),
            "p_value": round(float(p_next), 6),
            "spearman": round(float(s_next), 4),
            "spearman_p_value": round(float(sp_next), 6),
            "sample_size": len(next_day_valid),
            "statistically_significant": bool(p_next < 0.05),
        }
    else:
        next_day_res = {"pearson": 0.0, "p_value": 1.0, "spearman": 0.0, "sample_size": len(next_day_valid), "statistically_significant": False}

    return {
        "same_day_correlation": same_day_res,
        "next_day_correlation": next_day_res,
        "disclaimer": (
            "Statistical association between sentiment scores and asset returns does not "
            "constitute evidence of causal influence or financial market predictability."
        ),
    }


# =====================================================================
# 2. Return Statistics by Sentiment Regime
# =====================================================================

def calculate_regime_return_statistics(
    joined_df: pd.DataFrame,
    threshold: float = 0.15,
) -> Dict[str, Any]:
    """
    Calculate return distribution metrics across sentiment categories:
    Positive (sentiment > +threshold), Negative (sentiment < -threshold), Neutral.

    Parameters
    ----------
    joined_df : pd.DataFrame
        Joined sentiment and market return DataFrame.
    threshold : float, default 0.15
        Regime boundary.

    Returns
    -------
    dict
        Summary statistics per regime for both same-day and next-day returns.
    """
    if joined_df.empty or "average_sentiment" not in joined_df.columns:
        return {}

    df = joined_df.copy()

    # Categorize daily sentiment
    conditions = [
        df["average_sentiment"] > threshold,
        df["average_sentiment"] < -threshold,
    ]
    choices = ["Positive", "Negative"]
    df["sentiment_category"] = np.select(conditions, choices, default="Neutral")

    categories = ["Positive", "Neutral", "Negative"]
    stats_dict: Dict[str, Any] = {}

    for cat in categories:
        subset = df[df["sentiment_category"] == cat]
        returns_same = subset["daily_return"].dropna()
        returns_next = subset["next_day_return"].dropna()

        stats_dict[cat] = {
            "day_count": len(subset),
            "same_day_return": {
                "mean": round(float(returns_same.mean()), 6) if not returns_same.empty else 0.0,
                "std": round(float(returns_same.std()), 6) if len(returns_same) > 1 else 0.0,
                "median": round(float(returns_same.median()), 6) if not returns_same.empty else 0.0,
                "win_rate": round(float((returns_same > 0).mean()), 4) if not returns_same.empty else 0.0,
            },
            "next_day_return": {
                "mean": round(float(returns_next.mean()), 6) if not returns_next.empty else 0.0,
                "std": round(float(returns_next.std()), 6) if len(returns_next) > 1 else 0.0,
                "median": round(float(returns_next.median()), 6) if not returns_next.empty else 0.0,
                "win_rate": round(float((returns_next > 0).mean()), 4) if not returns_next.empty else 0.0,
            },
        }

    return stats_dict


# =====================================================================
# 3. Stock Sentiment Frequency & Coverage
# =====================================================================

def calculate_stock_sentiment_breakdown(article_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize sentiment distribution per stock symbol.

    Returns
    -------
    pd.DataFrame
        Columns: ['symbol', 'total_articles', 'positive_count', 'neutral_count',
                  'negative_count', 'mean_sentiment', 'sentiment_label']
    """
    if article_df.empty:
        return pd.DataFrame()

    records = []
    for symbol, group in article_df.groupby("symbol"):
        total = len(group)
        pos = int((group["sentiment_label"] == "Positive").sum())
        neu = int((group["sentiment_label"] == "Neutral").sum())
        neg = int((group["sentiment_label"] == "Negative").sum())
        mean_score = float(group["sentiment_score"].mean())

        records.append({
            "symbol": symbol,
            "total_articles": total,
            "positive_count": pos,
            "neutral_count": neu,
            "negative_count": neg,
            "mean_sentiment": round(mean_score, 4),
            "dominant_regime": "Positive" if pos > max(neg, neu) else ("Negative" if neg > max(pos, neu) else "Neutral"),
        })

    return pd.DataFrame(records).sort_values("total_articles", ascending=False).reset_index(drop=True)


# =====================================================================
# 4. Visualization Data Generators (For Phase 8 Dashboard)
# =====================================================================

def generate_visualization_payload(
    article_df: pd.DataFrame,
    daily_sentiment_df: pd.DataFrame,
    joined_df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Generate pre-formatted data structures ready for rendering in Streamlit (Phase 8).
    """
    # 1. Overall sentiment label distribution
    dist = article_df["sentiment_label"].value_counts().to_dict() if not article_df.empty else {}

    # 2. Timeline series
    timeline = []
    if not daily_sentiment_df.empty:
        time_agg = daily_sentiment_df.groupby("date")["average_sentiment"].mean().reset_index()
        timeline = time_agg.rename(columns={"average_sentiment": "market_sentiment"}).to_dict(orient="records")

    # 3. Stock breakdown
    stock_summary = calculate_stock_sentiment_breakdown(article_df).to_dict(orient="records")

    # 4. Correlations
    corrs = calculate_sentiment_correlations(joined_df)

    # 5. Regime stats
    regimes = calculate_regime_return_statistics(joined_df)

    return {
        "sentiment_distribution": dist,
        "sentiment_timeline": timeline,
        "stock_breakdown": stock_summary,
        "correlations": corrs,
        "regime_statistics": regimes,
    }
