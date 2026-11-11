"""
Daily Sentiment Aggregation and Market Return Integration.

Aggregates article-level sentiment by stock symbol and date,
and performs joins with market OHLCV returns, volatility, and volume
with strict lead-lag (no lookahead) alignment.
"""

from typing import Optional
import numpy as np
import pandas as pd

from src.utils.logger import logger


# =====================================================================
# 1. Daily Sentiment Aggregation
# =====================================================================

def aggregate_daily_sentiment(article_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate article sentiment by (symbol, date).

    Parameters
    ----------
    article_df : pd.DataFrame
        DataFrame containing 'symbol', 'date', 'sentiment_score', 'sentiment_label'.

    Returns
    -------
    pd.DataFrame
        Daily aggregated DataFrame with columns:
        [
            'date', 'symbol', 'article_count', 'average_sentiment',
            'positive_count', 'negative_count', 'neutral_count',
            'positive_ratio', 'negative_ratio', 'neutral_ratio',
            'sentiment_strength'
        ]
    """
    required = {"symbol", "date", "sentiment_score", "sentiment_label"}
    missing = required - set(article_df.columns)
    if missing:
        raise ValueError(f"Missing required columns for aggregation: {sorted(missing)}")

    if article_df.empty:
        return pd.DataFrame(columns=[
            "date", "symbol", "article_count", "average_sentiment",
            "positive_count", "negative_count", "neutral_count",
            "positive_ratio", "negative_ratio", "neutral_ratio",
            "sentiment_strength",
        ])

    df = article_df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

    records = []
    for (symbol, date), group in df.groupby(["symbol", "date"]):
        count = len(group)
        avg_score = float(group["sentiment_score"].mean())

        pos_cnt = int((group["sentiment_label"] == "Positive").sum())
        neg_cnt = int((group["sentiment_label"] == "Negative").sum())
        neu_cnt = int((group["sentiment_label"] == "Neutral").sum())

        pos_ratio = round(pos_cnt / count, 4)
        neg_ratio = round(neg_cnt / count, 4)
        neu_ratio = round(neu_cnt / count, 4)

        # Sentiment strength reflects directional conviction weighted by non-neutral share
        # strength = |avg_sentiment| * (1 - neutral_ratio)
        strength = round(abs(avg_score) * (1.0 - neu_ratio), 4)

        records.append({
            "date": date,
            "symbol": symbol,
            "article_count": count,
            "average_sentiment": round(avg_score, 4),
            "positive_count": pos_cnt,
            "negative_count": neg_cnt,
            "neutral_count": neu_cnt,
            "positive_ratio": pos_ratio,
            "negative_ratio": neg_ratio,
            "neutral_ratio": neu_ratio,
            "sentiment_strength": strength,
        })

    daily_df = pd.DataFrame(records).sort_values(["symbol", "date"]).reset_index(drop=True)
    logger.info(f"Aggregated {len(article_df)} articles into {len(daily_df)} daily stock sentiment records.")
    return daily_df


# =====================================================================
# 2. Market + Sentiment Join (Lead-Lag, No Leakage)
# =====================================================================

def join_sentiment_with_market(
    daily_sentiment_df: pd.DataFrame,
    market_df: pd.DataFrame,
    how: str = "inner",
) -> pd.DataFrame:
    """
    Join daily sentiment records with market returns and volatility.

    CRITICAL DATA LEAKAGE PREVENTION:
    - Same-day pairing: sentiment[t] matched with daily_return[t] for contemporaneous impact.
    - Predictive pairing: next_day_return[t] is strictly market daily_return[t+1]
      calculated within the full chronological market timeline per symbol.
    - Day t sentiment NEVER uses information from t+1.

    Parameters
    ----------
    daily_sentiment_df : pd.DataFrame
        Daily aggregated sentiment DataFrame.
    market_df : pd.DataFrame
        Market features DataFrame (must contain 'date', 'symbol', 'daily_return', 'log_return', 'volume', 'volatility').
    how : str, default 'inner'
        Join type ('inner' or 'left').

    Returns
    -------
    pd.DataFrame
        Joined DataFrame with aligned market metrics and next-day return.
    """
    req_market = {"date", "symbol", "daily_return", "log_return", "volume", "volatility"}
    missing = req_market - set(market_df.columns)
    if missing:
        raise ValueError(f"Missing required market columns for join: {sorted(missing)}")

    # Standardize market dates and calculate next_day_return on full market timeline
    mkt = market_df.copy()
    mkt["date"] = pd.to_datetime(mkt["date"]).dt.strftime("%Y-%m-%d")
    mkt = mkt.sort_values(["symbol", "date"]).reset_index(drop=True)

    # Next day return: daily_return shifted by -1 per stock symbol
    mkt["next_day_return"] = mkt.groupby("symbol")["daily_return"].shift(-1)

    # Standardize sentiment dates
    snt = daily_sentiment_df.copy()
    snt["date"] = pd.to_datetime(snt["date"]).dt.strftime("%Y-%m-%d")

    # Perform join
    market_cols = ["date", "symbol", "daily_return", "log_return", "volume", "volatility", "next_day_return"]
    joined = pd.merge(snt, mkt[market_cols], on=["symbol", "date"], how=how)

    joined = joined.sort_values(["symbol", "date"]).reset_index(drop=True)
    logger.info(
        f"Joined {len(snt)} daily sentiment records with market features: "
        f"{len(joined)} paired observations produced ({how} join)."
    )
    return joined
