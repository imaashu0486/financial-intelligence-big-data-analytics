"""Tests for Phase 5 NLP & Financial Sentiment Analysis Layer."""

import numpy as np
import pandas as pd
import pytest
from src.nlp.aggregation import aggregate_daily_sentiment, join_sentiment_with_market
from src.nlp.analytics import (
    calculate_regime_return_statistics,
    calculate_sentiment_correlations,
    calculate_stock_sentiment_breakdown,
)
from src.nlp.lexicon import score_financial_text
from src.nlp.sentiment import (
    FinancialLexiconAnalyzer,
    analyze_news_dataframe,
    get_sentiment_analyzer,
    prepare_news_text,
)


# =====================================================================
# 1. Text Preparation & Edge Cases
# =====================================================================

class TestTextPreparation:
    """Unit tests for news text preparation and edge cases."""

    def test_prepare_news_text_both_title_and_description(self):
        title = "TCS beats profit estimates"
        desc = "The IT giant reported robust revenue growth."
        res = prepare_news_text(title, desc)
        assert res == "TCS beats profit estimates. The IT giant reported robust revenue growth."

    def test_prepare_news_text_already_has_punctuation(self):
        title = "RELIANCE expands into green energy."
        desc = "Investment exceeds 5000 crores."
        res = prepare_news_text(title, desc)
        assert res == "RELIANCE expands into green energy. Investment exceeds 5000 crores."

    def test_prepare_news_text_title_only(self):
        res = prepare_news_text("TCS reports Q3 results", None)
        assert res == "TCS reports Q3 results"

    def test_prepare_news_text_desc_only(self):
        res = prepare_news_text(None, "Robust revenue expansion noted across segments.")
        assert res == "Robust revenue expansion noted across segments."

    def test_prepare_news_text_empty_and_nan(self):
        assert prepare_news_text("", "") == ""
        assert prepare_news_text(np.nan, np.nan) == ""
        assert prepare_news_text(None, None) == ""


# =====================================================================
# 2. Financial Lexicon & Sentiment Scoring
# =====================================================================

class TestSentimentScoring:
    """Unit tests for financial sentiment lexicon and scoring normalization."""

    def test_positive_financial_sentiment(self):
        text = "TCS profits surge to record high following strong revenue growth and dividend hike."
        res = score_financial_text(text)
        assert res["sentiment_score"] > 0.30
        assert res["sentiment_label"] == "Positive"
        assert -1.0 <= res["sentiment_score"] <= 1.0

    def test_negative_financial_sentiment(self):
        text = "WIPRO shares plunge amid heavy losses, revenue drop, and regulatory investigation."
        res = score_financial_text(text)
        assert res["sentiment_score"] < -0.30
        assert res["sentiment_label"] == "Negative"
        assert -1.0 <= res["sentiment_score"] <= 1.0

    def test_neutral_financial_sentiment(self):
        text = "Board of directors will hold regular meeting next Tuesday at headquarters."
        res = score_financial_text(text)
        assert abs(res["sentiment_score"]) <= 0.15
        assert res["sentiment_label"] == "Neutral"

    def test_negation_inversion(self):
        # "profit" is positive, but "no profit" / "did not grow" should act as negative
        text_pos = "Company showed strong profit."
        text_neg = "Company showed no profit and failed to grow."
        res_pos = score_financial_text(text_pos)
        res_neg = score_financial_text(text_neg)
        assert res_pos["sentiment_score"] > 0
        assert res_neg["sentiment_score"] < 0

    def test_empty_text_returns_neutral(self):
        res = score_financial_text("")
        assert res["sentiment_score"] == 0.0
        assert res["sentiment_label"] == "Neutral"


# =====================================================================
# 3. Batch DataFrame Analysis
# =====================================================================

class TestDataFrameAnalysis:
    """Unit tests for batch DataFrame processing."""

    @pytest.fixture
    def sample_news_df(self) -> pd.DataFrame:
        return pd.DataFrame([
            {
                "date": "2023-05-01",
                "symbol": "INFY",
                "title": "INFY reports massive profit surge and dividend increase",
                "description": "Quarterly earnings beat all street estimates.",
                "source": "MoneyControl",
            },
            {
                "date": "2023-05-01",
                "symbol": "INFY",
                "title": "INFY faces regulatory penalty over compliance lapse",
                "description": "Investigation opened into internal controls.",
                "source": "Reuters",
            },
            {
                "date": "2023-05-02",
                "symbol": "TCS",
                "title": "TCS management holds routine annual conference",
                "description": "General discussions held with institutional holders.",
                "source": "Bloomberg",
            },
        ])

    def test_analyze_news_dataframe_columns(self, sample_news_df):
        analyzer = FinancialLexiconAnalyzer()
        res = analyze_news_dataframe(sample_news_df, analyzer=analyzer)

        expected_cols = {"combined_text", "sentiment_score", "sentiment_label", "confidence", "sentiment_model"}
        assert expected_cols.issubset(set(res.columns))
        assert len(res) == len(sample_news_df)
        assert (res["sentiment_model"] == "financial_lexicon").all()
        assert res.loc[0, "sentiment_label"] == "Positive"
        assert res.loc[1, "sentiment_label"] == "Negative"


# =====================================================================
# 4. Daily Aggregation
# =====================================================================

class TestDailyAggregation:
    """Unit tests for daily sentiment aggregation per (symbol, date)."""

    def test_aggregate_daily_sentiment_counts_and_ratios(self):
        article_df = pd.DataFrame([
            {"symbol": "RELIANCE", "date": "2023-06-01", "sentiment_score": 0.8, "sentiment_label": "Positive"},
            {"symbol": "RELIANCE", "date": "2023-06-01", "sentiment_score": -0.6, "sentiment_label": "Negative"},
            {"symbol": "RELIANCE", "date": "2023-06-01", "sentiment_score": 0.1, "sentiment_label": "Neutral"},
            {"symbol": "TCS", "date": "2023-06-01", "sentiment_score": 0.5, "sentiment_label": "Positive"},
        ])

        daily = aggregate_daily_sentiment(article_df)
        assert len(daily) == 2  # RELIANCE on 06-01 and TCS on 06-01

        rel = daily[daily["symbol"] == "RELIANCE"].iloc[0]
        assert rel["article_count"] == 3
        assert rel["positive_count"] == 1
        assert rel["negative_count"] == 1
        assert rel["neutral_count"] == 1
        assert np.isclose(rel["positive_ratio"] + rel["negative_ratio"] + rel["neutral_ratio"], 1.0, atol=1e-3)
        assert np.isclose(rel["average_sentiment"], (0.8 - 0.6 + 0.1) / 3, atol=1e-3)
        assert rel["sentiment_strength"] >= 0


# =====================================================================
# 5. Market + Sentiment Join & Leak-Free Lead-Lag Alignment
# =====================================================================

class TestMarketJoinAndAnalytics:
    """Unit tests for market return join, correlation, and regime statistics."""

    @pytest.fixture
    def sample_daily_sentiment(self) -> pd.DataFrame:
        return pd.DataFrame([
            {"date": "2023-01-02", "symbol": "SBIN", "average_sentiment": 0.6, "article_count": 2, "positive_ratio": 1.0, "negative_ratio": 0.0, "neutral_ratio": 0.0, "sentiment_strength": 0.6},
            {"date": "2023-01-03", "symbol": "SBIN", "average_sentiment": -0.5, "article_count": 1, "positive_ratio": 0.0, "negative_ratio": 1.0, "neutral_ratio": 0.0, "sentiment_strength": 0.5},
            {"date": "2023-01-04", "symbol": "SBIN", "average_sentiment": 0.0, "article_count": 1, "positive_ratio": 0.0, "negative_ratio": 0.0, "neutral_ratio": 1.0, "sentiment_strength": 0.0},
        ])

    @pytest.fixture
    def sample_market_data(self) -> pd.DataFrame:
        return pd.DataFrame([
            {"date": "2023-01-02", "symbol": "SBIN", "daily_return": 0.015, "log_return": 0.0148, "volume": 1_000_000, "volatility": 0.012},
            {"date": "2023-01-03", "symbol": "SBIN", "daily_return": -0.010, "log_return": -0.0101, "volume": 1_200_000, "volatility": 0.013},
            {"date": "2023-01-04", "symbol": "SBIN", "daily_return": 0.002, "log_return": 0.0020, "volume": 900_000, "volatility": 0.011},
            {"date": "2023-01-05", "symbol": "SBIN", "daily_return": 0.008, "log_return": 0.0079, "volume": 950_000, "volatility": 0.011},
        ])

    def test_join_sentiment_with_market_alignment(self, sample_daily_sentiment, sample_market_data):
        joined = join_sentiment_with_market(sample_daily_sentiment, sample_market_data)

        assert len(joined) == 3
        # Same-day return check on 2023-01-02
        assert np.isclose(joined.loc[0, "daily_return"], 0.015)

        # Lead-lag check: next_day_return on 2023-01-02 should equal daily_return on 2023-01-03 (-0.010)
        assert np.isclose(joined.loc[0, "next_day_return"], -0.010)

    def test_calculate_sentiment_correlations(self, sample_daily_sentiment, sample_market_data):
        joined = join_sentiment_with_market(sample_daily_sentiment, sample_market_data)
        corrs = calculate_sentiment_correlations(joined)

        assert "same_day_correlation" in corrs
        assert "next_day_correlation" in corrs
        assert "disclaimer" in corrs
        assert -1.0 <= corrs["same_day_correlation"]["pearson"] <= 1.0

    def test_calculate_regime_return_statistics(self, sample_daily_sentiment, sample_market_data):
        joined = join_sentiment_with_market(sample_daily_sentiment, sample_market_data)
        stats_dict = calculate_regime_return_statistics(joined, threshold=0.15)

        for regime in ["Positive", "Neutral", "Negative"]:
            assert regime in stats_dict
            assert "same_day_return" in stats_dict[regime]
            assert "next_day_return" in stats_dict[regime]
