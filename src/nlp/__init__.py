"""
Natural Language Processing and Sentiment Analysis Layer for Financial Intelligence.

Exports:
- Sentiment analysis interfaces (FinancialLexiconAnalyzer, FinBERTSentimentAnalyzer)
- Text preparation and batch processing
- Daily aggregation and market return integration
- Statistical correlation and regime analytics
"""

from src.nlp.lexicon import score_financial_text
from src.nlp.sentiment import (
    BaseSentimentAnalyzer,
    FinancialLexiconAnalyzer,
    FinBERTSentimentAnalyzer,
    analyze_news_dataframe,
    get_sentiment_analyzer,
    prepare_news_text,
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

__all__ = [
    "score_financial_text",
    "BaseSentimentAnalyzer",
    "FinancialLexiconAnalyzer",
    "FinBERTSentimentAnalyzer",
    "analyze_news_dataframe",
    "get_sentiment_analyzer",
    "prepare_news_text",
    "aggregate_daily_sentiment",
    "join_sentiment_with_market",
    "calculate_sentiment_correlations",
    "calculate_regime_return_statistics",
    "calculate_stock_sentiment_breakdown",
    "generate_visualization_payload",
]
