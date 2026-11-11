"""
Sentiment Analysis Core for Financial News.

Provides:
- BaseSentimentAnalyzer interface
- FinancialLexiconAnalyzer: 100% offline, CPU-friendly deterministic financial engine
- FinBERTSentimentAnalyzer: Contextual transformer pipeline (ProsusAI/finbert)
- Factory get_sentiment_analyzer with automatic graceful fallback
- News text preprocessor and batch DataFrame analyzer
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd

from src.nlp.lexicon import score_financial_text
from src.utils.logger import logger


# =====================================================================
# 1. Text Preprocessing Helpers
# =====================================================================

def prepare_news_text(
    title: Optional[str] = None,
    description: Optional[str] = None,
) -> str:
    """
    Safely combine headline and description for sentiment processing.

    Handles:
    - title only
    - description only
    - title + description
    - None / NaN / empty strings

    Parameters
    ----------
    title : str, optional
        News headline.
    description : str, optional
        News body or lead paragraph.

    Returns
    -------
    str
        Clean combined text string.
    """
    t = str(title).strip() if pd.notna(title) and title is not None else ""
    d = str(description).strip() if pd.notna(description) and description is not None else ""

    if t and d:
        # Check if description already repeats title
        if d.startswith(t):
            combined = d
        else:
            sep = " " if t.endswith((".", "!", "?", ":")) else ". "
            combined = f"{t}{sep}{d}"
    elif t:
        combined = t
    elif d:
        combined = d
    else:
        combined = ""

    return combined


# =====================================================================
# 2. Base & Specialized Analyzers
# =====================================================================

class BaseSentimentAnalyzer(ABC):
    """Abstract base class for financial sentiment analyzers."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Name/identifier of the underlying sentiment model."""
        pass

    @abstractmethod
    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Analyze a single text string.

        Returns
        -------
        dict
            {
                "sentiment_score": float,  # [-1.0, 1.0]
                "sentiment_label": str,    # 'Positive' | 'Neutral' | 'Negative'
                "confidence": float,       # [0.0, 1.0]
                "sentiment_model": str
            }
        """
        pass

    def analyze_batch(self, texts: List[str], batch_size: int = 32) -> List[Dict[str, Any]]:
        """Batch process a list of text strings."""
        return [self.analyze(t) for t in texts]


class FinancialLexiconAnalyzer(BaseSentimentAnalyzer):
    """
    Lightweight, deterministic financial sentiment analyzer based on the
    Loughran-McDonald financial dictionary and contextual negation weighting.
    """

    def __init__(self, neutral_threshold: float = 0.15):
        self.neutral_threshold = neutral_threshold
        self._model_name = "financial_lexicon"

    @property
    def model_name(self) -> str:
        return self._model_name

    def analyze(self, text: str) -> Dict[str, Any]:
        res = score_financial_text(text, neutral_threshold=self.neutral_threshold)
        res["sentiment_model"] = self.model_name
        return res


class FinBERTSentimentAnalyzer(BaseSentimentAnalyzer):
    """
    Transformer-based financial sentiment analyzer using ProsusAI/finbert.

    Falls back to FinancialLexiconAnalyzer if transformers or model weights
    are unavailable.
    """

    def __init__(
        self,
        model_name: str = "ProsusAI/finbert",
        neutral_threshold: float = 0.15,
        max_length: int = 512,
    ):
        self._model_name = model_name
        self.neutral_threshold = neutral_threshold
        self.max_length = max_length
        self._pipeline = None
        self._fallback = FinancialLexiconAnalyzer(neutral_threshold=neutral_threshold)
        self._init_transformer()

    def _init_transformer(self) -> None:
        """Attempt to load transformers pipeline on CPU."""
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

            logger.info(f"Loading transformer model '{self._model_name}' on CPU...")
            tokenizer = AutoTokenizer.from_pretrained(self._model_name)
            model = AutoModelForSequenceClassification.from_pretrained(self._model_name)
            self._pipeline = pipeline(
                "sentiment-analysis",
                model=model,
                tokenizer=tokenizer,
                device=-1,  # Force CPU
                top_k=None,  # Return all class probabilities
                max_length=self.max_length,
                truncation=True,
            )
            logger.info(f"Successfully loaded FinBERT model '{self._model_name}'.")
        except Exception as e:
            logger.warning(
                f"Could not load transformer model '{self._model_name}' ({e}). "
                f"Falling back to local FinancialLexiconAnalyzer."
            )
            self._pipeline = None

    @property
    def model_name(self) -> str:
        return self._model_name if self._pipeline is not None else "financial_lexicon_fallback"

    def analyze(self, text: str) -> Dict[str, Any]:
        if self._pipeline is None or not text.strip():
            res = self._fallback.analyze(text)
            res["sentiment_model"] = self.model_name
            return res

        try:
            results = self._pipeline(text)[0]  # list of dicts: [{'label': 'positive', 'score': 0.8}, ...]
            probs = {item["label"].lower(): item["score"] for item in results}

            p_pos = probs.get("positive", 0.0)
            p_neg = probs.get("negative", 0.0)
            p_neu = probs.get("neutral", 0.0)

            # Continuous normalized score in [-1.0, 1.0]
            score = round(float(p_pos - p_neg), 4)

            # Assign label with neutral threshold or argmax
            if score > self.neutral_threshold:
                label = "Positive"
                conf = p_pos
            elif score < -self.neutral_threshold:
                label = "Negative"
                conf = p_neg
            else:
                label = "Neutral"
                conf = p_neu

            return {
                "sentiment_score": score,
                "sentiment_label": label,
                "confidence": round(float(conf), 4),
                "positive_score": round(float(p_pos), 4),
                "negative_score": round(float(p_neg), 4),
                "sentiment_model": self.model_name,
            }
        except Exception as e:
            logger.warning(f"Error during FinBERT inference ({e}). Using lexicon fallback.")
            res = self._fallback.analyze(text)
            res["sentiment_model"] = "financial_lexicon_fallback"
            return res

    def analyze_batch(self, texts: List[str], batch_size: int = 32) -> List[Dict[str, Any]]:
        if self._pipeline is None:
            return [self.analyze(t) for t in texts]

        outputs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            for text in batch:
                outputs.append(self.analyze(text))
        return outputs


# =====================================================================
# 3. Factory & Batch DataFrame Processing
# =====================================================================

def get_sentiment_analyzer(
    use_finbert: bool = False,
    model_name: str = "ProsusAI/finbert",
    neutral_threshold: float = 0.15,
) -> BaseSentimentAnalyzer:
    """
    Factory to retrieve configured sentiment analyzer.

    Parameters
    ----------
    use_finbert : bool, default False
        If True, attempts to initialize FinBERT transformer pipeline.
        If False, uses the fast CPU-friendly FinancialLexiconAnalyzer.
    model_name : str
        HuggingFace model ID.
    neutral_threshold : float
        Neutral score boundary in [-threshold, +threshold].
    """
    if use_finbert:
        return FinBERTSentimentAnalyzer(
            model_name=model_name,
            neutral_threshold=neutral_threshold,
        )
    return FinancialLexiconAnalyzer(neutral_threshold=neutral_threshold)


def analyze_news_dataframe(
    df: pd.DataFrame,
    analyzer: Optional[BaseSentimentAnalyzer] = None,
    use_finbert: bool = False,
    neutral_threshold: float = 0.15,
) -> pd.DataFrame:
    """
    Run sentiment analysis over a news DataFrame and append sentiment columns.

    Parameters
    ----------
    df : pd.DataFrame
        News DataFrame containing 'date', 'symbol', and optionally 'title', 'description'.
    analyzer : BaseSentimentAnalyzer, optional
        Pre-initialized sentiment analyzer instance.
    use_finbert : bool, default False
        Flag passed to factory if analyzer is None.
    neutral_threshold : float, default 0.15
        Neutral threshold passed to factory.

    Returns
    -------
    pd.DataFrame
        DataFrame with appended columns:
        ['combined_text', 'sentiment_score', 'sentiment_label', 'confidence', 'sentiment_model']
    """
    if df.empty:
        res = df.copy()
        for col in ["combined_text", "sentiment_score", "sentiment_label", "confidence", "sentiment_model"]:
            res[col] = []
        return res

    active_analyzer = analyzer or get_sentiment_analyzer(
        use_finbert=use_finbert,
        neutral_threshold=neutral_threshold,
    )
    logger.info(f"Analyzing {len(df)} news articles using model '{active_analyzer.model_name}'...")

    data = df.copy()

    # Create combined text
    combined_texts = []
    has_title = "title" in data.columns
    has_desc = "description" in data.columns

    for _, row in data.iterrows():
        t = row["title"] if has_title else ""
        d = row["description"] if has_desc else ""
        combined_texts.append(prepare_news_text(t, d))

    data["combined_text"] = combined_texts

    # Batch sentiment inference
    sentiment_results = active_analyzer.analyze_batch(combined_texts)

    data["sentiment_score"] = [r["sentiment_score"] for r in sentiment_results]
    data["sentiment_label"] = [r["sentiment_label"] for r in sentiment_results]
    data["confidence"] = [r["confidence"] for r in sentiment_results]
    data["sentiment_model"] = [r["sentiment_model"] for r in sentiment_results]

    logger.info(
        f"Completed sentiment analysis: "
        f"Positive={sum(data['sentiment_label'] == 'Positive')}, "
        f"Neutral={sum(data['sentiment_label'] == 'Neutral')}, "
        f"Negative={sum(data['sentiment_label'] == 'Negative')}"
    )

    return data
