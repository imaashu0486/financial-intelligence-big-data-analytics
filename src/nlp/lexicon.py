"""
Financial Lexicon and Contextual Sentiment Engine.

Implements a domain-specific financial sentiment analyzer tailored for
market news, earnings releases, and analyst commentary using Loughran-McDonald
financial sentiment vocabulary, negation detection, and intensifier weighting.

Runs 100% locally and deterministically with zero external dependencies.
"""

import math
import re
from typing import Any, Dict, List, Set, Tuple


# =====================================================================
# Financial Sentiment Vocabularies (Loughran-McDonald adapted)
# =====================================================================

FINANCIAL_POSITIVE: Dict[str, float] = {
    "surge": 1.5, "surges": 1.5, "surged": 1.5, "surging": 1.5,
    "profit": 1.4, "profits": 1.4, "profitable": 1.4, "profitability": 1.4,
    "rally": 1.4, "rallies": 1.4, "rallied": 1.4, "rallying": 1.4,
    "outperform": 1.5, "outperforms": 1.5, "outperformed": 1.5, "outperforming": 1.5,
    "growth": 1.2, "grow": 1.1, "grows": 1.1, "growing": 1.1, "grew": 1.1,
    "gain": 1.2, "gains": 1.2, "gained": 1.2, "gaining": 1.2,
    "beat": 1.4, "beats": 1.4, "beating": 1.4,
    "exceed": 1.3, "exceeds": 1.3, "exceeded": 1.3, "exceeding": 1.3,
    "upgrade": 1.5, "upgrades": 1.5, "upgraded": 1.5, "upgrading": 1.5,
    "bullish": 1.5, "bull": 1.2, "bulls": 1.2,
    "dividend": 1.1, "dividends": 1.1,
    "rebound": 1.3, "rebounds": 1.3, "rebounded": 1.3,
    "breakthrough": 1.4, "record": 1.3, "boom": 1.4, "booming": 1.4,
    "strong": 1.2, "strength": 1.2, "strengthens": 1.2,
    "recovery": 1.3, "recovers": 1.3, "recovered": 1.3,
    "momentum": 1.1, "buyback": 1.3, "buybacks": 1.3,
    "expansion": 1.2, "expands": 1.2, "expanded": 1.2, "expanding": 1.2,
    "soaring": 1.5, "soar": 1.5, "soared": 1.5,
    "jump": 1.2, "jumps": 1.2, "jumped": 1.2,
    "upside": 1.3, "positive": 1.2, "optimistic": 1.3, "optimism": 1.3,
    "high": 1.0, "higher": 1.1, "highest": 1.2,
    "lucrative": 1.4, "win": 1.2, "wins": 1.2, "won": 1.2, "award": 1.1,
    "increase": 1.1, "increases": 1.1, "increased": 1.1, "increasing": 1.1,
    "boost": 1.3, "boosts": 1.3, "boosted": 1.3,
    "milestone": 1.2, "partnership": 1.1, "acquisition": 1.1,
}

FINANCIAL_NEGATIVE: Dict[str, float] = {
    "plunge": 1.5, "plunges": 1.5, "plunged": 1.5, "plunging": 1.5,
    "slump": 1.5, "slumps": 1.5, "slumped": 1.5, "slumping": 1.5,
    "crash": 1.6, "crashes": 1.6, "crashed": 1.6, "crashing": 1.6,
    "loss": 1.4, "losses": 1.4, "losing": 1.3, "lost": 1.3,
    "drop": 1.2, "drops": 1.2, "dropped": 1.2, "dropping": 1.2,
    "decline": 1.2, "declines": 1.2, "declined": 1.2, "declining": 1.2,
    "fall": 1.2, "falls": 1.2, "fell": 1.2, "falling": 1.2,
    "miss": 1.3, "misses": 1.3, "missed": 1.3, "missing": 1.3,
    "downgrade": 1.5, "downgrades": 1.5, "downgraded": 1.5, "downgrading": 1.5,
    "bearish": 1.5, "bear": 1.2, "bears": 1.2,
    "probe": 1.4, "probes": 1.4, "probed": 1.4,
    "investigation": 1.4, "investigated": 1.4, "investigating": 1.4,
    "lawsuit": 1.3, "lawsuits": 1.3, "sued": 1.3, "suing": 1.3,
    "penalty": 1.4, "penalties": 1.4, "fine": 1.3, "fined": 1.3, "fines": 1.3,
    "fraud": 1.6, "scam": 1.6, "default": 1.5, "defaults": 1.5, "defaulted": 1.5,
    "debt": 1.2, "debts": 1.2, "deficit": 1.3,
    "crisis": 1.5, "risk": 1.1, "risks": 1.1, "risky": 1.2,
    "cut": 1.2, "cuts": 1.2, "cutting": 1.2,
    "layoff": 1.4, "layoffs": 1.4, "laid": 1.2,
    "resign": 1.3, "resigns": 1.3, "resigned": 1.3, "resignation": 1.3,
    "warning": 1.3, "warns": 1.3, "warned": 1.3,
    "bankruptcy": 1.7, "bankrupt": 1.7, "insolvent": 1.6,
    "slowdown": 1.3, "slows": 1.1, "slowed": 1.1,
    "weak": 1.2, "weaker": 1.2, "weakest": 1.3, "weakness": 1.3,
    "selloff": 1.5, "selloffs": 1.5,
    "turmoil": 1.4, "setback": 1.3, "setbacks": 1.3,
    "underperform": 1.5, "underperforms": 1.5, "underperformed": 1.5,
    "negative": 1.2, "pessimistic": 1.3, "pessimism": 1.3,
    "low": 1.0, "lower": 1.1, "lowest": 1.2,
}

NEGATION_TERMS: Set[str] = {
    "not", "no", "never", "without", "barely", "hardly", "scarcely",
    "fails", "failed", "failing", "neither", "nor", "cannot", "couldnt", "didnt", "wont"
}

INTENSIFIERS: Dict[str, float] = {
    "substantially": 1.5, "substantive": 1.3,
    "significantly": 1.5, "significant": 1.3,
    "sharply": 1.5, "sharp": 1.3,
    "dramatically": 1.6, "dramatic": 1.4,
    "massively": 1.6, "massive": 1.4,
    "strongly": 1.4, "strong": 1.2,
    "heavily": 1.4, "heavy": 1.2,
    "widely": 1.2, "huge": 1.3,
    "slightly": 0.5, "marginal": 0.5, "marginally": 0.5, "partially": 0.6,
}


# =====================================================================
# Sentiment Evaluation Logic
# =====================================================================

def tokenize_financial_text(text: str) -> List[str]:
    """Clean and tokenize text preserving contractions and hyphenated terms."""
    if not isinstance(text, str):
        return []
    cleaned = text.lower()
    # Replace non-alphanumeric with spaces except apostrophes
    cleaned = re.sub(r"[^\w\s']", " ", cleaned)
    tokens = cleaned.split()
    return tokens


def score_financial_text(
    text: str,
    neutral_threshold: float = 0.15,
) -> Dict[str, Any]:
    """
    Score financial text using weighted lexicon, negation window, and intensifiers.

    Returns normalized score in [-1.0, +1.0] and label in {Positive, Neutral, Negative}.

    Parameters
    ----------
    text : str
        Input headline and/or description.
    neutral_threshold : float, default 0.15
        Boundary around 0 classified as 'Neutral'.

    Returns
    -------
    dict
        {
            "sentiment_score": float,       # [-1.0, 1.0]
            "sentiment_label": str,         # 'Positive' | 'Neutral' | 'Negative'
            "confidence": float,            # [0.0, 1.0]
            "positive_score": float,
            "negative_score": float,
            "tokens_matched": int
        }
    """
    tokens = tokenize_financial_text(text)
    if not tokens:
        return {
            "sentiment_score": 0.0,
            "sentiment_label": "Neutral",
            "confidence": 0.0,
            "positive_score": 0.0,
            "negative_score": 0.0,
            "tokens_matched": 0,
        }

    pos_total = 0.0
    neg_total = 0.0
    matched_count = 0

    n = len(tokens)
    for i, token in enumerate(tokens):
        # Look behind for negation up to 3 words
        negated = False
        lookback_start = max(0, i - 3)
        for j in range(lookback_start, i):
            if tokens[j] in NEGATION_TERMS:
                negated = True
                break

        # Look behind for intensifier up to 2 words
        multiplier = 1.0
        for j in range(max(0, i - 2), i):
            if tokens[j] in INTENSIFIERS:
                multiplier = INTENSIFIERS[tokens[j]]
                break

        if token in FINANCIAL_POSITIVE:
            w = FINANCIAL_POSITIVE[token] * multiplier
            matched_count += 1
            if negated:
                neg_total += w * 0.9  # Negated positive acts as negative
            else:
                pos_total += w

        elif token in FINANCIAL_NEGATIVE:
            w = FINANCIAL_NEGATIVE[token] * multiplier
            matched_count += 1
            if negated:
                pos_total += w * 0.8  # Negated negative acts as positive
            else:
                neg_total += w

    raw_diff = pos_total - neg_total

    # Non-linear hyperbolic tangent normalization ensuring smooth bounded [-1, 1]
    # S = raw_diff / sqrt(1 + (raw_diff / 2.5)^2) or tanh(raw_diff / 2.5)
    if matched_count == 0:
        norm_score = 0.0
        confidence = 0.50  # Neutral baseline confidence
    else:
        norm_score = math.tanh(raw_diff / 2.5)
        # Confidence increases with token evidence and score magnitude
        evidence_factor = min(1.0, matched_count / 3.0)
        confidence = 0.50 + 0.45 * abs(norm_score) * evidence_factor

    norm_score = round(norm_score, 4)
    confidence = round(confidence, 4)

    if norm_score > neutral_threshold:
        label = "Positive"
    elif norm_score < -neutral_threshold:
        label = "Negative"
    else:
        label = "Neutral"

    return {
        "sentiment_score": norm_score,
        "sentiment_label": label,
        "confidence": confidence,
        "positive_score": round(pos_total, 4),
        "negative_score": round(neg_total, 4),
        "tokens_matched": matched_count,
    }
