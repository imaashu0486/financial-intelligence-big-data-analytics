"""
Lexical overlap reranker for the RAG pipeline.

Combines semantic similarity scores (from embedding vector search) with
lexical/token-overlap scores (BM25-inspired) for hybrid ranking.

Why reranking helps:
- Embedding models capture semantic meaning but may miss exact keywords (e.g. specific numbers, tickers).
- Lexical overlap captures exact keyword presence.
- Hybrid combination balances semantic recall with exact-match precision.
"""

import re
from typing import List, Set
from dataclasses import replace

from src.rag.vectorstore import SearchResult
from src.utils.logger import logger


def _tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase alphanumeric words."""
    return re.findall(r"\b\w+\b", text.lower())


def compute_lexical_score(query_tokens: Set[str], chunk_text: str) -> float:
    """
    Compute lexical overlap score between query tokens and document chunk.

    Parameters
    ----------
    query_tokens : Set[str]
        Set of unique words in query.
    chunk_text : str
        Document chunk content.

    Returns
    -------
    float
        Overlap score in range [0.0, 1.0].
    """
    if not query_tokens:
        return 0.0

    chunk_tokens = set(_tokenize(chunk_text))
    if not chunk_tokens:
        return 0.0

    overlap = query_tokens.intersection(chunk_tokens)
    # Jaccard-like or recall score: fraction of query terms present in chunk
    score = len(overlap) / len(query_tokens)
    return float(score)


def lexical_rerank(
    query: str,
    results: List[SearchResult],
    alpha: float = 0.5,
) -> List[SearchResult]:
    """
    Rerank search results using a hybrid of semantic and lexical overlap scores.

    combined_score = alpha * semantic_score + (1 - alpha) * lexical_score

    Parameters
    ----------
    query : str
        The search question / query.
    results : List[SearchResult]
        Initial results from vector search.
    alpha : float, default=0.5
        Weight for semantic score vs lexical score (0.0 = pure lexical, 1.0 = pure semantic).

    Returns
    -------
    List[SearchResult]
        Reranked results sorted by descending combined score with updated ranks.
    """
    if not results:
        return []

    # Clamp alpha to [0, 1]
    alpha = max(0.0, min(1.0, float(alpha)))

    query_tokens = set(_tokenize(query))

    reranked: List[SearchResult] = []
    for res in results:
        lex_score = compute_lexical_score(query_tokens, res.chunk.text)
        combined = alpha * float(res.score) + (1.0 - alpha) * lex_score

        # Create new SearchResult with combined score
        reranked.append(
            SearchResult(
                chunk=res.chunk,
                score=float(combined),
                rank=0,  # will update after sort
            )
        )

    # Sort descending by combined score
    reranked.sort(key=lambda x: x.score, reverse=True)

    # Assign new ranks (1-indexed)
    for i, item in enumerate(reranked, start=1):
        item.rank = i

    logger.debug(
        f"Reranked {len(reranked)} results with alpha={alpha:.2f} "
        f"(top score: {reranked[0].score:.4f})"
    )
    return reranked
