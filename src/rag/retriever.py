"""
Retrieval orchestration for the RAG pipeline.

Coordinates embedding the query, performing vector similarity search,
and optionally applying lexical overlap reranking.
"""

import time
from dataclasses import dataclass
from typing import List, Optional

from src.rag.embeddings import EmbeddingModel
from src.rag.reranker import lexical_rerank
from src.rag.vectorstore import SearchResult, VectorStore
from src.utils.logger import logger


@dataclass
class RetrievalResult:
    """Container for retrieval outputs and metadata."""

    query: str
    results: List[SearchResult]
    retrieval_time_ms: float
    reranked: bool = False

    @property
    def has_matches(self) -> bool:
        """True if at least one document chunk was retrieved."""
        return len(self.results) > 0

    @property
    def top_chunk(self) -> Optional[SearchResult]:
        """Top-ranked search result or None."""
        return self.results[0] if self.results else None


class Retriever:
    """
    Orchestrates embedding-based search and optional reranking over a VectorStore.
    """

    def __init__(
        self,
        vectorstore: VectorStore,
        embedder: EmbeddingModel,
        top_k: int = 5,
        min_score: float = 0.3,
        rerank: bool = True,
        rerank_alpha: float = 0.5,
    ) -> None:
        """
        Parameters
        ----------
        vectorstore : VectorStore
            Populated vector index.
        embedder : EmbeddingModel
            Embedding model for query encoding.
        top_k : int, default=5
            Max results to return.
        min_score : float, default=0.3
            Minimum similarity score threshold.
        rerank : bool, default=True
            Whether to apply lexical reranking after vector search.
        rerank_alpha : float, default=0.5
            Weight between semantic (alpha) and lexical (1-alpha) scores.
        """
        self.vectorstore = vectorstore
        self.embedder = embedder
        self.top_k = top_k
        self.min_score = min_score
        self.rerank = rerank
        self.rerank_alpha = rerank_alpha

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
    ) -> RetrievalResult:
        """
        Retrieve relevant document chunks for a query.

        Parameters
        ----------
        query : str
            The input question or search query.
        top_k : Optional[int]
            Override default top_k if provided.
        min_score : Optional[float]
            Override default min_score if provided.

        Returns
        -------
        RetrievalResult
            Search results with timing and metadata.
        """
        k = top_k if top_k is not None else self.top_k
        score_thresh = min_score if min_score is not None else self.min_score

        start_t = time.perf_counter()

        if not query or not query.strip():
            return RetrievalResult(
                query=query,
                results=[],
                retrieval_time_ms=0.0,
                reranked=False,
            )

        if len(self.vectorstore) == 0:
            logger.warning("Vector store is empty, returning 0 results")
            return RetrievalResult(
                query=query,
                results=[],
                retrieval_time_ms=(time.perf_counter() - start_t) * 1000.0,
                reranked=False,
            )

        # 1. Encode query
        # Embedder encode accepts List[str], returns (N, D)
        query_vec = self.embedder.encode([query])

        # 2. Vector search - retrieve a larger candidate pool if reranking
        candidate_k = max(k * 2, 10) if self.rerank else k
        raw_results = self.vectorstore.search(
            query_embedding=query_vec[0],
            top_k=candidate_k,
            min_score=score_thresh,
        )

        # 3. Optional reranking
        did_rerank = False
        if self.rerank and len(raw_results) > 0:
            reranked_results = lexical_rerank(
                query=query,
                results=raw_results,
                alpha=self.rerank_alpha,
            )
            final_results = reranked_results[:k]
            did_rerank = True
        else:
            final_results = raw_results[:k]

        elapsed_ms = (time.perf_counter() - start_t) * 1000.0

        logger.info(
            f"Retrieved {len(final_results)} chunks for query: '{query[:40]}...' "
            f"in {elapsed_ms:.2f}ms (reranked={did_rerank})"
        )

        return RetrievalResult(
            query=query,
            results=final_results,
            retrieval_time_ms=elapsed_ms,
            reranked=did_rerank,
        )
