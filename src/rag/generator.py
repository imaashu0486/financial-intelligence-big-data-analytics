"""
Answer generation for the RAG pipeline.

Provides:
- ExtractiveGenerator: Offline, CPU-friendly extractive answer synthesizer
- MockGenerator: Deterministic fixture generator for unit testing
- Factory get_generator(mode=...)
- GeneratedAnswer data container with full provenance
"""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.rag.prompting import (
    FINANCIAL_ADVICE_DISCLAIMER,
    build_rag_prompt,
    format_citations,
    format_citation_dicts,
)
from src.rag.vectorstore import SearchResult
from src.utils.logger import logger

INSUFFICIENT_INFO_MSG = (
    "Based on the provided documents, there is insufficient information to answer this question."
)


@dataclass
class GeneratedAnswer:
    """Complete RAG response with answer text, citations, and provenance."""

    query: str
    answer: str
    sources: List[str]
    citations: List[Dict[str, Any]]
    retrieved_chunks: List[SearchResult] = field(repr=False)
    generator_mode: str
    confidence_score: float
    is_supported: bool
    disclaimer: str = FINANCIAL_ADVICE_DISCLAIMER

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization or MongoDB storage."""
        return {
            "query": self.query,
            "answer": self.answer,
            "sources": self.sources,
            "citations": self.citations,
            "generator_mode": self.generator_mode,
            "confidence_score": round(self.confidence_score, 4),
            "is_supported": self.is_supported,
            "disclaimer": self.disclaimer,
            "retrieved_chunk_count": len(self.retrieved_chunks),
        }


class BaseGenerator(ABC):
    """Abstract base class for RAG answer generators."""

    @abstractmethod
    def generate(
        self,
        query: str,
        retrieved_chunks: List[SearchResult],
    ) -> GeneratedAnswer:
        """Generate a grounded answer given a query and retrieved chunks."""
        pass


class ExtractiveGenerator(BaseGenerator):
    """
    Offline extractive generator that synthesizes answers directly from
    the highest-ranking retrieved context chunks without calling external APIs.
    """

    def __init__(self, min_confidence: float = 0.25) -> None:
        self.min_confidence = min_confidence

    def generate(
        self,
        query: str,
        retrieved_chunks: List[SearchResult],
    ) -> GeneratedAnswer:
        sources = format_citations(retrieved_chunks)
        citation_dicts = format_citation_dicts(retrieved_chunks)

        # Check if context is completely empty or score is below threshold
        if not retrieved_chunks or retrieved_chunks[0].score < self.min_confidence:
            return GeneratedAnswer(
                query=query,
                answer=INSUFFICIENT_INFO_MSG,
                sources=[],
                citations=[],
                retrieved_chunks=retrieved_chunks,
                generator_mode="extractive",
                confidence_score=0.0,
                is_supported=False,
            )

        top_result = retrieved_chunks[0]
        top_chunk = top_result.chunk
        confidence = float(top_result.score)

        # Extract query words excluding common question stopwords
        stopwords = {
            "what", "when", "where", "which", "who", "whom", "whose", "why", "how",
            "are", "were", "was", "is", "be", "been", "being", "the", "and", "for",
            "with", "from", "that", "this", "these", "those", "have", "has", "had",
            "does", "did", "can", "could", "should", "would", "about", "into", "over",
        }
        all_query_words = set(re.findall(r"\b\w{3,}\b", query.lower()))
        content_query_words = {w for w in all_query_words if w not in stopwords}

        sentences = re.split(r"(?<=[.?!])\s+", top_chunk.text.strip())

        scored_sentences = []
        for s in sentences:
            s_clean = s.strip()
            if not s_clean or len(s_clean) < 15:
                continue
            s_words = set(re.findall(r"\b\w{3,}\b", s_clean.lower()))
            overlap = len(content_query_words.intersection(s_words))
            scored_sentences.append((overlap, s_clean))

        # Sort by overlap, keep top 2-3 most relevant sentences
        scored_sentences.sort(key=lambda x: x[0], reverse=True)
        relevant_sentences = [s for score, s in scored_sentences if score > 0][:3]

        # Anti-hallucination guard 1: Check if topic words (excluding company tokens) appear in chunk
        company_tokens = {
            "acme", "globex", "initech", "company", "corp", "corporation",
            "ltd", "limited", "inc", "group", "report", "presentation",
        }
        topic_words = {w for w in content_query_words if w not in company_tokens}
        chunk_words = set(re.findall(r"\b\w{3,}\b", top_chunk.text.lower()))

        matched_topic_words = topic_words.intersection(chunk_words)
        if topic_words:
            topic_match_ratio = len(matched_topic_words) / len(topic_words)
            if len(matched_topic_words) == 0 or (len(topic_words) >= 3 and len(matched_topic_words) <= 1 and topic_match_ratio < 0.30):
                return GeneratedAnswer(
                    query=query,
                    answer=INSUFFICIENT_INFO_MSG,
                    sources=[],
                    citations=[],
                    retrieved_chunks=retrieved_chunks,
                    generator_mode="extractive",
                    confidence_score=confidence,
                    is_supported=False,
                )

        # Anti-hallucination guard 2: if query had specific content words but none matched any sentence
        if content_query_words and not relevant_sentences:
            return GeneratedAnswer(
                query=query,
                answer=INSUFFICIENT_INFO_MSG,
                sources=[],
                citations=[],
                retrieved_chunks=retrieved_chunks,
                generator_mode="extractive",
                confidence_score=confidence,
                is_supported=False,
            )

        if relevant_sentences:
            extracted_text = " ".join(relevant_sentences)
        else:
            extracted_text = top_chunk.text.strip()[:350]
            if len(top_chunk.text.strip()) > 350:
                extracted_text += "..."

        page_info = f"page {top_chunk.page_number}" if top_chunk.page_number is not None else "document"
        answer = (
            f"According to {top_chunk.document_name} ({page_info}):\n"
            f"\"{extracted_text}\"\n\n"
            f"Sources cited: {', '.join(sources[:2])}"
        )

        return GeneratedAnswer(
            query=query,
            answer=answer,
            sources=sources,
            citations=citation_dicts,
            retrieved_chunks=retrieved_chunks,
            generator_mode="extractive",
            confidence_score=confidence,
            is_supported=True,
        )


class MockGenerator(BaseGenerator):
    """
    Deterministic mock generator for unit tests.
    Returns predictable outputs without randomness or model dependencies.
    """

    def __init__(self, fixed_answer: Optional[str] = None) -> None:
        self.fixed_answer = fixed_answer

    def generate(
        self,
        query: str,
        retrieved_chunks: List[SearchResult],
    ) -> GeneratedAnswer:
        sources = format_citations(retrieved_chunks)
        citation_dicts = format_citation_dicts(retrieved_chunks)

        if not retrieved_chunks:
            return GeneratedAnswer(
                query=query,
                answer=INSUFFICIENT_INFO_MSG,
                sources=[],
                citations=[],
                retrieved_chunks=[],
                generator_mode="mock",
                confidence_score=0.0,
                is_supported=False,
            )

        if self.fixed_answer:
            ans_text = self.fixed_answer
        else:
            top = retrieved_chunks[0]
            ans_text = f"[MOCK ANSWER] Sourced from {top.chunk.document_name}: {top.chunk.text[:100]}..."

        return GeneratedAnswer(
            query=query,
            answer=ans_text,
            sources=sources,
            citations=citation_dicts,
            retrieved_chunks=retrieved_chunks,
            generator_mode="mock",
            confidence_score=retrieved_chunks[0].score,
            is_supported=True,
        )


def get_generator(mode: str = "extractive", **kwargs: Any) -> BaseGenerator:
    """
    Factory function to obtain a generator instance.

    Parameters
    ----------
    mode : str
        'extractive' or 'mock'
    **kwargs
        Passed to generator constructor.

    Returns
    -------
    BaseGenerator
    """
    mode_lower = mode.strip().lower()
    if mode_lower == "mock":
        return MockGenerator(**kwargs)
    elif mode_lower == "extractive":
        return ExtractiveGenerator(**kwargs)
    else:
        logger.warning(f"Unknown generator mode '{mode}', defaulting to 'extractive'")
        return ExtractiveGenerator(**kwargs)
