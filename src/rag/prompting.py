"""
Prompt construction and citation formatting for the RAG pipeline.

Constructs grounded prompts enforcing:
- Sourcing only from provided context
- Strict citation of document names and page numbers
- Explicit rejection of unsupported questions (anti-hallucination)
- Clear financial advice disclaimer
"""

from typing import Any, Dict, List
from src.rag.vectorstore import SearchResult

FINANCIAL_ADVICE_DISCLAIMER = (
    "DISCLAIMER: This system provides academic financial document analytics for educational "
    "purposes only. It does NOT provide financial advice, trading signals, or investment recommendations."
)

SYSTEM_INSTRUCTION = (
    "You are a financial document intelligence assistant. Your task is to answer questions strictly and "
    "accurately based on the provided document excerpts below.\n\n"
    "RULES:\n"
    "1. SOURCING: Answer ONLY using information directly stated in the context.\n"
    "2. CITATIONS: Cite every claim with the exact document name and page number from the context.\n"
    "3. UNSUPPORTED QUERIES: If the provided excerpts do not contain enough information to answer the question "
    "faithfully, state clearly: 'Based on the provided documents, there is insufficient information to answer this question.'\n"
    "4. DO NOT hallucinate, infer speculative conclusions, or extrapolate beyond the provided text.\n"
    "5. DISCLAIMER: Always note that the response does not constitute financial advice.\n"
)


def format_citations(results: List[SearchResult]) -> List[str]:
    """
    Format human-readable citation strings from search results.

    Returns
    -------
    List[str]
        List of formatted citations like:
        '[Doc: acme_corp_annual_report_2024.txt, Page: 1, Chunk: 0] (score: 0.8521)'
    """
    citations = []
    for res in results:
        chunk = res.chunk
        page_str = f"Page: {chunk.page_number}" if chunk.page_number is not None else "Page: N/A"
        citation = f"[Doc: {chunk.document_name}, {page_str}, Chunk: {chunk.chunk_index}] (score: {res.score:.4f})"
        if citation not in citations:
            citations.append(citation)
    return citations


def format_citation_dicts(results: List[SearchResult]) -> List[Dict[str, Any]]:
    """
    Format structured citation records suitable for MongoDB and API/UI consumers.
    """
    records = []
    for res in results:
        c = res.chunk
        records.append({
            "document_name": c.document_name,
            "document_id": c.document_id,
            "page_number": c.page_number,
            "chunk_id": c.chunk_id,
            "chunk_index": c.chunk_index,
            "score": round(float(res.score), 4),
            "rank": res.rank,
            "snippet": c.text[:200] + ("..." if len(c.text) > 200 else ""),
        })
    return records


def build_rag_prompt(query: str, retrieved_chunks: List[SearchResult]) -> str:
    """
    Build a structured prompt containing system instructions, retrieved context,
    and user query.

    Parameters
    ----------
    query : str
        User question.
    retrieved_chunks : List[SearchResult]
        Top retrieved chunks from Retriever.

    Returns
    -------
    str
        Full prompt string.
    """
    context_blocks = []
    for i, res in enumerate(retrieved_chunks, start=1):
        c = res.chunk
        page_str = f"Page {c.page_number}" if c.page_number is not None else "Page N/A"
        block = (
            f"--- Context Excerpt [{i}] ---\n"
            f"Document: {c.document_name}\n"
            f"Page: {page_str}\n"
            f"Score: {res.score:.4f}\n"
            f"Content:\n{c.text.strip()}\n"
        )
        context_blocks.append(block)

    context_str = "\n".join(context_blocks) if context_blocks else "No relevant context found."

    prompt = (
        f"{SYSTEM_INSTRUCTION}\n"
        f"CONTEXT EXCERPTS:\n"
        f"{context_str}\n\n"
        f"USER QUESTION: {query.strip()}\n\n"
        f"GROUNDED ANSWER (with citations):\n"
    )
    return prompt
