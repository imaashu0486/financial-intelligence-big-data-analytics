"""
Text chunking for the RAG pipeline.

Splits document pages into overlapping chunks of configurable size,
preserving document and page metadata for retrieval citation.

Why chunking is required:
- Embedding models have limited input length (~512 tokens for MiniLM).
- Smaller chunks improve retrieval precision by isolating topics.
- Overlap ensures sentences at chunk boundaries are not lost.
- Chunk-level metadata enables accurate source citations.
"""

from dataclasses import dataclass
from typing import List

from src.rag.loaders import DocumentPage
from src.utils.logger import logger


@dataclass
class DocumentChunk:
    """A chunk of text from a document page, with full provenance metadata."""

    document_id: str
    document_name: str
    source_path: str
    page_number: int
    chunk_id: str
    chunk_index: int
    text: str


def chunk_text(
    text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> List[str]:
    """
    Split text into overlapping chunks by character count.

    Parameters
    ----------
    text : str
        The input text to split.
    chunk_size : int
        Maximum characters per chunk.
    chunk_overlap : int
        Number of overlapping characters between consecutive chunks.

    Returns
    -------
    List[str]
        List of text chunks (empty strings are filtered out).
    """
    if not text or not text.strip():
        return []

    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if chunk_overlap < 0:
        raise ValueError(f"chunk_overlap must be non-negative, got {chunk_overlap}")
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be less than "
            f"chunk_size ({chunk_size})"
        )

    text = text.strip()
    if len(text) <= chunk_size:
        return [text]

    chunks: List[str] = []
    start = 0
    step = chunk_size - chunk_overlap

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += step

    return chunks


def chunk_document_pages(
    pages: List[DocumentPage],
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> List[DocumentChunk]:
    """
    Chunk a list of document pages into DocumentChunks.

    Each chunk inherits the metadata of its source page and receives
    a unique chunk_id and sequential chunk_index.

    Parameters
    ----------
    pages : List[DocumentPage]
        Pages from document loaders.
    chunk_size : int
        Maximum characters per chunk.
    chunk_overlap : int
        Overlap between consecutive chunks.

    Returns
    -------
    List[DocumentChunk]
        Ordered list of document chunks with metadata.
    """
    all_chunks: List[DocumentChunk] = []
    global_index = 0

    for page in pages:
        text_chunks = chunk_text(page.text, chunk_size, chunk_overlap)

        for i, text in enumerate(text_chunks):
            chunk_id = f"{page.document_id}_p{page.page_number}_c{i}"
            chunk = DocumentChunk(
                document_id=page.document_id,
                document_name=page.document_name,
                source_path=page.source_path,
                page_number=page.page_number,
                chunk_id=chunk_id,
                chunk_index=global_index,
                text=text,
            )
            all_chunks.append(chunk)
            global_index += 1

    logger.info(
        f"Chunking complete: {len(pages)} pages → {len(all_chunks)} chunks "
        f"(size={chunk_size}, overlap={chunk_overlap})"
    )
    return all_chunks
