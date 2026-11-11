"""
Local vector store for the RAG pipeline.

Uses NumPy cosine similarity for transparent, zero-dependency (beyond NumPy)
similarity search. Suitable for student projects with <10k chunks.

Features:
- Build index from document chunks + embeddings
- Save/load index to/from disk
- Top-k similarity search with minimum score filtering
- Metadata preservation on every result
"""

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.rag.chunking import DocumentChunk
from src.utils.logger import logger


@dataclass
class SearchResult:
    """A single search result with chunk metadata and similarity score."""

    chunk: DocumentChunk
    score: float
    rank: int


class VectorStore:
    """
    In-memory vector index with NumPy cosine similarity search.

    Stores document chunk embeddings and metadata. Supports save/load
    for persistence across sessions.
    """

    def __init__(self) -> None:
        self._embeddings: Optional[np.ndarray] = None  # (N, D)
        self._chunks: List[DocumentChunk] = []

    def __len__(self) -> int:
        return len(self._chunks)

    @property
    def dimension(self) -> Optional[int]:
        """Embedding dimension, or None if empty."""
        if self._embeddings is not None and self._embeddings.shape[0] > 0:
            return self._embeddings.shape[1]
        return None

    def clear(self) -> None:
        """Reset and empty all chunks and stored vectors."""
        self._embeddings = None
        self._chunks = []

    def add(
        self,
        chunks: List[DocumentChunk],
        embeddings: np.ndarray,
        deduplicate: bool = True,
    ) -> None:
        """
        Add document chunks and their embeddings to the index.

        Parameters
        ----------
        chunks : List[DocumentChunk]
            Document chunks with metadata.
        embeddings : np.ndarray
            Shape (len(chunks), dimension) float32 embeddings.
        deduplicate : bool, default=True
            If True, updates existing chunks with matching chunk_id in place
            rather than appending duplicate vectors.

        Raises
        ------
        ValueError
            If chunks and embeddings counts don't match.
        """
        if len(chunks) != embeddings.shape[0]:
            raise ValueError(
                f"Chunk count ({len(chunks)}) doesn't match "
                f"embedding count ({embeddings.shape[0]})"
            )

        if not chunks:
            return

        if self._embeddings is None or self._embeddings.shape[0] == 0:
            if deduplicate:
                seen = {}
                for i, c in enumerate(chunks):
                    seen[c.chunk_id] = (c, embeddings[i])
                self._chunks = [item[0] for item in seen.values()]
                self._embeddings = np.array([item[1] for item in seen.values()], dtype=np.float32)
            else:
                self._embeddings = embeddings.astype(np.float32)
                self._chunks = list(chunks)
        else:
            if embeddings.shape[1] != self._embeddings.shape[1]:
                raise ValueError(
                    f"Embedding dimension mismatch: existing={self._embeddings.shape[1]}, "
                    f"new={embeddings.shape[1]}"
                )
            if deduplicate:
                existing_map = {c.chunk_id: i for i, c in enumerate(self._chunks)}
                new_chunks = []
                new_embs = []
                for c, emb in zip(chunks, embeddings):
                    if c.chunk_id in existing_map:
                        idx = existing_map[c.chunk_id]
                        self._chunks[idx] = c
                        self._embeddings[idx] = emb.astype(np.float32)
                    else:
                        existing_map[c.chunk_id] = len(self._chunks) + len(new_chunks)
                        new_chunks.append(c)
                        new_embs.append(emb)
                if new_chunks:
                    self._embeddings = np.vstack(
                        [self._embeddings, np.array(new_embs, dtype=np.float32)]
                    )
                    self._chunks.extend(new_chunks)
            else:
                self._embeddings = np.vstack(
                    [self._embeddings, embeddings.astype(np.float32)]
                )
                self._chunks.extend(chunks)

        logger.info(
            f"Indexed {len(chunks)} chunks into vector store "
            f"(unique total: {len(self._chunks)})"
        )

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> List[SearchResult]:
        """
        Search for most similar chunks using cosine similarity.

        Parameters
        ----------
        query_embedding : np.ndarray
            Shape (dimension,) or (1, dimension) query vector.
        top_k : int
            Maximum number of results to return.
        min_score : float
            Minimum similarity score threshold.

        Returns
        -------
        List[SearchResult]
            Top-k results sorted by descending similarity score.
        """
        if self._embeddings is None or len(self._chunks) == 0:
            return []

        # Ensure query is 1D
        q = query_embedding.flatten().astype(np.float32)

        # Cosine similarity: dot product of normalized vectors
        # Normalize query
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            return []
        q = q / q_norm

        # Normalize stored embeddings (row-wise)
        norms = np.linalg.norm(self._embeddings, axis=1, keepdims=True)
        # Avoid division by zero
        safe_norms = np.where(norms == 0, 1.0, norms)
        normed_embeddings = self._embeddings / safe_norms

        # Compute similarities
        similarities = normed_embeddings @ q  # (N,)

        # Filter by min_score
        valid_mask = similarities >= min_score
        valid_indices = np.where(valid_mask)[0]

        if len(valid_indices) == 0:
            return []

        # Get top-k indices
        valid_scores = similarities[valid_indices]
        k = min(top_k, len(valid_indices))
        top_k_local = np.argsort(valid_scores)[::-1][:k]
        top_k_indices = valid_indices[top_k_local]

        results: List[SearchResult] = []
        for rank, idx in enumerate(top_k_indices):
            results.append(
                SearchResult(
                    chunk=self._chunks[idx],
                    score=float(similarities[idx]),
                    rank=rank + 1,
                )
            )

        return results

    def save(self, directory: Path) -> None:
        """
        Save the vector store to disk.

        Saves:
        - embeddings.npy: the embedding matrix
        - metadata.json: chunk metadata list

        Parameters
        ----------
        directory : Path
            Directory to save to (created if not exists).
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        if self._embeddings is not None:
            np.save(directory / "embeddings.npy", self._embeddings)

        metadata = []
        for chunk in self._chunks:
            metadata.append({
                "document_id": chunk.document_id,
                "document_name": chunk.document_name,
                "source_path": chunk.source_path,
                "page_number": chunk.page_number,
                "chunk_id": chunk.chunk_id,
                "chunk_index": chunk.chunk_index,
                "text": chunk.text,
            })

        with open(directory / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        logger.info(
            f"Vector store saved to {directory}: "
            f"{len(self._chunks)} chunks, "
            f"embeddings shape={self._embeddings.shape if self._embeddings is not None else 'None'}"
        )

    def load(self, directory: Path) -> None:
        """
        Load the vector store from disk.

        Parameters
        ----------
        directory : Path
            Directory containing embeddings.npy and metadata.json.

        Raises
        ------
        FileNotFoundError
            If required files are missing.
        """
        directory = Path(directory)
        emb_path = directory / "embeddings.npy"
        meta_path = directory / "metadata.json"

        if not emb_path.exists():
            raise FileNotFoundError(f"Embeddings file not found: {emb_path}")
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")

        self._embeddings = np.load(emb_path).astype(np.float32)

        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        self._chunks = []
        for m in metadata:
            self._chunks.append(
                DocumentChunk(
                    document_id=m["document_id"],
                    document_name=m["document_name"],
                    source_path=m["source_path"],
                    page_number=m["page_number"],
                    chunk_id=m["chunk_id"],
                    chunk_index=m["chunk_index"],
                    text=m["text"],
                )
            )

        if len(self._chunks) != self._embeddings.shape[0]:
            raise ValueError(
                f"Metadata count ({len(self._chunks)}) doesn't match "
                f"embedding count ({self._embeddings.shape[0]})"
            )

        logger.info(
            f"Vector store loaded from {directory}: "
            f"{len(self._chunks)} chunks, dim={self._embeddings.shape[1]}"
        )
