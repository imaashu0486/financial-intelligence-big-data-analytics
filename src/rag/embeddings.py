"""
Embedding generation for the RAG pipeline.

Provides:
- SentenceTransformerEmbedder: uses sentence-transformers/all-MiniLM-L6-v2
- MockEmbedder: deterministic hash-based embeddings for tests (no downloads)
- get_embedder(): factory with auto-detection and graceful fallback
"""

import hashlib
from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np

from src.utils.logger import logger

# Default embedding dimension (matches all-MiniLM-L6-v2)
DEFAULT_DIMENSION = 384


class EmbeddingModel(ABC):
    """Abstract base class for embedding models."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding vector dimension."""
        ...

    @abstractmethod
    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Encode a list of texts into embedding vectors.

        Parameters
        ----------
        texts : List[str]
            Input texts to embed.

        Returns
        -------
        np.ndarray
            Shape (len(texts), dimension) float32 array.
        """
        ...


class MockEmbedder(EmbeddingModel):
    """
    Deterministic hash-based embedder for testing.

    Produces consistent embeddings for the same input text without
    requiring any model download or GPU. Embeddings are unit-normalized
    so cosine similarity works correctly.
    """

    def __init__(self, dimension: int = DEFAULT_DIMENSION) -> None:
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def encode(self, texts: List[str]) -> np.ndarray:
        """Generate deterministic embeddings from text hashes."""
        if not texts:
            return np.empty((0, self._dimension), dtype=np.float32)

        embeddings = np.zeros((len(texts), self._dimension), dtype=np.float32)
        for i, text in enumerate(texts):
            # Use SHA-256 hash to seed a deterministic random vector
            hash_bytes = hashlib.sha256(text.encode("utf-8")).digest()
            seed = int.from_bytes(hash_bytes[:4], "big")
            rng = np.random.RandomState(seed)
            vec = rng.randn(self._dimension).astype(np.float32)
            # L2-normalize for cosine similarity
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            embeddings[i] = vec

        return embeddings


class SentenceTransformerEmbedder(EmbeddingModel):
    """
    Embedding model using sentence-transformers.

    Default model: all-MiniLM-L6-v2 (384-dim, ~80 MB, CPU-friendly).
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 32,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required. "
                "Install with: pip install sentence-transformers>=2.2.0"
            )

        logger.info(f"Loading embedding model: {model_name}")
        try:
            self._model = SentenceTransformer(model_name, local_files_only=True)
        except Exception:
            self._model = SentenceTransformer(model_name)
        self._batch_size = batch_size
        if hasattr(self._model, "get_embedding_dimension"):
            self._dimension = self._model.get_embedding_dimension()
        else:
            self._dimension = self._model.get_sentence_embedding_dimension()
        logger.info(
            f"Embedding model loaded: dim={self._dimension}, "
            f"batch_size={self._batch_size}"
        )

    @property
    def dimension(self) -> int:
        return self._dimension

    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts using the sentence-transformer model."""
        if not texts:
            return np.empty((0, self._dimension), dtype=np.float32)

        embeddings = self._model.encode(
            texts,
            batch_size=self._batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return np.array(embeddings, dtype=np.float32)


def get_embedder(
    mode: str = "auto",
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 32,
    dimension: int = DEFAULT_DIMENSION,
) -> EmbeddingModel:
    """
    Factory function to get an embedding model.

    Parameters
    ----------
    mode : str
        - "auto": try sentence-transformers, fall back to mock with warning
        - "mock": always use MockEmbedder
        - "sentence-transformer": require sentence-transformers (raise if unavailable)
    model_name : str
        Model name for sentence-transformers.
    batch_size : int
        Batch size for encoding.
    dimension : int
        Dimension for mock embedder.

    Returns
    -------
    EmbeddingModel
        An initialized embedding model.
    """
    if mode == "mock":
        logger.info("Using MockEmbedder (deterministic hash-based embeddings)")
        return MockEmbedder(dimension=dimension)

    if mode == "sentence-transformer":
        return SentenceTransformerEmbedder(
            model_name=model_name, batch_size=batch_size
        )

    # mode == "auto"
    try:
        return SentenceTransformerEmbedder(
            model_name=model_name, batch_size=batch_size
        )
    except (ImportError, Exception) as e:
        logger.warning(
            f"Could not load sentence-transformer model: {e}. "
            f"Falling back to MockEmbedder."
        )
        return MockEmbedder(dimension=dimension)
