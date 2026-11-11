"""
End-to-end RAG Pipeline orchestration.

Coordinates:
1. Document ingestion (loading from PDF/TXT)
2. Text chunking with overlap and metadata retention
3. Vector embedding generation
4. Local vector store index persistence
5. Query retrieval (semantic vector search + lexical reranking)
6. Grounded answer generation with citations and disclaimers
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from src.rag.chunking import DocumentChunk, chunk_document_pages
from src.rag.embeddings import EmbeddingModel, get_embedder
from src.rag.generator import BaseGenerator, GeneratedAnswer, get_generator
from src.rag.loaders import DocumentPage, load_directory
from src.rag.retriever import Retriever
from src.rag.vectorstore import VectorStore
from src.utils.logger import logger


class RAGPipeline:
    """
    End-to-end Financial Document RAG Pipeline.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        embedder: Optional[EmbeddingModel] = None,
        generator: Optional[BaseGenerator] = None,
    ) -> None:
        """
        Initialize pipeline components from configuration dictionary or defaults.
        """
        self.config = config or {}
        rag_cfg = self.config.get("rag", {})

        self.chunk_size = int(rag_cfg.get("chunk_size", 800))
        self.chunk_overlap = int(rag_cfg.get("chunk_overlap", 120))
        self.top_k = int(rag_cfg.get("top_k", 5))
        self.min_similarity = float(rag_cfg.get("min_similarity", 0.3))
        self.reranking_enabled = bool(rag_cfg.get("reranking_enabled", True))
        self.rerank_alpha = float(rag_cfg.get("rerank_alpha", 0.5))

        self.documents_dir = Path(rag_cfg.get("documents_dir", "data/documents"))
        self.vectorstore_dir = Path(rag_cfg.get("vectorstore_dir", "data/vectorstore"))

        # Initialize Embedder
        embedding_mode = rag_cfg.get("embedding_mode", "auto")
        model_name = rag_cfg.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
        batch_size = int(rag_cfg.get("embedding_batch_size", 32))

        self.embedder = embedder or get_embedder(
            mode=embedding_mode,
            model_name=model_name,
            batch_size=batch_size,
        )

        # Initialize VectorStore
        self.vectorstore = VectorStore()

        # Initialize Generator
        generator_mode = rag_cfg.get("generator_mode", "extractive")
        self.generator = generator or get_generator(mode=generator_mode)

        # Initialize Retriever
        self.retriever = Retriever(
            vectorstore=self.vectorstore,
            embedder=self.embedder,
            top_k=self.top_k,
            min_score=self.min_similarity,
            rerank=self.reranking_enabled,
            rerank_alpha=self.rerank_alpha,
        )

        self._indexed_documents: List[str] = []

    def ingest_documents(
        self,
        documents_dir: Optional[Union[str, Path]] = None,
        reset: bool = True,
    ) -> int:
        """
        Load all supported documents from disk, chunk them, embed them,
        and add them to the vector store.

        Parameters
        ----------
        documents_dir : Optional[Union[str, Path]]
            Directory containing source documents.
        reset : bool, default=True
            Whether to clear existing in-memory vectors before ingestion,
            guaranteeing an idempotent rebuild.

        Returns
        -------
        int
            Number of chunks indexed.
        """
        dir_path = Path(documents_dir or self.documents_dir)
        logger.info(f"Ingesting documents from directory: {dir_path}")

        if reset:
            self.vectorstore.clear()

        pages: List[DocumentPage] = load_directory(dir_path)
        if not pages:
            logger.warning(f"No document pages found in {dir_path}")
            return 0

        # Chunk pages
        chunks: List[DocumentChunk] = chunk_document_pages(
            pages,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        if not chunks:
            logger.warning("Chunking yielded 0 chunks")
            return 0

        # Generate embeddings
        chunk_texts = [c.text for c in chunks]
        embeddings = self.embedder.encode(chunk_texts)

        # Populate vectorstore (idempotently deduplicates by chunk_id)
        self.vectorstore.add(chunks, embeddings, deduplicate=True)

        self._indexed_documents = sorted(list({p.document_name for p in pages}))
        logger.info(
            f"Successfully indexed {len(chunks)} chunks across "
            f"{len(self._indexed_documents)} documents (store total: {len(self.vectorstore)})"
        )
        return len(chunks)

    def save_index(self, directory: Optional[Union[str, Path]] = None) -> Path:
        """Save vector store to disk."""
        target_dir = Path(directory or self.vectorstore_dir)
        self.vectorstore.save(target_dir)
        return target_dir

    def load_index(self, directory: Optional[Union[str, Path]] = None) -> bool:
        """Load vector store from disk."""
        target_dir = Path(directory or self.vectorstore_dir)
        if not target_dir.exists():
            logger.warning(f"Vector store directory does not exist: {target_dir}")
            return False
        try:
            self.vectorstore.load(target_dir)
            self._indexed_documents = sorted(
                list({c.document_name for c in self.vectorstore._chunks})
            )
            return True
        except Exception as e:
            logger.error(f"Failed to load vector store from {target_dir}: {e}")
            return False

    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
    ) -> GeneratedAnswer:
        """
        Execute full RAG flow: retrieval -> answer generation.

        Parameters
        ----------
        question : str
            User query.
        top_k : Optional[int]
            Optional top_k override.
        min_score : Optional[float]
            Optional minimum similarity score override.

        Returns
        -------
        GeneratedAnswer
            Grounded answer with citations and confidence score.
        """
        retrieval_res = self.retriever.retrieve(
            query=question,
            top_k=top_k,
            min_score=min_score,
        )

        answer = self.generator.generate(
            query=question,
            retrieved_chunks=retrieval_res.results,
        )
        return answer

    def get_status(self) -> Dict[str, Any]:
        """Return pipeline status summary."""
        return {
            "chunk_count": len(self.vectorstore),
            "document_count": len(self._indexed_documents),
            "indexed_documents": self._indexed_documents,
            "embedder_dimension": self.embedder.dimension,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "reranking_enabled": self.reranking_enabled,
        }
