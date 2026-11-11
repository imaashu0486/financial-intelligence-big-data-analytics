"""
Financial RAG service layer (Phases 7 & 8).
Connects to the existing RAGPipeline, vector store index,
and empirical evaluation benchmarks.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent

from src.rag.pipeline import RAGPipeline
from src.utils.logger import logger


class RAGService:
    """Service facade for the Financial Document RAG system."""

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        embedder: Optional[Any] = None,
        mock: bool = False,
    ) -> None:
        self.root = data_dir or ROOT
        self.vectorstore_dir = self.root / "data" / "vectorstore"
        self.documents_dir = self.root / "data" / "documents"
        self.eval_dir = self.root / "data" / "evaluation"
        self.embedder = embedder
        self.mock = mock
        self._pipeline: Optional[RAGPipeline] = None

    def get_index_status(self) -> Dict[str, Any]:
        """
        Check vector index existence, chunk count, and document metadata on disk.
        """
        embeddings_path = self.vectorstore_dir / "embeddings.npy"
        metadata_path = self.vectorstore_dir / "metadata.json"

        is_indexed = embeddings_path.exists() and metadata_path.exists()
        chunk_count = 0
        documents = []
        embedding_dim = 0

        if is_indexed:
            try:
                emb = np.load(embeddings_path)
                chunk_count = len(emb)
                embedding_dim = int(emb.shape[1]) if len(emb.shape) > 1 else 0

                with open(metadata_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                    if isinstance(meta, list):
                        chunks = meta
                    elif isinstance(meta, dict):
                        chunks = meta.get("chunks", [])
                    else:
                        chunks = []
                    documents = sorted(list({c.get("document_name") for c in chunks if isinstance(c, dict) and c.get("document_name")}))
                    unique_chunks = len({c.get("chunk_id") for c in chunks if isinstance(c, dict) and c.get("chunk_id")})
            except Exception as e:
                logger.warning(f"Error reading vector index metadata: {e}")

        # Scan raw documents folder
        raw_docs = []
        if self.documents_dir.exists():
            raw_docs = [p.name for p in self.documents_dir.glob("*.*") if p.is_file() and not p.name.startswith(".")]

        return {
            "is_indexed": is_indexed,
            "chunk_count": chunk_count,
            "unique_chunk_count": unique_chunks if unique_chunks > 0 else chunk_count,
            "embedding_dim": embedding_dim,
            "indexed_documents": documents,
            "raw_documents": sorted(raw_docs),
            "vectorstore_path": str(self.vectorstore_dir),
        }

    def get_pipeline(self) -> RAGPipeline:
        """
        Retrieve or initialize the RAGPipeline singleton with loaded vector index.
        """
        if self._pipeline is not None:
            return self._pipeline

        config = {}
        if self.mock:
            config = {"rag": {"embedding_mode": "mock"}}

        pipeline = RAGPipeline(config=config, embedder=self.embedder)
        if (self.vectorstore_dir / "embeddings.npy").exists() and not self.mock:
            success = pipeline.load_index(self.vectorstore_dir)
            if not success:
                logger.warning("Could not load vector index from disk; ingesting documents afresh.")
                pipeline.ingest_documents(self.documents_dir)
        else:
            logger.info(f"Ingesting documents in-memory (mock={self.mock}).")
            pipeline.ingest_documents(self.documents_dir)

        self._pipeline = pipeline
        return self._pipeline

    def query(
        self,
        question: str,
        top_k: int = 5,
        min_score: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Execute full RAG retrieval and grounded generation flow.
        """
        if not question or not question.strip():
            return {
                "error": "Question cannot be empty.",
                "is_supported": False,
            }

        pipeline = self.get_pipeline()
        effective_min_score = min_score
        if effective_min_score is None:
            if self.mock or getattr(pipeline.embedder, "is_mock", False) or pipeline.embedder.__class__.__name__ == "MockEmbedder":
                effective_min_score = 0.05

        ans = pipeline.query(question=question.strip(), top_k=top_k, min_score=effective_min_score)

        # Format retrieved chunks for presentation
        chunks_formatted = []
        for res in ans.retrieved_chunks:
            chunk = res.chunk
            score_val = float(getattr(res, "score", 0.0))
            combined_val = float(getattr(res, "combined_score", score_val))
            lex_val = float(getattr(res, "lexical_score", 0.0))
            chunks_formatted.append({
                "chunk_id": chunk.chunk_id,
                "document_name": chunk.document_name,
                "page_number": chunk.page_number,
                "text": chunk.text,
                "similarity_score": round(score_val, 4),
                "lexical_score": round(lex_val, 4),
                "combined_score": round(combined_val, 4),
            })

        return {
            "query": ans.query,
            "answer": ans.answer,
            "sources": ans.sources,
            "citations": ans.citations,
            "is_supported": ans.is_supported,
            "confidence_score": round(float(ans.confidence_score), 4),
            "generator_mode": ans.generator_mode,
            "disclaimer": ans.disclaimer,
            "retrieved_chunks": chunks_formatted,
        }

    def get_evaluation_benchmarks(self) -> Dict[str, Any]:
        """
        Load empirical Phase 8 evaluation benchmarks from results.json.
        """
        path = self.eval_dir / "results.json"
        if not path.exists():
            return {}

        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
