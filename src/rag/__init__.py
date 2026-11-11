"""
Financial Document RAG package for Big Data Analytics Platform.

Provides:
- Document loading (TXT, PDF)
- Text chunking with metadata propagation
- Embeddings (SentenceTransformer and Mock)
- In-memory NumPy VectorStore with cosine similarity
- Lexical overlap reranker
- Retriever orchestration
- Prompting and citation formatting
- Grounded answer generators (Extractive, Mock)
- End-to-end RAGPipeline
"""

from src.rag.chunking import DocumentChunk, chunk_document_pages
from src.rag.embeddings import (
    EmbeddingModel,
    MockEmbedder,
    SentenceTransformerEmbedder,
    get_embedder,
)
from src.rag.generator import (
    BaseGenerator,
    ExtractiveGenerator,
    GeneratedAnswer,
    MockGenerator,
    get_generator,
)
from src.rag.loaders import DocumentPage, load_directory, load_pdf, load_txt
from src.rag.pipeline import RAGPipeline
from src.rag.prompting import (
    FINANCIAL_ADVICE_DISCLAIMER,
    build_rag_prompt,
    format_citations,
    format_citation_dicts,
)
from src.rag.retriever import RetrievalResult, Retriever
from src.rag.vectorstore import SearchResult, VectorStore
from src.rag.evaluation import (
    AggregateEvaluationReport,
    AnswerMetrics,
    EvaluationSample,
    QueryEvaluationResult,
    RAGEvaluator,
    RetrievalMetrics,
    evaluate_answer,
    evaluate_retrieval,
    load_evaluation_dataset,
)

__all__ = [
    # Data structures
    "DocumentPage",
    "DocumentChunk",
    "SearchResult",
    "RetrievalResult",
    "GeneratedAnswer",
    "FINANCIAL_ADVICE_DISCLAIMER",
    # Loaders & Chunking
    "load_txt",
    "load_pdf",
    "load_directory",
    "chunk_document_pages",
    # Embeddings
    "EmbeddingModel",
    "MockEmbedder",
    "SentenceTransformerEmbedder",
    "get_embedder",
    # Vector store & Retrieval
    "VectorStore",
    "Retriever",
    "lexical_rerank",
    # Prompting & Generation
    "build_rag_prompt",
    "format_citations",
    "format_citation_dicts",
    "BaseGenerator",
    "ExtractiveGenerator",
    "MockGenerator",
    "get_generator",
    # Pipeline
    "RAGPipeline",
    # Evaluation (Phase 8)
    "EvaluationSample",
    "RetrievalMetrics",
    "AnswerMetrics",
    "QueryEvaluationResult",
    "AggregateEvaluationReport",
    "RAGEvaluator",
    "load_evaluation_dataset",
    "evaluate_retrieval",
    "evaluate_answer",
]
