"""
Comprehensive unit and integration test suite for Phase 7: Financial Document RAG.

Tests all components:
- Document loaders (TXT, PDF, directory, error handling, metadata)
- Text chunking (size, overlap, metadata propagation, edge cases)
- Embeddings (MockEmbedder determinism, batching, dimensions)
- Vector store (Index add, search, top-k, min_score, save/load)
- Retriever & Reranker (Hybrid retrieval, lexical scoring, ranking)
- Prompting & Citations (Prompt structure, citation integrity, disclaimers)
- Answer generation (Extractive, Mock, unsupported questions)
- MongoDB operations (Queries and manifest logging)
- End-to-end RAG pipeline

Runs offline without requiring live MongoDB, GPUs, or paid APIs.
"""

import os
import shutil
import tempfile
from pathlib import Path
from typing import List

import mongomock
import numpy as np
import pytest

from src.database.indexes import INDEX_DEFINITIONS, create_indexes
from src.database.queries import (
    get_rag_documents,
    get_recent_rag_queries,
    store_rag_document_manifest,
    store_rag_query,
)
from src.rag.chunking import DocumentChunk, chunk_document_pages
from src.rag.embeddings import MockEmbedder, get_embedder
from src.rag.generator import (
    INSUFFICIENT_INFO_MSG,
    ExtractiveGenerator,
    GeneratedAnswer,
    MockGenerator,
    get_generator,
)
from src.rag.loaders import (
    DocumentPage,
    load_directory,
    load_pdf,
    load_txt,
)
from src.rag.pipeline import RAGPipeline
from src.rag.prompting import (
    FINANCIAL_ADVICE_DISCLAIMER,
    build_rag_prompt,
    format_citation_dicts,
    format_citations,
)
from src.rag.reranker import compute_lexical_score, lexical_rerank
from src.rag.retriever import Retriever
from src.rag.vectorstore import SearchResult, VectorStore


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def temp_doc_dir(tmp_path):
    """Create a temporary directory with sample text files for loader tests."""
    doc1 = tmp_path / "acme_report.txt"
    doc1.write_text(
        "Acme Corporation reported annual revenue of $12.5 billion in FY 2024. "
        "Operating profit increased by 15% due to cloud expansion.",
        encoding="utf-8",
    )

    doc2 = tmp_path / "globex_note.txt"
    doc2.write_text(
        "Globex Industries announced plans to reduce debt by $2 billion. "
        "Supply chain disruptions remain a primary operational risk.",
        encoding="utf-8",
    )

    doc_empty = tmp_path / "empty.txt"
    doc_empty.write_text("", encoding="utf-8")

    return tmp_path


@pytest.fixture
def sample_pages() -> List[DocumentPage]:
    """Sample DocumentPage objects."""
    return [
        DocumentPage(
            document_id="doc_01",
            document_name="doc_01.txt",
            source_path="/path/to/doc_01.txt",
            page_number=1,
            text=(
                "Acme Corp revenue grew 14 percent to reach 12 billion dollars in FY 2024. "
                "Cloud infrastructure services were the largest growth driver. "
                "Capital expenditures rose to 1.8 billion dollars."
            ),
        ),
        DocumentPage(
            document_id="doc_01",
            document_name="doc_01.txt",
            source_path="/path/to/doc_01.txt",
            page_number=2,
            text=(
                "Risk factors for Acme include cybersecurity threats and regulatory changes. "
                "Foreign exchange volatility could impact overseas earnings."
            ),
        ),
    ]


@pytest.fixture
def mock_embedder() -> MockEmbedder:
    """Deterministic MockEmbedder for fast testing."""
    return MockEmbedder(dimension=384)


@pytest.fixture
def populated_vectorstore(sample_pages, mock_embedder) -> VectorStore:
    """A vector store populated with sample chunks and mock embeddings."""
    chunks = chunk_document_pages(sample_pages, chunk_size=100, chunk_overlap=20)
    texts = [c.text for c in chunks]
    embeddings = mock_embedder.encode(texts)

    store = VectorStore()
    store.add(chunks, embeddings)
    return store


@pytest.fixture
def mock_db():
    """mongomock database instance with created indexes."""
    client = mongomock.MongoClient()
    db = client["test_financial_analytics"]
    create_indexes(db)
    return db


# ── Loaders Tests ────────────────────────────────────────────────────


class TestDocumentLoaders:
    def test_load_txt_success(self, temp_doc_dir):
        path = temp_doc_dir / "acme_report.txt"
        pages = load_txt(path)
        assert len(pages) == 1
        page = pages[0]
        assert page.document_name == "acme_report.txt"
        assert page.page_number == 1
        assert "12.5 billion" in page.text
        assert page.document_id is not None and len(page.document_id) > 0

    def test_load_txt_empty_file(self, temp_doc_dir):
        path = temp_doc_dir / "empty.txt"
        pages = load_txt(path)
        assert len(pages) == 0

    def test_load_txt_nonexistent_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_txt(tmp_path / "non_existent.txt")

    def test_load_directory(self, temp_doc_dir):
        pages = load_directory(temp_doc_dir, extensions=[".txt"])
        # 2 non-empty files
        assert len(pages) == 2
        names = {p.document_name for p in pages}
        assert "acme_report.txt" in names
        assert "globex_note.txt" in names

    def test_load_pdf_invalid(self, tmp_path):
        bad_pdf = tmp_path / "corrupt.pdf"
        bad_pdf.write_bytes(b"not a valid pdf content")
        pages = load_pdf(bad_pdf)
        assert len(pages) == 0


# ── Chunking Tests ───────────────────────────────────────────────────


class TestChunking:
    def test_chunk_document_pages_basic(self, sample_pages):
        chunks = chunk_document_pages(sample_pages, chunk_size=80, chunk_overlap=15)
        assert len(chunks) > len(sample_pages)
        for chunk in chunks:
            assert isinstance(chunk, DocumentChunk)
            assert chunk.document_name == "doc_01.txt"
            assert chunk.chunk_id != ""
            assert len(chunk.text) <= 80

    def test_chunking_preserves_page_numbers(self, sample_pages):
        chunks = chunk_document_pages(sample_pages, chunk_size=60, chunk_overlap=10)
        pages_seen = {c.page_number for c in chunks}
        assert 1 in pages_seen
        assert 2 in pages_seen

    def test_chunking_short_document(self):
        page = DocumentPage(
            document_id="short_doc",
            document_name="short.txt",
            source_path="/tmp/short.txt",
            page_number=1,
            text="Short text.",
        )
        chunks = chunk_document_pages([page], chunk_size=500, chunk_overlap=50)
        assert len(chunks) == 1
        assert chunks[0].text == "Short text."
        assert chunks[0].chunk_index == 0

    def test_chunking_empty_pages(self):
        page = DocumentPage(
            document_id="empty",
            document_name="empty.txt",
            source_path="/tmp/empty.txt",
            page_number=1,
            text="   \n   ",
        )
        chunks = chunk_document_pages([page])
        assert len(chunks) == 0


# ── Embeddings Tests ─────────────────────────────────────────────────


class TestEmbeddings:
    def test_mock_embedder_shape(self, mock_embedder):
        texts = ["Hello world", "Financial revenue growth", "Risk factors"]
        embs = mock_embedder.encode(texts)
        assert isinstance(embs, np.ndarray)
        assert embs.shape == (3, 384)

    def test_mock_embedder_determinism(self, mock_embedder):
        text = ["Identical sentence for determinism testing."]
        emb1 = mock_embedder.encode(text)
        emb2 = mock_embedder.encode(text)
        np.testing.assert_allclose(emb1, emb2)

    def test_mock_embedder_normalized(self, mock_embedder):
        texts = ["Check unit norm for cosine similarity."]
        embs = mock_embedder.encode(texts)
        norm = np.linalg.norm(embs[0])
        assert np.isclose(norm, 1.0, atol=1e-5)

    def test_mock_embedder_empty_input(self, mock_embedder):
        embs = mock_embedder.encode([])
        assert embs.shape == (0, 384)

    def test_get_embedder_factory(self):
        embedder = get_embedder(mode="mock")
        assert isinstance(embedder, MockEmbedder)
        assert embedder.dimension == 384


# ── VectorStore Tests ────────────────────────────────────────────────


class TestVectorStore:
    def test_vectorstore_empty(self):
        store = VectorStore()
        assert len(store) == 0
        assert store.dimension is None
        results = store.search(np.zeros(384))
        assert results == []

    def test_vectorstore_add_and_search(self, populated_vectorstore, mock_embedder):
        assert len(populated_vectorstore) > 0
        assert populated_vectorstore.dimension == 384

        q_vec = mock_embedder.encode(["Acme revenue growth"])[0]
        results = populated_vectorstore.search(q_vec, top_k=3)

        assert len(results) <= 3
        assert len(results) > 0
        assert isinstance(results[0], SearchResult)
        assert results[0].rank == 1
        # Highest score first
        if len(results) > 1:
            assert results[0].score >= results[1].score

    def test_vectorstore_min_score_filter(self, populated_vectorstore, mock_embedder):
        q_vec = mock_embedder.encode(["Acme revenue growth"])[0]
        results = populated_vectorstore.search(q_vec, min_score=0.9999)
        # Virtually impossible with mock hash unless exact match
        assert len(results) <= 1

    def test_vectorstore_dimension_mismatch(self, populated_vectorstore):
        wrong_emb = np.zeros((1, 128), dtype=np.float32)
        dummy_chunk = DocumentChunk(
            document_id="d",
            document_name="d.txt",
            source_path="/d.txt",
            page_number=1,
            chunk_id="c_0",
            chunk_index=0,
            text="text",
        )
        with pytest.raises(ValueError, match="dimension mismatch"):
            populated_vectorstore.add([dummy_chunk], wrong_emb)

    def test_vectorstore_save_and_load(self, populated_vectorstore, tmp_path, mock_embedder):
        save_dir = tmp_path / "vstore"
        populated_vectorstore.save(save_dir)

        assert (save_dir / "embeddings.npy").exists()
        assert (save_dir / "metadata.json").exists()

        new_store = VectorStore()
        new_store.load(save_dir)

        assert len(new_store) == len(populated_vectorstore)
        assert new_store.dimension == populated_vectorstore.dimension

        q_vec = mock_embedder.encode(["cloud infrastructure"])[0]
        res1 = populated_vectorstore.search(q_vec, top_k=2)
        res2 = new_store.search(q_vec, top_k=2)

        assert len(res1) == len(res2)
        assert res1[0].chunk.chunk_id == res2[0].chunk.chunk_id

    def test_vectorstore_idempotent_deduplication(self, populated_vectorstore, mock_embedder):
        initial_len = len(populated_vectorstore)
        assert initial_len > 0

        # Adding the same chunks and embeddings must not duplicate entries
        same_chunks = list(populated_vectorstore._chunks)
        same_embs = np.copy(populated_vectorstore._embeddings)
        populated_vectorstore.add(same_chunks, same_embs, deduplicate=True)
        assert len(populated_vectorstore) == initial_len
        assert populated_vectorstore._embeddings.shape[0] == initial_len

    def test_vectorstore_clear(self, populated_vectorstore):
        assert len(populated_vectorstore) > 0
        populated_vectorstore.clear()
        assert len(populated_vectorstore) == 0
        assert populated_vectorstore.dimension is None


# ── Reranker Tests ───────────────────────────────────────────────────


class TestReranker:
    def test_compute_lexical_score(self):
        query_tokens = {"acme", "revenue"}
        text_with_both = "Acme reported strong revenue growth."
        text_with_one = "Acme experienced operating profit."
        text_with_none = "Globex reported quarterly earnings."

        assert compute_lexical_score(query_tokens, text_with_both) == 1.0
        assert compute_lexical_score(query_tokens, text_with_one) == 0.5
        assert compute_lexical_score(query_tokens, text_with_none) == 0.0

    def test_lexical_rerank(self):
        c1 = DocumentChunk("1", "doc1.txt", "/1", 1, "c1", 0, "Weather is sunny today.")
        c2 = DocumentChunk("2", "doc2.txt", "/2", 1, "c2", 0, "Acme revenue and profit numbers.")

        res1 = SearchResult(chunk=c1, score=0.8, rank=1)
        res2 = SearchResult(chunk=c2, score=0.7, rank=2)

        # Query: "Acme revenue"
        reranked = lexical_rerank("Acme revenue", [res1, res2], alpha=0.3)
        assert len(reranked) == 2
        # c2 has high lexical overlap, should rank #1 despite lower semantic score
        assert reranked[0].chunk.chunk_id == "c2"
        assert reranked[0].rank == 1
        assert reranked[1].rank == 2


# ── Retriever Tests ──────────────────────────────────────────────────


class TestRetriever:
    def test_retriever_empty_query(self, populated_vectorstore, mock_embedder):
        retriever = Retriever(populated_vectorstore, mock_embedder)
        result = retriever.retrieve("")
        assert len(result.results) == 0
        assert not result.has_matches

    def test_retriever_normal_flow(self, populated_vectorstore, mock_embedder):
        retriever = Retriever(
            populated_vectorstore,
            mock_embedder,
            top_k=2,
            min_score=0.0,
            rerank=True,
        )
        res = retriever.retrieve("What is Acme's revenue?")
        assert res.has_matches
        assert len(res.results) <= 2
        assert res.retrieval_time_ms >= 0.0
        assert res.reranked is True


# ── Prompting & Citations Tests ──────────────────────────────────────


class TestPromptingAndCitations:
    def test_format_citations(self):
        c = DocumentChunk("id1", "annual_report.txt", "/p", 3, "c_0", 0, "Text snippet")
        res = [SearchResult(chunk=c, score=0.8523, rank=1)]
        citations = format_citations(res)
        assert len(citations) == 1
        assert "annual_report.txt" in citations[0]
        assert "Page: 3" in citations[0]
        assert "0.8523" in citations[0]

    def test_format_citation_dicts(self):
        c = DocumentChunk("id1", "annual_report.txt", "/p", 3, "c_0", 0, "Sample content")
        res = [SearchResult(chunk=c, score=0.8523, rank=1)]
        records = format_citation_dicts(res)
        assert len(records) == 1
        assert records[0]["document_name"] == "annual_report.txt"
        assert records[0]["page_number"] == 3
        assert records[0]["rank"] == 1

    def test_build_rag_prompt(self):
        c = DocumentChunk("id1", "report.txt", "/p", 1, "c_0", 0, "Revenue was 10M.")
        res = [SearchResult(chunk=c, score=0.9, rank=1)]
        prompt = build_rag_prompt("What was revenue?", res)
        assert "USER QUESTION: What was revenue?" in prompt
        assert "report.txt" in prompt
        assert "Revenue was 10M." in prompt
        assert "RULES:" in prompt


# ── Generator Tests ──────────────────────────────────────────────────


class TestGenerator:
    def test_extractive_generator_supported(self):
        gen = ExtractiveGenerator(min_confidence=0.1)
        c = DocumentChunk(
            "id", "acme.txt", "/p", 1, "c_1", 0,
            "Acme revenue was $12.5 billion for FY 2024."
        )
        res = [SearchResult(chunk=c, score=0.8, rank=1)]
        answer = gen.generate("What was Acme revenue?", res)
        assert isinstance(answer, GeneratedAnswer)
        assert answer.is_supported is True
        assert answer.confidence_score == 0.8
        assert "acme.txt" in answer.answer
        assert len(answer.sources) == 1
        assert answer.disclaimer == FINANCIAL_ADVICE_DISCLAIMER

    def test_extractive_generator_unsupported(self):
        gen = ExtractiveGenerator(min_confidence=0.5)
        # Empty context
        ans = gen.generate("Unknown query", [])
        assert ans.is_supported is False
        assert ans.answer == INSUFFICIENT_INFO_MSG
        assert ans.confidence_score == 0.0

        # Low score context
        c = DocumentChunk("id", "acme.txt", "/p", 1, "c_1", 0, "Irrelevant snippet")
        low_res = [SearchResult(chunk=c, score=0.2, rank=1)]
        ans_low = gen.generate("Unknown query", low_res)
        assert ans_low.is_supported is False
        assert ans_low.answer == INSUFFICIENT_INFO_MSG

    def test_mock_generator(self):
        gen = MockGenerator(fixed_answer="Custom mock answer.")
        c = DocumentChunk("id", "doc.txt", "/p", 1, "c_1", 0, "Text")
        res = [SearchResult(chunk=c, score=0.9, rank=1)]
        ans = gen.generate("Test query", res)
        assert ans.answer == "Custom mock answer."
        assert ans.is_supported is True


# ── MongoDB Operations Tests ─────────────────────────────────────────


class TestMongoOperations:
    def test_store_rag_query_and_get_recent(self, mock_db):
        query_id = store_rag_query(
            query="What is the net income?",
            answer="Net income was $1.2B.",
            sources=["[Doc: test.txt, Page: 1]"],
            citations=[{"document_name": "test.txt", "page_number": 1, "score": 0.88}],
            generator_mode="extractive",
            confidence_score=0.88,
            is_supported=True,
            db=mock_db,
        )
        assert query_id is not None

        recent = get_recent_rag_queries(limit=5, db=mock_db)
        assert len(recent) == 1
        assert recent[0]["query"] == "What is the net income?"
        assert recent[0]["confidence_score"] == 0.88
        assert recent[0]["is_supported"] is True

    def test_store_and_get_rag_manifest(self, mock_db):
        manifest = [
            {
                "document_id": "report_2024",
                "document_name": "report_2024.txt",
                "chunk_count": 12,
            },
            {
                "document_id": "notes_2024",
                "document_name": "notes_2024.txt",
                "chunk_count": 5,
            },
        ]
        count = store_rag_document_manifest(manifest, db=mock_db)
        assert count == 2

        docs = get_rag_documents(db=mock_db)
        assert len(docs) == 2
        assert docs[0]["document_id"] in ("notes_2024", "report_2024")


# ── End-to-End Pipeline Integration Test ─────────────────────────────


class TestRAGPipelineIntegration:
    def test_end_to_end_pipeline(self, temp_doc_dir, tmp_path):
        vstore_dir = tmp_path / "vstore_test"

        config = {
            "rag": {
                "documents_dir": str(temp_doc_dir),
                "vectorstore_dir": str(vstore_dir),
                "embedding_mode": "mock",
                "chunk_size": 150,
                "chunk_overlap": 30,
                "top_k": 3,
                "min_similarity": 0.0,
                "reranking_enabled": True,
                "rerank_alpha": 0.5,
                "generator_mode": "extractive",
            }
        }

        pipeline = RAGPipeline(config=config)

        # 1. Ingest
        chunk_count = pipeline.ingest_documents()
        assert chunk_count > 0

        # 2. Save
        saved_path = pipeline.save_index()
        assert saved_path.exists()

        # 3. Query
        answer = pipeline.query("What was Acme Corporation revenue in FY 2024?")
        assert isinstance(answer, GeneratedAnswer)
        assert answer.is_supported is True
        assert len(answer.sources) > 0
        assert "Acme" in answer.answer or "acme" in answer.answer.lower()
        assert answer.disclaimer == FINANCIAL_ADVICE_DISCLAIMER

        # 4. Status
        status = pipeline.get_status()
        assert status["chunk_count"] == chunk_count
        assert status["document_count"] == 2
