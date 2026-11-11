"""
Unit and integration test suite for Phase 8: RAG Evaluation Framework.

Tests:
- Dataset loader and schema validation (missing fields, invalid JSON, types)
- Retrieval metric calculations (Recall@K, Precision@K, Hit Rate@K, MRR)
- Answer metric calculations (Answerability accuracy, keyword coverage, citation validity)
- Edge cases (division by zero, empty results, unanswerable queries, K > chunks)
- RAGEvaluator end-to-end evaluation runner and JSON serialization
"""

import json
from pathlib import Path
from typing import List

import numpy as np
import pytest

from src.rag.chunking import DocumentChunk
from src.rag.embeddings import MockEmbedder
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
from src.rag.generator import GeneratedAnswer, MockGenerator
from src.rag.pipeline import RAGPipeline
from src.rag.vectorstore import SearchResult, VectorStore


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def sample_answerable_item() -> EvaluationSample:
    """An answerable evaluation sample."""
    return EvaluationSample(
        id="test_01",
        question="What was ACME's total revenue?",
        expected_answer="ACME reported total revenue of INR 18,450 crore.",
        expected_keywords=["18,450", "revenue"],
        relevant_documents=["acme_corp_annual_report_2024.txt"],
        relevant_pages=[1],
        answerable=True,
        category="Financial performance",
    )


@pytest.fixture
def sample_unsupported_item() -> EvaluationSample:
    """An unanswerable evaluation sample."""
    return EvaluationSample(
        id="test_02",
        question="What is the launch date of ACME mission to Mars?",
        expected_answer="Based on the provided documents, there is insufficient information.",
        expected_keywords=["insufficient information"],
        relevant_documents=[],
        relevant_pages=[],
        answerable=False,
        category="Unsupported questions",
    )


@pytest.fixture
def mock_retrieved_results() -> List[SearchResult]:
    """Sample search results for testing metric computation."""
    c1 = DocumentChunk("id1", "acme_corp_annual_report_2024.txt", "/p", 1, "c1", 0, "ACME revenue was 18,450 crore.")
    c2 = DocumentChunk("id2", "globex_investor_presentation_2024.txt", "/p", 1, "c2", 1, "Globex revenue was 22,800 crore.")
    c3 = DocumentChunk("id3", "acme_corp_annual_report_2024.txt", "/p", 1, "c3", 2, "EBITDA was 3,690 crore.")

    return [
        SearchResult(chunk=c1, score=0.88, rank=1),
        SearchResult(chunk=c2, score=0.75, rank=2),
        SearchResult(chunk=c3, score=0.62, rank=3),
    ]


# ── Dataset Loading Tests ────────────────────────────────────────────


class TestDatasetLoader:
    def test_load_real_evaluation_dataset(self):
        dataset_path = Path("data/evaluation/rag_eval_dataset.json")
        assert dataset_path.exists()
        dataset = load_evaluation_dataset(dataset_path)
        assert len(dataset) >= 20

        # Check sample schema integrity
        for item in dataset:
            assert isinstance(item, EvaluationSample)
            assert item.id.startswith("eval_")
            assert len(item.question) > 10
            assert isinstance(item.answerable, bool)
            if not item.answerable:
                assert len(item.relevant_documents) == 0

    def test_load_nonexistent_dataset(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_evaluation_dataset(tmp_path / "missing.json")

    def test_load_invalid_json(self, tmp_path):
        bad_json = tmp_path / "bad.json"
        bad_json.write_text("{not a valid json}", encoding="utf-8")
        with pytest.raises(ValueError, match="Invalid JSON format"):
            load_evaluation_dataset(bad_json)

    def test_load_non_array_json(self, tmp_path):
        obj_json = tmp_path / "obj.json"
        obj_json.write_text('{"key": "value"}', encoding="utf-8")
        with pytest.raises(ValueError, match="JSON array"):
            load_evaluation_dataset(obj_json)

    def test_load_missing_required_field(self, tmp_path):
        invalid_item = tmp_path / "invalid_item.json"
        invalid_item.write_text('[{"id": "01", "question": "Q"}]', encoding="utf-8")
        with pytest.raises(ValueError, match="Missing required field"):
            load_evaluation_dataset(invalid_item)


# ── Retrieval Metrics Tests ──────────────────────────────────────────


class TestRetrievalMetrics:
    def test_evaluate_retrieval_perfect_match(self, sample_answerable_item, mock_retrieved_results):
        # mock_retrieved_results has acme at rank 1 and rank 3
        metrics = evaluate_retrieval(sample_answerable_item, mock_retrieved_results, k_values=[1, 3, 5])

        assert metrics.hit_rate_at_k[1] == 1.0
        assert metrics.hit_rate_at_k[3] == 1.0
        assert metrics.recall_at_k[1] == 1.0  # only 1 relevant doc, and it was found at rank 1
        assert metrics.precision_at_k[1] == 1.0  # 1/1
        assert metrics.precision_at_k[3] == 2.0 / 3.0  # 2 chunks from acme out of 3
        assert metrics.mrr == 1.0  # first match at rank 1
        assert metrics.retrieved_count == 3
        assert metrics.relevant_retrieved_count == 1

    def test_evaluate_retrieval_no_match(self, mock_retrieved_results):
        sample = EvaluationSample(
            id="t3",
            question="Initech question?",
            expected_answer="Initech answer",
            expected_keywords=[],
            relevant_documents=["initech_risk_disclosure_2024.txt"],
            answerable=True,
        )
        # mock_retrieved_results has only acme and globex
        metrics = evaluate_retrieval(sample, mock_retrieved_results, k_values=[1, 3])
        assert metrics.hit_rate_at_k[1] == 0.0
        assert metrics.hit_rate_at_k[3] == 0.0
        assert metrics.recall_at_k[1] == 0.0
        assert metrics.precision_at_k[1] == 0.0
        assert metrics.mrr == 0.0

    def test_evaluate_retrieval_match_at_rank_2(self):
        sample = EvaluationSample(
            id="t4",
            question="Globex question",
            expected_answer="Globex answer",
            expected_keywords=[],
            relevant_documents=["globex_investor_presentation_2024.txt"],
            answerable=True,
        )
        c1 = DocumentChunk("1", "acme_corp_annual_report_2024.txt", "/p", 1, "c1", 0, "A")
        c2 = DocumentChunk("2", "globex_investor_presentation_2024.txt", "/p", 1, "c2", 1, "G")
        results = [SearchResult(c1, 0.8, 1), SearchResult(c2, 0.7, 2)]

        metrics = evaluate_retrieval(sample, results, k_values=[1, 2])
        assert metrics.hit_rate_at_k[1] == 0.0
        assert metrics.hit_rate_at_k[2] == 1.0
        assert metrics.recall_at_k[1] == 0.0
        assert metrics.recall_at_k[2] == 1.0
        assert metrics.precision_at_k[1] == 0.0
        assert metrics.precision_at_k[2] == 0.5
        assert metrics.mrr == 0.5  # 1 / 2

    def test_evaluate_retrieval_multi_doc_query(self):
        sample = EvaluationSample(
            id="t5",
            question="Compare ACME and Globex",
            expected_answer="Comparison",
            expected_keywords=[],
            relevant_documents=["acme_corp_annual_report_2024.txt", "globex_investor_presentation_2024.txt"],
            answerable=True,
        )
        c1 = DocumentChunk("1", "acme_corp_annual_report_2024.txt", "/p", 1, "c1", 0, "A")
        c2 = DocumentChunk("2", "initech_risk_disclosure_2024.txt", "/p", 1, "c2", 1, "I")
        c3 = DocumentChunk("3", "globex_investor_presentation_2024.txt", "/p", 1, "c3", 2, "G")
        results = [SearchResult(c1, 0.9, 1), SearchResult(c2, 0.8, 2), SearchResult(c3, 0.7, 3)]

        metrics = evaluate_retrieval(sample, results, k_values=[1, 3])
        # At K=1: only acme found (1/2 = 0.5)
        assert metrics.recall_at_k[1] == 0.5
        assert metrics.hit_rate_at_k[1] == 1.0
        assert metrics.precision_at_k[1] == 1.0
        # At K=3: both acme and globex found (2/2 = 1.0)
        assert metrics.recall_at_k[3] == 1.0
        assert metrics.hit_rate_at_k[3] == 1.0
        assert metrics.precision_at_k[3] == 2.0 / 3.0

    def test_evaluate_retrieval_empty_results(self, sample_answerable_item):
        metrics = evaluate_retrieval(sample_answerable_item, [], k_values=[1, 3, 5])
        assert metrics.hit_rate_at_k[1] == 0.0
        assert metrics.recall_at_k[1] == 0.0
        assert metrics.precision_at_k[1] == 0.0
        assert metrics.mrr == 0.0
        assert metrics.retrieved_count == 0

    def test_evaluate_retrieval_unanswerable_query(self, sample_unsupported_item):
        metrics = evaluate_retrieval(sample_unsupported_item, [], k_values=[1, 3])
        assert metrics.retrieved_count == 0
        assert metrics.relevant_retrieved_count == 0
        assert metrics.mrr == 0.0

    def test_evaluate_retrieval_k1_3_5_explicit(self):
        sample = EvaluationSample(
            id="k_test",
            question="Compare Acme and Globex",
            expected_answer="Ans",
            expected_keywords=[],
            relevant_documents=["acme_corp_annual_report_2024.txt", "globex_investor_presentation_2024.txt"],
            answerable=True,
        )
        c1 = DocumentChunk("1", "acme_corp_annual_report_2024.txt", "/p", 1, "c1", 0, "A")
        c2 = DocumentChunk("2", "initech_risk_disclosure_2024.txt", "/p", 1, "c2", 1, "I1")
        c3 = DocumentChunk("3", "acme_corp_annual_report_2024.txt", "/p", 1, "c3", 2, "A2")
        c4 = DocumentChunk("4", "globex_investor_presentation_2024.txt", "/p", 1, "c4", 3, "G")
        c5 = DocumentChunk("5", "initech_risk_disclosure_2024.txt", "/p", 1, "c5", 4, "I2")
        results = [
            SearchResult(c1, 0.95, 1),
            SearchResult(c2, 0.85, 2),
            SearchResult(c3, 0.75, 3),
            SearchResult(c4, 0.65, 4),
            SearchResult(c5, 0.55, 5),
        ]

        metrics = evaluate_retrieval(sample, results, k_values=[1, 3, 5])
        # K=1
        assert metrics.hit_rate_at_k[1] == 1.0
        assert metrics.recall_at_k[1] == 0.5
        assert metrics.precision_at_k[1] == 1.0
        # K=3
        assert metrics.hit_rate_at_k[3] == 1.0
        assert metrics.recall_at_k[3] == 0.5
        assert round(metrics.precision_at_k[3], 4) == round(2.0 / 3.0, 4)
        # K=5
        assert metrics.hit_rate_at_k[5] == 1.0
        assert metrics.recall_at_k[5] == 1.0
        assert round(metrics.precision_at_k[5], 4) == 0.6
        assert metrics.mrr == 1.0
        assert metrics.relevant_retrieved_count == 2
        assert metrics.retrieved_count == 5


# ── Answer Metrics Tests ─────────────────────────────────────────────


class TestAnswerMetrics:
    def test_evaluate_answer_supported_correct(self, sample_answerable_item, mock_retrieved_results):
        answer = GeneratedAnswer(
            query=sample_answerable_item.question,
            answer="According to acme_corp_annual_report_2024.txt: ACME revenue was INR 18,450 crore.",
            sources=["[Doc: acme_corp_annual_report_2024.txt, Page: 1]"],
            citations=[{"document_name": "acme_corp_annual_report_2024.txt", "score": 0.88}],
            retrieved_chunks=mock_retrieved_results,
            generator_mode="extractive",
            confidence_score=0.88,
            is_supported=True,
        )

        metrics = evaluate_answer(
            sample=sample_answerable_item,
            generated_answer=answer,
            retrieval_latency_ms=10.0,
            generation_latency_ms=5.0,
        )

        assert metrics.is_supported is True
        assert metrics.answerability_correct is True
        assert metrics.keyword_coverage == 1.0  # both "18,450" and "revenue" present
        assert metrics.citation_present is True
        assert metrics.citation_validity == 1.0
        assert metrics.grounded_in_context is True
        assert metrics.total_latency_ms == 15.0

    def test_evaluate_answer_unsupported_rejected(self, sample_unsupported_item):
        answer = GeneratedAnswer(
            query=sample_unsupported_item.question,
            answer="Based on the provided documents, there is insufficient information to answer this question.",
            sources=[],
            citations=[],
            retrieved_chunks=[],
            generator_mode="extractive",
            confidence_score=0.0,
            is_supported=False,
        )

        metrics = evaluate_answer(sample=sample_unsupported_item, generated_answer=answer)
        assert metrics.is_supported is False
        assert metrics.answerability_correct is True
        assert metrics.keyword_coverage == 1.0  # "insufficient information" matched
        assert metrics.citation_present is False
        assert metrics.citation_validity == 1.0  # 0 citations expected for unanswerable
        assert metrics.grounded_in_context is True

    def test_evaluate_answer_partial_keyword_coverage(self, sample_answerable_item, mock_retrieved_results):
        # sample has ["18,450", "revenue"], answer only has "revenue"
        answer = GeneratedAnswer(
            query=sample_answerable_item.question,
            answer="ACME reported high revenue.",
            sources=["[Doc: acme_corp_annual_report_2024.txt]"],
            citations=[{"document_name": "acme_corp_annual_report_2024.txt"}],
            retrieved_chunks=mock_retrieved_results,
            generator_mode="extractive",
            confidence_score=0.8,
            is_supported=True,
        )

        metrics = evaluate_answer(sample=sample_answerable_item, generated_answer=answer)
        assert metrics.keyword_coverage == 0.5  # 1 out of 2 keywords matched

    def test_evaluate_answer_invalid_citation(self, sample_answerable_item, mock_retrieved_results):
        # Citation points to wrong doc (initech instead of acme)
        answer = GeneratedAnswer(
            query=sample_answerable_item.question,
            answer="Revenue answer.",
            sources=["[Doc: initech_risk_disclosure_2024.txt]"],
            citations=[{"document_name": "initech_risk_disclosure_2024.txt"}],
            retrieved_chunks=mock_retrieved_results,
            generator_mode="extractive",
            confidence_score=0.7,
            is_supported=True,
        )

        metrics = evaluate_answer(sample=sample_answerable_item, generated_answer=answer)
        assert metrics.citation_validity == 0.0


# ── RAGEvaluator End-to-End Tests ────────────────────────────────────


class TestRAGEvaluator:
    def test_evaluator_end_to_end(self, tmp_path):
        # Build mini-pipeline with MockEmbedder
        doc_dir = tmp_path / "docs"
        doc_dir.mkdir()
        doc_file = doc_dir / "acme_corp_annual_report_2024.txt"
        doc_file.write_text(
            "ACME Corporation reported total revenue of INR 18,450 crore for FY 2024. "
            "Net profit after tax reached INR 2,214 crore.",
            encoding="utf-8",
        )

        vstore_dir = tmp_path / "vstore"

        cfg = {
            "rag": {
                "documents_dir": str(doc_dir),
                "vectorstore_dir": str(vstore_dir),
                "embedding_mode": "mock",
                "chunk_size": 200,
                "chunk_overlap": 20,
                "top_k": 3,
                "min_similarity": 0.0,
                "generator_mode": "extractive",
            }
        }

        pipeline = RAGPipeline(config=cfg)
        pipeline.ingest_documents()

        sample1 = EvaluationSample(
            id="s1",
            question="What was ACME Corporation total revenue?",
            expected_answer="INR 18,450 crore",
            expected_keywords=["18,450", "revenue"],
            relevant_documents=["acme_corp_annual_report_2024.txt"],
            relevant_pages=[1],
            answerable=True,
            category="Financial performance",
        )
        sample2 = EvaluationSample(
            id="s2",
            question="What is the launch date of ACME mission to Mars?",
            expected_answer="Insufficient information",
            expected_keywords=["insufficient information"],
            relevant_documents=[],
            relevant_pages=[],
            answerable=False,
            category="Unsupported questions",
        )

        evaluator = RAGEvaluator(pipeline)
        report = evaluator.evaluate([sample1, sample2], k_values=[1, 3], verbose=True)

        assert isinstance(report, AggregateEvaluationReport)
        assert report.total_samples == 2
        assert report.answerable_samples == 1
        assert report.unsupported_samples == 1

        # Retrieval check
        assert report.hit_rate_at_k[1] == 1.0
        assert report.mrr == 1.0

        # Answer quality check
        assert report.answerability_accuracy == 1.0
        assert report.supported_answer_rate == 1.0
        assert report.unsupported_rejection_rate == 1.0

        # Serialization check
        report_dict = report.to_dict()
        assert "summary" in report_dict
        assert "queries" in report_dict
        assert len(report_dict["queries"]) == 2


# ── Edge Cases & Robustness Tests ────────────────────────────────────


class TestEvaluationEdgeCases:
    def test_k_greater_than_number_of_chunks(self, sample_answerable_item):
        c1 = DocumentChunk("1", "acme_corp_annual_report_2024.txt", "/p", 1, "c1", 0, "A")
        results = [SearchResult(c1, 0.9, 1)]  # Only 1 chunk available

        # Evaluate with K=5, which is > len(results)
        metrics = evaluate_retrieval(sample_answerable_item, results, k_values=[1, 5, 10])
        assert metrics.hit_rate_at_k[5] == 1.0
        assert metrics.recall_at_k[5] == 1.0
        assert metrics.precision_at_k[5] == 1.0  # 1 relevant chunk out of min(5, 1) = 1
        assert metrics.mrr == 1.0

    def test_duplicate_chunks_from_same_document(self, sample_answerable_item):
        c1 = DocumentChunk("1", "acme_corp_annual_report_2024.txt", "/p", 1, "c1", 0, "A")
        c2 = DocumentChunk("2", "acme_corp_annual_report_2024.txt", "/p", 1, "c2", 1, "B")
        results = [SearchResult(c1, 0.9, 1), SearchResult(c2, 0.8, 2)]

        metrics = evaluate_retrieval(sample_answerable_item, results, k_values=[2])
        # Both chunks from relevant doc: recall is 1.0, precision is 2/2 = 1.0
        assert metrics.recall_at_k[2] == 1.0
        assert metrics.precision_at_k[2] == 1.0
        assert metrics.relevant_retrieved_count == 1  # 1 unique relevant document

    def test_missing_or_empty_metadata_fields(self, sample_answerable_item):
        # Chunk with empty document name
        c_anon = DocumentChunk("anon", "", "", None, "c_anon", 0, "Anonymous text")
        results = [SearchResult(c_anon, 0.5, 1)]

        metrics = evaluate_retrieval(sample_answerable_item, results, k_values=[1])
        assert metrics.hit_rate_at_k[1] == 0.0
        assert metrics.recall_at_k[1] == 0.0
        assert metrics.precision_at_k[1] == 0.0
        assert metrics.mrr == 0.0

    def test_empty_expected_keywords(self, sample_answerable_item):
        sample = EvaluationSample(
            id="nokw",
            question="No keywords query",
            expected_answer="Ans",
            expected_keywords=[],
            relevant_documents=["acme_corp_annual_report_2024.txt"],
            answerable=True,
        )
        answer = GeneratedAnswer(
            query="No keywords query",
            answer="Ans",
            sources=[],
            citations=[],
            retrieved_chunks=[],
            generator_mode="mock",
            confidence_score=0.8,
            is_supported=True,
        )
        metrics = evaluate_answer(sample, answer)
        assert metrics.keyword_coverage == 1.0

    def test_json_report_file_generation(self, tmp_path):
        report = AggregateEvaluationReport(
            total_samples=1,
            answerable_samples=1,
            unsupported_samples=0,
            recall_at_k={1: 1.0},
            precision_at_k={1: 1.0},
            hit_rate_at_k={1: 1.0},
            mrr=1.0,
            avg_retrieval_score=0.85,
            answerability_accuracy=1.0,
            supported_answer_rate=1.0,
            unsupported_rejection_rate=1.0,
            avg_keyword_coverage=1.0,
            citation_presence_rate=1.0,
            citation_validity_rate=1.0,
            avg_retrieval_latency_ms=5.0,
            avg_generation_latency_ms=2.0,
            avg_total_latency_ms=7.0,
            query_results=[],
        )

        out_file = tmp_path / "eval_out.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2)

        assert out_file.exists()
        loaded = json.loads(out_file.read_text(encoding="utf-8"))
        assert loaded["summary"]["total_samples"] == 1
        assert loaded["summary"]["retrieval"]["mrr"] == 1.0
