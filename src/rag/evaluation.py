"""
Evaluation framework for Financial Document RAG Pipeline.

Provides:
- Evaluation data structures (EvaluationSample, RetrievalMetrics, AnswerMetrics, EvaluationReport)
- Dataset loader and validator
- Mathematical metrics computation (Recall@K, Precision@K, Hit Rate@K, MRR)
- Answer quality assessment (Answerability accuracy, keyword coverage, citation validity)
- RAGEvaluator engine for orchestrating end-to-end evaluation
"""

import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

from src.rag.generator import GeneratedAnswer
from src.rag.pipeline import RAGPipeline
from src.rag.retriever import RetrievalResult
from src.rag.vectorstore import SearchResult
from src.utils.logger import logger


@dataclass
class EvaluationSample:
    """A single evaluation benchmark question with ground-truth annotations."""

    id: str
    question: str
    expected_answer: str
    expected_keywords: List[str]
    relevant_documents: List[str]
    relevant_pages: List[int] = field(default_factory=list)
    answerable: bool = True
    category: str = "General"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluationSample":
        """Validate and construct an EvaluationSample from a dictionary."""
        required = ["id", "question", "expected_answer", "relevant_documents", "answerable"]
        for field_name in required:
            if field_name not in data:
                raise ValueError(f"Missing required field in evaluation sample: '{field_name}'")

        return cls(
            id=str(data["id"]),
            question=str(data["question"]),
            expected_answer=str(data["expected_answer"]),
            expected_keywords=[str(k) for k in data.get("expected_keywords", [])],
            relevant_documents=[str(d) for d in data.get("relevant_documents", [])],
            relevant_pages=[int(p) for p in data.get("relevant_pages", [])],
            answerable=bool(data.get("answerable", True)),
            category=str(data.get("category", "General")),
        )


@dataclass
class RetrievalMetrics:
    """Retrieval evaluation metrics for a query across varying K cutoffs."""

    k_values: List[int]
    recall_at_k: Dict[int, float]
    precision_at_k: Dict[int, float]
    hit_rate_at_k: Dict[int, float]
    mrr: float
    avg_similarity_score: float
    retrieved_count: int
    relevant_retrieved_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "k_values": self.k_values,
            "recall_at_k": {str(k): round(v, 4) for k, v in self.recall_at_k.items()},
            "precision_at_k": {str(k): round(v, 4) for k, v in self.precision_at_k.items()},
            "hit_rate_at_k": {str(k): round(v, 4) for k, v in self.hit_rate_at_k.items()},
            "mrr": round(self.mrr, 4),
            "avg_similarity_score": round(self.avg_similarity_score, 4),
            "retrieved_count": self.retrieved_count,
            "relevant_retrieved_count": self.relevant_retrieved_count,
        }


@dataclass
class AnswerMetrics:
    """Answer quality and citation evaluation metrics for a query."""

    is_supported: bool
    answerability_correct: bool
    keyword_coverage: float
    citation_present: bool
    citation_validity: float
    grounded_in_context: bool
    retrieval_latency_ms: float
    generation_latency_ms: float
    total_latency_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_supported": self.is_supported,
            "answerability_correct": self.answerability_correct,
            "keyword_coverage": round(self.keyword_coverage, 4),
            "citation_present": self.citation_present,
            "citation_validity": round(self.citation_validity, 4),
            "grounded_in_context": self.grounded_in_context,
            "retrieval_latency_ms": round(self.retrieval_latency_ms, 2),
            "generation_latency_ms": round(self.generation_latency_ms, 2),
            "total_latency_ms": round(self.total_latency_ms, 2),
        }


@dataclass
class QueryEvaluationResult:
    """Combined evaluation result for a single benchmark question."""

    sample: EvaluationSample
    retrieval_metrics: RetrievalMetrics
    answer_metrics: AnswerMetrics
    generated_answer: str
    citations: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample.id,
            "question": self.sample.question,
            "category": self.sample.category,
            "answerable": self.sample.answerable,
            "retrieval": self.retrieval_metrics.to_dict(),
            "answer": self.answer_metrics.to_dict(),
            "generated_answer": self.generated_answer,
            "citations": self.citations,
        }


@dataclass
class AggregateEvaluationReport:
    """Aggregated evaluation metrics across all evaluated benchmark samples."""

    total_samples: int
    answerable_samples: int
    unsupported_samples: int

    # Retrieval aggregates (evaluated over answerable samples)
    recall_at_k: Dict[int, float]
    precision_at_k: Dict[int, float]
    hit_rate_at_k: Dict[int, float]
    mrr: float
    avg_retrieval_score: float

    # Answer quality aggregates
    answerability_accuracy: float
    supported_answer_rate: float
    unsupported_rejection_rate: float
    avg_keyword_coverage: float
    citation_presence_rate: float
    citation_validity_rate: float

    # Latency aggregates
    avg_retrieval_latency_ms: float
    avg_generation_latency_ms: float
    avg_total_latency_ms: float

    # Detailed results
    query_results: List[QueryEvaluationResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": {
                "total_samples": self.total_samples,
                "answerable_samples": self.answerable_samples,
                "unsupported_samples": self.unsupported_samples,
                "retrieval": {
                    "recall_at_k": {str(k): round(v, 4) for k, v in self.recall_at_k.items()},
                    "precision_at_k": {str(k): round(v, 4) for k, v in self.precision_at_k.items()},
                    "hit_rate_at_k": {str(k): round(v, 4) for k, v in self.hit_rate_at_k.items()},
                    "mrr": round(self.mrr, 4),
                    "avg_retrieval_score": round(self.avg_retrieval_score, 4),
                },
                "answer_quality": {
                    "answerability_accuracy": round(self.answerability_accuracy, 4),
                    "supported_answer_rate": round(self.supported_answer_rate, 4),
                    "unsupported_rejection_rate": round(self.unsupported_rejection_rate, 4),
                    "avg_keyword_coverage": round(self.avg_keyword_coverage, 4),
                    "citation_presence_rate": round(self.citation_presence_rate, 4),
                    "citation_validity_rate": round(self.citation_validity_rate, 4),
                },
                "latency_ms": {
                    "avg_retrieval": round(self.avg_retrieval_latency_ms, 2),
                    "avg_generation": round(self.avg_generation_latency_ms, 2),
                    "avg_total": round(self.avg_total_latency_ms, 2),
                },
            },
            "queries": [r.to_dict() for r in self.query_results],
        }


def load_evaluation_dataset(path: Union[str, Path]) -> List[EvaluationSample]:
    """
    Load and validate an evaluation benchmark dataset from JSON.

    Parameters
    ----------
    path : str or Path
        Path to dataset JSON file.

    Returns
    -------
    List[EvaluationSample]
        Validated list of samples.

    Raises
    ------
    FileNotFoundError
        If file does not exist.
    ValueError
        If JSON is malformed or missing required schema elements.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        try:
            raw_data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format in evaluation dataset: {e}")

    if not isinstance(raw_data, list):
        raise ValueError("Evaluation dataset root must be a JSON array of objects.")

    samples = []
    for idx, item in enumerate(raw_data):
        if not isinstance(item, dict):
            raise ValueError(f"Item at index {idx} is not a valid JSON object.")
        try:
            samples.append(EvaluationSample.from_dict(item))
        except Exception as e:
            raise ValueError(f"Error parsing sample at index {idx}: {e}")

    logger.info(f"Loaded {len(samples)} valid evaluation samples from {file_path}")
    return samples


def evaluate_retrieval(
    sample: EvaluationSample,
    retrieved_results: List[SearchResult],
    k_values: Optional[List[int]] = None,
) -> RetrievalMetrics:
    """
    Compute retrieval metrics (Recall@K, Precision@K, Hit Rate@K, MRR) for a query.

    Parameters
    ----------
    sample : EvaluationSample
        The benchmark query sample with relevant_documents.
    retrieved_results : List[SearchResult]
        Top retrieved chunks from Retriever.
    k_values : List[int], optional
        List of K cutoff thresholds (default: [1, 3, 5]).

    Returns
    -------
    RetrievalMetrics
    """
    if k_values is None:
        k_values = [1, 3, 5]

    sorted_k = sorted(list(set(k_values)))
    relevant_docs: Set[str] = set(sample.relevant_documents)

    recall_at_k: Dict[int, float] = {}
    precision_at_k: Dict[int, float] = {}
    hit_rate_at_k: Dict[int, float] = {}

    # If unanswerable / no relevant documents defined
    if not relevant_docs:
        for k in sorted_k:
            recall_at_k[k] = 1.0 if not retrieved_results else 0.0
            precision_at_k[k] = 0.0
            hit_rate_at_k[k] = 0.0
        return RetrievalMetrics(
            k_values=sorted_k,
            recall_at_k=recall_at_k,
            precision_at_k=precision_at_k,
            hit_rate_at_k=hit_rate_at_k,
            mrr=0.0,
            avg_similarity_score=0.0,
            retrieved_count=len(retrieved_results),
            relevant_retrieved_count=0,
        )

    # Document names of retrieved chunks in order of rank
    retrieved_doc_names = [res.chunk.document_name for res in retrieved_results]

    # Calculate Reciprocal Rank (MRR component)
    rr = 0.0
    for rank_idx, doc_name in enumerate(retrieved_doc_names, start=1):
        if doc_name in relevant_docs:
            rr = 1.0 / rank_idx
            break

    # Calculate metrics at each K
    for k in sorted_k:
        # Candidate window at K
        top_k_docs = retrieved_doc_names[:k]

        # Distinct relevant documents found in top K
        relevant_found_in_k = set(top_k_docs).intersection(relevant_docs)

        # Hit Rate @ K: 1 if any relevant doc is present in top K
        hit_rate_at_k[k] = 1.0 if len(relevant_found_in_k) > 0 else 0.0

        # Recall @ K: proportion of expected relevant docs retrieved
        recall_at_k[k] = len(relevant_found_in_k) / len(relevant_docs) if relevant_docs else 0.0

        # Precision @ K: fraction of retrieved chunks at K from relevant docs
        # Number of chunks from relevant docs in top-K
        relevant_chunks_in_k = sum(1 for d in top_k_docs if d in relevant_docs)
        denom = min(k, len(top_k_docs)) if len(top_k_docs) > 0 else k
        precision_at_k[k] = relevant_chunks_in_k / denom if denom > 0 else 0.0

    # Overall similarity score
    scores = [res.score for res in retrieved_results]
    avg_score = float(sum(scores) / len(scores)) if scores else 0.0

    total_relevant_retrieved = len(set(retrieved_doc_names).intersection(relevant_docs))

    return RetrievalMetrics(
        k_values=sorted_k,
        recall_at_k=recall_at_k,
        precision_at_k=precision_at_k,
        hit_rate_at_k=hit_rate_at_k,
        mrr=rr,
        avg_similarity_score=avg_score,
        retrieved_count=len(retrieved_results),
        relevant_retrieved_count=total_relevant_retrieved,
    )


def evaluate_answer(
    sample: EvaluationSample,
    generated_answer: GeneratedAnswer,
    retrieval_latency_ms: float = 0.0,
    generation_latency_ms: float = 0.0,
) -> AnswerMetrics:
    """
    Evaluate generated answer correctness, keyword coverage, citation validity, and latency.

    Parameters
    ----------
    sample : EvaluationSample
        Annotated benchmark sample.
    generated_answer : GeneratedAnswer
        Output from RAG pipeline.
    retrieval_latency_ms : float
        Retrieval latency in milliseconds.
    generation_latency_ms : float
        Generation latency in milliseconds.

    Returns
    -------
    AnswerMetrics
    """
    # 1. Answerability classification accuracy
    # System correctly accepted if answerable=True and is_supported=True,
    # OR correctly rejected if answerable=False and is_supported=False.
    answerability_correct = (generated_answer.is_supported == sample.answerable)

    ans_text_lower = generated_answer.answer.lower()

    # 2. Keyword coverage
    if sample.expected_keywords:
        matched_kw = 0
        for kw in sample.expected_keywords:
            kw_clean = kw.strip().lower()
            if kw_clean in ans_text_lower:
                matched_kw += 1
        keyword_coverage = matched_kw / len(sample.expected_keywords)
    else:
        keyword_coverage = 1.0

    # 3. Citation presence & validity
    citations = generated_answer.citations or []
    sources = generated_answer.sources or []
    has_citations = len(citations) > 0 or len(sources) > 0

    if not sample.answerable:
        # For unsupported questions, ideal citation count is 0
        citation_validity = 1.0 if not has_citations else 0.0
    else:
        if not citations:
            citation_validity = 0.0
        else:
            valid_count = 0
            rel_docs = set(sample.relevant_documents)
            for c in citations:
                doc_name = c.get("document_name", "")
                if doc_name in rel_docs:
                    valid_count += 1
            citation_validity = valid_count / len(citations) if citations else 0.0

    # 4. Groundedness in retrieved context
    grounded_in_context = False
    if generated_answer.is_supported and generated_answer.retrieved_chunks:
        # Check if answer quotes or excerpts overlap with any retrieved chunk
        for chunk_res in generated_answer.retrieved_chunks:
            chunk_words = set(re.findall(r"\b\w{4,}\b", chunk_res.chunk.text.lower()))
            ans_words = set(re.findall(r"\b\w{4,}\b", ans_text_lower))
            if len(chunk_words.intersection(ans_words)) >= 3:
                grounded_in_context = True
                break
    elif not generated_answer.is_supported:
        grounded_in_context = True  # Correctly stated insufficient info

    total_latency_ms = retrieval_latency_ms + generation_latency_ms

    return AnswerMetrics(
        is_supported=generated_answer.is_supported,
        answerability_correct=answerability_correct,
        keyword_coverage=keyword_coverage,
        citation_present=has_citations,
        citation_validity=citation_validity,
        grounded_in_context=grounded_in_context,
        retrieval_latency_ms=retrieval_latency_ms,
        generation_latency_ms=generation_latency_ms,
        total_latency_ms=total_latency_ms,
    )


class RAGEvaluator:
    """
    Orchestrator for evaluating a RAGPipeline against benchmark datasets.
    """

    def __init__(self, pipeline: RAGPipeline) -> None:
        self.pipeline = pipeline

    def evaluate_sample(
        self,
        sample: EvaluationSample,
        k_values: Optional[List[int]] = None,
    ) -> QueryEvaluationResult:
        """Evaluate a single query sample through retrieval and generation."""
        if k_values is None:
            k_values = [1, 3, 5]

        max_k = max(k_values)

        # 1. Measure retrieval
        t0 = time.perf_counter()
        retrieval_res: RetrievalResult = self.pipeline.retriever.retrieve(
            query=sample.question,
            top_k=max_k,
            min_score=-1.0,  # Evaluate ranking across all candidates (cosine similarity in [-1, 1])
        )
        retrieval_ms = (time.perf_counter() - t0) * 1000.0

        # Compute retrieval metrics
        ret_metrics = evaluate_retrieval(
            sample=sample,
            retrieved_results=retrieval_res.results,
            k_values=k_values,
        )

        # 2. Measure generation
        t1 = time.perf_counter()
        generated_answer: GeneratedAnswer = self.pipeline.generator.generate(
            query=sample.question,
            retrieved_chunks=retrieval_res.results,
        )
        generation_ms = (time.perf_counter() - t1) * 1000.0

        # Compute answer metrics
        ans_metrics = evaluate_answer(
            sample=sample,
            generated_answer=generated_answer,
            retrieval_latency_ms=retrieval_ms,
            generation_latency_ms=generation_ms,
        )

        return QueryEvaluationResult(
            sample=sample,
            retrieval_metrics=ret_metrics,
            answer_metrics=ans_metrics,
            generated_answer=generated_answer.answer,
            citations=generated_answer.citations,
        )

    def evaluate(
        self,
        dataset: List[EvaluationSample],
        k_values: Optional[List[int]] = None,
        verbose: bool = False,
    ) -> AggregateEvaluationReport:
        """
        Run complete evaluation across all benchmark samples and aggregate results.

        Parameters
        ----------
        dataset : List[EvaluationSample]
            List of evaluation questions.
        k_values : List[int], optional
            Cutoff levels for Recall/Precision/HitRate (default: [1, 3, 5]).
        verbose : bool, default False
            If True, logs per-sample evaluation results.

        Returns
        -------
        AggregateEvaluationReport
        """
        if k_values is None:
            k_values = [1, 3, 5]

        sorted_k = sorted(list(set(k_values)))
        query_results: List[QueryEvaluationResult] = []

        for idx, sample in enumerate(dataset, start=1):
            res = self.evaluate_sample(sample, k_values=sorted_k)
            query_results.append(res)

            if verbose:
                status_str = "PASS" if res.answer_metrics.answerability_correct else "FAIL"
                hit1 = res.retrieval_metrics.hit_rate_at_k.get(1, 0.0)
                logger.info(
                    f"[{idx:02d}/{len(dataset):02d}] ({status_str}) {sample.id} "
                    f"[{sample.category}] Hit@1={hit1:.0f} | "
                    f"Coverage={res.answer_metrics.keyword_coverage:.2f} | "
                    f"Q: '{sample.question[:45]}...'"
                )

        # Aggregate metrics
        total_samples = len(dataset)
        answerable_samples = [r for r in query_results if r.sample.answerable]
        unsupported_samples = [r for r in query_results if not r.sample.answerable]

        num_ans = len(answerable_samples)
        num_unsupp = len(unsupported_samples)

        # Retrieval metrics are averaged over answerable queries
        agg_recall: Dict[int, float] = {}
        agg_precision: Dict[int, float] = {}
        agg_hit_rate: Dict[int, float] = {}

        for k in sorted_k:
            if num_ans > 0:
                agg_recall[k] = sum(r.retrieval_metrics.recall_at_k.get(k, 0.0) for r in answerable_samples) / num_ans
                agg_precision[k] = sum(r.retrieval_metrics.precision_at_k.get(k, 0.0) for r in answerable_samples) / num_ans
                agg_hit_rate[k] = sum(r.retrieval_metrics.hit_rate_at_k.get(k, 0.0) for r in answerable_samples) / num_ans
            else:
                agg_recall[k] = 0.0
                agg_precision[k] = 0.0
                agg_hit_rate[k] = 0.0

        mrr = sum(r.retrieval_metrics.mrr for r in answerable_samples) / num_ans if num_ans > 0 else 0.0
        avg_ret_score = sum(r.retrieval_metrics.avg_similarity_score for r in answerable_samples) / num_ans if num_ans > 0 else 0.0

        # Answer metrics aggregates
        correct_answerability = sum(1 for r in query_results if r.answer_metrics.answerability_correct)
        answerability_acc = correct_answerability / total_samples if total_samples > 0 else 0.0

        supported_ans_count = sum(1 for r in answerable_samples if r.answer_metrics.is_supported)
        supported_answer_rate = supported_ans_count / num_ans if num_ans > 0 else 0.0

        rejected_unsupp_count = sum(1 for r in unsupported_samples if not r.answer_metrics.is_supported)
        unsupported_rejection_rate = rejected_unsupp_count / num_unsupp if num_unsupp > 0 else 1.0

        avg_kw_coverage = sum(r.answer_metrics.keyword_coverage for r in answerable_samples) / num_ans if num_ans > 0 else 1.0
        citation_presence_rate = sum(1 for r in answerable_samples if r.answer_metrics.citation_present) / num_ans if num_ans > 0 else 0.0
        citation_validity_rate = sum(r.answer_metrics.citation_validity for r in answerable_samples) / num_ans if num_ans > 0 else 1.0

        # Latencies
        avg_ret_lat = sum(r.answer_metrics.retrieval_latency_ms for r in query_results) / total_samples if total_samples > 0 else 0.0
        avg_gen_lat = sum(r.answer_metrics.generation_latency_ms for r in query_results) / total_samples if total_samples > 0 else 0.0
        avg_tot_lat = sum(r.answer_metrics.total_latency_ms for r in query_results) / total_samples if total_samples > 0 else 0.0

        report = AggregateEvaluationReport(
            total_samples=total_samples,
            answerable_samples=num_ans,
            unsupported_samples=num_unsupp,
            recall_at_k=agg_recall,
            precision_at_k=agg_precision,
            hit_rate_at_k=agg_hit_rate,
            mrr=mrr,
            avg_retrieval_score=avg_ret_score,
            answerability_accuracy=answerability_acc,
            supported_answer_rate=supported_answer_rate,
            unsupported_rejection_rate=unsupported_rejection_rate,
            avg_keyword_coverage=avg_kw_coverage,
            citation_presence_rate=citation_presence_rate,
            citation_validity_rate=citation_validity_rate,
            avg_retrieval_latency_ms=avg_ret_lat,
            avg_generation_latency_ms=avg_gen_lat,
            avg_total_latency_ms=avg_tot_lat,
            query_results=query_results,
        )

        return report
