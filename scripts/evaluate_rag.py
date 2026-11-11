"""
CLI evaluation runner for Financial Document RAG Pipeline (Phase 8).

Usage:
    # Run evaluation with default parameters in mock mode (fast, deterministic, offline)
    python scripts/evaluate_rag.py --mock

    # Run with custom K cutoffs and verbose logging
    python scripts/evaluate_rag.py --mock --k 1,3,5 --verbose

    # Save machine-readable evaluation report
    python scripts/evaluate_rag.py --mock --json-output data/evaluation/results.json

    # Run with production sentence-transformers embeddings
    python scripts/evaluate_rag.py --verbose --json-output data/evaluation/results_minilm.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.rag.evaluation import (
    AggregateEvaluationReport,
    RAGEvaluator,
    load_evaluation_dataset,
)
from src.rag.pipeline import RAGPipeline
from src.utils.logger import logger


def load_config() -> Dict[str, Any]:
    """Load configuration from config/config.yaml."""
    config_path = ROOT / "config" / "config.yaml"
    if not config_path.exists():
        logger.warning(f"Config file not found at {config_path}, using defaults")
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def format_table_row(col1: str, col2: str, col3: str = "", width: int = 68) -> str:
    """Format a consistent aligned table row."""
    if col3:
        return f"  {col1:<28} | {col2:<18} | {col3}"
    return f"  {col1:<38} : {col2}"


def print_evaluation_summary(report: AggregateEvaluationReport) -> None:
    """Print an aesthetically formatted summary report of RAG metrics."""
    sep = "=" * 70
    subsep = "-" * 70

    print("\n" + sep)
    print("FINANCIAL DOCUMENT RAG EVALUATION REPORT (PHASE 8)")
    print(sep)

    print("\n[1] DATASET BENCHMARK SUMMARY")
    print(subsep)
    print(format_table_row("Total Benchmark Samples", str(report.total_samples)))
    print(format_table_row("Answerable Questions", str(report.answerable_samples)))
    print(format_table_row("Unsupported / Unanswerable Questions", str(report.unsupported_samples)))

    print("\n[2] RETRIEVAL PERFORMANCE (Answerable Samples)")
    print(subsep)
    print(format_table_row("Metric", "Cutoff (K)", "Value"))
    print(f"  {'-'*28}-|-{'-'*18}-|-{'-'*16}")

    for k in sorted(report.recall_at_k.keys()):
        rec = report.recall_at_k[k]
        prec = report.precision_at_k[k]
        hit = report.hit_rate_at_k[k]
        print(f"  Recall@{k:<21} | K={k:<16} | {rec * 100:.1f}%")
        print(f"  Precision@{k:<18} | K={k:<16} | {prec * 100:.1f}%")
        print(f"  Hit Rate@{k:<19} | K={k:<16} | {hit * 100:.1f}%")

    print(f"  {'-'*28}-|-{'-'*18}-|-{'-'*16}")
    print(format_table_row("Mean Reciprocal Rank (MRR)", f"{report.mrr:.4f}"))
    print(format_table_row("Average Retrieval Similarity Score", f"{report.avg_retrieval_score:.4f}"))

    print("\n[3] ANSWER QUALITY & GROUNDEDNESS")
    print(subsep)
    print(format_table_row("Answerability Classification Accuracy", f"{report.answerability_accuracy * 100:.1f}%"))
    print(format_table_row("Supported-Question Answer Rate", f"{report.supported_answer_rate * 100:.1f}%"))
    print(format_table_row("Unsupported-Question Rejection Rate", f"{report.unsupported_rejection_rate * 100:.1f}%"))
    print(format_table_row("Average Keyword Coverage", f"{report.avg_keyword_coverage * 100:.1f}%"))
    print(format_table_row("Citation Presence Rate (Answerable)", f"{report.citation_presence_rate * 100:.1f}%"))
    print(format_table_row("Citation Document Validity Rate", f"{report.citation_validity_rate * 100:.1f}%"))

    print("\n[4] SYSTEM LATENCIES")
    print(subsep)
    print(format_table_row("Average Retrieval Latency", f"{report.avg_retrieval_latency_ms:.2f} ms"))
    print(format_table_row("Average Generation Latency", f"{report.avg_generation_latency_ms:.2f} ms"))
    print(format_table_row("Average Total Round-Trip Latency", f"{report.avg_total_latency_ms:.2f} ms"))

    print("\n" + sep)
    print("NOTE: Evaluation is grounded in synthetic demo documents. Keyword matching is an")
    print("automated proxy metric and does not substitute for expert financial analysis.")
    print(sep + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Financial Document RAG Pipeline (Phase 8)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=str(ROOT / "data" / "evaluation" / "rag_eval_dataset.json"),
        help="Path to evaluation benchmark dataset JSON",
    )
    parser.add_argument(
        "--k",
        type=str,
        default="1,3,5",
        help="Comma-separated K cutoff values for retrieval metrics (default: 1,3,5)",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use deterministic mock embedder for rapid offline evaluation",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed per-sample evaluation breakdown",
    )
    parser.add_argument(
        "--json-output",
        type=str,
        default=None,
        help="Path to save full evaluation report as JSON",
    )
    parser.add_argument(
        "--build-index",
        action="store_true",
        help="Rebuild vector index from data/documents before evaluating",
    )

    args = parser.parse_args()

    # Parse K values
    try:
        k_values = [int(x.strip()) for x in args.k.split(",") if x.strip()]
        if not k_values:
            k_values = [1, 3, 5]
    except ValueError:
        logger.error(f"Invalid --k parameter: {args.k}. Must be comma-separated integers.")
        sys.exit(1)

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        logger.error(f"Evaluation dataset does not exist: {dataset_path}")
        sys.exit(1)

    print(f"Loading benchmark dataset from: {dataset_path}")
    dataset = load_evaluation_dataset(dataset_path)

    # Configure pipeline
    config = load_config()
    if args.mock:
        config.setdefault("rag", {})["embedding_mode"] = "mock"
        config.setdefault("rag", {})["generator_mode"] = "extractive"

    pipeline = RAGPipeline(config=config)

    # Build or load index
    index_loaded = pipeline.load_index()
    if args.build_index or not index_loaded or len(pipeline.vectorstore) == 0:
        print("[INFO] Building vector index from documents...")
        chunks_indexed = pipeline.ingest_documents()
        pipeline.save_index()
        print(f"[INFO] Indexed {chunks_indexed} chunks into vector store.")

    print(f"[INFO] Vector store initialized with {len(pipeline.vectorstore)} chunks.")
    print(f"[INFO] Running evaluation over {len(dataset)} samples with K={k_values} (mock={args.mock})...\n")

    evaluator = RAGEvaluator(pipeline)
    report = evaluator.evaluate(
        dataset=dataset,
        k_values=k_values,
        verbose=args.verbose,
    )

    # Print formatted console report
    print_evaluation_summary(report)

    # Optional JSON output
    if args.json_output:
        out_path = Path(args.json_output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        report_dict = report.to_dict()
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report_dict, f, indent=2)
        print(f"[SUCCESS] Machine-readable report saved to: {out_path}\n")


if __name__ == "__main__":
    main()
