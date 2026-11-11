"""
CLI script to build, manage, and query the Financial Document RAG Pipeline.

Usage:
    # Build vector index from documents in data/documents
    python scripts/run_rag.py --build-index

    # Ask a financial question
    python scripts/run_rag.py --query "What were ACME Corp's financial results and revenue for FY 2024?"

    # Query using mock database (no live MongoDB required)
    python scripts/run_rag.py --query "What are Globex's growth strategies?" --mock

    # Run in interactive mode
    python scripts/run_rag.py --interactive --mock

    # Check index status
    python scripts/run_rag.py --status
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.database.mongodb import check_connection, get_database
from src.database.queries import (
    get_recent_rag_queries,
    store_rag_document_manifest,
    store_rag_query,
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


def init_database(use_mock: bool = False) -> Any:
    """Initialize MongoDB connection or mongomock client."""
    if use_mock:
        import mongomock
        logger.info("Using mongomock in-memory database")
        return mongomock.MongoClient()["financial_analytics"]

    health = check_connection()
    if isinstance(health, dict) and health.get("connected", False):
        logger.info("Connected to live MongoDB instance")
        return get_database()
    else:
        import mongomock
        logger.warning("Live MongoDB not available. Falling back to mongomock.")
        return mongomock.MongoClient()["financial_analytics"]


def build_index(pipeline: RAGPipeline, db: Any) -> int:
    """Index all documents in the configured documents directory."""
    print("\n" + "=" * 70)
    print("Building Financial Document Vector Index")
    print("=" * 70)

    num_chunks = pipeline.ingest_documents()
    if num_chunks == 0:
        print("[WARNING] No chunks were indexed. Check that documents exist in data/documents/")
        return 0

    save_dir = pipeline.save_index()
    print(f"[SUCCESS] Indexed {num_chunks} chunks.")
    print(f"[SUCCESS] Vector store index saved to: {save_dir}")

    # Build manifest for MongoDB
    manifest = []
    for doc_name in pipeline._indexed_documents:
        manifest.append({
            "document_id": doc_name.replace(".", "_"),
            "document_name": doc_name,
            "source_dir": str(pipeline.documents_dir),
            "chunk_count": sum(1 for c in pipeline.vectorstore._chunks if c.document_name == doc_name),
        })

    try:
        saved_count = store_rag_document_manifest(manifest, db=db)
        print(f"[SUCCESS] Document manifest ({saved_count} documents) logged to MongoDB.")
    except Exception as e:
        logger.warning(f"Could not log manifest to MongoDB: {e}")

    return num_chunks


def run_query(pipeline: RAGPipeline, question: str, db: Any) -> None:
    """Run a query through the RAG pipeline and display results."""
    print("\n" + "=" * 70)
    print(f"QUESTION: {question}")
    print("=" * 70)

    answer = pipeline.query(question)

    print("\n--- GROUNDED ANSWER ---")
    print(answer.answer)

    print("\n--- SOURCES & CITATIONS ---")
    if answer.sources:
        for s in answer.sources:
            print(f"  • {s}")
    else:
        print("  (No relevant sources found)")

    print(f"\nConfidence Score : {answer.confidence_score:.4f}")
    print(f"Supported Status : {'Supported' if answer.is_supported else 'Insufficient Information'}")
    print(f"Generator Mode   : {answer.generator_mode}")

    print("\n" + "-" * 70)
    print(answer.disclaimer)
    print("-" * 70)

    # Store query in MongoDB
    try:
        query_id = store_rag_query(
            query=question,
            answer=answer.answer,
            sources=answer.sources,
            citations=answer.citations,
            generator_mode=answer.generator_mode,
            confidence_score=answer.confidence_score,
            is_supported=answer.is_supported,
            db=db,
        )
        logger.info(f"Query logged to MongoDB (id={query_id})")
    except Exception as e:
        logger.warning(f"Could not log query to MongoDB: {e}")


def interactive_mode(pipeline: RAGPipeline, db: Any) -> None:
    """Run an interactive Q&A loop."""
    print("\n" + "=" * 70)
    print("Financial Document Intelligence Platform — Interactive RAG")
    print("Type your question below, or 'exit' / 'quit' to end.")
    print("=" * 70)

    while True:
        try:
            query_text = input("\nEnter question > ").strip()
            if not query_text:
                continue
            if query_text.lower() in ("exit", "quit", "q"):
                print("Exiting RAG interactive session. Goodbye!")
                break
            run_query(pipeline, query_text, db)
        except (KeyboardInterrupt, EOFError):
            print("\nSession interrupted.")
            break


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Financial Document RAG Pipeline CLI"
    )
    parser.add_argument(
        "--build-index",
        "--rebuild",
        action="store_true",
        dest="build_index",
        help="Build or rebuild vector index from documents in data/documents",
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Query the RAG pipeline with a financial question",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Start interactive conversational Q&A prompt",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use in-memory mongomock instead of live MongoDB",
    )
    parser.add_argument(
        "--embedding-mode",
        type=str,
        choices=["auto", "mock", "sentence-transformer"],
        default=None,
        help="Embedding mode override",
    )
    parser.add_argument(
        "--generator-mode",
        type=str,
        choices=["extractive", "mock"],
        default=None,
        help="Generator mode override",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Print status of vector index and recent queries",
    )

    args = parser.parse_args()

    config = load_config()

    # Apply CLI overrides to config
    if args.embedding_mode:
        config.setdefault("rag", {})["embedding_mode"] = args.embedding_mode
    if args.generator_mode:
        config.setdefault("rag", {})["generator_mode"] = args.generator_mode

    db = init_database(use_mock=args.mock)

    pipeline = RAGPipeline(config=config)

    # If index exists on disk, load it
    index_loaded = pipeline.load_index()

    if args.build_index or not index_loaded:
        if not index_loaded and (args.query or args.interactive):
            print("[INFO] No existing index found on disk. Building index first...")
        build_index(pipeline, db)

    if args.status:
        print("\n" + "=" * 50)
        print("RAG Pipeline Status")
        print("=" * 50)
        status = pipeline.get_status()
        for k, v in status.items():
            print(f"  {k:22}: {v}")

        try:
            recent = get_recent_rag_queries(limit=5, db=db)
            print(f"\nRecent Queries in DB ({len(recent)}):")
            for q in recent:
                print(f"  [{q.get('timestamp', 'N/A')[:19]}] Q: {q.get('query', '')[:50]}...")
        except Exception as e:
            logger.debug(f"Could not fetch recent queries: {e}")
        return

    if args.query:
        run_query(pipeline, args.query, db)
    elif args.interactive:
        interactive_mode(pipeline, db)
    elif not args.build_index:
        parser.print_help()


if __name__ == "__main__":
    main()
