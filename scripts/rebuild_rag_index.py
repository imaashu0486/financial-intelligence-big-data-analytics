"""
Rebuild Financial Document RAG Vector Index.
Idempotently indexes documents in data/documents into data/vectorstore using
SentenceTransformer (all-MiniLM-L6-v2, 384 dimensions).
Ensures exactly 28 unique chunks and 28 embedding vectors.
"""

import sys
from pathlib import Path
import numpy as np
import json

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rag.pipeline import RAGPipeline
from src.utils.logger import logger


def rebuild() -> None:
    print("=" * 70)
    print("REBUILDING RAG VECTOR STORE INDEX")
    print("=" * 70)

    docs_dir = ROOT / "data" / "documents"
    vectorstore_dir = ROOT / "data" / "vectorstore"

    # Initialize RAGPipeline with default production config (SentenceTransformers)
    config = {
        "rag": {
            "documents_dir": str(docs_dir),
            "vectorstore_dir": str(vectorstore_dir),
            "embedding_mode": "auto",
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "chunk_size": 800,
            "chunk_overlap": 120,
        }
    }

    pipeline = RAGPipeline(config=config)
    print(f"Loaded embedder: {pipeline.embedder.__class__.__name__} (dim={pipeline.embedder.dimension})")

    # Ingest with reset=True
    num_chunks = pipeline.ingest_documents(documents_dir=docs_dir, reset=True)
    print(f"Indexed chunks count: {num_chunks}")
    print(f"Vector store length: {len(pipeline.vectorstore)}")

    # Verify idempotency: calling ingest_documents again should not duplicate
    num_chunks_2 = pipeline.ingest_documents(documents_dir=docs_dir, reset=True)
    assert len(pipeline.vectorstore) == num_chunks, "Idempotent rebuild failed!"

    # Save index
    saved_path = pipeline.save_index(vectorstore_dir)
    print(f"Index successfully saved to: {saved_path}")

    # Verify persisted files
    emb_path = vectorstore_dir / "embeddings.npy"
    meta_path = vectorstore_dir / "metadata.json"

    assert emb_path.exists(), "embeddings.npy does not exist!"
    assert meta_path.exists(), "metadata.json does not exist!"

    embeddings = np.load(emb_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    print(f"\n--- VERIFICATION OF SAVED INDEX ---")
    print(f"Embeddings shape : {embeddings.shape}")
    print(f"Metadata entries : {len(metadata)}")
    chunk_ids = [m["chunk_id"] for m in metadata]
    unique_ids = set(chunk_ids)
    print(f"Unique chunk IDs : {len(unique_ids)}")
    assert embeddings.shape == (28, 384), f"Expected (28, 384), got {embeddings.shape}"
    assert len(metadata) == 28, f"Expected 28 metadata items, got {len(metadata)}"
    assert len(unique_ids) == 28, f"Expected 28 unique chunk IDs, got {len(unique_ids)}"

    # Test Reloading in a fresh pipeline
    print(f"\n--- TESTING INDEX RELOAD ---")
    reload_pipeline = RAGPipeline(config=config)
    reloaded = reload_pipeline.load_index(vectorstore_dir)
    assert reloaded is True, "Failed to reload saved index!"
    assert len(reload_pipeline.vectorstore) == 28, f"Reloaded count {len(reload_pipeline.vectorstore)} != 28"
    assert reload_pipeline.vectorstore.dimension == 384, f"Reloaded dim {reload_pipeline.vectorstore.dimension} != 384"
    print("SUCCESS: Reloaded index verified with 28 vectors of dimension 384.")

    # Test Querying
    print(f"\n--- TESTING QUERY EXECUTION ---")
    q_supp = "What were ACME Corporation's operating profit and revenue for FY 2024?"
    ans_supp = reload_pipeline.query(q_supp)
    print(f"Supported Query     : '{q_supp}'")
    print(f"Is Supported        : {ans_supp.is_supported}")
    print(f"Confidence Score    : {ans_supp.confidence_score:.4f}")
    print(f"Sources Found       : {len(ans_supp.sources)}")

    q_unsupp = "What is the projected price of Bitcoin in 2045?"
    ans_unsupp = reload_pipeline.query(q_unsupp)
    print(f"Unsupported Query   : '{q_unsupp}'")
    print(f"Is Supported        : {ans_unsupp.is_supported}")
    print(f"Answer              : {ans_unsupp.answer}")
    assert ans_unsupp.is_supported is False, "Unsupported query was not rejected!"

    print("\n" + "=" * 70)
    print("RAG VECTOR STORE INDEX REBUILT AND VERIFIED: EXACTLY 28 VECTORS & 28 CHUNKS")
    print("=" * 70)


if __name__ == "__main__":
    rebuild()

