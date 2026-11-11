"""
HDFS Data Organization and Ingestion Manager for Real Financial Data (Step 7).

Manages the real-data HDFS hierarchy:
    /financial-intelligence/real/
        raw/
            market/
            news/
            documents/
        processed/
            market/
            benchmark/
            news/
            documents/
        metadata/

Generates cryptographic ingestion manifests with SHA-256 checksums,
file sizes, target HDFS paths, and replication factor 2.
"""

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from src.utils.logger import logger

HDFS_PROJECT_ROOT = "/financial-intelligence/real"
DEFAULT_MANIFEST_PATH = Path("data/real/metadata/hdfs_ingestion_manifest.json")


def compute_sha256(filepath: Union[str, Path]) -> str:
    """Compute SHA-256 hash of a local file."""
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()


def generate_hdfs_file_catalog() -> List[Dict[str, Any]]:
    """
    Build the complete catalog of real files to be ingested into HDFS.
    """
    catalog = []

    # 1. Raw Market Files (51 CSVs)
    raw_market_dir = Path("data/real/raw/market")
    if raw_market_dir.exists():
        for f in sorted(raw_market_dir.glob("*.csv")):
            catalog.append({
                "local_path": str(f),
                "original_filename": f.name,
                "hdfs_destination": f"{HDFS_PROJECT_ROOT}/raw/market/{f.name}",
                "source_dataset": "real_raw_market",
                "category": "raw",
            })

    # 2. Raw News (financial_news.csv)
    raw_news = Path("data/real/raw/news/financial_news.csv")
    if raw_news.exists():
        catalog.append({
            "local_path": str(raw_news),
            "original_filename": raw_news.name,
            "hdfs_destination": f"{HDFS_PROJECT_ROOT}/raw/news/{raw_news.name}",
            "source_dataset": "real_raw_news",
            "category": "raw",
        })

    # 3. Raw Documents (4 text files)
    raw_docs_dir = Path("data/real/raw/documents")
    if raw_docs_dir.exists():
        for f in sorted(raw_docs_dir.glob("*.txt")):
            catalog.append({
                "local_path": str(f),
                "original_filename": f.name,
                "hdfs_destination": f"{HDFS_PROJECT_ROOT}/raw/documents/{f.name}",
                "source_dataset": "real_raw_documents",
                "category": "raw",
            })

    # 4. Processed Market Parquet & CSVs
    proc_market_dir = Path("data/real/processed/market")
    if proc_market_dir.exists():
        for name in ["consolidated_market_features.parquet", "consolidated_market_features.csv"]:
            p = proc_market_dir / name
            if p.exists():
                catalog.append({
                    "local_path": str(p),
                    "original_filename": p.name,
                    "hdfs_destination": f"{HDFS_PROJECT_ROOT}/processed/market/{p.name}",
                    "source_dataset": "real_processed_market",
                    "category": "processed",
                })

    # 5. Processed Benchmark Parquet & CSV
    bench_dir = Path("data/real/processed/market")
    if bench_dir.exists():
        for name in ["benchmark_features.parquet", "benchmark_features.csv"]:
            p = bench_dir / name
            if p.exists():
                catalog.append({
                    "local_path": str(p),
                    "original_filename": p.name,
                    "hdfs_destination": f"{HDFS_PROJECT_ROOT}/processed/benchmark/{p.name}",
                    "source_dataset": "real_processed_benchmark",
                    "category": "processed",
                })

    # 6. Processed News Files
    proc_news_dir = Path("data/real/processed/news")
    if proc_news_dir.exists():
        for f in sorted(proc_news_dir.glob("*.csv")):
            catalog.append({
                "local_path": str(f),
                "original_filename": f.name,
                "hdfs_destination": f"{HDFS_PROJECT_ROOT}/processed/news/{f.name}",
                "source_dataset": "real_processed_news",
                "category": "processed",
            })

    # 7. Processed Clean Documents
    proc_docs_dir = Path("data/real/processed/documents/cleaned_corporate_documents")
    if proc_docs_dir.exists():
        for f in sorted(proc_docs_dir.glob("*.txt")):
            catalog.append({
                "local_path": str(f),
                "original_filename": f.name,
                "hdfs_destination": f"{HDFS_PROJECT_ROOT}/processed/documents/{f.name}",
                "source_dataset": "real_processed_documents",
                "category": "processed",
            })

    # 8. Metadata
    meta_dir = Path("data/real/metadata")
    if meta_dir.exists():
        for f in sorted(meta_dir.glob("*.json")) + sorted(meta_dir.glob("*.csv")):
            if f.name != "hdfs_ingestion_manifest.json":
                catalog.append({
                    "local_path": str(f),
                    "original_filename": f.name,
                    "hdfs_destination": f"{HDFS_PROJECT_ROOT}/metadata/{f.name}",
                    "source_dataset": "real_metadata",
                    "category": "metadata",
                })

    # Enrich catalog with file sizes and SHA-256 hashes
    enriched = []
    for item in catalog:
        lp = Path(item["local_path"])
        enriched.append({
            "original_filename": item["original_filename"],
            "local_path": str(lp),
            "hdfs_destination": item["hdfs_destination"],
            "source_dataset": item["source_dataset"],
            "category": item["category"],
            "file_size_bytes": lp.stat().st_size,
            "sha256_checksum": compute_sha256(lp),
            "upload_timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "target_replication": 2,
            "ingestion_status": "PREPARED",
        })

    return enriched


def generate_hdfs_manifest(manifest_path: Union[str, Path] = DEFAULT_MANIFEST_PATH) -> Dict[str, Any]:
    """
    Generate and save the master HDFS ingestion manifest.
    """
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    catalog = generate_hdfs_file_catalog()

    total_files = len(catalog)
    total_bytes = sum(c["file_size_bytes"] for c in catalog)
    categories = {}
    for c in catalog:
        cat = c["category"]
        categories[cat] = categories.get(cat, 0) + 1

    manifest = {
        "manifest_timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "hdfs_project_root": HDFS_PROJECT_ROOT,
        "replication_factor": 2,
        "summary": {
            "total_files_cataloged": total_files,
            "total_bytes": total_bytes,
            "total_mb": round(total_bytes / (1024 * 1024), 2),
            "categories_breakdown": categories,
        },
        "target_directories": [
            f"{HDFS_PROJECT_ROOT}/raw/market",
            f"{HDFS_PROJECT_ROOT}/raw/news",
            f"{HDFS_PROJECT_ROOT}/raw/documents",
            f"{HDFS_PROJECT_ROOT}/processed/market",
            f"{HDFS_PROJECT_ROOT}/processed/benchmark",
            f"{HDFS_PROJECT_ROOT}/processed/news",
            f"{HDFS_PROJECT_ROOT}/processed/documents",
            f"{HDFS_PROJECT_ROOT}/metadata",
        ],
        "files": catalog,
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info(
        f"HDFS ingestion manifest written to {manifest_path} "
        f"({total_files} files, {round(total_bytes / (1024 * 1024), 2)} MB)"
    )
    return manifest


if __name__ == "__main__":
    generate_hdfs_manifest()

