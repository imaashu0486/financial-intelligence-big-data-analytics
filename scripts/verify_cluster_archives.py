"""
Verification Engine for Local Apache Cluster Installation Archives.

Verifies the cryptographic integrity of:
1. docker/cluster/dist/hadoop-3.3.6.tar.gz (SHA-512)
2. docker/cluster/dist/spark-3.5.1-bin-hadoop3.tgz (SHA-512)
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DIST_DIR = PROJECT_ROOT / "docker" / "cluster" / "dist"

OFFICIAL_SPEC = {
    "hadoop-3.3.6.tar.gz": {
        "version": "3.3.6",
        "expected_sha512": "de3eaca2e0517e4b569a88b63c89fae19cb8ac6c01ff990f1ff8f0cc0f3128c8e8a23db01577ca562a0e0bb1b4a3889f8c74384e609cd55e537aada3dcaa9f8a",
        "official_source": "https://archive.apache.org/dist/hadoop/common/hadoop-3.3.6/hadoop-3.3.6.tar.gz",
    },
    "spark-3.5.1-bin-hadoop3.tgz": {
        "version": "3.5.1",
        "expected_sha512": "3d8e3f082c602027d540771e9eba9987f8ea955e978afc29e1349eb6e3f9fe32543e3d3de52dff048ebbd789730454c96447c86ff5b60a98d72620a0f082b355",
        "official_source": "https://archive.apache.org/dist/spark/spark-3.5.1/spark-3.5.1-bin-hadoop3.tgz",
    },
}


def compute_sha512(filepath: Path) -> str:
    h = hashlib.sha512()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024 * 4):  # 4MB chunks
            h.update(chunk)
    return h.hexdigest().lower()


def audit_staged_archives() -> dict:
    results = {}
    all_ready = True

    for filename, spec in OFFICIAL_SPEC.items():
        archive_path = DIST_DIR / filename
        if not archive_path.exists():
            results[filename] = {
                "present": False,
                "verified": False,
                "file_size_bytes": 0,
                "sha512": None,
                "expected_sha512": spec["expected_sha512"],
                "official_source": spec["official_source"],
                "error": "Archive file not found in docker/cluster/dist/",
            }
            all_ready = False
            continue

        size = archive_path.stat().st_size
        logger.info(f"Computing SHA-512 for {filename} ({size / (1024*1024):.1f} MB)...")
        actual_sha512 = compute_sha512(archive_path)
        is_valid = actual_sha512 == spec["expected_sha512"].lower()

        results[filename] = {
            "present": True,
            "verified": is_valid,
            "file_size_bytes": size,
            "sha512": actual_sha512,
            "expected_sha512": spec["expected_sha512"],
            "official_source": spec["official_source"],
            "status": "PASS" if is_valid else "CORRUPT",
        }
        if not is_valid:
            all_ready = False

    report = {
        "dist_directory": str(DIST_DIR.relative_to(PROJECT_ROOT)),
        "all_archives_verified": all_ready,
        "archives": results,
    }
    return report


if __name__ == "__main__":
    rep = audit_staged_archives()
    print(json.dumps(rep, indent=2))
    sys.exit(0 if rep["all_archives_verified"] else 1)

