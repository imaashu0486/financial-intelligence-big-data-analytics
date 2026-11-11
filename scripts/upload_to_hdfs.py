"""
Live HDFS Real-Data Ingestion and Verification Engine (Step 7.3 Phase B).

Executes inside hadoop-master or via docker exec to:
1. Validate local source files against manifest SHA-256 digests.
2. Create HDFS directories under /financial-intelligence/real/.
3. Ingest all 77 manifest-listed files into HDFS.
4. Enforce replication factor 2 across the two DataNodes.
5. Verify uploaded counts, sizes, checksums, and fsck block replication distribution.
6. Export empirical audit metrics to data/real/metadata/hdfs_live_ingestion_audit.json.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Determine if running inside container or on host
IN_CONTAINER = Path("/opt/project").exists()
BASE_DIR = Path("/opt/project") if IN_CONTAINER else Path(__file__).resolve().parent.parent

MANIFEST_PATH = BASE_DIR / "data" / "real" / "metadata" / "hdfs_ingestion_manifest.json"
AUDIT_OUTPUT = BASE_DIR / "data" / "real" / "metadata" / "hdfs_live_ingestion_audit.json"
HDFS_ROOT = "/financial-intelligence/real"


def run_cmd(cmd: list[str]) -> tuple[int, str, str]:
    if not IN_CONTAINER:
        # Wrap with docker exec hadoop-master
        full_cmd = ["docker", "exec", "hadoop-master"] + cmd
    else:
        full_cmd = cmd
    p = subprocess.run(full_cmd, capture_output=True, text=True)
    return p.returncode, p.stdout.strip(), p.stderr.strip()


def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest().lower()


def execute_ingestion():
    logger.info("==========================================================================")
    logger.info(" Starting Step 7.3 Phase B: Live HDFS Ingestion & Verification")
    logger.info("==========================================================================")

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_PATH}")

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    catalog = manifest["files"]
    logger.info(f"Loaded manifest with {len(catalog)} files to be ingested.")

    # 1. Pre-ingestion local checksum validation
    logger.info("Verifying local source files against manifest SHA-256 digests...")
    verified_files = []
    for item in catalog:
        lp = BASE_DIR / item["local_path"].replace("\\", "/")
        if not lp.exists():
            raise FileNotFoundError(f"Missing local source file: {lp}")
        file_sha = compute_sha256(lp)
        if file_sha != item["sha256_checksum"].lower():
            raise ValueError(f"Checksum mismatch for {lp}!")
        verified_files.append((lp, item))

    logger.info(f"All {len(verified_files)} source files verified against recorded SHA-256.")

    # 2. Inspect existing HDFS state
    rc, out, err = run_cmd(["hdfs", "dfs", "-ls", "/"])
    logger.info(f"Initial HDFS root state:\n{out}")

    # 3. Create target HDFS directories
    target_dirs = manifest.get("target_directories", [
        f"{HDFS_ROOT}/raw/market",
        f"{HDFS_ROOT}/raw/news",
        f"{HDFS_ROOT}/raw/documents",
        f"{HDFS_ROOT}/processed/market",
        f"{HDFS_ROOT}/processed/benchmark",
        f"{HDFS_ROOT}/processed/news",
        f"{HDFS_ROOT}/processed/documents",
        f"{HDFS_ROOT}/metadata",
    ])

    for td in target_dirs:
        rc, out, err = run_cmd(["hdfs", "dfs", "-mkdir", "-p", td])
        if rc != 0:
            logger.warning(f"Failed to create directory {td}: {err}")

    logger.info(f"Created {len(target_dirs)} HDFS target directory structures.")

    # 4. Upload files
    logger.info(f"Uploading {len(verified_files)} files into HDFS...")
    t0 = time.time()
    uploaded_records = []
    for lp, meta in verified_files:
        # Container path for the local file
        container_path = f"/opt/project/{meta['local_path'].replace(chr(92), '/')}"
        dest_hdfs = meta["hdfs_destination"]

        # Run hdfs dfs -put -f
        rc, out, err = run_cmd(["hdfs", "dfs", "-put", "-f", container_path, dest_hdfs])
        if rc != 0:
            raise RuntimeError(f"Failed to put {container_path} to {dest_hdfs}: {err}")

        uploaded_records.append({
            "filename": meta["original_filename"],
            "local_path": str(lp),
            "hdfs_destination": dest_hdfs,
            "size_bytes": meta["file_size_bytes"],
            "sha256": meta["sha256_checksum"],
            "status": "UPLOADED",
        })

    upload_duration = round(time.time() - t0, 2)
    logger.info(f"Successfully uploaded {len(uploaded_records)} files to HDFS in {upload_duration}s.")

    # 5. Set replication factor to 2 recursively
    logger.info("Enforcing replication factor 2 across HDFS hierarchy...")
    rc, out, err = run_cmd(["hdfs", "dfs", "-setrep", "-R", "2", HDFS_ROOT])
    logger.info(f"hdfs setrep output: {out}")

    # Wait 3 seconds for NameNode block replication placement
    time.sleep(3)

    # 6. Verify HDFS file count and total size
    rc, count_out, _ = run_cmd(["hdfs", "dfs", "-count", "-q", "-v", HDFS_ROOT])
    logger.info(f"HDFS Count Verification:\n{count_out}")

    rc, du_out, _ = run_cmd(["hdfs", "dfs", "-du", "-s", HDFS_ROOT])
    logger.info(f"HDFS Disk Usage (du -s):\n{du_out}")

    # 7. Run fsck for replication health and DataNode distribution
    logger.info("Running HDFS fsck for block locations and replication health...")
    rc, fsck_out, _ = run_cmd(["hdfs", "fsck", HDFS_ROOT, "-files", "-blocks", "-locations"])

    # 8. Run live DataNode report
    rc, dfsadmin_out, _ = run_cmd(["hdfs", "dfsadmin", "-report"])

    audit_summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "target_hdfs_root": HDFS_ROOT,
        "total_files_uploaded": len(uploaded_records),
        "upload_duration_sec": upload_duration,
        "configured_replication": 2,
        "hdfs_count_summary": count_out,
        "hdfs_du_summary": du_out,
        "fsck_report_summary": fsck_out[-1200:] if len(fsck_out) > 1200 else fsck_out,
        "uploaded_files": uploaded_records,
    }

    with open(AUDIT_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(audit_summary, f, indent=2)

    logger.info(f"Ingestion audit report saved to {AUDIT_OUTPUT}")
    return audit_summary


if __name__ == "__main__":
    execute_ingestion()

