"""
MapReduce Execution Engine for Real Financial Datasets.

Executes the 5 Hadoop MapReduce jobs across the real financial market data:
1. Job 1: 10-Year Trading Volume Aggregated by Ticker
2. Job 2: Cross-Sectional Daily Market Return
3. Job 3: Company-Level 10-Year Average Daily Return
4. Job 4: Yearly Trading Volume per Ticker
5. Job 5: Market Data Quality & Integrity Summary

Supports:
- Local streaming execution (Mapper -> Sort -> Reducer) on the host or inside containers.
- Generation of standard Hadoop Streaming YARN submission commands.
- Generation of execution manifests and validation outputs in data/real/distributed/mapreduce/.
"""

from __future__ import annotations

import csv
import io
import json
import logging
from pathlib import Path
import subprocess
import sys
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
INPUT_CSV = PROJECT_ROOT / "data" / "real" / "processed" / "market" / "consolidated_market_features.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "real" / "distributed" / "mapreduce"
MAPREDUCE_DIR = PROJECT_ROOT / "src" / "bigdata" / "mapreduce"

JOBS = [
    {
        "id": "job1",
        "name": "Volume Aggregation by Ticker",
        "mapper": MAPREDUCE_DIR / "volume_by_ticker_mapper.py",
        "reducer": MAPREDUCE_DIR / "volume_by_ticker_reducer.py",
        "output_file": OUTPUT_DIR / "job1_volume_by_ticker.tsv",
        "columns": ["symbol", "total_volume", "record_count"],
        "hdfs_input": "/financial-intelligence/real/processed/market/consolidated_market_features.csv",
        "hdfs_output": "/financial-intelligence/real/distributed/mapreduce/job1_volume_by_ticker",
    },
    {
        "id": "job2",
        "name": "Cross-Sectional Daily Market Return",
        "mapper": MAPREDUCE_DIR / "daily_market_return_mapper.py",
        "reducer": MAPREDUCE_DIR / "daily_market_return_reducer.py",
        "output_file": OUTPUT_DIR / "job2_daily_market_return.tsv",
        "columns": ["date", "mean_daily_return", "constituent_count"],
        "hdfs_input": "/financial-intelligence/real/processed/market/consolidated_market_features.csv",
        "hdfs_output": "/financial-intelligence/real/distributed/mapreduce/job2_daily_market_return",
    },
    {
        "id": "job3",
        "name": "Company-Level 10-Year Average Daily Return",
        "mapper": MAPREDUCE_DIR / "company_avg_return_mapper.py",
        "reducer": MAPREDUCE_DIR / "company_avg_return_reducer.py",
        "output_file": OUTPUT_DIR / "job3_company_avg_return.tsv",
        "columns": ["symbol", "mean_daily_return", "observation_count"],
        "hdfs_input": "/financial-intelligence/real/processed/market/consolidated_market_features.csv",
        "hdfs_output": "/financial-intelligence/real/distributed/mapreduce/job3_company_avg_return",
    },
    {
        "id": "job4",
        "name": "Yearly Trading Volume per Ticker",
        "mapper": MAPREDUCE_DIR / "yearly_volume_mapper.py",
        "reducer": MAPREDUCE_DIR / "yearly_volume_reducer.py",
        "output_file": OUTPUT_DIR / "job4_yearly_volume.tsv",
        "columns": ["symbol", "year", "total_volume", "trading_days"],
        "hdfs_input": "/financial-intelligence/real/processed/market/consolidated_market_features.csv",
        "hdfs_output": "/financial-intelligence/real/distributed/mapreduce/job4_yearly_volume",
    },
    {
        "id": "job5",
        "name": "Market Data Quality & Integrity Summary",
        "mapper": MAPREDUCE_DIR / "market_quality_mapper.py",
        "reducer": MAPREDUCE_DIR / "market_quality_reducer.py",
        "output_file": OUTPUT_DIR / "job5_market_quality_summary.tsv",
        "columns": ["metric_name", "total_count"],
        "hdfs_input": "/financial-intelligence/real/processed/market/consolidated_market_features.csv",
        "hdfs_output": "/financial-intelligence/real/distributed/mapreduce/job5_market_quality_summary",
    },
]


def execute_job_locally(job: dict, input_path: Path) -> dict:
    """
    Simulate distributed MapReduce streaming locally:
    1. Stream input file through mapper.py
    2. Sort mapper intermediate output by key (shuffle & sort phase)
    3. Stream sorted intermediate output through reducer.py
    4. Write reducer output to TSV
    """
    logger.info(f"Running MapReduce {job['id']}: {job['name']}...")
    t0 = time.time()

    # Phase 1: Mapper
    python_exe = sys.executable
    p_map = subprocess.Popen(
        [python_exe, str(job["mapper"])],
        stdin=open(input_path, "r", encoding="utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    map_out, map_err = p_map.communicate()
    if p_map.returncode != 0:
        raise RuntimeError(f"Mapper failed for {job['id']}: {map_err}")

    # Phase 2: Shuffle & Sort
    lines = [line for line in map_out.splitlines() if line.strip()]
    lines.sort()
    sorted_intermediate = "\n".join(lines) + "\n"

    # Phase 3: Reducer
    p_red = subprocess.Popen(
        [python_exe, str(job["reducer"])],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    red_out, red_err = p_red.communicate(input=sorted_intermediate)
    if p_red.returncode != 0:
        raise RuntimeError(f"Reducer failed for {job['id']}: {red_err}")

    # Phase 4: Save output
    job["output_file"].parent.mkdir(parents=True, exist_ok=True)
    header = "\t".join(job["columns"]) + "\n"
    with open(job["output_file"], "w", encoding="utf-8") as f:
        f.write(header)
        f.write(red_out)

    elapsed = time.time() - t0
    output_rows = len([l for l in red_out.splitlines() if l.strip()])
    logger.info(f"Finished {job['id']} in {elapsed:.2f}s ({output_rows} result records)")

    yarn_command = (
        f"hadoop jar /opt/hadoop/share/hadoop/tools/lib/hadoop-streaming-*.jar \\\n"
        f"  -files {job['mapper'].name},{job['reducer'].name} \\\n"
        f"  -input {job['hdfs_input']} \\\n"
        f"  -output {job['hdfs_output']} \\\n"
        f"  -mapper \"python3 {job['mapper'].name}\" \\\n"
        f"  -reducer \"python3 {job['reducer'].name}\""
    )

    return {
        "job_id": job["id"],
        "name": job["name"],
        "output_file": str(job["output_file"].relative_to(PROJECT_ROOT)),
        "row_count": output_rows,
        "execution_time_sec": round(elapsed, 3),
        "yarn_command": yarn_command,
    }


def run_all_jobs() -> dict:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input market data not found: {INPUT_CSV}")

    results = []
    t_start = time.time()
    for job in JOBS:
        res = execute_job_locally(job, INPUT_CSV)
        results.append(res)

    manifest = {
        "execution_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "input_dataset": str(INPUT_CSV.relative_to(PROJECT_ROOT)),
        "total_jobs": len(JOBS),
        "total_duration_sec": round(time.time() - t_start, 3),
        "execution_mode": "Hadoop Streaming Local Reference Pipeline",
        "cluster_compatible": True,
        "jobs": results,
    }

    manifest_path = OUTPUT_DIR / "mapreduce_execution_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"MapReduce manifest saved to {manifest_path}")

    return manifest


if __name__ == "__main__":
    run_all_jobs()

