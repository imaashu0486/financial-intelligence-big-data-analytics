#!/usr/bin/env python3
"""
Run MapReduce jobs locally (simulating Hadoop Streaming).

Executes mapper → sort → reducer using subprocess pipes, exactly as
Hadoop Streaming would, but without requiring a Hadoop cluster.

Also prints the equivalent Hadoop Streaming commands for cluster execution.

Usage:
    python scripts/run_mapreduce.py
    python scripts/run_mapreduce.py --job wordcount
    python scripts/run_mapreduce.py --job volume
    python scripts/run_mapreduce.py --job all
"""

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.logger import logger

# Paths to mapper/reducer scripts
MAPREDUCE_DIR = ROOT / "src" / "bigdata" / "mapreduce"


def run_local_mapreduce(
    input_file: Path,
    mapper_script: Path,
    reducer_script: Path,
    output_file: Path,
    job_name: str,
) -> bool:
    """
    Simulate Hadoop Streaming locally using subprocess pipes.

    Pipeline: cat input | python mapper | sort | python reducer > output

    This is functionally identical to what Hadoop Streaming does:
    1. MAP phase: Each mapper receives a portion of input via stdin
    2. SHUFFLE/SORT: Hadoop sorts mapper output by key
    3. REDUCE phase: Reducer receives sorted key-value pairs via stdin
    """
    logger.info(f"{'=' * 60}")
    logger.info(f"MapReduce Job: {job_name}")
    logger.info(f"{'=' * 60}")
    logger.info(f"  Input:   {input_file}")
    logger.info(f"  Mapper:  {mapper_script.name}")
    logger.info(f"  Reducer: {reducer_script.name}")
    logger.info(f"  Output:  {output_file}")

    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        return False

    python = sys.executable
    output_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Step 1: MAP phase — read input through mapper
        logger.info("  ▸ Step 1/3: MAP phase (mapper processing input)")
        with open(input_file, "r", encoding="utf-8") as f_in:
            mapper_proc = subprocess.run(
                [python, str(mapper_script)],
                stdin=f_in,
                capture_output=True,
                text=True,
                timeout=120,
            )

        if mapper_proc.returncode != 0:
            logger.error(f"  Mapper failed: {mapper_proc.stderr}")
            return False

        mapper_output = mapper_proc.stdout
        mapper_lines = mapper_output.strip().count("\n") + 1 if mapper_output.strip() else 0
        logger.info(f"    Mapper emitted {mapper_lines:,} key-value pairs")

        # Step 2: SHUFFLE/SORT phase — sort mapper output by key
        logger.info("  ▸ Step 2/3: SHUFFLE/SORT phase (sorting by key)")
        sort_proc = subprocess.run(
            ["sort"],
            input=mapper_output,
            capture_output=True,
            text=True,
            timeout=60,
            shell=True,  # Use shell sort on Windows
        )

        sorted_output = sort_proc.stdout

        # Step 3: REDUCE phase — process sorted output through reducer
        logger.info("  ▸ Step 3/3: REDUCE phase (reducer aggregating)")
        reducer_proc = subprocess.run(
            [python, str(reducer_script)],
            input=sorted_output,
            capture_output=True,
            text=True,
            timeout=120,
        )

        if reducer_proc.returncode != 0:
            logger.error(f"  Reducer failed: {reducer_proc.stderr}")
            return False

        # Save output
        with open(output_file, "w", encoding="utf-8") as f_out:
            f_out.write(reducer_proc.stdout)

        result_lines = reducer_proc.stdout.strip().count("\n") + 1 if reducer_proc.stdout.strip() else 0
        logger.info(f"    Reducer produced {result_lines:,} aggregated results")
        logger.info(f"  ✅ Output saved: {output_file}")

        # Show top results
        lines = reducer_proc.stdout.strip().split("\n")[:10]
        logger.info(f"\n  ── Top {min(10, len(lines))} Results ──")
        for line in lines:
            logger.info(f"    {line}")

        return True

    except subprocess.TimeoutExpired:
        logger.error("  MapReduce job timed out")
        return False
    except Exception as e:
        logger.error(f"  MapReduce job failed: {e}")
        return False


def print_hadoop_command(
    input_hdfs: str,
    output_hdfs: str,
    mapper_script: str,
    reducer_script: str,
    job_name: str,
) -> None:
    """Print the equivalent Hadoop Streaming command for cluster execution."""
    logger.info(f"\n  ── Hadoop Streaming Command ({job_name}) ──")
    logger.info(
        f"  hadoop jar $HADOOP_HOME/share/hadoop/tools/lib/hadoop-streaming-*.jar \\\n"
        f"      -input {input_hdfs} \\\n"
        f"      -output {output_hdfs} \\\n"
        f"      -mapper \"python3 {mapper_script}\" \\\n"
        f"      -reducer \"python3 {reducer_script}\" \\\n"
        f"      -file {mapper_script} \\\n"
        f"      -file {reducer_script}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run MapReduce jobs locally (simulating Hadoop Streaming)"
    )
    parser.add_argument(
        "--job",
        choices=["wordcount", "volume", "all"],
        default="all",
        help="Which MapReduce job to run (default: all)",
    )
    args = parser.parse_args()

    with open(ROOT / "config" / "config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    mr_output = Path(ROOT / cfg.get("mapreduce", {}).get("output_path", "data/processed/mapreduce"))

    jobs_run = 0
    jobs_passed = 0

    # ── Word Count Job ───────────────────────────────────
    if args.job in ("wordcount", "all"):
        news_input = ROOT / cfg["paths"]["processed_news"]
        success = run_local_mapreduce(
            input_file=news_input,
            mapper_script=MAPREDUCE_DIR / "news_mapper.py",
            reducer_script=MAPREDUCE_DIR / "news_reducer.py",
            output_file=mr_output / "word_counts.txt",
            job_name="Word Count (Financial News)",
        )
        jobs_run += 1
        if success:
            jobs_passed += 1

        print_hadoop_command(
            input_hdfs="/financial_analytics/raw/news_cleaned.csv",
            output_hdfs="/financial_analytics/processed/word_count",
            mapper_script="news_mapper.py",
            reducer_script="news_reducer.py",
            job_name="Word Count",
        )

    # ── Volume Aggregation Job ───────────────────────────
    if args.job in ("volume", "all"):
        market_input = ROOT / cfg["paths"]["processed_market"]
        success = run_local_mapreduce(
            input_file=market_input,
            mapper_script=MAPREDUCE_DIR / "volume_mapper.py",
            reducer_script=MAPREDUCE_DIR / "volume_reducer.py",
            output_file=mr_output / "volume_totals.txt",
            job_name="Volume Aggregation (Market Data)",
        )
        jobs_run += 1
        if success:
            jobs_passed += 1

        print_hadoop_command(
            input_hdfs="/financial_analytics/raw/market_features.csv",
            output_hdfs="/financial_analytics/processed/volume_totals",
            mapper_script="volume_mapper.py",
            reducer_script="volume_reducer.py",
            job_name="Volume Aggregation",
        )

    # Summary
    logger.info(f"\n{'=' * 60}")
    logger.info(f"MapReduce Summary: {jobs_passed}/{jobs_run} jobs completed")
    logger.info(f"Output directory: {mr_output}")
    logger.info(f"{'=' * 60}")


if __name__ == "__main__":
    main()
