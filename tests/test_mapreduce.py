"""Tests for MapReduce mapper and reducer scripts."""

import csv
import io
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
MAPREDUCE_DIR = ROOT / "src" / "bigdata" / "mapreduce"


# ─────────────────────────────────────────────────────────────
# Helper: Run a mapper/reducer script with given input
# ─────────────────────────────────────────────────────────────

def run_script(script: Path, input_text: str) -> str:
    """Run a Python script with input on stdin, return stdout."""
    result = subprocess.run(
        [sys.executable, str(script)],
        input=input_text,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, f"Script failed: {result.stderr}"
    return result.stdout


def run_mapreduce_pipeline(mapper: Path, reducer: Path, input_text: str) -> str:
    """Simulate mapper | sort | reducer pipeline."""
    # Map
    mapper_output = run_script(mapper, input_text)
    # Sort (simulate Hadoop shuffle/sort)
    sorted_lines = sorted(mapper_output.strip().split("\n"))
    sorted_text = "\n".join(sorted_lines) + "\n"
    # Reduce
    reducer_output = run_script(reducer, sorted_text)
    return reducer_output


# ─────────────────────────────────────────────────────────────
# Test Data
# ─────────────────────────────────────────────────────────────

SAMPLE_NEWS_CSV = """date,title,description,source,symbol
2023-01-02,TCS shares surge on strong results,TCS reported strong quarterly results with revenue growth,Economic Times,TCS
2023-01-03,INFY hits new high amid rally,INFY shares hit a new high as the sector rallied strongly,Moneycontrol,INFY
2023-01-04,TCS wins major contract,TCS wins a major contract worth several crores for TCS expansion,LiveMint,TCS
"""

SAMPLE_MARKET_CSV = """date,symbol,open,high,low,close,volume,outlier_flag,daily_return,log_return,ma_7,ma_20,ma_50,volatility,volume_change,rsi_14,macd,macd_signal
2023-01-02,TCS,3500.0,3550.0,3480.0,3530.0,1000000,False,0.01,0.00995,3530.0,3530.0,3530.0,0.005,0.0,60.0,1.0,0.5
2023-01-03,TCS,3520.0,3560.0,3500.0,3540.0,1200000,False,0.00283,0.00283,3535.0,3535.0,3535.0,0.004,0.2,62.0,1.2,0.6
2023-01-02,INFY,1500.0,1520.0,1490.0,1510.0,2000000,False,0.02,0.0198,1510.0,1510.0,1510.0,0.008,0.0,65.0,2.0,1.0
2023-01-03,INFY,1510.0,1530.0,1500.0,1525.0,2200000,False,0.00993,0.00988,1517.5,1517.5,1517.5,0.007,0.1,68.0,2.5,1.2
"""


# ─────────────────────────────────────────────────────────────
# News Mapper Tests
# ─────────────────────────────────────────────────────────────

class TestNewsMapper:
    """Tests for news_mapper.py."""

    def test_output_format(self):
        """Each line should be word\\t1."""
        output = run_script(MAPREDUCE_DIR / "news_mapper.py", SAMPLE_NEWS_CSV)
        for line in output.strip().split("\n"):
            parts = line.split("\t")
            assert len(parts) == 2, f"Expected tab-separated pair, got: {line}"
            assert parts[1] == "1", f"Count should be 1, got: {parts[1]}"

    def test_words_are_lowercase(self):
        """All emitted words should be lowercase."""
        output = run_script(MAPREDUCE_DIR / "news_mapper.py", SAMPLE_NEWS_CSV)
        for line in output.strip().split("\n"):
            word = line.split("\t")[0]
            assert word == word.lower(), f"Word should be lowercase: {word}"

    def test_stopwords_removed(self):
        """Common stopwords should not appear in output."""
        output = run_script(MAPREDUCE_DIR / "news_mapper.py", SAMPLE_NEWS_CSV)
        words = {line.split("\t")[0] for line in output.strip().split("\n")}
        # These are in our stopword list
        for stopword in ["the", "and", "with", "for"]:
            assert stopword not in words, f"Stopword '{stopword}' should be removed"

    def test_short_words_removed(self):
        """Words shorter than 3 characters should be removed."""
        output = run_script(MAPREDUCE_DIR / "news_mapper.py", SAMPLE_NEWS_CSV)
        for line in output.strip().split("\n"):
            word = line.split("\t")[0]
            assert len(word) >= 3, f"Word too short: '{word}'"

    def test_extracts_from_title_and_description(self):
        """Mapper should extract words from both title and description."""
        output = run_script(MAPREDUCE_DIR / "news_mapper.py", SAMPLE_NEWS_CSV)
        words = {line.split("\t")[0] for line in output.strip().split("\n")}
        # "surge" is in the title, "revenue" is in the description
        assert "surge" in words, "Should find 'surge' from title"
        assert "revenue" in words, "Should find 'revenue' from description"


# ─────────────────────────────────────────────────────────────
# News Reducer Tests
# ─────────────────────────────────────────────────────────────

class TestNewsReducer:
    """Tests for news_reducer.py."""

    def test_aggregates_counts(self):
        """Reducer should sum counts for the same word."""
        input_text = "apple\t1\napple\t1\napple\t1\nbanana\t1\nbanana\t1\n"
        output = run_script(MAPREDUCE_DIR / "news_reducer.py", input_text)
        results = dict(line.split("\t") for line in output.strip().split("\n"))
        assert results["apple"] == "3"
        assert results["banana"] == "2"

    def test_single_occurrence(self):
        """Words appearing once should have count 1."""
        input_text = "unique\t1\n"
        output = run_script(MAPREDUCE_DIR / "news_reducer.py", input_text)
        assert output.strip() == "unique\t1"

    def test_output_format(self):
        """Output should be word\\tcount."""
        input_text = "word1\t1\nword1\t1\nword2\t1\n"
        output = run_script(MAPREDUCE_DIR / "news_reducer.py", input_text)
        for line in output.strip().split("\n"):
            parts = line.split("\t")
            assert len(parts) == 2
            assert parts[1].isdigit()


# ─────────────────────────────────────────────────────────────
# Volume Mapper Tests
# ─────────────────────────────────────────────────────────────

class TestVolumeMapper:
    """Tests for volume_mapper.py."""

    def test_output_format(self):
        """Each line should be SYMBOL\\tvolume."""
        output = run_script(MAPREDUCE_DIR / "volume_mapper.py", SAMPLE_MARKET_CSV)
        for line in output.strip().split("\n"):
            parts = line.split("\t")
            assert len(parts) == 2, f"Expected tab-separated pair, got: {line}"
            assert parts[0].isupper(), f"Symbol should be uppercase: {parts[0]}"
            assert parts[1].isdigit(), f"Volume should be integer: {parts[1]}"

    def test_correct_volumes(self):
        """Volumes should match the CSV data."""
        output = run_script(MAPREDUCE_DIR / "volume_mapper.py", SAMPLE_MARKET_CSV)
        lines = output.strip().split("\n")
        volumes = [(l.split("\t")[0], int(l.split("\t")[1])) for l in lines]

        tcs_volumes = [v for s, v in volumes if s == "TCS"]
        assert 1000000 in tcs_volumes
        assert 1200000 in tcs_volumes


# ─────────────────────────────────────────────────────────────
# Volume Reducer Tests
# ─────────────────────────────────────────────────────────────

class TestVolumeReducer:
    """Tests for volume_reducer.py."""

    def test_aggregates_volume(self):
        """Reducer should sum volumes per symbol."""
        input_text = "INFY\t2000000\nINFY\t2200000\nTCS\t1000000\nTCS\t1200000\n"
        output = run_script(MAPREDUCE_DIR / "volume_reducer.py", input_text)
        results = dict(line.split("\t") for line in output.strip().split("\n"))
        assert results["INFY"] == "4200000"
        assert results["TCS"] == "2200000"


# ─────────────────────────────────────────────────────────────
# End-to-End Pipeline Tests
# ─────────────────────────────────────────────────────────────

class TestMapReducePipeline:
    """End-to-end tests: mapper | sort | reducer."""

    def test_word_count_pipeline(self):
        """Full word count pipeline should produce aggregated counts."""
        output = run_mapreduce_pipeline(
            MAPREDUCE_DIR / "news_mapper.py",
            MAPREDUCE_DIR / "news_reducer.py",
            SAMPLE_NEWS_CSV,
        )
        results = dict(line.split("\t") for line in output.strip().split("\n"))
        # "tcs" appears multiple times in the sample data
        assert "tcs" in results
        assert int(results["tcs"]) >= 2

    def test_volume_pipeline(self):
        """Full volume pipeline should produce correct totals."""
        output = run_mapreduce_pipeline(
            MAPREDUCE_DIR / "volume_mapper.py",
            MAPREDUCE_DIR / "volume_reducer.py",
            SAMPLE_MARKET_CSV,
        )
        results = dict(line.split("\t") for line in output.strip().split("\n"))
        assert results["TCS"] == "2200000"    # 1M + 1.2M
        assert results["INFY"] == "4200000"   # 2M + 2.2M
