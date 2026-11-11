import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Ensure Spark workers and driver use the active Python executable on Windows
if "PYSPARK_PYTHON" not in os.environ:
    os.environ["PYSPARK_PYTHON"] = sys.executable
if "PYSPARK_DRIVER_PYTHON" not in os.environ:
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

if "HADOOP_HOME" not in os.environ:
    hadoop_dir = ROOT / "hadoop"
    if hadoop_dir.exists():
        os.environ["HADOOP_HOME"] = str(hadoop_dir)
        bin_dir = str(hadoop_dir / "bin")
        if bin_dir not in os.environ.get("PATH", ""):
            os.environ["PATH"] = f"{bin_dir};{os.environ.get('PATH', '')}"

# PySpark imports — skip all tests if pyspark is not available
pyspark = pytest.importorskip("pyspark")

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from src.bigdata.spark_jobs import SparkPipeline


@pytest.fixture(scope="module")
def spark():
    """Create a module-scoped SparkSession for testing."""
    session = (
        SparkSession.builder
        .appName("TestFinancialAnalytics")
        .master("local[1]")
        .config("spark.ui.showConsoleProgress", "false")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .getOrCreate()
    )
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()


@pytest.fixture
def sample_cfg():
    """Standard config for testing."""
    return {
        "spark": {
            "app_name": "TestFinancialAnalytics",
            "master": "local[1]",
            "output_path": "data/processed/spark",
        },
        "paths": {
            "processed_market": "data/processed/market_features.csv",
        },
    }


@pytest.fixture
def sample_df(spark):
    """Create a small Spark DataFrame for testing."""
    data = [
        ("2023-01-02", "TCS", 3500.0, 3550.0, 3480.0, 3530.0, 1000000, "False", 0.01, 0.00995, 3530.0, 3530.0, 3530.0, 0.005, 0.0, 60.0, 1.0, 0.5),
        ("2023-01-03", "TCS", 3520.0, 3560.0, 3500.0, 3540.0, 1200000, "False", 0.00283, 0.00283, 3535.0, 3535.0, 3535.0, 0.004, 0.2, 62.0, 1.2, 0.6),
        ("2023-01-04", "TCS", 3510.0, 3540.0, 3490.0, 3520.0, 900000, "False", -0.00565, -0.00566, 3530.0, 3530.0, 3530.0, 0.006, -0.25, 45.0, -0.5, -0.3),
        ("2023-01-02", "INFY", 1500.0, 1520.0, 1490.0, 1510.0, 2000000, "False", 0.02, 0.0198, 1510.0, 1510.0, 1510.0, 0.008, 0.0, 65.0, 2.0, 1.0),
        ("2023-01-03", "INFY", 1510.0, 1530.0, 1500.0, 1525.0, 2200000, "False", 0.00993, 0.00988, 1517.5, 1517.5, 1517.5, 0.007, 0.1, 68.0, 2.5, 1.2),
        ("2023-01-04", "INFY", 1520.0, 1540.0, 1510.0, 1530.0, 1800000, "False", 0.00328, 0.00327, 1521.7, 1521.7, 1521.7, 0.006, -0.182, 70.0, 3.0, 1.5),
    ]
    columns = [
        "date", "symbol", "open", "high", "low", "close", "volume",
        "outlier_flag", "daily_return", "log_return", "ma_7", "ma_20", "ma_50",
        "volatility", "volume_change", "rsi_14", "macd", "macd_signal",
    ]
    df = spark.createDataFrame(data, columns)
    df = df.withColumn("date", F.to_date(F.col("date")))
    return df


class TestSparkPipeline:
    """Tests for the SparkPipeline class."""

    def test_quality_checks(self, spark, sample_df, sample_cfg):
        """Quality checks should return correct counts."""
        pipeline = SparkPipeline(sample_cfg, spark=spark)
        checks = pipeline.run_quality_checks(sample_df)

        assert checks["total_rows"] == 6
        assert checks["symbol_count"] == 2
        assert checks["negative_prices"] == 0
        assert checks["null_counts"]["close"] == 0

    def test_stock_statistics_schema(self, spark, sample_df, sample_cfg):
        """Stock statistics should have all required columns."""
        pipeline = SparkPipeline(sample_cfg, spark=spark)
        stats = pipeline.compute_stock_statistics(sample_df)

        expected_cols = {
            "symbol", "avg_daily_return", "avg_log_return", "avg_volatility",
            "avg_volume", "total_volume", "max_price", "min_price", "trading_days",
        }
        assert expected_cols.issubset(set(stats.columns))

    def test_stock_statistics_values(self, spark, sample_df, sample_cfg):
        """Stock statistics should compute correct aggregations."""
        pipeline = SparkPipeline(sample_cfg, spark=spark)
        stats = pipeline.compute_stock_statistics(sample_df)

        # Convert to pandas for easy assertions
        pdf = stats.toPandas()

        tcs = pdf[pdf["symbol"] == "TCS"].iloc[0]
        assert tcs["trading_days"] == 3
        assert tcs["max_price"] == 3540.0
        assert tcs["min_price"] == 3520.0
        assert tcs["total_volume"] == 3100000

        infy = pdf[pdf["symbol"] == "INFY"].iloc[0]
        assert infy["trading_days"] == 3
        assert infy["max_price"] == 1530.0
        assert infy["total_volume"] == 6000000

    def test_daily_statistics_schema(self, spark, sample_df, sample_cfg):
        """Daily statistics should have correct columns."""
        pipeline = SparkPipeline(sample_cfg, spark=spark)
        daily = pipeline.compute_daily_statistics(sample_df)

        expected_cols = {
            "date", "market_avg_return", "market_total_volume",
            "stocks_traded", "avg_close_price",
        }
        assert expected_cols.issubset(set(daily.columns))

    def test_daily_statistics_values(self, spark, sample_df, sample_cfg):
        """Daily stats should aggregate across all stocks for each date."""
        pipeline = SparkPipeline(sample_cfg, spark=spark)
        daily = pipeline.compute_daily_statistics(sample_df)

        pdf = daily.toPandas()
        assert len(pdf) == 3  # 3 trading days

        # Jan 2: TCS volume=1M + INFY volume=2M = 3M
        jan2 = pdf[pdf["date"].astype(str) == "2023-01-02"].iloc[0]
        assert jan2["market_total_volume"] == 3000000
        assert jan2["stocks_traded"] == 2

    def test_stock_rankings(self, spark, sample_df, sample_cfg):
        """Stock rankings should rank by volatility."""
        pipeline = SparkPipeline(sample_cfg, spark=spark)
        rankings = pipeline.compute_stock_rankings(sample_df)

        pdf = rankings.toPandas()
        assert len(pdf) == 2  # 2 stocks
        assert "volatility_rank" in pdf.columns
        assert pdf["volatility_rank"].tolist() == [1, 2]

    def test_enriched_statistics_join(self, spark, sample_df, sample_cfg):
        """Enriched stats should join stock stats with rankings."""
        pipeline = SparkPipeline(sample_cfg, spark=spark)
        stock_stats = pipeline.compute_stock_statistics(sample_df)
        rankings = pipeline.compute_stock_rankings(sample_df)
        enriched = pipeline.compute_enriched_statistics(stock_stats, rankings)

        pdf = enriched.toPandas()
        assert len(pdf) == 2
        assert "volatility_rank" in pdf.columns
        assert "avg_daily_return" in pdf.columns
        assert "total_volume" in pdf.columns

    def test_save_creates_parquet(self, spark, sample_df, sample_cfg, tmp_path):
        """Save should create Parquet files."""
        sample_cfg["spark"]["output_path"] = str(tmp_path / "spark_output")
        pipeline = SparkPipeline(sample_cfg, spark=spark)

        stock_stats = pipeline.compute_stock_statistics(sample_df)
        daily_stats = pipeline.compute_daily_statistics(sample_df)
        rankings = pipeline.compute_stock_rankings(sample_df)
        enriched = pipeline.compute_enriched_statistics(stock_stats, rankings)

        saved = pipeline.save_results(stock_stats, daily_stats, rankings, enriched)

        for name in ["stock_statistics", "daily_statistics", "stock_rankings", "enriched_statistics"]:
            out_dir = Path(saved[name])
            assert out_dir.exists(), f"{name} directory should exist"
            parquet_files = list(out_dir.glob("*.parquet"))
            assert len(parquet_files) > 0, f"{name} should have parquet files"
