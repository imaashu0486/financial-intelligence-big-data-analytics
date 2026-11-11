"""
Unit and integration tests for Frontend Service Facade Layer.
Validates that services correctly interface with backend modules and data artifacts.
"""

from pathlib import Path
import pandas as pd
import pytest

from app.services.data_service import DataService
from app.services.analytics_service import AnalyticsService
from app.services.ml_service import MLService
from app.services.sentiment_service import SentimentService
from app.services.mining_service import MiningService
from app.services.rag_service import RAGService


@pytest.fixture
def data_service() -> DataService:
    return DataService()


@pytest.fixture
def analytics_service() -> AnalyticsService:
    return AnalyticsService()


@pytest.fixture
def ml_service() -> MLService:
    return MLService()


@pytest.fixture
def sentiment_service() -> SentimentService:
    return SentimentService()


@pytest.fixture
def mining_service() -> MiningService:
    return MiningService()


@pytest.fixture
def rag_service() -> RAGService:
    return RAGService(mock=True)


# --- DATA SERVICE TESTS ---

class TestDataService:
    def test_load_market_features_real_default(self, data_service: DataService):
        df = data_service.load_market_features()
        assert not df.empty
        assert "symbol" in df.columns
        assert "date" in df.columns
        assert "company_name" in df.columns
        assert "sector" in df.columns
        assert "daily_return" in df.columns
        assert "volatility" in df.columns
        assert len(df) == 121962
        assert df["symbol"].nunique() == 50

    def test_load_market_features_synthetic_explicit(self, data_service: DataService):
        df = data_service.load_market_features(scope="Synthetic Baseline (10 Equities)")
        assert not df.empty
        assert len(df) == 5200
        assert df["symbol"].nunique() == 10

    def test_get_market_kpis(self, data_service: DataService):
        # Real default KPIs
        kpis_real = data_service.get_market_kpis()
        assert kpis_real["total_records"] == 121962
        assert kpis_real["symbols_count"] == 50
        assert "RELIANCE" in kpis_real["symbols"]
        assert kpis_real["trading_days"] == 2467

        # Synthetic KPIs
        df_synth = data_service.load_market_features(scope="Synthetic Baseline (10 Equities)")
        kpis_synth = data_service.get_market_kpis(df_synth)
        assert kpis_synth["total_records"] == 5200
        assert kpis_synth["symbols_count"] == 10
        assert kpis_synth["trading_days"] == 520

    def test_get_ticker_data_with_indicators(self, data_service: DataService):
        df_reliance = data_service.get_ticker_data("RELIANCE")
        assert not df_reliance.empty
        assert (df_reliance["symbol"] == "RELIANCE").all()
        assert "rsi_14" in df_reliance.columns
        assert "macd" in df_reliance.columns
        assert "macd_signal" in df_reliance.columns
        assert len(df_reliance) == 2467

        # Date filter check
        df_filtered = data_service.get_ticker_data("RELIANCE", start_date="2024-01-01", end_date="2024-06-30")
        assert not df_filtered.empty
        assert df_filtered["date"].min() >= pd.to_datetime("2024-01-01")
        assert df_filtered["date"].max() <= pd.to_datetime("2024-06-30")

    def test_get_latest_snapshot(self, data_service: DataService):
        # Real snapshot
        snapshot_real = data_service.get_latest_snapshot()
        assert len(snapshot_real) == 50
        assert snapshot_real["symbol"].nunique() == 50
        assert "rsi_14" in snapshot_real.columns

        # Synthetic snapshot
        df_synth = data_service.load_market_features(scope="Synthetic Baseline (10 Equities)")
        snapshot_synth = data_service.get_latest_snapshot(df_synth)
        assert len(snapshot_synth) == 10

    def test_missing_data_no_silent_fallback(self, tmp_path: Path):
        empty_svc = DataService(data_dir=tmp_path)
        # Real scope must raise FileNotFoundError and NEVER silently fall back
        with pytest.raises(FileNotFoundError, match="Real dataset scope must not silently fall back"):
            empty_svc.load_market_features(scope="Real Market History (NIFTY 50)")

        # Synthetic scope returns empty dataframe
        df_synth = empty_svc.load_market_features(scope="Synthetic Baseline (10 Equities)")
        assert df_synth.empty


# --- ANALYTICS SERVICE TESTS ---

class TestAnalyticsService:
    def test_get_mapreduce_volume(self, analytics_service: AnalyticsService):
        # Real default (50 tickers)
        df_real = analytics_service.get_mapreduce_volume()
        assert not df_real.empty
        assert "symbol" in df_real.columns
        assert "total_volume" in df_real.columns
        assert len(df_real) == 50
        assert df_real["symbol"].nunique() == 50

        # Synthetic explicit (10 tickers)
        df_synth = analytics_service.get_mapreduce_volume(scope="Synthetic Baseline (10 Equities)")
        assert not df_synth.empty
        assert len(df_synth) == 10
        assert df_synth["symbol"].nunique() == 10

    def test_get_mapreduce_word_counts(self, analytics_service: AnalyticsService):
        df = analytics_service.get_mapreduce_word_counts(top_n=15)
        assert not df.empty
        assert "word" in df.columns
        assert "count" in df.columns
        assert len(df) <= 15
        assert (df["count"] >= 0).all()

    def test_get_spark_parquet_stats(self, analytics_service: AnalyticsService):
        # Real default (50 tickers)
        df_stats = analytics_service.get_spark_stock_statistics()
        assert not df_stats.empty
        assert "symbol" in df_stats.columns
        assert "avg_daily_return" in df_stats.columns
        assert len(df_stats) == 50
        assert df_stats["symbol"].nunique() == 50

        df_rank = analytics_service.get_spark_stock_rankings()
        assert not df_rank.empty
        assert len(df_rank) == 50

        df_daily = analytics_service.get_spark_daily_statistics()
        assert not df_daily.empty
        assert len(df_daily) == 2467

        # Synthetic explicit (10 tickers)
        df_stats_synth = analytics_service.get_spark_stock_statistics(scope="Synthetic Baseline (10 Equities)")
        assert len(df_stats_synth) == 10
        df_rank_synth = analytics_service.get_spark_stock_rankings(scope="Synthetic Baseline (10 Equities)")
        assert len(df_rank_synth) == 10

    def test_get_cluster_architecture(self, analytics_service: AnalyticsService):
        arch = analytics_service.get_cluster_architecture()
        assert "cluster_nodes" in arch
        assert len(arch["cluster_nodes"]) == 4
        assert "academic_statement" in arch
        assert "three logical Hadoop/Spark nodes" in arch["academic_statement"]


# --- ML SERVICE TESTS ---

class TestMLService:
    def test_get_metrics(self, ml_service: MLService):
        metrics = ml_service.get_metrics()
        assert "regression" in metrics
        assert "classification" in metrics
        assert "clustering" in metrics

    def test_classification_and_regression_predictions(self, ml_service: MLService):
        clf_df = ml_service.get_classification_predictions()
        assert not clf_df.empty

        reg_df = ml_service.get_regression_predictions()
        assert not reg_df.empty

    def test_get_stock_clusters(self, ml_service: MLService):
        clusters = ml_service.get_stock_clusters()
        assert not clusters.empty
        assert "cluster" in clusters.columns
        assert len(clusters) == 10

    def test_get_feature_importances(self, ml_service: MLService):
        feat_df = ml_service.get_feature_importances()
        assert not feat_df.empty
        assert "feature" in feat_df.columns
        assert "importance" in feat_df.columns

    def test_predict_market_direction_inference(self, ml_service: MLService):
        features = {
            "daily_return": 0.015,
            "log_return": 0.014,
            "ma_7": 1500.0,
            "ma_20": 1480.0,
            "ma_50": 1450.0,
            "volatility": 0.012,
            "volume_change": 0.05,
            "rsi_14": 62.0,
            "macd": 1.5,
            "macd_signal": 1.0,
        }
        res = ml_service.predict_market_direction(features)
        assert "prediction" in res
        assert res["prediction"] in ["Bullish", "Bearish", "Neutral"]
        assert "confidence" in res
        assert 0.0 <= res["confidence"] <= 1.0


# --- SENTIMENT SERVICE TESTS ---

class TestSentimentService:
    def test_get_sentiment_analytics(self, sentiment_service: SentimentService):
        stats = sentiment_service.get_sentiment_analytics()
        assert "correlations" in stats
        assert "regime_statistics" in stats
        assert "sentiment_distribution" in stats

    def test_get_article_sentiments(self, sentiment_service: SentimentService):
        articles = sentiment_service.get_article_sentiments()
        assert not articles.empty
        assert len(articles) == 500

        # Filter by symbol
        tcs_articles = sentiment_service.get_article_sentiments(symbol="TCS")
        assert not tcs_articles.empty
        assert (tcs_articles["symbol"] == "TCS").all()

    def test_get_daily_sentiments(self, sentiment_service: SentimentService):
        daily = sentiment_service.get_daily_sentiments()
        assert not daily.empty
        assert "average_sentiment" in daily.columns


# --- MINING SERVICE TESTS ---

class TestMiningService:
    def test_get_association_rules(self, mining_service: MiningService):
        rules = mining_service.get_association_rules()
        assert not rules.empty
        assert "antecedent" in rules.columns
        assert "consequent" in rules.columns
        assert "lift" in rules.columns

        # Filtered rules
        filtered = mining_service.get_association_rules(min_lift=1.2)
        assert (filtered["lift"] >= 1.2).all()

    def test_get_frequent_itemsets(self, mining_service: MiningService):
        itemsets = mining_service.get_frequent_itemsets()
        assert not itemsets.empty
        assert "itemset" in itemsets.columns

    def test_get_mining_statistics(self, mining_service: MiningService):
        stats = mining_service.get_mining_statistics()
        assert stats.get("num_rules") == 10
        assert stats.get("num_frequent_itemsets") == 47
        tx_summary = mining_service.get_transaction_summary()
        assert tx_summary.get("num_transactions") == 482
        assert tx_summary.get("num_unique_items") == 10


# --- RAG SERVICE TESTS ---

class TestRAGService:
    def test_get_index_status(self, rag_service: RAGService):
        status = rag_service.get_index_status()
        assert status["is_indexed"] is True
        assert status["chunk_count"] == 28
        assert status["unique_chunk_count"] == 28
        assert len(status["indexed_documents"]) == 3

    def test_get_evaluation_benchmarks(self, rag_service: RAGService):
        bench = rag_service.get_evaluation_benchmarks()
        assert "summary" in bench or "retrieval_metrics" in bench
        summary = bench.get("summary", bench)
        assert "answer_quality" in summary

    def test_query_grounded_answer(self, rag_service: RAGService):
        res = rag_service.query("What were ACME Corp's financial results?", top_k=2, min_score=0.0)
        assert "answer" in res
        assert "citations" in res
        assert res["is_supported"] is True
        assert len(res["retrieved_chunks"]) <= 2

    def test_unsupported_query_rejection(self, rag_service: RAGService):
        res = rag_service.query("What was Apple iPhone sales in 2030?")
        assert res["is_supported"] is False
        assert "insufficient information" in res["answer"].lower()


# --- REGRESSION TESTS FOR DASHBOARD BUG FIXES ---

class TestBugFixesRegression:
    def test_bug1_big_data_scatter_schema(self, analytics_service: AnalyticsService):
        """Verify Big Data Analytics Page scatter handles avg_volatility."""
        spark_stats = analytics_service.get_spark_enriched_statistics()
        if not spark_stats.empty:
            vol_col = "avg_volatility" if "avg_volatility" in spark_stats.columns else ("volatility" if "volatility" in spark_stats.columns else None)
            assert vol_col is not None
            assert vol_col in spark_stats.columns
            assert "avg_daily_return" in spark_stats.columns

    def test_bug2_sentiment_chart_schema(self, sentiment_service: SentimentService):
        """Verify sentiment timeline chart handles average_sentiment with multi-symbol rows."""
        from app.components.charts import create_sentiment_time_chart
        import pandas as pd

        # Test with real service output
        daily_df = sentiment_service.get_daily_sentiments()
        assert not daily_df.empty
        fig = create_sentiment_time_chart(daily_df)
        assert len(fig.data) >= 1

        # Test with empty dataframe
        fig_empty = create_sentiment_time_chart(pd.DataFrame())
        assert fig_empty is not None

        # Test with legacy column name
        mock_legacy = pd.DataFrame({
            "date": ["2023-01-01", "2023-01-02"],
            "mean_sentiment": [0.15, -0.20],
        })
        fig_legacy = create_sentiment_time_chart(mock_legacy)
        assert len(fig_legacy.data) >= 1

    def test_bug3_multi_factor_screening_drawdown(self, analytics_service: AnalyticsService):
        """Verify multi-factor screening works with real risk metrics and drawdown factor."""
        from src.analytics.screening import compute_multi_factor_rankings

        df_risk = analytics_service.get_real_risk_return_metrics()
        df_bench = analytics_service.get_real_benchmark_metrics()

        assert not df_risk.empty
        assert not df_bench.empty
        assert "max_drawdown" in df_risk.columns

        rankings = compute_multi_factor_rankings(df_risk, df_bench)
        assert not rankings.empty
        assert len(rankings) == 50
        assert "rank" in rankings.columns
        assert "composite_score" in rankings.columns
        assert "score_drawdown" in rankings.columns
        assert "score_sharpe" in rankings.columns
        assert "score_alpha" in rankings.columns
        assert rankings["rank"].min() == 1
        assert rankings["rank"].max() == 50

    def test_bug4_big_data_scope_switching_and_isolation(self, analytics_service: AnalyticsService):
        """Verify Big Data Analytics produces 50 tickers in real mode, 10 in synthetic mode, and does not reuse stale cache."""
        # Real mode
        real_stats = analytics_service.get_spark_stock_statistics(scope="Real Market History (NIFTY 50)")
        assert len(real_stats) == 50
        assert real_stats["symbol"].nunique() == 50

        # Synthetic mode (must NOT reuse real cached 50 tickers)
        synth_stats = analytics_service.get_spark_stock_statistics(scope="Synthetic Baseline (10 Equities)")
        assert len(synth_stats) == 10
        assert synth_stats["symbol"].nunique() == 10

        # Switch back to Real
        real_stats_again = analytics_service.get_spark_stock_statistics(scope="Real Market History (NIFTY 50)")
        assert len(real_stats_again) == 50

    def test_big_data_missing_data_no_silent_fallback(self, tmp_path: Path):
        """Verify missing real Big Data assets raise FileNotFoundError rather than fallback."""
        empty_svc = AnalyticsService(data_dir=tmp_path)
        with pytest.raises(FileNotFoundError, match="Real MapReduce volume dataset not found"):
            empty_svc.get_mapreduce_volume(scope="Real Market History (NIFTY 50)")

        with pytest.raises(FileNotFoundError, match="Real Spark stock rankings not found"):
            empty_svc.get_spark_stock_statistics(scope="Real Market History (NIFTY 50)")
