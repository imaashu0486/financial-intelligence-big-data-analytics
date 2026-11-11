"""
End-to-end Backend Integration Smoke Test.
Verifies all 6 analytical pipelines, MongoDB offline fallback, and Streamlit service facades.
"""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def run_smoke_test():
    print("=" * 70)
    print("FINANCIAL INTELLIGENCE PLATFORM — BACKEND INTEGRATION SMOKE TEST")
    print("=" * 70)

    # 1. Market Data Service
    print("\n[1] Auditing DataService (Phase 1)...")
    from app.services.data_service import DataService
    ds = DataService()
    df = ds.load_market_features()
    kpis = ds.get_market_kpis(df)
    assert len(df) == 5200, f"Expected 5200 rows, got {len(df)}"
    assert kpis["symbols_count"] == 10, f"Expected 10 symbols, got {kpis['symbols_count']}"
    assert kpis["trading_days"] == 520, f"Expected 520 trading days, got {kpis['trading_days']}"
    print(f"  PASS: {len(df)} market rows, 10 symbols, {kpis['min_date']} to {kpis['max_date']}")

    # 2. Big Data Analytics Service
    print("\n[2] Auditing AnalyticsService (Phase 2)...")
    from app.services.analytics_service import AnalyticsService
    ans = AnalyticsService()
    vol = ans.get_mapreduce_volume()
    words = ans.get_mapreduce_word_counts(top_n=5)
    spark_stats = ans.get_spark_stock_statistics()
    spark_ranks = ans.get_spark_stock_rankings()
    spark_daily = ans.get_spark_daily_statistics()
    assert len(vol) == 10, f"Expected 10 ticker volumes, got {len(vol)}"
    assert len(words) == 5, f"Expected 5 top words, got {len(words)}"
    assert len(spark_stats) == 10, f"Expected 10 spark statistics rows, got {len(spark_stats)}"
    assert len(spark_ranks) == 10, f"Expected 10 spark ranking rows, got {len(spark_ranks)}"
    assert len(spark_daily) == 519, f"Expected 519 daily rows, got {len(spark_daily)}"
    print(f"  PASS: MapReduce & PySpark Parquet datasets loaded cleanly ({len(vol)} tickers, {len(words)} top words, {len(spark_stats)} statistics)")

    # 3. Machine Learning Service
    print("\n[3] Auditing MLService (Phase 4)...")
    from app.services.ml_service import MLService
    mls = MLService()
    metrics = mls.get_metrics()
    reg_preds = mls.get_regression_predictions()
    clf_preds = mls.get_classification_predictions()
    clusters = mls.get_stock_clusters()
    features = mls.get_feature_importances()
    assert "regression" in metrics and "classification" in metrics and "clustering" in metrics
    assert len(reg_preds) == 1480, f"Expected 1480 regression test preds, got {len(reg_preds)}"
    assert len(clf_preds) == 1480, f"Expected 1480 classification test preds, got {len(clf_preds)}"
    assert len(clusters) == 10, f"Expected 10 stock clusters, got {len(clusters)}"
    assert len(features) > 0, "Feature importance table is empty"
    infer = mls.predict_market_direction({"daily_return": 0.01, "rsi_14": 62.0, "macd": 1.5})
    assert "prediction" in infer and "probabilities" in infer
    print(f"  PASS: ML pipelines verified (1480 test rows, {len(clusters)} clusters, inference -> {infer['prediction']} at {infer['confidence']*100:.1f}%)")

    # 4. Sentiment Intelligence Service
    print("\n[4] Auditing SentimentService (Phase 5)...")
    from app.services.sentiment_service import SentimentService
    ss = SentimentService()
    dist = ss.get_sentiment_distribution()
    articles = ss.get_article_sentiments()
    daily = ss.get_daily_sentiments()
    joined = ss.get_sentiment_market_joined()
    corrs = ss.get_correlations()
    assert len(articles) == 500, f"Expected 500 articles, got {len(articles)}"
    assert len(daily) == 469, f"Expected 469 daily sentiment rows, got {len(daily)}"
    assert len(joined) == 469, f"Expected 469 joined rows, got {len(joined)}"
    assert dist == {"Neutral": 233, "Positive": 174, "Negative": 93}
    print(f"  PASS: 500 news articles, 469 daily records, distribution {dist}")

    # 5. Mining Service
    print("\n[5] Auditing MiningService (Phase 6)...")
    from app.services.mining_service import MiningService
    ms = MiningService()
    rules = ms.get_association_rules()
    itemsets = ms.get_frequent_itemsets()
    m_stats = ms.get_mining_statistics()
    tx_summary = ms.get_transaction_summary()
    assert len(rules) == 10, f"Expected 10 association rules, got {len(rules)}"
    assert len(itemsets) == 47, f"Expected 47 frequent itemsets, got {len(itemsets)}"
    assert tx_summary["num_transactions"] == 482, f"Expected 482 transactions, got {tx_summary['num_transactions']}"
    assert m_stats["num_rules"] == 10, f"Expected 10 rules in stats, got {m_stats['num_rules']}"
    print(f"  PASS: 10 association rules, 47 frequent itemsets, 482 market basket transactions")

    # 6. RAG Service
    print("\n[6] Auditing RAGService (Phases 7 & 8)...")
    from app.services.rag_service import RAGService
    rs = RAGService(mock=True)
    idx = rs.get_index_status()
    eval_b = rs.get_evaluation_benchmarks()
    assert idx["chunk_count"] == 28, f"Expected 28 vectors in index, got {idx['chunk_count']}"
    assert idx["unique_chunk_count"] == 28, f"Expected 28 unique chunks, got {idx['unique_chunk_count']}"
    assert len(idx["indexed_documents"]) == 3, f"Expected 3 indexed docs, got {len(idx['indexed_documents'])}"
    assert eval_b.get("summary", {}).get("total_samples") == 24, "Expected 24 evaluation samples"
    
    # Query test
    q_supp = "What were ACME Corporation's operating profit (EBITDA) and net profit after tax for FY 2024?"
    ans_supp = rs.query(q_supp)
    assert ans_supp["is_supported"] is True, f"Expected supported answer, got {ans_supp}"
    assert len(ans_supp["citations"]) >= 1, "Expected citations in supported answer"

    q_unsupp = "What will be the price of Bitcoin in 2050?"
    ans_unsupp = rs.query(q_unsupp)
    assert ans_unsupp["is_supported"] is False, "Expected unsupported question to be rejected"
    assert "insufficient information" in ans_unsupp["answer"].lower()
    print(f"  PASS: RAG index ({idx['chunk_count']} vectors / {idx['unique_chunk_count']} unique chunks across 3 docs), supported query answered with citations, unsupported query rejected")

    # 7. Database Fallback
    print("\n[7] Auditing MongoDB Offline Fallback...")
    from src.database.mongodb import check_connection
    health = check_connection(timeout_ms=200)
    assert isinstance(health, dict), "Health report must be a dict"
    assert "connected" in health, "Health report must contain 'connected' key"
    print(f"  PASS: Connection health check returned valid dictionary (connected={health['connected']})")

    print("\n" + "=" * 70)
    print("ALL 7 SERVICE INTEGRATION WORKFLOWS PASSED VERIFICATION!")
    print("=" * 70)

if __name__ == "__main__":
    run_smoke_test()

