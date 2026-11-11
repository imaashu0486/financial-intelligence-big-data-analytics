"""
Unified REST API Service for the Financial Intelligence Platform.
Built on Starlette and Uvicorn for asynchronous, high-throughput service delivery.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import numpy as np
import pandas as pd
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

ROOT = Path(__file__).resolve().parent.parent.parent

from app.services.data_service import DataService
from app.services.analytics_service import AnalyticsService
from app.services.ml_service import MLService
from app.services.sentiment_service import SentimentService
from app.services.mining_service import MiningService
from app.services.rag_service import RAGService
from src.database.mongodb import check_connection

# Initialize Services
data_service = DataService()
analytics_service = AnalyticsService()
ml_service = MLService()
sentiment_service = SentimentService()
mining_service = MiningService()
rag_service = RAGService()


def _sanitize(val: Any) -> Any:
    """Recursively convert float NaN/Inf, NumPy types, and Timestamps to JSON-safe primitives."""
    if isinstance(val, (float, np.floating)):
        if np.isnan(val) or np.isinf(val):
            return None
        return float(val)
    elif isinstance(val, (int, np.integer)):
        return int(val)
    elif isinstance(val, (pd.Timestamp, np.datetime64)):
        return str(val)
    elif isinstance(val, dict):
        return {k: _sanitize(v) for k, v in val.items()}
    elif isinstance(val, (list, tuple)):
        return [_sanitize(v) for v in val]
    return val


def _df_to_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Convert DataFrame to JSON-safe dictionary records."""
    if df.empty:
        return []
    records = df.to_dict(orient="records")
    return _sanitize(records)


# --- ROUTE HANDLERS ---

async def health_check(request: Request) -> JSONResponse:
    """GET /api/health: Check overall backend, storage, database, and model health."""
    mongo_status = check_connection(timeout_ms=300)
    parquet_ready = (ROOT / "data" / "processed" / "spark" / "stock_statistics").exists()
    models_ready = (ROOT / "data" / "models" / "classification_random_forest.joblib").exists()
    rag_status = rag_service.get_index_status()

    return JSONResponse({
        "status": "healthy",
        "service": "Financial Intelligence REST API",
        "version": "1.0.0",
        "database": {
            "mongodb_connected": bool(mongo_status.get("connected", False)),
            "fallback_active": not bool(mongo_status.get("connected", False)),
        },
        "storage": {
            "spark_parquet_ready": parquet_ready,
            "vectorstore_ready": rag_status["is_indexed"],
            "chunk_count": rag_status["chunk_count"],
        },
        "models": {
            "classifiers_loaded": models_ready,
        },
    })


# --- MARKET ROUTES ---

async def get_market_kpis(request: Request) -> JSONResponse:
    """GET /api/market/kpis: Global market record metrics."""
    kpis = data_service.get_market_kpis()
    return JSONResponse(_sanitize(kpis))


async def get_market_tickers(request: Request) -> JSONResponse:
    """GET /api/market/tickers: List of tracked stock symbols and metadata."""
    kpis = data_service.get_market_kpis()
    df = data_service.load_market_features()
    mapping = {}
    if not df.empty and "company_name" in df.columns:
        sub = df[["symbol", "company_name", "sector"]].drop_duplicates()
        mapping = {row["symbol"]: {"company_name": row["company_name"], "sector": row["sector"]} for _, row in sub.iterrows()}

    return JSONResponse({
        "symbols_count": kpis["symbols_count"],
        "symbols": kpis["symbols"],
        "metadata": mapping,
    })


async def get_market_snapshot(request: Request) -> JSONResponse:
    """GET /api/market/snapshot: Latest trading snapshot across all 10 stocks."""
    snapshot = data_service.get_latest_snapshot()
    return JSONResponse({"count": len(snapshot), "snapshot": _df_to_records(snapshot)})


async def get_ticker_history(request: Request) -> JSONResponse:
    """GET /api/market/{symbol}: Historical prices & technicals for a single ticker."""
    symbol = request.path_params["symbol"].upper()
    start_date = request.query_params.get("start_date")
    end_date = request.query_params.get("end_date")

    df = data_service.get_ticker_data(symbol, start_date=start_date, end_date=end_date)
    if df.empty:
        return JSONResponse({"error": f"Symbol '{symbol}' not found in dataset."}, status_code=404)

    return JSONResponse({
        "symbol": symbol,
        "count": len(df),
        "data": _df_to_records(df),
    })


# --- BIG DATA ROUTES ---

async def get_mapreduce_volume(request: Request) -> JSONResponse:
    """GET /api/bigdata/mapreduce/volume: MapReduce ticker cumulative volume."""
    df = analytics_service.get_mapreduce_volume()
    return JSONResponse({"count": len(df), "volumes": _df_to_records(df)})


async def get_mapreduce_words(request: Request) -> JSONResponse:
    """GET /api/bigdata/mapreduce/words: MapReduce news term frequencies."""
    top_n = int(request.query_params.get("top_n", 25))
    df = analytics_service.get_mapreduce_word_counts(top_n=top_n)
    return JSONResponse({"count": len(df), "terms": _df_to_records(df)})


async def get_spark_statistics(request: Request) -> JSONResponse:
    """GET /api/bigdata/spark/statistics: Spark stock statistics (Parquet)."""
    df = analytics_service.get_spark_stock_statistics()
    return JSONResponse({"count": len(df), "statistics": _df_to_records(df)})


async def get_spark_rankings(request: Request) -> JSONResponse:
    """GET /api/bigdata/spark/rankings: Spark window function stock rankings."""
    df = analytics_service.get_spark_stock_rankings()
    return JSONResponse({"count": len(df), "rankings": _df_to_records(df)})


async def get_cluster_specs(request: Request) -> JSONResponse:
    """GET /api/bigdata/cluster: Docker cluster nodes, topology, and academic statement."""
    info = analytics_service.get_cluster_architecture()
    return JSONResponse(info)


# --- MACHINE LEARNING ROUTES ---

async def get_ml_metrics(request: Request) -> JSONResponse:
    """GET /api/ml/metrics: Performance metrics for regression, classification, clustering."""
    metrics = ml_service.get_metrics()
    return JSONResponse(_sanitize(metrics))


async def get_ml_features(request: Request) -> JSONResponse:
    """GET /api/ml/features: Random Forest feature importances."""
    df = ml_service.get_feature_importances()
    return JSONResponse({"features": _df_to_records(df)})


async def get_ml_clusters(request: Request) -> JSONResponse:
    """GET /api/ml/clusters: Stock risk-return K-Means cluster assignments."""
    df = ml_service.get_stock_clusters()
    profiles = ml_service.get_clustering_metrics().get("cluster_profiles", [])
    return JSONResponse({"clusters": _df_to_records(df), "profiles": profiles})


async def predict_direction(request: Request) -> JSONResponse:
    """POST /api/ml/predict: Run Random Forest inference on technical indicators."""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body provided."}, status_code=400)

    result = ml_service.predict_market_direction(body)
    if "error" in result:
        return JSONResponse(result, status_code=400)
    return JSONResponse(_sanitize(result))


# --- SENTIMENT ROUTES ---

async def get_sentiment_analytics(request: Request) -> JSONResponse:
    """GET /api/sentiment/analytics: News sentiment analytics, correlations, and regimes."""
    data = sentiment_service.get_sentiment_analytics()
    return JSONResponse(_sanitize(data))


async def get_sentiment_articles(request: Request) -> JSONResponse:
    """GET /api/sentiment/articles: Article-level sentiment classifications."""
    symbol = request.query_params.get("symbol")
    df = sentiment_service.get_article_sentiments(symbol=symbol)
    return JSONResponse({"count": len(df), "articles": _df_to_records(df)})


async def get_daily_sentiment(request: Request) -> JSONResponse:
    """GET /api/sentiment/daily: Daily aggregated news sentiment timelines."""
    df = sentiment_service.get_daily_sentiments()
    return JSONResponse({"count": len(df), "timeline": _df_to_records(df)})


# --- ASSOCIATION MINING ROUTES ---

async def get_mining_rules(request: Request) -> JSONResponse:
    """GET /api/mining/rules: Apriori association rules with dynamic threshold filters."""
    min_supp = float(request.query_params.get("min_support", 0.05))
    min_conf = float(request.query_params.get("min_confidence", 0.30))
    min_lift = float(request.query_params.get("min_lift", 1.00))

    df = mining_service.get_association_rules(
        min_support=min_supp,
        min_confidence=min_conf,
        min_lift=min_lift,
    )
    return JSONResponse({"count": len(df), "rules": _df_to_records(df)})


async def get_mining_itemsets(request: Request) -> JSONResponse:
    """GET /api/mining/itemsets: Frequent itemsets discovered by Apriori."""
    min_supp = float(request.query_params.get("min_support", 0.05))
    df = mining_service.get_frequent_itemsets(min_support=min_supp)
    return JSONResponse({"count": len(df), "itemsets": _df_to_records(df)})


# --- RAG ROUTES ---

async def get_rag_status(request: Request) -> JSONResponse:
    """GET /api/rag/status: Vector store indexing status, document manifest, chunk count."""
    status = rag_service.get_index_status()
    return JSONResponse(_sanitize(status))


async def query_rag(request: Request) -> JSONResponse:
    """POST /api/rag/query: Submit financial question and receive grounded extractive answer."""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body provided."}, status_code=400)

    question = body.get("question", "")
    top_k = int(body.get("top_k", 5))

    if not question:
        return JSONResponse({"error": "Field 'question' is required."}, status_code=400)

    response = rag_service.query(question=question, top_k=top_k)
    return JSONResponse(_sanitize(response))


async def get_rag_benchmarks(request: Request) -> JSONResponse:
    """GET /api/rag/benchmarks: Phase 8 empirical evaluation benchmarks."""
    bench = rag_service.get_evaluation_benchmarks()
    return JSONResponse(_sanitize(bench))


# --- APPLICATION ROUTING DEFINITION ---

routes = [
    # Health & Meta
    Route("/api/health", endpoint=health_check, methods=["GET"]),
    Route("/health", endpoint=health_check, methods=["GET"]),

    # Market Endpoints
    Route("/api/market/kpis", endpoint=get_market_kpis, methods=["GET"]),
    Route("/api/market/tickers", endpoint=get_market_tickers, methods=["GET"]),
    Route("/api/market/snapshot", endpoint=get_market_snapshot, methods=["GET"]),
    Route("/api/market/{symbol}", endpoint=get_ticker_history, methods=["GET"]),

    # Big Data Endpoints
    Route("/api/bigdata/mapreduce/volume", endpoint=get_mapreduce_volume, methods=["GET"]),
    Route("/api/bigdata/mapreduce/words", endpoint=get_mapreduce_words, methods=["GET"]),
    Route("/api/bigdata/spark/statistics", endpoint=get_spark_statistics, methods=["GET"]),
    Route("/api/bigdata/spark/rankings", endpoint=get_spark_rankings, methods=["GET"]),
    Route("/api/bigdata/cluster", endpoint=get_cluster_specs, methods=["GET"]),

    # Machine Learning Endpoints
    Route("/api/ml/metrics", endpoint=get_ml_metrics, methods=["GET"]),
    Route("/api/ml/features", endpoint=get_ml_features, methods=["GET"]),
    Route("/api/ml/clusters", endpoint=get_ml_clusters, methods=["GET"]),
    Route("/api/ml/predict", endpoint=predict_direction, methods=["POST"]),

    # Sentiment Endpoints
    Route("/api/sentiment/analytics", endpoint=get_sentiment_analytics, methods=["GET"]),
    Route("/api/sentiment/articles", endpoint=get_sentiment_articles, methods=["GET"]),
    Route("/api/sentiment/daily", endpoint=get_daily_sentiment, methods=["GET"]),

    # Association Mining Endpoints
    Route("/api/mining/rules", endpoint=get_mining_rules, methods=["GET"]),
    Route("/api/mining/itemsets", endpoint=get_mining_itemsets, methods=["GET"]),

    # RAG Endpoints
    Route("/api/rag/status", endpoint=get_rag_status, methods=["GET"]),
    Route("/api/rag/query", endpoint=query_rag, methods=["POST"]),
    Route("/api/rag/benchmarks", endpoint=get_rag_benchmarks, methods=["GET"]),
]

middleware = [
    Middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
]

app = Starlette(debug=False, routes=routes, middleware=middleware)

