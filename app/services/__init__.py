"""
Service facades connecting Streamlit UI pages to existing project modules and data artifacts.
"""

from app.services.data_service import DataService
from app.services.analytics_service import AnalyticsService
from app.services.ml_service import MLService
from app.services.sentiment_service import SentimentService
from app.services.mining_service import MiningService
from app.services.rag_service import RAGService

__all__ = [
    "DataService",
    "AnalyticsService",
    "MLService",
    "SentimentService",
    "MiningService",
    "RAGService",
]

