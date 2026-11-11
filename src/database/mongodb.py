"""
MongoDB connection and client management for the Financial Intelligence Platform.

Handles client lifecycle, connection pooling, configuration loading from
config.yaml and environment variables, and health checks.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.errors import ConnectionFailure, PyMongoError, ServerSelectionTimeoutError

ROOT = Path(__file__).resolve().parent.parent.parent
from src.utils.logger import logger

# Singleton client storage
_CLIENT_INSTANCE: Optional[MongoClient] = None


def load_mongo_config() -> Dict[str, Any]:
    """
    Load MongoDB settings with precedence:
    1. Environment variables (MONGODB_URI, MONGODB_DATABASE)
    2. config/config.yaml (mongodb section)
    3. Safe defaults
    """
    config_path = ROOT / "config" / "config.yaml"
    cfg: Dict[str, Any] = {}

    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                full_cfg = yaml.safe_load(f) or {}
                cfg = full_cfg.get("mongodb", {})
        except Exception as e:
            logger.warning(f"Could not read config.yaml for MongoDB settings: {e}")

    uri = os.environ.get("MONGODB_URI") or cfg.get("uri") or "mongodb://localhost:27017"
    database = os.environ.get("MONGODB_DATABASE") or cfg.get("database") or "financial_analytics"
    timeout_ms = int(os.environ.get("MONGODB_TIMEOUT_MS") or cfg.get("timeout_ms") or 5000)

    return {
        "uri": uri,
        "database": database,
        "timeout_ms": timeout_ms,
        "collections": cfg.get("collections", {
            "market_data": "market_data",
            "news": "news",
            "stock_features": "stock_features",
            "predictions": "predictions",
            "sentiment": "sentiment",
        }),
    }


def get_client(
    uri: Optional[str] = None,
    timeout_ms: Optional[int] = None,
    force_new: bool = False,
) -> MongoClient:
    """
    Get or initialize a PyMongo MongoClient singleton.

    Parameters
    ----------
    uri : str, optional
        MongoDB connection URI. If None, loaded from environment/config.
    timeout_ms : int, optional
        Server selection timeout in milliseconds (default 5000ms).
    force_new : bool, optional
        If True, creates a new MongoClient without caching.

    Returns
    -------
    MongoClient
        Active PyMongo client instance.
    """
    global _CLIENT_INSTANCE

    if not force_new and _CLIENT_INSTANCE is not None:
        return _CLIENT_INSTANCE

    mongo_cfg = load_mongo_config()
    target_uri = uri or mongo_cfg["uri"]
    target_timeout = timeout_ms if timeout_ms is not None else mongo_cfg["timeout_ms"]

    try:
        client: MongoClient = MongoClient(
            target_uri,
            serverSelectionTimeoutMS=target_timeout,
            connectTimeoutMS=target_timeout,
            socketTimeoutMS=target_timeout,
        )
        if not force_new:
            _CLIENT_INSTANCE = client
        return client
    except PyMongoError as e:
        logger.error(f"Failed to initialize MongoClient with URI '{target_uri}': {e}")
        raise


def get_database(
    name: Optional[str] = None,
    client: Optional[MongoClient] = None,
) -> Database:
    """
    Get MongoDB database instance.

    Parameters
    ----------
    name : str, optional
        Database name (defaults to configured database name).
    client : MongoClient, optional
        Optional custom client instance.
    """
    active_client = client or get_client()
    mongo_cfg = load_mongo_config()
    db_name = name or mongo_cfg["database"]
    return active_client[db_name]


def get_collection(
    collection_name: str,
    db_name: Optional[str] = None,
    client: Optional[MongoClient] = None,
) -> Collection:
    """
    Get a specific MongoDB collection.

    Parameters
    ----------
    collection_name : str
        Name of the collection (e.g. 'market_data', 'news', 'stock_features').
    db_name : str, optional
        Database name override.
    client : MongoClient, optional
        Optional custom client instance.
    """
    db = get_database(name=db_name, client=client)
    return db[collection_name]


def check_connection(
    client: Optional[MongoClient] = None,
    timeout_ms: int = 2000,
) -> Dict[str, Any]:
    """
    Perform a health check on the MongoDB server.

    Returns
    -------
    dict
        Health status dictionary containing:
        - connected: bool
        - version: str or None
        - database: str
        - uri: str
        - error: str or None
    """
    mongo_cfg = load_mongo_config()
    uri = mongo_cfg["uri"]
    database = mongo_cfg["database"]

    health_report: Dict[str, Any] = {
        "connected": False,
        "version": None,
        "database": database,
        "uri": uri,
        "error": None,
    }

    try:
        # Use a short timeout client for health check if no client supplied
        test_client = client or get_client(timeout_ms=timeout_ms, force_new=True)
        # ping admin command to test live responsiveness
        info = test_client.admin.command("ping")
        server_info = test_client.server_info()
        health_report["connected"] = bool(info.get("ok") == 1.0)
        health_report["version"] = server_info.get("version")
        logger.info(f"MongoDB connection healthy: version {health_report['version']} at {uri}")
    except (ServerSelectionTimeoutError, ConnectionFailure) as e:
        msg = f"MongoDB connection failed: Server unavailable at {uri} ({e.__class__.__name__})"
        logger.warning(msg)
        health_report["error"] = msg
    except Exception as e:
        msg = f"MongoDB health check error: {e}"
        logger.warning(msg)
        health_report["error"] = msg

    return health_report


def close_connection() -> None:
    """Close and reset the singleton MongoClient."""
    global _CLIENT_INSTANCE
    if _CLIENT_INSTANCE is not None:
        try:
            _CLIENT_INSTANCE.close()
            logger.debug("MongoDB connection closed.")
        finally:
            _CLIENT_INSTANCE = None
