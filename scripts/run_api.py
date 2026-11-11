"""
CLI Runner for the Unified Financial Intelligence REST API Server.

Usage:
    python scripts/run_api.py
    python scripts/run_api.py --port 8080 --host 0.0.0.0
"""

import argparse
import sys
from pathlib import Path
import uvicorn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.logger import logger


def main() -> None:
    parser = argparse.ArgumentParser(description="Financial Intelligence REST API Server")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host interface (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Listening port (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload on code change")
    args = parser.parse_args()

    logger.info("=" * 65)
    logger.info("Starting Financial Intelligence REST API Server")
    logger.info(f"Target: http://{args.host}:{args.port}")
    logger.info(f"Health: http://{args.host}:{args.port}/api/health")
    logger.info("=" * 65)

    uvicorn.run(
        "src.api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()

