"""
Logging utility — thin wrapper around loguru.

Usage:
    from src.utils.logger import logger
    logger.info("Pipeline started")
"""

import sys
from pathlib import Path

from loguru import logger

# ── Remove default handler ───────────────────────────────────
logger.remove()

# ── Console sink (colourised, INFO+) ─────────────────────────
logger.add(
    sys.stderr,
    level="INFO",
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> — "
        "<level>{message}</level>"
    ),
    colorize=True,
)

# ── File sink (DEBUG+, rotation) ─────────────────────────────
_log_dir = Path("logs")
_log_dir.mkdir(exist_ok=True)

logger.add(
    _log_dir / "pipeline.log",
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} — {message}",
    rotation="5 MB",
    retention="7 days",
    encoding="utf-8",
)

__all__ = ["logger"]
