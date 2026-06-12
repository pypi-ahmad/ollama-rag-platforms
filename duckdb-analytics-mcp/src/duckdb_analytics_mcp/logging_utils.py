"""Logging configuration utilities."""

from __future__ import annotations

import sys

from loguru import logger


def configure_logging(level: str) -> None:
    """Configure loguru to write logs to stderr.

    Using stderr avoids corrupting stdio-based MCP transport traffic.
    """
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
        backtrace=False,
        diagnose=False,
    )
