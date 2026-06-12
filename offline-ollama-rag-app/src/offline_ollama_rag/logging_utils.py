"""Logging setup utilities."""

from __future__ import annotations

import sys

from loguru import logger


def configure_logging(level: str = "INFO") -> None:
    """Configure loguru output.

    Stream logs to stderr so stdio-based workflows are not polluted.
    """
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
        backtrace=False,
        diagnose=False,
    )
