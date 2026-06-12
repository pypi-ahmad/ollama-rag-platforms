"""Centralized logger setup for consistent structured logs."""

from __future__ import annotations

import sys

from loguru import logger

_configured = False


def configure_logging() -> None:
    """Configure loguru once for CLI/API processes."""
    global _configured
    if _configured:
        return

    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
        level="INFO",
    )
    _configured = True


__all__ = ["configure_logging", "logger"]
