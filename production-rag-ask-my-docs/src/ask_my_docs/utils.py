"""General utility helpers."""

from __future__ import annotations

import re
from pathlib import Path

import orjson

_WORD_RE = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase alphanumeric tokens."""

    return [match.group(0).lower() for match in _WORD_RE.finditer(text)]


def ensure_dir(path: Path) -> None:
    """Create a directory path when missing."""

    path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> dict[str, object]:
    """Read a JSON object from disk."""

    payload = orjson.loads(path.read_bytes())
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}, found {type(payload).__name__}")
    return {str(key): value for key, value in payload.items()}


def write_json(path: Path, payload: object) -> None:
    """Write a JSON payload with stable formatting."""

    ensure_dir(path.parent)
    path.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS))
