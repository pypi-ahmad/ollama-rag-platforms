"""Text utilities: tokenization, chunking, citation parsing, and sentence extraction."""

from __future__ import annotations

import re
from collections.abc import Iterable

_TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_]+")
_SENTENCE_PATTERN = re.compile(r"[^.!?]+[.!?]")
_CITATION_PATTERN = re.compile(r"\[(\d+)]")


def simple_tokenize(text: str) -> list[str]:
    """Lowercase tokenization suitable for BM25 indexing."""
    return [m.group(0).lower() for m in _TOKEN_PATTERN.finditer(text)]


def split_into_sentences(text: str) -> list[str]:
    """Split text into basic sentence units."""
    sentences = [s.strip() for s in _SENTENCE_PATTERN.findall(text)]
    return sentences if sentences else [text.strip()]


def sliding_word_chunks(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Create overlapping chunks by word-count windows."""
    words = text.split()
    if not words:
        return []

    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")

    chunks: list[str] = []
    step = chunk_size - overlap
    for start in range(0, len(words), step):
        end = start + chunk_size
        chunk_words = words[start:end]
        if not chunk_words:
            continue
        chunks.append(" ".join(chunk_words))
        if end >= len(words):
            break
    return chunks


def extract_citation_numbers(answer: str) -> list[int]:
    """Extract citation numbers from an answer string."""
    return [int(m.group(1)) for m in _CITATION_PATTERN.finditer(answer)]


def ensure_line_citations(lines: Iterable[str], default_citation: int = 1) -> list[str]:
    """Enforce at least one citation marker on each non-empty line."""
    output: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if _CITATION_PATTERN.search(stripped):
            output.append(stripped)
        else:
            output.append(f"{stripped} [{default_citation}]")
    return output
