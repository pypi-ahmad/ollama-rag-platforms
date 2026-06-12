"""Shared typed models for retrieval, answering, and observability."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime


@dataclass(slots=True)
class Document:
    """Represents one source document.

    Attributes:
        doc_id: Stable document identifier.
        text: Full document text.
        metadata: Source metadata (path, tags, etc.).
    """

    doc_id: str
    text: str
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class Chunk:
    """Represents one chunk from a source document."""

    chunk_id: str
    doc_id: str
    text: str
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class RetrievedChunk:
    """Represents one ranked retrieval result."""

    chunk: Chunk
    score: float
    lexical_score: float
    semantic_score: float
    rank: int


@dataclass(slots=True)
class RAGAnswer:
    """Represents one fully observed RAG response."""

    request_id: str
    trace_id: str | None
    model_name: str
    question: str
    answer: str
    citations: list[str]
    retrieved_doc_ids: list[str]
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost_usd: float
    latency_ms: float
    retrieval_latency_ms: float
    generation_latency_ms: float
    retrieval_recall_at_k: float | None = None
    timestamp_utc: str = field(
        default_factory=lambda: datetime.now(tz=UTC).isoformat(timespec="seconds")
    )
