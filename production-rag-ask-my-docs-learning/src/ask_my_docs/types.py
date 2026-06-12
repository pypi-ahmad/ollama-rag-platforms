"""Data models shared across ingestion, retrieval, answering, and evaluation."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """A single indexed chunk from a source document."""

    chunk_id: str
    doc_id: str
    title: str
    source_path: str
    text: str
    token_count: int


class RetrievedChunk(Chunk):
    """Chunk enriched with retrieval-time scores and rank."""

    bm25_score: float | None = None
    vector_score: float | None = None
    hybrid_score: float | None = None
    rerank_score: float | None = None
    rank: int


class AskResponse(BaseModel):
    """Final answer payload with supporting chunks."""

    question: str
    answer: str
    citations: list[RetrievedChunk]
    mode: str = "extractive-cited"


class EvalExample(BaseModel):
    """Single QA evaluation row."""

    question: str
    gold_chunk_ids: list[str] = Field(default_factory=list)
    expected_keywords: list[str] = Field(default_factory=list)


class EvalDataset(BaseModel):
    """Collection of evaluation questions for CI gating."""

    queries: list[EvalExample]
