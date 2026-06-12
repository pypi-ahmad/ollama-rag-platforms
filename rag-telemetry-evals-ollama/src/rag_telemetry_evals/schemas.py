"""Typed models for documents, telemetry, and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class Document(BaseModel):
    """A source knowledge document."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    source: str
    title: str
    text: str


class DocumentChunk(BaseModel):
    """A chunked document segment."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    source: str
    title: str
    text: str
    start_word: int
    end_word: int


@dataclass(frozen=True)
class RetrievedChunk:
    """Retrieved chunk with similarity score."""

    chunk: DocumentChunk
    score: float


class QAExample(BaseModel):
    """Evaluation question with source expectation and answer cues."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    question_id: str
    question: str
    reference_answer: str
    expected_source: str
    required_keywords: list[str] = Field(default_factory=list)


class ChatResult(BaseModel):
    """Model response plus metadata returned by Ollama."""

    model_config = ConfigDict(extra="forbid")

    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_duration_ns: int = 0
    prompt_eval_duration_ns: int = 0
    eval_duration_ns: int = 0
    load_duration_ns: int = 0
    done_reason: str = ""

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def latency_ms(self) -> float:
        return self.total_duration_ns / 1_000_000 if self.total_duration_ns else 0.0


class TraceSpanRecord(BaseModel):
    """One telemetry span persisted as JSONL."""

    model_config = ConfigDict(extra="forbid")

    trace_id: str
    span_name: str
    status: Literal["ok", "error"]
    start_time_utc: str
    end_time_utc: str
    latency_ms: float
    attributes: dict[str, Any] = Field(default_factory=dict)


class EvalPrediction(BaseModel):
    """Prediction row for baseline and RAG responses."""

    model_config = ConfigDict(extra="forbid")

    question_id: str
    question: str
    expected_source: str
    reference_answer: str
    baseline_answer: str
    rag_answer: str
    baseline_keyword_recall: float
    rag_keyword_recall: float
    baseline_semantic_similarity: float
    rag_semantic_similarity: float
    retrieval_hit: bool
    rag_mentions_expected_source: bool
    baseline_latency_ms: float
    rag_latency_ms: float
    baseline_prompt_tokens: int
    rag_prompt_tokens: int
    baseline_completion_tokens: int
    rag_completion_tokens: int


class EvalSummary(BaseModel):
    """Aggregate metrics across all evaluation questions."""

    model_config = ConfigDict(extra="forbid")

    n_questions: int
    retrieval_hit_rate: float
    rag_mentions_expected_source_rate: float
    baseline_keyword_recall_mean: float
    rag_keyword_recall_mean: float
    keyword_recall_gain: float
    baseline_semantic_similarity_mean: float
    rag_semantic_similarity_mean: float
    semantic_similarity_gain: float
    baseline_latency_ms_mean: float
    rag_latency_ms_mean: float
    baseline_total_tokens_mean: float
    rag_total_tokens_mean: float
