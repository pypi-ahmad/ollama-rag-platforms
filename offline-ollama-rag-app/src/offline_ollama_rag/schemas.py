"""Typed models for documents, retrieval, and evaluation."""

from __future__ import annotations

from dataclasses import dataclass

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
    """Evaluation question with reference cues."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    question_id: str
    question: str
    reference_answer: str
    required_keywords: list[str] = Field(default_factory=list)


class EvalPrediction(BaseModel):
    """Prediction row for baseline and RAG responses."""

    model_config = ConfigDict(extra="forbid")

    question_id: str
    question: str
    reference_answer: str
    baseline_answer: str
    rag_answer: str
    baseline_keyword_recall: float
    rag_keyword_recall: float


class EvalSummary(BaseModel):
    """Aggregate evaluation metrics."""

    model_config = ConfigDict(extra="forbid")

    n_questions: int
    baseline_keyword_recall_mean: float
    rag_keyword_recall_mean: float
    keyword_recall_gain: float
    baseline_nonempty_rate: float
    rag_nonempty_rate: float
