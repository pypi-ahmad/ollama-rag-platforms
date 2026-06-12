"""Typed models for tasks, retrieval, agent outputs, and evaluations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class KnowledgeDoc(BaseModel):
    """Knowledge base document."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    source: str
    title: str
    text: str


@dataclass(frozen=True)
class RetrievedDoc:
    """Retrieved doc with lexical relevance score."""

    doc: KnowledgeDoc
    score: float


class TaskExample(BaseModel):
    """Evaluation task definition."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    task_id: str
    question: str
    reference_answer: str
    expected_source: str
    required_keywords: list[str] = Field(default_factory=list)


class ChatResult(BaseModel):
    """LLM response plus request metadata."""

    model_config = ConfigDict(extra="forbid")

    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_duration_ns: int = 0
    done_reason: str = ""
    fallback_used: bool = False

    @property
    def latency_ms(self) -> float:
        return self.total_duration_ns / 1_000_000 if self.total_duration_ns else 0.0


class AgentRoute(BaseModel):
    """Router output."""

    model_config = ConfigDict(extra="forbid")

    intent: Literal["incident", "release", "support", "governance", "general"]
    severity: Literal["S0", "S1", "S2", "S3", "NA"]
    reason: str


class AgentRunResult(BaseModel):
    """Output of one multi-agent coordinated run."""

    model_config = ConfigDict(extra="forbid")

    trace_id: str
    question: str
    route: AgentRoute
    retrieved: list[dict[str, Any]]
    baseline_answer: str
    plan_answer: str
    final_answer: str
    planner_fallback_used: bool
    reviewer_fallback_used: bool
    total_latency_ms: float


class TraceSpanRecord(BaseModel):
    """Telemetry span persisted in JSONL."""

    model_config = ConfigDict(extra="forbid")

    trace_id: str
    span_name: str
    status: Literal["ok", "error"]
    start_time_utc: str
    end_time_utc: str
    latency_ms: float
    attributes: dict[str, Any] = Field(default_factory=dict)


class EvalPrediction(BaseModel):
    """Per-task baseline vs multi-agent row."""

    model_config = ConfigDict(extra="forbid")

    task_id: str
    question: str
    expected_source: str
    baseline_answer: str
    final_answer: str
    baseline_keyword_recall: float
    final_keyword_recall: float
    keyword_gain: float
    retrieval_hit: bool
    source_cited_in_final: bool
    planner_fallback_used: bool
    reviewer_fallback_used: bool
    total_latency_ms: float


class EvalSummary(BaseModel):
    """Aggregate evaluation metrics."""

    model_config = ConfigDict(extra="forbid")

    n_tasks: int
    baseline_keyword_recall_mean: float
    multi_agent_keyword_recall_mean: float
    keyword_recall_gain: float
    retrieval_hit_rate: float
    source_citation_rate: float
    planner_fallback_rate: float
    reviewer_fallback_rate: float
    avg_total_latency_ms: float
