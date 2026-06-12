"""DuckDB-backed storage for request observability metrics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import duckdb

from ask_my_docs.utils import ensure_dir


@dataclass(slots=True)
class RequestMetricRecord:
    """One request metrics row stored in DuckDB."""

    request_id: str
    trace_id: str | None
    model_name: str
    question: str
    latency_ms: float
    retrieval_latency_ms: float
    generation_latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost_usd: float
    retrieval_recall_at_k: float | None
    answer_f1: float | None
    exact_match: float | None
    timestamp_utc: str


class MetricsStore:
    """Persist and summarize request metrics."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        ensure_dir(db_path.parent)
        self._init_schema()

    def _connect(self) -> duckdb.DuckDBPyConnection:
        return duckdb.connect(str(self._db_path))

    def _init_schema(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS rag_request_metrics (
                    request_id TEXT,
                    trace_id TEXT,
                    model_name TEXT,
                    question TEXT,
                    latency_ms DOUBLE,
                    retrieval_latency_ms DOUBLE,
                    generation_latency_ms DOUBLE,
                    prompt_tokens BIGINT,
                    completion_tokens BIGINT,
                    total_tokens BIGINT,
                    estimated_cost_usd DOUBLE,
                    retrieval_recall_at_k DOUBLE,
                    answer_f1 DOUBLE,
                    exact_match DOUBLE,
                    timestamp_utc TEXT
                )
                """
            )
            connection.execute(
                "ALTER TABLE rag_request_metrics ADD COLUMN IF NOT EXISTS model_name TEXT"
            )

    def record(self, record: RequestMetricRecord) -> None:
        """Insert one request metric row."""

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO rag_request_metrics (
                    request_id,
                    trace_id,
                    model_name,
                    question,
                    latency_ms,
                    retrieval_latency_ms,
                    generation_latency_ms,
                    prompt_tokens,
                    completion_tokens,
                    total_tokens,
                    estimated_cost_usd,
                    retrieval_recall_at_k,
                    answer_f1,
                    exact_match,
                    timestamp_utc
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    record.request_id,
                    record.trace_id,
                    record.model_name,
                    record.question,
                    record.latency_ms,
                    record.retrieval_latency_ms,
                    record.generation_latency_ms,
                    record.prompt_tokens,
                    record.completion_tokens,
                    record.total_tokens,
                    record.estimated_cost_usd,
                    record.retrieval_recall_at_k,
                    record.answer_f1,
                    record.exact_match,
                    record.timestamp_utc,
                ],
            )

    def update_quality(self, request_id: str, answer_f1: float, exact_match: float) -> None:
        """Backfill quality fields once scoring is complete."""

        with self._connect() as connection:
            connection.execute(
                """
                UPDATE rag_request_metrics
                SET answer_f1 = ?, exact_match = ?
                WHERE request_id = ?
                """,
                [answer_f1, exact_match, request_id],
            )

    def summarize(self, limit: int | None = None) -> dict[str, float]:
        """Compute aggregate observability metrics including p50/p95 latency."""

        base_query = "SELECT * FROM rag_request_metrics"
        if limit is not None:
            source = f"{base_query} ORDER BY timestamp_utc DESC LIMIT {int(limit)}"
        else:
            source = base_query

        query = f"""
            WITH source AS ({source})
            SELECT
                COUNT(*)::DOUBLE AS request_count,
                COALESCE(quantile_cont(latency_ms, 0.50), 0.0) AS latency_p50_ms,
                COALESCE(quantile_cont(latency_ms, 0.95), 0.0) AS latency_p95_ms,
                COALESCE(AVG(estimated_cost_usd), 0.0) AS avg_cost_usd,
                COALESCE(AVG(retrieval_recall_at_k), 0.0) AS avg_retrieval_recall_at_k,
                COALESCE(AVG(answer_f1), 0.0) AS avg_answer_f1,
                COALESCE(AVG(exact_match), 0.0) AS avg_exact_match
            FROM source
        """

        with self._connect() as connection:
            row = connection.execute(query).fetchone()

        if row is None:
            return {
                "request_count": 0.0,
                "latency_p50_ms": 0.0,
                "latency_p95_ms": 0.0,
                "avg_cost_usd": 0.0,
                "avg_retrieval_recall_at_k": 0.0,
                "avg_answer_f1": 0.0,
                "avg_exact_match": 0.0,
            }

        return {
            "request_count": float(row[0]),
            "latency_p50_ms": float(row[1]),
            "latency_p95_ms": float(row[2]),
            "avg_cost_usd": float(row[3]),
            "avg_retrieval_recall_at_k": float(row[4]),
            "avg_answer_f1": float(row[5]),
            "avg_exact_match": float(row[6]),
        }
