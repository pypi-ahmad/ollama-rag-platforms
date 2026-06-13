"""DuckDB-backed storage for request observability metrics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import RLock

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
        self._lock = RLock()
        self._connection = duckdb.connect(str(self._db_path))
        self._closed = False
        self._init_schema()

    def _table_exists(self) -> bool:
        row = self._connection.execute(
            """
            SELECT COUNT(*)
            FROM information_schema.tables
            WHERE table_schema = 'main'
              AND table_name = 'rag_request_metrics'
            """
        ).fetchone()
        return bool(row and int(row[0]) > 0)

    def _has_primary_key_on_request_id(self) -> bool:
        rows = self._connection.execute("PRAGMA table_info('rag_request_metrics')").fetchall()
        for row in rows:
            column_name = str(row[1])
            is_primary_key = int(row[5]) == 1
            if column_name == "request_id" and is_primary_key:
                return True
        return False

    def _create_schema(self) -> None:
        self._connection.execute(
            """
            CREATE TABLE rag_request_metrics (
                request_id TEXT PRIMARY KEY,
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

    def _migrate_schema_to_v2(self) -> None:
        self._connection.execute(
            "ALTER TABLE rag_request_metrics ADD COLUMN IF NOT EXISTS model_name TEXT"
        )

        self._connection.execute(
            """
            CREATE TABLE rag_request_metrics_v2 (
                request_id TEXT PRIMARY KEY,
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

        self._connection.execute(
            """
            INSERT INTO rag_request_metrics_v2 (
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
            SELECT
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
            FROM (
                SELECT
                    *,
                    ROW_NUMBER() OVER (
                        PARTITION BY request_id
                        ORDER BY timestamp_utc DESC
                    ) AS row_num
                FROM rag_request_metrics
            ) AS deduped
            WHERE row_num = 1
            """
        )

        self._connection.execute("DROP TABLE rag_request_metrics")
        self._connection.execute("ALTER TABLE rag_request_metrics_v2 RENAME TO rag_request_metrics")

    def _ensure_indexes(self) -> None:
        self._connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_rag_request_metrics_timestamp_utc
            ON rag_request_metrics(timestamp_utc)
            """
        )
        self._connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_rag_request_metrics_model_name
            ON rag_request_metrics(model_name)
            """
        )

    def _init_schema(self) -> None:
        with self._lock:
            if not self._table_exists():
                self._create_schema()
            else:
                self._connection.execute(
                    "ALTER TABLE rag_request_metrics ADD COLUMN IF NOT EXISTS model_name TEXT"
                )
                if not self._has_primary_key_on_request_id():
                    self._migrate_schema_to_v2()

            self._ensure_indexes()

    def record(self, record: RequestMetricRecord) -> None:
        """Insert one request metric row."""

        with self._lock:
            self._connection.execute(
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
                ON CONFLICT(request_id) DO UPDATE SET
                    trace_id = excluded.trace_id,
                    model_name = excluded.model_name,
                    question = excluded.question,
                    latency_ms = excluded.latency_ms,
                    retrieval_latency_ms = excluded.retrieval_latency_ms,
                    generation_latency_ms = excluded.generation_latency_ms,
                    prompt_tokens = excluded.prompt_tokens,
                    completion_tokens = excluded.completion_tokens,
                    total_tokens = excluded.total_tokens,
                    estimated_cost_usd = excluded.estimated_cost_usd,
                    retrieval_recall_at_k = excluded.retrieval_recall_at_k,
                    answer_f1 = excluded.answer_f1,
                    exact_match = excluded.exact_match,
                    timestamp_utc = excluded.timestamp_utc
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

        with self._lock:
            self._connection.execute(
                """
                UPDATE rag_request_metrics
                SET answer_f1 = ?, exact_match = ?
                WHERE request_id = ?
                """,
                [answer_f1, exact_match, request_id],
            )

    def summarize(self, limit: int | None = None) -> dict[str, float]:
        """Compute aggregate observability metrics including p50/p95 latency."""

        source = "SELECT * FROM rag_request_metrics"
        if limit is not None:
            if limit <= 0:
                raise ValueError("limit must be greater than zero")
            source = f"{source} ORDER BY timestamp_utc DESC LIMIT {int(limit)}"

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

        with self._lock:
            row = self._connection.execute(query).fetchone()

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

    def close(self) -> None:
        """Close the DuckDB connection."""

        with self._lock:
            if self._closed:
                return
            self._connection.close()
            self._closed = True

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            return
