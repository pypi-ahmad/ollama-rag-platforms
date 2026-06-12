"""Core analytics service used by MCP tools/resources."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from datetime import UTC, datetime
from typing import TypeVar

import duckdb
from loguru import logger

from duckdb_analytics_mcp.catalog import DatasetCatalog, DatasetEntry
from duckdb_analytics_mcp.config import Settings
from duckdb_analytics_mcp.models import (
    DatasetColumn,
    DatasetDescription,
    HealthStatus,
    PaginatedDatasetsResult,
    QueryResult,
)
from duckdb_analytics_mcp.security import SQLGuard

T = TypeVar("T")


class AnalyticsService:
    """Read-only analytics operations against local datasets."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._catalog = DatasetCatalog(settings.resolved_dataset_dir)
        self._guard = SQLGuard(max_query_chars=settings.max_query_chars)

    def list_datasets(self, limit: int, offset: int) -> PaginatedDatasetsResult:
        """Return paginated dataset catalog."""
        all_items = [entry.to_summary() for entry in self._catalog.scan()]
        bounded_limit = min(limit, self._settings.max_limit)

        page = all_items[offset : offset + bounded_limit]
        total_count = len(all_items)
        has_more = offset + len(page) < total_count
        next_offset = offset + len(page) if has_more else None

        return PaginatedDatasetsResult(
            total_count=total_count,
            count=len(page),
            limit=bounded_limit,
            offset=offset,
            has_more=has_more,
            next_offset=next_offset,
            datasets=page,
        )

    def describe_dataset(self, dataset_name: str, sample_rows: int) -> DatasetDescription:
        """Return schema, row count, and sample rows for a dataset."""
        entry = self._catalog.get(dataset_name)
        bounded_sample_rows = min(sample_rows, self._settings.max_sample_rows)

        def _task() -> DatasetDescription:
            with self._connect() as con:
                self._register_source(con, entry)

                schema_rows = con.execute("DESCRIBE SELECT * FROM source").fetchall()
                columns = [
                    DatasetColumn(
                        name=str(row[0]),
                        data_type=str(row[1]),
                        nullable=str(row[2]) if row[2] is not None else None,
                    )
                    for row in schema_rows
                ]

                row_count_row = con.execute("SELECT COUNT(*) FROM source").fetchone()
                if row_count_row is None:
                    raise ValueError("Failed to read dataset row count")
                row_count = int(row_count_row[0])
                _, rows = self._fetch_rows(
                    con,
                    "SELECT * FROM source LIMIT ?",
                    [bounded_sample_rows],
                )

                return DatasetDescription(
                    dataset=entry.to_summary(),
                    row_count=row_count,
                    columns=columns,
                    sample_rows=rows,
                )

        return self._run_with_timeout(_task)

    def query_dataset(self, dataset_name: str, sql: str, limit: int, offset: int) -> QueryResult:
        """Execute a guarded read-only SQL query against a dataset."""
        entry = self._catalog.get(dataset_name)
        safe_sql = self._guard.validate(sql)
        bounded_limit = min(limit, self._settings.max_limit)

        def _task() -> QueryResult:
            with self._connect() as con:
                self._register_source(con, entry)

                total_count_row = con.execute(
                    f"SELECT COUNT(*) FROM ({safe_sql}) AS guarded_query"
                ).fetchone()
                if total_count_row is None:
                    raise ValueError("Failed to read query row count")
                total_count = int(total_count_row[0])
                columns, rows = self._fetch_rows(
                    con,
                    f"SELECT * FROM ({safe_sql}) AS guarded_query LIMIT ? OFFSET ?",
                    [bounded_limit, offset],
                )

                has_more = offset + len(rows) < total_count
                next_offset = offset + len(rows) if has_more else None

                return QueryResult(
                    dataset=dataset_name,
                    sql=safe_sql,
                    total_count=total_count,
                    count=len(rows),
                    limit=bounded_limit,
                    offset=offset,
                    has_more=has_more,
                    next_offset=next_offset,
                    columns=columns,
                    rows=rows,
                )

        return self._run_with_timeout(_task)

    def health(self) -> HealthStatus:
        """Return service health metadata."""
        return HealthStatus(
            status="ok",
            server=self._settings.server_name,
            dataset_dir=self._settings.resolved_dataset_dir.as_posix(),
            dataset_count=len(self._catalog.scan()),
            checked_at=datetime.now(tz=UTC),
        )

    def _connect(self) -> duckdb.DuckDBPyConnection:
        """Create a constrained DuckDB connection."""
        return duckdb.connect(":memory:", config=self._settings.duckdb_config)

    def _register_source(self, con: duckdb.DuckDBPyConnection, entry: DatasetEntry) -> None:
        path_literal = self._sql_literal(entry.path.as_posix())
        if entry.file_format == "csv":
            con.execute(
                f"CREATE OR REPLACE TEMP VIEW source AS "
                f"SELECT * FROM read_csv_auto({path_literal}, sample_size=-1)"
            )
            return
        if entry.file_format == "parquet":
            con.execute(
                f"CREATE OR REPLACE TEMP VIEW source AS SELECT * FROM read_parquet({path_literal})"
            )
            return
        if entry.file_format in {"json", "jsonl"}:
            con.execute(
                f"CREATE OR REPLACE TEMP VIEW source AS SELECT * FROM read_json_auto({path_literal})"
            )
            return

        raise ValueError(f"Unsupported dataset format for {entry.name}: {entry.file_format}")

    @staticmethod
    def _sql_literal(value: str) -> str:
        escaped = value.replace("'", "''")
        return f"'{escaped}'"

    @staticmethod
    def _fetch_rows(
        con: duckdb.DuckDBPyConnection,
        sql: str,
        params: list[object],
    ) -> tuple[list[str], list[dict[str, object]]]:
        cursor = con.execute(sql, params)
        description = cursor.description
        if description is None:
            return [], []

        columns = [str(column[0]) for column in description]
        raw_rows = cursor.fetchall()

        rows: list[dict[str, object]] = []
        for raw_row in raw_rows:
            row = {col: raw_row[idx] for idx, col in enumerate(columns)}
            rows.append(row)

        return columns, rows

    def _run_with_timeout(self, fn: Callable[[], T]) -> T:
        """Run blocking DB work in a bounded-time worker thread."""
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(fn)
            try:
                return future.result(timeout=self._settings.query_timeout_seconds)
            except FuturesTimeoutError as exc:
                future.cancel()
                logger.warning("Query timed out after {} seconds", self._settings.query_timeout_seconds)
                raise TimeoutError(
                    f"Operation timed out after {self._settings.query_timeout_seconds} seconds"
                ) from exc
