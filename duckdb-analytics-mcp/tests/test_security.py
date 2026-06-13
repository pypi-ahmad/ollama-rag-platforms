from __future__ import annotations

import pytest

from duckdb_analytics_mcp.security import SQLGuard, SQLValidationError


def test_sql_guard_accepts_select() -> None:
    guard = SQLGuard(max_query_chars=200)
    sql = guard.validate("SELECT region, SUM(units) FROM source GROUP BY region;")

    assert sql == "SELECT region, SUM(units) FROM source GROUP BY region"


def test_sql_guard_allows_comment_tokens_inside_literals() -> None:
    guard = SQLGuard(max_query_chars=200)
    sql = guard.validate("SELECT '--' AS token FROM source LIMIT 1")

    assert sql == "SELECT '--' AS token FROM source LIMIT 1"


def test_sql_guard_allows_source_ctes() -> None:
    guard = SQLGuard(max_query_chars=400)
    sql = guard.validate(
        "WITH x AS (SELECT region, units FROM source), y AS (SELECT * FROM x) SELECT * FROM y"
    )

    assert sql.startswith("WITH x AS")


@pytest.mark.parametrize(
    "bad_sql",
    [
        "UPDATE source SET units = 1",
        "SELECT * FROM source; DELETE FROM source",
        "SELECT * FROM source --comment",
        "PRAGMA show_tables",
        "INSERT INTO x VALUES (1)",
        "SELECT * FROM read_csv_auto('/etc/passwd')",
        "SELECT * FROM read_parquet('/tmp/data.parquet')",
        "SELECT * FROM other",
        "SELECT * FROM source; DELETE FROM source",
        "SELECT read_csv_auto('/etc/passwd')",
    ],
)
def test_sql_guard_rejects_non_read_only(bad_sql: str) -> None:
    guard = SQLGuard(max_query_chars=200)

    with pytest.raises(SQLValidationError):
        guard.validate(bad_sql)
