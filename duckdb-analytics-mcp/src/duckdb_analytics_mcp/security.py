"""Security and validation guardrails for SQL execution."""

from __future__ import annotations

from sqlglot import expressions as exp
from sqlglot import parse, parse_one
from sqlglot.errors import ParseError


class SQLValidationError(ValueError):
    """Raised when SQL violates read-only safety rules."""


class SQLGuard:
    """Enforce read-only, single-statement SQL constraints."""

    def __init__(self, max_query_chars: int) -> None:
        self._max_query_chars = max_query_chars

    def validate(self, sql: str) -> str:
        """Validate and normalize SQL.

        Args:
            sql: Raw SQL text.

        Returns:
            Normalized SQL string without trailing semicolon.

        Raises:
            SQLValidationError: If SQL is not read-only or is malformed.
        """
        candidate = sql.strip()
        if not candidate:
            raise SQLValidationError("SQL cannot be empty")

        if len(candidate) > self._max_query_chars:
            raise SQLValidationError(
                f"SQL exceeds max length ({self._max_query_chars} characters)"
            )

        if self._contains_sql_comment(candidate):
            raise SQLValidationError("SQL comments are not allowed")

        try:
            parsed_statements = parse(candidate, read="duckdb")
        except ParseError as exc:
            raise SQLValidationError("SQL is not valid DuckDB syntax") from exc

        if len(parsed_statements) != 1:
            raise SQLValidationError("Only a single SQL statement is allowed")

        candidate = candidate.rstrip(";").strip()

        try:
            parsed = parse_one(candidate, read="duckdb")
        except ParseError as exc:
            raise SQLValidationError("SQL is not valid DuckDB syntax") from exc

        if not isinstance(parsed, exp.Query):
            raise SQLValidationError("Only SELECT/WITH read-only queries are allowed")

        self._validate_table_references(parsed)
        self._validate_function_calls(parsed)

        return candidate

    @staticmethod
    def _contains_sql_comment(sql: str) -> bool:
        in_single_quote = False
        in_double_quote = False
        idx = 0

        while idx < len(sql):
            ch = sql[idx]
            next_ch = sql[idx + 1] if idx + 1 < len(sql) else ""

            if in_single_quote:
                if ch == "'" and next_ch == "'":
                    idx += 2
                    continue
                if ch == "'":
                    in_single_quote = False
                idx += 1
                continue

            if in_double_quote:
                if ch == '"' and next_ch == '"':
                    idx += 2
                    continue
                if ch == '"':
                    in_double_quote = False
                idx += 1
                continue

            if ch == "'":
                in_single_quote = True
                idx += 1
                continue
            if ch == '"':
                in_double_quote = True
                idx += 1
                continue

            if (ch == "-" and next_ch == "-") or (ch == "/" and next_ch == "*") or (
                ch == "*" and next_ch == "/"
            ):
                return True

            idx += 1

        return False

    @staticmethod
    def _validate_table_references(parsed: exp.Query) -> None:
        cte_names = {cte.alias_or_name.lower() for cte in parsed.find_all(exp.CTE)}
        allowed_table_names = cte_names | {"source"}

        for table in parsed.find_all(exp.Table):
            # DuckDB table functions and external relations appear as non-Identifier table expressions.
            if not isinstance(table.this, exp.Identifier):
                raise SQLValidationError(
                    "External table functions and external datasets are not allowed. Use only `source`."
                )

            table_name = table.name.lower()
            if table_name not in allowed_table_names:
                raise SQLValidationError(
                    f"Only the `source` alias and in-query CTEs are allowed (found `{table.name}`)."
                )

            if table.db or table.catalog:
                raise SQLValidationError("Schema/catalog-qualified tables are not allowed in read-only mode")

    @staticmethod
    def _validate_function_calls(parsed: exp.Query) -> None:
        for func in parsed.find_all(exp.Anonymous):
            function_name = func.name.lower()
            if function_name.startswith("read_") or function_name.endswith("_scan"):
                raise SQLValidationError(
                    f"Function `{function_name}` is not allowed. Query only from `source`."
                )
