"""Security and validation guardrails for SQL execution."""

from __future__ import annotations

import re


class SQLValidationError(ValueError):
    """Raised when SQL violates read-only safety rules."""


class SQLGuard:
    """Enforce read-only, single-statement SQL constraints."""

    _BANNED_KEYWORDS = re.compile(
        r"\b(insert|update|delete|merge|drop|alter|create|attach|detach|copy|call|"
        r"pragma|set|reset|install|load|export|import|vacuum|transaction|commit|rollback|truncate)\b",
        flags=re.IGNORECASE,
    )
    _COMMENT_PATTERN = re.compile(r"(--|/\*|\*/)")

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

        if self._COMMENT_PATTERN.search(candidate):
            raise SQLValidationError("SQL comments are not allowed")

        if ";" in candidate.rstrip(";"):
            raise SQLValidationError("Only a single SQL statement is allowed")

        candidate = candidate.rstrip(";").strip()
        lowered = candidate.lower()
        if not (lowered.startswith("select") or lowered.startswith("with")):
            raise SQLValidationError("Only SELECT/WITH read-only queries are allowed")

        banned = self._BANNED_KEYWORDS.search(candidate)
        if banned:
            raise SQLValidationError(
                f"Keyword '{banned.group(0)}' is not allowed in read-only mode"
            )

        return candidate
