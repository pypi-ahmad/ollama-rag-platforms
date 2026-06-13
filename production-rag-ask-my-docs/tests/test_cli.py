"""CLI behavior tests for error handling and command decoupling."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from ask_my_docs.cli import app
from ask_my_docs.settings import get_settings

_runner = CliRunner()


@pytest.fixture(autouse=True)
def _clear_settings_cache() -> None:
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_metrics_summary_does_not_require_retrieval_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ASK_MY_DOCS_METRICS_DB_PATH", str(tmp_path / "metrics.duckdb"))
    monkeypatch.setenv("ASK_MY_DOCS_INDEX_DIR", str(tmp_path / "missing-index"))

    result = _runner.invoke(app, ["metrics-summary"])

    assert result.exit_code == 0
    assert "request_count=0" in result.output


def test_metrics_summary_rejects_negative_limit() -> None:
    result = _runner.invoke(app, ["metrics-summary", "--limit", "-1"])
    assert result.exit_code != 0


def test_ask_reports_clean_error_without_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ASK_MY_DOCS_INDEX_DIR", str(tmp_path / "missing-index"))
    monkeypatch.setenv("ASK_MY_DOCS_METRICS_DB_PATH", str(tmp_path / "metrics.duckdb"))

    result = _runner.invoke(app, ["ask", "What is policy?"])

    assert result.exit_code == 1
    assert "error: Index artifacts not found" in result.output
    assert "Traceback" not in result.output
