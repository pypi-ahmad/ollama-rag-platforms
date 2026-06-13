from __future__ import annotations

from pathlib import Path
from typing import Any

import anyio

from duckdb_analytics_mcp.config import Settings
from duckdb_analytics_mcp.server import build_server


def _unwrap_result_payload(result: Any) -> dict[str, Any]:
    if isinstance(result, tuple) and len(result) == 2:
        structured = result[1]
    else:
        structured = getattr(result, "structuredContent", None)
    if not isinstance(structured, dict):
        raise AssertionError("Expected structuredContent to be a dict")
    payload = structured.get("result")
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, str):
        return {"error": payload}
    raise AssertionError("Expected structuredContent['result'] to be a dict or string")


def _build_runtime(tmp_path: Path) -> Settings:
    return Settings(
        dataset_dir=tmp_path,
        query_timeout_seconds=5.0,
        max_limit=50,
        default_limit=10,
        max_sample_rows=20,
        catalog_cache_ttl_seconds=0.0,
    )


def test_health_invalid_response_format_returns_tool_error_payload(tmp_path: Path) -> None:
    async def _run() -> None:
        server = build_server(_build_runtime(tmp_path))
        result = await server.call_tool(
            "duckdb_analytics_health",
            {"response_format": "xml"},
        )
        payload = _unwrap_result_payload(result)
        assert "error" in payload
        assert "response_format" in payload["error"]

    anyio.run(_run)


def test_query_validation_error_is_returned_as_payload(tmp_path: Path) -> None:
    (tmp_path / "orders.csv").write_text(
        "id,region,units\n1,North,5\n2,South,9\n",
        encoding="utf-8",
    )

    async def _run() -> None:
        server = build_server(_build_runtime(tmp_path))
        result = await server.call_tool(
            "duckdb_analytics_query_dataset",
            {
                "dataset": "orders.csv",
                "sql": "SELECT * FROM source",
                "limit": -1,
                "offset": 0,
                "response_format": "json",
            },
        )
        payload = _unwrap_result_payload(result)
        assert "error" in payload
        assert "limit" in payload["error"]

    anyio.run(_run)
