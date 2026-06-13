from __future__ import annotations

import os
import socket
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import anyio
import httpx
import pytest
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.streamable_http import streamable_http_client

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOKEN = "e2e-static-token"


def _clear_auth_env(base_env: dict[str, str]) -> dict[str, str]:
    env = dict(base_env)
    for key in (
        "STATIC_BEARER_TOKEN",
        "AUTH_ISSUER_URL",
        "AUTH_RESOURCE_SERVER_URL",
        "AUTH_REQUIRED_SCOPE",
    ):
        env.pop(key, None)
    return env


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_http_ready(url: str, process: subprocess.Popen[str], timeout_seconds: float = 20.0) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if process.poll() is not None:
            stderr = process.stderr.read() if process.stderr else ""
            raise AssertionError(f"HTTP MCP server exited before ready: {stderr}")
        try:
            urllib.request.urlopen(url, timeout=0.5)
            return
        except urllib.error.HTTPError:
            return
        except Exception:
            time.sleep(0.2)
    stderr = process.stderr.read() if process.stderr else ""
    raise AssertionError(f"HTTP MCP server did not become ready: {stderr}")


@contextmanager
def _run_http_server(*, auth_enabled: bool) -> Generator[tuple[str, dict[str, str]], None, None]:
    port = _pick_free_port()
    url = f"http://127.0.0.1:{port}/mcp"
    env = _clear_auth_env(dict(os.environ))
    env.update(
        {
            "HOST": "127.0.0.1",
            "PORT": str(port),
            "TRANSPORT": "streamable-http",
        }
    )
    if auth_enabled:
        env.update(
            {
                "STATIC_BEARER_TOKEN": TOKEN,
                "AUTH_ISSUER_URL": "https://auth.example.com",
                "AUTH_RESOURCE_SERVER_URL": "https://mcp.example.com",
                "AUTH_REQUIRED_SCOPE": "analytics.read",
            }
        )

    process = subprocess.Popen(
        ["uv", "run", "duckdb-analytics-mcp", "run", "--transport", "streamable-http"],
        cwd=PROJECT_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        _wait_for_http_ready(url, process)
        yield url, env
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()


def _unwrap_result_payload(result: Any) -> dict[str, Any]:
    structured = getattr(result, "structuredContent", None)
    if not isinstance(structured, dict):
        raise AssertionError("Expected structuredContent dictionary")
    payload = structured.get("result")
    if not isinstance(payload, dict):
        raise AssertionError("Expected structuredContent['result'] dictionary")
    return payload


async def _call_http_tool(
    url: str,
    tool_name: str,
    arguments: dict[str, Any],
    headers: dict[str, str] | None = None,
) -> Any:
    if headers is None:
        async with (
            streamable_http_client(url) as (read, write, _),
            ClientSession(read, write) as session,
        ):
            await session.initialize()
            return await session.call_tool(tool_name, arguments)

    async with (
        httpx.AsyncClient(headers=headers) as http_client,
        streamable_http_client(url, http_client=http_client) as (read, write, _),
        ClientSession(read, write) as session,
    ):
        await session.initialize()
        return await session.call_tool(tool_name, arguments)


async def _call_stdio_tool(
    tool_name: str,
    arguments: dict[str, Any],
    env: dict[str, str],
) -> Any:
    server = StdioServerParameters(
        command="uv",
        args=["run", "duckdb-analytics-mcp", "run", "--transport", "stdio"],
        cwd=PROJECT_ROOT,
        env=env,
    )
    async with (
        stdio_client(server) as (read, write),
        ClientSession(read, write) as session,
    ):
        await session.initialize()
        tools = await session.list_tools()
        tool_names = {tool.name for tool in tools.tools}
        assert "duckdb_analytics_query_dataset" in tool_names
        return await session.call_tool(tool_name, arguments)


def _select_first_dataset(catalog_payload: dict[str, Any]) -> str:
    datasets = catalog_payload.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise AssertionError("No datasets returned by catalog during E2E run")
    first = datasets[0]
    if not isinstance(first, dict) or not isinstance(first.get("name"), str):
        raise AssertionError("Unexpected dataset payload shape")
    return first["name"]


@pytest.mark.e2e
def test_e2e_stdio_transport_no_auth() -> None:
    env = _clear_auth_env(dict(os.environ))

    health_result = anyio.run(
        _call_stdio_tool,
        "duckdb_analytics_health",
        {"response_format": "json"},
        env,
    )
    health_payload = _unwrap_result_payload(health_result)
    assert health_payload["status"] == "ok"

    catalog_result = anyio.run(
        _call_stdio_tool,
        "duckdb_analytics_list_datasets",
        {"limit": 10, "offset": 0, "response_format": "json"},
        env,
    )
    catalog_payload = _unwrap_result_payload(catalog_result)
    dataset_name = _select_first_dataset(catalog_payload)

    describe_result = anyio.run(
        _call_stdio_tool,
        "duckdb_analytics_describe_dataset",
        {"dataset": dataset_name, "sample_rows": 2, "response_format": "json"},
        env,
    )
    describe_payload = _unwrap_result_payload(describe_result)
    assert describe_payload["dataset"]["name"] == dataset_name

    query_result = anyio.run(
        _call_stdio_tool,
        "duckdb_analytics_query_dataset",
        {
            "dataset": dataset_name,
            "sql": "SELECT * FROM source LIMIT 2",
            "limit": 2,
            "offset": 0,
            "response_format": "json",
        },
        env,
    )
    query_payload = _unwrap_result_payload(query_result)
    assert query_payload["count"] <= 2


@pytest.mark.e2e
def test_e2e_http_transport_no_auth_and_security_guard() -> None:
    with _run_http_server(auth_enabled=False) as (url, _):
        health_result = anyio.run(
            _call_http_tool,
            url,
            "duckdb_analytics_health",
            {"response_format": "json"},
            None,
        )
        health_payload = _unwrap_result_payload(health_result)
        assert health_payload["status"] == "ok"

        catalog_result = anyio.run(
            _call_http_tool,
            url,
            "duckdb_analytics_list_datasets",
            {"limit": 10, "offset": 0, "response_format": "json"},
            None,
        )
        catalog_payload = _unwrap_result_payload(catalog_result)
        dataset_name = _select_first_dataset(catalog_payload)

        blocked_result = anyio.run(
            _call_http_tool,
            url,
            "duckdb_analytics_query_dataset",
            {
                "dataset": dataset_name,
                "sql": "SELECT * FROM read_csv_auto('/etc/passwd')",
                "limit": 2,
                "offset": 0,
                "response_format": "json",
            },
            None,
        )
        blocked_payload = _unwrap_result_payload(blocked_result)
        assert "error" in blocked_payload
        assert "not allowed" in blocked_payload["error"].lower()


@pytest.mark.e2e
def test_e2e_http_transport_with_auth() -> None:
    with _run_http_server(auth_enabled=True) as (url, _):
        with pytest.raises(ExceptionGroup):
            anyio.run(
                _call_http_tool,
                url,
                "duckdb_analytics_health",
                {"response_format": "json"},
                None,
            )

        authorized_result = anyio.run(
            _call_http_tool,
            url,
            "duckdb_analytics_health",
            {"response_format": "json"},
            {"Authorization": f"Bearer {TOKEN}"},
        )
        authorized_payload = _unwrap_result_payload(authorized_result)
        assert authorized_payload["status"] == "ok"
