"""MCP server composition and tool/resource registration."""

from __future__ import annotations

import json
from typing import Any, cast

from loguru import logger
from mcp.server.auth.provider import AccessToken, TokenVerifier
from mcp.server.auth.settings import AuthSettings
from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations
from pydantic import BaseModel

from duckdb_analytics_mcp.config import Settings, Transport, get_settings
from duckdb_analytics_mcp.formatter import (
    render_catalog_markdown,
    render_description_markdown,
    render_health_markdown,
    render_query_markdown,
)
from duckdb_analytics_mcp.models import ResponseFormat
from duckdb_analytics_mcp.service import AnalyticsService

READ_ONLY_TOOL = ToolAnnotations(
    title="Read-only analytics tool",
    readOnlyHint=True,
    destructiveHint=False,
    idempotentHint=True,
    openWorldHint=False,
)


class StaticTokenVerifier(TokenVerifier):
    """Simple static bearer token verifier for minimal auth setups."""

    def __init__(self, token: str, scope: str) -> None:
        self._token = token
        self._scope = scope

    async def verify_token(self, token: str) -> AccessToken | None:
        if token != self._token:
            return None

        return AccessToken(
            token=token,
            client_id="duckdb-analytics-client",
            scopes=[self._scope],
        )


def _model_to_json(model: BaseModel) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(model.model_dump_json()))


def _format_response(model: BaseModel, response_format: ResponseFormat, markdown: str) -> str | dict[str, Any]:
    if response_format == ResponseFormat.JSON:
        return _model_to_json(model)
    return markdown


def _format_error(message: str, response_format: ResponseFormat) -> str | dict[str, Any]:
    if response_format == ResponseFormat.JSON:
        return {"error": message}
    return f"Error: {message}"


def _resolve_response_format(raw_format: str) -> ResponseFormat:
    try:
        return ResponseFormat(raw_format)
    except ValueError as exc:
        allowed = ", ".join(option.value for option in ResponseFormat)
        raise ValueError(f"response_format must be one of: {allowed}") from exc


def _auth_kwargs(settings: Settings) -> dict[str, Any]:
    if settings.static_bearer_token is None:
        return {}

    if settings.auth_issuer_url is None or settings.auth_resource_server_url is None:
        raise ValueError(
            "static_bearer_token requires auth_issuer_url and auth_resource_server_url to be set"
        )

    logger.info("Authentication enabled with required scope '{}'", settings.auth_required_scope)
    return {
        "token_verifier": StaticTokenVerifier(
            token=settings.static_bearer_token,
            scope=settings.auth_required_scope,
        ),
        "auth": AuthSettings(
            issuer_url=settings.auth_issuer_url,
            resource_server_url=settings.auth_resource_server_url,
            required_scopes=[settings.auth_required_scope],
        ),
    }


def build_server(settings: Settings | None = None) -> FastMCP:
    """Build and configure the MCP server instance."""
    runtime = settings or get_settings()
    service = AnalyticsService(runtime)

    mcp = FastMCP(
        runtime.server_name,
        instructions=(
            "Read-only DuckDB analytics tools over local datasets. "
            "Use `duckdb_analytics_list_datasets` first, then describe/query datasets."
        ),
        host=runtime.host,
        port=runtime.port,
        streamable_http_path="/mcp",
        mount_path="/",
        json_response=True,
        stateless_http=True,
        log_level=runtime.log_level,
        **_auth_kwargs(runtime),
    )

    @mcp.resource(
        "dataset://catalog",
        name="dataset_catalog",
        description="Catalog of available datasets under the configured dataset directory.",
    )
    def dataset_catalog_resource() -> str:
        catalog = service.list_datasets(limit=runtime.max_limit, offset=0)
        return render_catalog_markdown(catalog)

    @mcp.resource(
        "dataset://schema/{dataset_name}",
        name="dataset_schema",
        description="Schema and sample rows for a specific dataset.",
    )
    def dataset_schema_resource(dataset_name: str) -> str:
        description = service.describe_dataset(dataset_name=dataset_name, sample_rows=5)
        return render_description_markdown(description)

    @mcp.tool(name="duckdb_analytics_health", annotations=READ_ONLY_TOOL)
    def duckdb_analytics_health(response_format: str = "markdown") -> str | dict[str, Any]:
        """Get MCP server and dataset-catalog health.

        Args:
            response_format: `markdown` or `json`.

        Returns:
            Health payload in markdown or JSON.
        """
        fmt = _resolve_response_format(response_format)
        health = service.health()
        return _format_response(health, fmt, render_health_markdown(health))

    @mcp.tool(name="duckdb_analytics_list_datasets", annotations=READ_ONLY_TOOL)
    def duckdb_analytics_list_datasets(
        limit: int = 25,
        offset: int = 0,
        response_format: str = "markdown",
    ) -> str | dict[str, Any]:
        """List datasets with pagination.

        Args:
            limit: Max number of results to return.
            offset: Number of datasets to skip.
            response_format: `markdown` or `json`.

        Returns:
            Dataset catalog page.
        """
        fmt = _resolve_response_format(response_format)
        try:
            result = service.list_datasets(limit=limit, offset=offset)
            return _format_response(result, fmt, render_catalog_markdown(result))
        except (ValueError, TimeoutError) as exc:
            logger.warning("list_datasets failed: {}", exc)
            return _format_error(str(exc), fmt)

    @mcp.tool(name="duckdb_analytics_describe_dataset", annotations=READ_ONLY_TOOL)
    def duckdb_analytics_describe_dataset(
        dataset: str,
        sample_rows: int = 10,
        response_format: str = "markdown",
    ) -> str | dict[str, Any]:
        """Describe a dataset schema and show sample rows.

        Args:
            dataset: Dataset name from catalog (e.g., `sales/orders.csv`).
            sample_rows: Number of rows to sample.
            response_format: `markdown` or `json`.

        Returns:
            Dataset profile including schema and sample rows.
        """
        fmt = _resolve_response_format(response_format)
        try:
            result = service.describe_dataset(dataset_name=dataset, sample_rows=sample_rows)
            return _format_response(result, fmt, render_description_markdown(result))
        except (ValueError, TimeoutError) as exc:
            logger.warning("describe_dataset failed: {}", exc)
            return _format_error(str(exc), fmt)

    @mcp.tool(name="duckdb_analytics_query_dataset", annotations=READ_ONLY_TOOL)
    def duckdb_analytics_query_dataset(
        dataset: str,
        sql: str,
        limit: int = 25,
        offset: int = 0,
        response_format: str = "markdown",
    ) -> str | dict[str, Any]:
        """Run a read-only SQL query against a dataset using `source` as table alias.

        Args:
            dataset: Dataset name from catalog.
            sql: Read-only SQL query that starts with `SELECT` or `WITH`.
            limit: Max rows to return.
            offset: Rows to skip.
            response_format: `markdown` or `json`.

        Returns:
            Query results with pagination metadata.
        """
        fmt = _resolve_response_format(response_format)
        try:
            result = service.query_dataset(
                dataset_name=dataset,
                sql=sql,
                limit=limit,
                offset=offset,
            )
            return _format_response(result, fmt, render_query_markdown(result))
        except (ValueError, TimeoutError) as exc:
            logger.warning("query_dataset failed: {}", exc)
            return _format_error(str(exc), fmt)

    return mcp


def run_server(transport: Transport | None = None, settings: Settings | None = None) -> None:
    """Run MCP server with configured transport."""
    runtime = settings or get_settings()
    runtime_transport = transport or runtime.transport

    server = build_server(runtime)
    if runtime_transport == "stdio":
        logger.info("Starting {} via stdio transport", runtime.server_name)
        server.run(transport="stdio")
        return

    logger.info(
        "Starting {} via streamable-http at http://{}:{}/mcp",
        runtime.server_name,
        runtime.host,
        runtime.port,
    )
    server.run(transport="streamable-http")
