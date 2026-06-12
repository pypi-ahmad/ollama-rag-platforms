"""CLI entrypoint for duckdb_analytics_mcp."""

from __future__ import annotations

import json

import typer

from duckdb_analytics_mcp.config import Transport, get_settings
from duckdb_analytics_mcp.logging_utils import configure_logging
from duckdb_analytics_mcp.server import run_server
from duckdb_analytics_mcp.service import AnalyticsService

app = typer.Typer(help="DuckDB Analytics MCP Server")


@app.callback()
def _callback() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)


@app.command("run")
def run_command(
    transport: Transport | None = None,
) -> None:
    """Run MCP server."""
    run_server(transport=transport)


@app.command("doctor")
def doctor_command() -> None:
    """Run local diagnostics for deployment readiness."""
    settings = get_settings()
    service = AnalyticsService(settings)

    health = service.health()
    catalog = service.list_datasets(limit=min(settings.max_limit, 1000), offset=0)

    payload = {
        "health": json.loads(health.model_dump_json()),
        "dataset_preview_count": catalog.count,
        "dataset_total": catalog.total_count,
    }
    typer.echo(json.dumps(payload, indent=2))


if __name__ == "__main__":
    app()
