"""Format structured payloads as markdown."""

from __future__ import annotations

import json

from duckdb_analytics_mcp.models import (
    DatasetDescription,
    HealthStatus,
    PaginatedDatasetsResult,
    QueryResult,
)


def _json_block(payload: object) -> str:
    return f"```json\n{json.dumps(payload, indent=2, default=str)}\n```"


def render_health_markdown(health: HealthStatus) -> str:
    """Render health status as markdown."""
    lines = [
        "# Server Health",
        "",
        f"- **Status**: {health.status}",
        f"- **Server**: `{health.server}`",
        f"- **Dataset Dir**: `{health.dataset_dir}`",
        f"- **Datasets Found**: {health.dataset_count}",
        f"- **Checked At**: {health.checked_at.isoformat()}",
    ]
    return "\n".join(lines)


def render_catalog_markdown(result: PaginatedDatasetsResult) -> str:
    """Render dataset catalog page as markdown."""
    lines = [
        "# Dataset Catalog",
        "",
        f"Showing {result.count} of {result.total_count} datasets (offset={result.offset}, limit={result.limit}).",
        "",
    ]

    if not result.datasets:
        lines.append("No datasets found.")
        return "\n".join(lines)

    for item in result.datasets:
        lines.extend(
            [
                f"## {item.name}",
                f"- **Format**: `{item.file_format}`",
                f"- **Size (bytes)**: {item.size_bytes}",
                f"- **Modified**: {item.modified_at.isoformat()}",
                f"- **Path**: `{item.path}`",
                "",
            ]
        )

    lines.append(
        f"Pagination: has_more={result.has_more}, next_offset={result.next_offset if result.next_offset is not None else 'null'}"
    )
    return "\n".join(lines)


def render_description_markdown(result: DatasetDescription) -> str:
    """Render dataset schema/profile result as markdown."""
    lines = [
        f"# Dataset Description: {result.dataset.name}",
        "",
        f"- **Format**: `{result.dataset.file_format}`",
        f"- **Rows**: {result.row_count}",
        f"- **Columns**: {len(result.columns)}",
        "",
        "## Columns",
        "",
    ]

    for column in result.columns:
        nullable = f", nullable={column.nullable}" if column.nullable else ""
        lines.append(f"- `{column.name}`: `{column.data_type}`{nullable}")

    lines.extend(["", "## Sample Rows", _json_block(result.sample_rows)])
    return "\n".join(lines)


def render_query_markdown(result: QueryResult) -> str:
    """Render query result as markdown."""
    lines = [
        f"# Query Result: {result.dataset}",
        "",
        f"- **Rows Returned**: {result.count}",
        f"- **Total Rows (before pagination)**: {result.total_count}",
        f"- **Offset**: {result.offset}",
        f"- **Limit**: {result.limit}",
        f"- **Has More**: {result.has_more}",
        f"- **Next Offset**: {result.next_offset if result.next_offset is not None else 'null'}",
        "",
        "## SQL",
        "",
        "```sql",
        result.sql,
        "```",
        "",
        "## Rows",
        _json_block(result.rows),
    ]
    return "\n".join(lines)
