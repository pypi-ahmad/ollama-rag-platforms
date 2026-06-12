"""Local smoke test for service logic without MCP client wiring."""

from __future__ import annotations

import json

from duckdb_analytics_mcp.config import get_settings
from duckdb_analytics_mcp.service import AnalyticsService


def main() -> None:
    settings = get_settings()
    service = AnalyticsService(settings)

    health = service.health()
    catalog = service.list_datasets(limit=5, offset=0)

    payload = {
        "health": json.loads(health.model_dump_json()),
        "catalog": json.loads(catalog.model_dump_json()),
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
