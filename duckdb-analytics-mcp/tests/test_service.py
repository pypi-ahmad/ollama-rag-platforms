from __future__ import annotations

from pathlib import Path

from duckdb_analytics_mcp.config import Settings
from duckdb_analytics_mcp.service import AnalyticsService


def _build_settings(dataset_dir: Path) -> Settings:
    return Settings(
        dataset_dir=dataset_dir,
        query_timeout_seconds=5.0,
        max_limit=50,
        default_limit=10,
        max_sample_rows=20,
    )


def test_service_query_dataset_with_pagination(tmp_path: Path) -> None:
    data_file = tmp_path / "orders.csv"
    data_file.write_text(
        "id,region,units\n1,North,5\n2,South,9\n3,North,7\n4,West,4\n",
        encoding="utf-8",
    )

    service = AnalyticsService(_build_settings(tmp_path))
    result = service.query_dataset(
        dataset_name="orders.csv",
        sql="SELECT region, SUM(units) AS total_units FROM source GROUP BY region ORDER BY total_units DESC",
        limit=2,
        offset=0,
    )

    assert result.count == 2
    assert result.total_count == 3
    assert result.has_more is True
    assert result.next_offset == 2
    assert result.rows[0]["region"] == "North"
    assert result.rows[0]["total_units"] == 12


def test_service_describe_dataset_returns_schema(tmp_path: Path) -> None:
    data_file = tmp_path / "orders.csv"
    data_file.write_text(
        "id,region,units\n1,North,5\n2,South,9\n",
        encoding="utf-8",
    )

    service = AnalyticsService(_build_settings(tmp_path))
    description = service.describe_dataset("orders.csv", sample_rows=1)

    assert description.row_count == 2
    assert [column.name for column in description.columns] == ["id", "region", "units"]
    assert len(description.sample_rows) == 1
