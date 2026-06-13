from __future__ import annotations

import os
from pathlib import Path

from duckdb_analytics_mcp.catalog import DatasetCatalog


def test_catalog_scans_supported_formats(tmp_path: Path) -> None:
    (tmp_path / "sales.csv").write_text("id,amount\n1,20\n", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("ignore", encoding="utf-8")
    (tmp_path / "events.jsonl").write_text('{"a": 1}\n', encoding="utf-8")

    catalog = DatasetCatalog(tmp_path)
    entries = catalog.scan()

    names = {entry.name for entry in entries}
    assert names == {"events.jsonl", "sales.csv"}


def test_catalog_get_missing_raises(tmp_path: Path) -> None:
    catalog = DatasetCatalog(tmp_path)

    try:
        catalog.get("missing.csv")
        raise AssertionError("expected missing dataset to raise")
    except ValueError as exc:
        assert "not found" in str(exc)


def test_catalog_skips_symlink_escaping_dataset_root(tmp_path: Path) -> None:
    (tmp_path / "ok.csv").write_text("id\n1\n", encoding="utf-8")

    escaped = tmp_path / "escaped.csv"
    os.symlink("/etc/passwd", escaped)

    catalog = DatasetCatalog(tmp_path)
    entries = catalog.scan()

    names = {entry.name for entry in entries}
    assert names == {"ok.csv"}
