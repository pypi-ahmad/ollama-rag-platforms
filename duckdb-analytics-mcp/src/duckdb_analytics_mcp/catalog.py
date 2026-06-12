"""Dataset catalog discovery and metadata utilities."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from duckdb_analytics_mcp.models import DatasetSummary

SUPPORTED_SUFFIXES: dict[str, str] = {
    ".csv": "csv",
    ".parquet": "parquet",
    ".json": "json",
    ".jsonl": "jsonl",
}


@dataclass(frozen=True)
class DatasetEntry:
    """Internal dataset representation."""

    name: str
    path: Path
    file_format: str
    size_bytes: int
    modified_at: datetime

    def to_summary(self) -> DatasetSummary:
        """Convert to public response model."""
        return DatasetSummary(
            name=self.name,
            path=self.path.as_posix(),
            file_format=self.file_format,
            size_bytes=self.size_bytes,
            modified_at=self.modified_at,
        )


class DatasetCatalog:
    """Discover and resolve datasets under a fixed directory."""

    def __init__(self, dataset_dir: Path) -> None:
        self._dataset_dir = dataset_dir.resolve()

    def scan(self) -> list[DatasetEntry]:
        """Scan dataset directory recursively and return supported files."""
        entries: list[DatasetEntry] = []
        for path in sorted(self._dataset_dir.rglob("*")):
            if not path.is_file():
                continue
            suffix = path.suffix.lower()
            if suffix not in SUPPORTED_SUFFIXES:
                continue
            relative = path.resolve().relative_to(self._dataset_dir).as_posix()
            stat = path.stat()
            entries.append(
                DatasetEntry(
                    name=relative,
                    path=path.resolve(),
                    file_format=SUPPORTED_SUFFIXES[suffix],
                    size_bytes=stat.st_size,
                    modified_at=datetime.fromtimestamp(stat.st_mtime, tz=UTC),
                )
            )
        return entries

    def get(self, dataset_name: str) -> DatasetEntry:
        """Return dataset by catalog name.

        Raises:
            ValueError: If dataset name does not exist or escapes dataset root.
        """
        for entry in self.scan():
            if entry.name == dataset_name:
                return entry
        raise ValueError(
            f"Dataset '{dataset_name}' not found. Use duckdb_analytics_list_datasets first."
        )

    def as_summaries(self) -> list[DatasetSummary]:
        """Return all datasets as public models."""
        return [entry.to_summary() for entry in self.scan()]
