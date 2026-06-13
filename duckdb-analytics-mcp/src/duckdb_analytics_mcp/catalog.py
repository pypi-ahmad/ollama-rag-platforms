"""Dataset catalog discovery and metadata utilities."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from time import monotonic

from loguru import logger

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

    def __init__(self, dataset_dir: Path, cache_ttl_seconds: float = 5.0) -> None:
        self._dataset_dir = dataset_dir.resolve()
        self._cache_ttl_seconds = max(cache_ttl_seconds, 0.0)
        self._lock = Lock()
        self._cached_entries: list[DatasetEntry] | None = None
        self._cached_index: dict[str, DatasetEntry] | None = None
        self._cached_at: float = 0.0

    def _scan_uncached(self) -> list[DatasetEntry]:
        entries: list[DatasetEntry] = []
        for path in sorted(self._dataset_dir.rglob("*")):
            if not path.is_file():
                continue

            suffix = path.suffix.lower()
            if suffix not in SUPPORTED_SUFFIXES:
                continue

            try:
                resolved_path = path.resolve()
                relative = resolved_path.relative_to(self._dataset_dir).as_posix()
                stat = resolved_path.stat()
            except (OSError, ValueError) as exc:
                logger.warning("Skipping dataset entry '{}': {}", path.as_posix(), exc)
                continue

            entries.append(
                DatasetEntry(
                    name=relative,
                    path=resolved_path,
                    file_format=SUPPORTED_SUFFIXES[suffix],
                    size_bytes=stat.st_size,
                    modified_at=datetime.fromtimestamp(stat.st_mtime, tz=UTC),
                )
            )
        return entries

    def _get_or_refresh_cache(self) -> tuple[list[DatasetEntry], dict[str, DatasetEntry]]:
        if self._cache_ttl_seconds <= 0:
            entries = self._scan_uncached()
            return entries, {entry.name: entry for entry in entries}

        now = monotonic()
        with self._lock:
            if (
                self._cached_entries is not None
                and self._cached_index is not None
                and now - self._cached_at < self._cache_ttl_seconds
            ):
                return list(self._cached_entries), dict(self._cached_index)

        entries = self._scan_uncached()
        index = {entry.name: entry for entry in entries}
        with self._lock:
            self._cached_entries = list(entries)
            self._cached_index = dict(index)
            self._cached_at = monotonic()
        return entries, index

    def scan(self) -> list[DatasetEntry]:
        """Scan dataset directory recursively and return supported files."""
        entries, _ = self._get_or_refresh_cache()
        return entries

    def get(self, dataset_name: str) -> DatasetEntry:
        """Return dataset by catalog name.

        Raises:
            ValueError: If dataset name does not exist or escapes dataset root.
        """
        _, index = self._get_or_refresh_cache()
        entry = index.get(dataset_name)
        if entry is not None:
            return entry
        raise ValueError(
            f"Dataset '{dataset_name}' not found. Use duckdb_analytics_list_datasets first."
        )

    def as_summaries(self) -> list[DatasetSummary]:
        """Return all datasets as public models."""
        return [entry.to_summary() for entry in self.scan()]
