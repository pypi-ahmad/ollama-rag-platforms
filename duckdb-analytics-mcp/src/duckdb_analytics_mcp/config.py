"""Environment-driven configuration for duckdb_analytics_mcp."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import AnyHttpUrl, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

Transport = Literal["stdio", "streamable-http"]


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables.

    Notes:
        - `transport` controls the default runtime transport.
        - `dataset_dir` is resolved relative to `project_root` when not absolute.
        - OAuth auth is enabled only when token + issuer/resource URLs are configured.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    server_name: str = "duckdb_analytics_mcp"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    transport: Transport = "streamable-http"

    project_root: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[2])
    dataset_dir: Path = Path("data")

    default_limit: int = Field(default=25, ge=1, le=1000)
    max_limit: int = Field(default=200, ge=1, le=5000)
    max_query_chars: int = Field(default=4000, ge=64, le=20000)
    query_timeout_seconds: float = Field(default=20.0, ge=1.0, le=120.0)
    max_sample_rows: int = Field(default=25, ge=1, le=200)
    catalog_cache_ttl_seconds: float = Field(default=5.0, ge=0.0, le=300.0)

    duckdb_threads: int = Field(default=4, ge=1, le=64)
    duckdb_memory_limit: str = "1GB"

    static_bearer_token: str | None = None
    auth_issuer_url: AnyHttpUrl | None = None
    auth_resource_server_url: AnyHttpUrl | None = None
    auth_required_scope: str = "analytics.read"

    @property
    def resolved_dataset_dir(self) -> Path:
        if self.dataset_dir.is_absolute():
            return self.dataset_dir
        return (self.project_root / self.dataset_dir).resolve()

    @property
    def duckdb_config(self) -> dict[str, str | bool | int | float | list[str]]:
        return {
            "threads": self.duckdb_threads,
            "memory_limit": self.duckdb_memory_limit,
            "autoinstall_known_extensions": False,
            "autoload_known_extensions": False,
            "allow_unsigned_extensions": False,
        }

    def ensure_runtime_paths(self) -> None:
        self.resolved_dataset_dir.mkdir(parents=True, exist_ok=True)


def get_settings() -> Settings:
    """Return validated settings."""
    settings = Settings()
    settings.ensure_runtime_paths()
    return settings
