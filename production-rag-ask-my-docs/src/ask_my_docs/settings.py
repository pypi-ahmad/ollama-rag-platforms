"""Application settings loaded from environment variables."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class PricingConfig(BaseModel):
    """Pricing assumptions used for cost estimation."""

    model_name: str = "ollama-local"
    prompt_cost_per_1k_tokens: float = 0.00015
    completion_cost_per_1k_tokens: float = 0.00060


class RetrievalConfig(BaseModel):
    """Hybrid retrieval and chunking settings."""

    chunk_size_tokens: int = 120
    chunk_overlap_tokens: int = 20
    top_k: int = 4
    lexical_weight: float = 0.45
    semantic_weight: float = 0.55
    embedding_dim: int = 256


class AppSettings(BaseSettings):
    """Typed runtime configuration for the project."""

    model_config = SettingsConfigDict(
        env_prefix="ASK_MY_DOCS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    service_name: str = "ask-my-docs-rag"

    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str | None = None
    ollama_timeout_seconds: float = 120.0
    ollama_list_timeout_seconds: float = 10.0

    docs_dir: Path = Path("data/docs")
    eval_path: Path = Path("data/eval/eval_set.jsonl")
    artifacts_dir: Path = Path("artifacts")
    index_dir: Path = Path("artifacts/index")
    traces_path: Path = Path("artifacts/observability/traces.jsonl")
    metrics_db_path: Path = Path("artifacts/observability/metrics.duckdb")
    thresholds_path: Path = Path("configs/regression_thresholds.yaml")
    observability_store_raw_questions: bool = False

    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    pricing: PricingConfig = Field(default_factory=PricingConfig)


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """Return cached application settings."""

    return AppSettings()
