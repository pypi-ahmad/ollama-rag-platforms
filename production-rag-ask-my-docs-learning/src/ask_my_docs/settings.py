"""Application configuration loaded from environment variables or defaults."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    """Typed runtime settings for ingestion, retrieval, and evaluation."""

    model_config = SettingsConfigDict(env_prefix="ASKDOCS_", env_file=".env", extra="ignore")

    docs_dir: Path = Path("data/docs")
    index_dir: Path = Path("artifacts/index")
    chunk_size_words: int = 180
    chunk_overlap_words: int = 40
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    hybrid_rrf_k: int = 60
    bm25_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    vector_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    candidate_pool_size: int = 30
    default_top_k: int = 5
    seed: int = 42


SETTINGS = AppSettings()
