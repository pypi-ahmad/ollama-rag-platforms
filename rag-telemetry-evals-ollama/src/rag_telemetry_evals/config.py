"""Environment-driven settings for the RAG telemetry project."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    project_root: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[2])

    ollama_host: str = "http://127.0.0.1:11434"
    chat_model: str = "phi3.5:3.8b"
    embedding_model: str = "embeddinggemma:latest"

    seed: int = 42
    chunk_size_words: int = Field(default=170, ge=50, le=1000)
    chunk_overlap_words: int = Field(default=35, ge=0, le=400)
    retrieval_top_k: int = Field(default=4, ge=1, le=20)

    generation_temperature: float = Field(default=0.1, ge=0.0, le=1.0)
    generation_max_tokens: int = Field(default=220, ge=32, le=2048)
    generation_timeout_seconds: float = Field(default=15.0, ge=1.0, le=180.0)
    embedding_timeout_seconds: float = Field(default=90.0, ge=1.0, le=180.0)

    knowledge_dir: Path = Path("data/knowledge")
    evaluation_file: Path = Path("data/eval/questions.json")
    artifacts_dir: Path = Path("artifacts")
    index_dir: Path = Path("artifacts/index")
    eval_dir: Path = Path("artifacts/evals")
    report_dir: Path = Path("artifacts/reports")
    telemetry_dir: Path = Path("artifacts/telemetry")

    def resolve(self, path: Path) -> Path:
        if path.is_absolute():
            return path
        return (self.project_root / path).resolve()

    @property
    def resolved_knowledge_dir(self) -> Path:
        return self.resolve(self.knowledge_dir)

    @property
    def resolved_evaluation_file(self) -> Path:
        return self.resolve(self.evaluation_file)

    @property
    def resolved_artifacts_dir(self) -> Path:
        return self.resolve(self.artifacts_dir)

    @property
    def resolved_index_dir(self) -> Path:
        return self.resolve(self.index_dir)

    @property
    def resolved_eval_dir(self) -> Path:
        return self.resolve(self.eval_dir)

    @property
    def resolved_report_dir(self) -> Path:
        return self.resolve(self.report_dir)

    @property
    def resolved_telemetry_dir(self) -> Path:
        return self.resolve(self.telemetry_dir)

    @property
    def embeddings_file(self) -> Path:
        return self.resolved_index_dir / "embeddings.npy"

    @property
    def chunks_file(self) -> Path:
        return self.resolved_index_dir / "chunks.json"

    @property
    def predictions_file(self) -> Path:
        return self.resolved_eval_dir / "predictions.csv"

    @property
    def summary_file(self) -> Path:
        return self.resolved_eval_dir / "summary.json"

    @property
    def report_file(self) -> Path:
        return self.resolved_report_dir / "rag_telemetry_eval_report.md"

    @property
    def trace_file(self) -> Path:
        return self.resolved_telemetry_dir / "traces.jsonl"

    @property
    def telemetry_summary_file(self) -> Path:
        return self.resolved_telemetry_dir / "summary.json"

    @property
    def run_summary_file(self) -> Path:
        return self.resolved_artifacts_dir / "run_summary.json"

    def ensure_dirs(self) -> None:
        self.resolved_knowledge_dir.mkdir(parents=True, exist_ok=True)
        self.resolved_artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.resolved_index_dir.mkdir(parents=True, exist_ok=True)
        self.resolved_eval_dir.mkdir(parents=True, exist_ok=True)
        self.resolved_report_dir.mkdir(parents=True, exist_ok=True)
        self.resolved_telemetry_dir.mkdir(parents=True, exist_ok=True)


def get_settings() -> Settings:
    """Load and validate project settings."""
    settings = Settings()
    settings.ensure_dirs()
    return settings
