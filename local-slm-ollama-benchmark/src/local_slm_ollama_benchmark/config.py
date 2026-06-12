"""Configuration models and loader for the benchmark suite."""

from __future__ import annotations

from pathlib import Path
import tomllib

from pydantic import BaseModel, Field, ValidationError, model_validator


class PromptCase(BaseModel):
    """One benchmark prompt and its quality-evaluation rubric."""

    id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    evaluation_type: str = Field(pattern="^(keywords|json_keys|exact_match)$")
    expected_keywords: list[str] = Field(default_factory=list)
    required_json_keys: list[str] = Field(default_factory=list)
    expected_answer: str | None = None
    max_words: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _validate_evaluation_fields(self) -> "PromptCase":
        """Ensure each evaluation type carries the required fields."""
        if self.evaluation_type == "keywords" and not self.expected_keywords:
            raise ValueError(
                f"Prompt '{self.id}' uses keyword evaluation but expected_keywords is empty."
            )

        if self.evaluation_type == "json_keys" and not self.required_json_keys:
            raise ValueError(
                f"Prompt '{self.id}' uses json_keys evaluation but required_json_keys is empty."
            )

        if self.evaluation_type == "exact_match" and not self.expected_answer:
            raise ValueError(
                f"Prompt '{self.id}' uses exact_match evaluation but expected_answer is missing."
            )

        return self


class GenerationConfig(BaseModel):
    """Ollama generation options used for every benchmark request."""

    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, gt=0.0, le=1.0)
    num_predict: int = Field(default=220, ge=16)
    think: bool | str | None = False
    seed: int = 42


class RuntimeConfig(BaseModel):
    """Execution controls for reproducible benchmarking."""

    repeat_count: int = Field(default=2, ge=1, le=20)
    request_timeout_sec: float = Field(default=180.0, gt=0.0)
    keep_alive: str = "20m"
    warmup_prompt: str = "Reply with the single token READY."


class CostConfig(BaseModel):
    """Optional assumptions for local-inference electricity estimation."""

    electricity_rate_usd_per_kwh: float | None = Field(default=None, gt=0.0)
    assumed_power_watts: float | None = Field(default=None, gt=0.0)


class BenchmarkConfig(BaseModel):
    """Top-level benchmark configuration."""

    ollama_host: str = "http://127.0.0.1:11434"
    models: list[str] = Field(min_length=3)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    cost: CostConfig = Field(default_factory=CostConfig)
    prompts: list[PromptCase] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_models(self) -> "BenchmarkConfig":
        """Ensure model list has unique names."""
        unique_models = set(self.models)
        if len(unique_models) != len(self.models):
            raise ValueError("Model names in `models` must be unique.")

        if len(self.models) != 3:
            raise ValueError("This project is designed to compare exactly 3 models.")

        return self


def load_benchmark_config(path: Path) -> BenchmarkConfig:
    """Load and validate benchmark configuration from a TOML file.

    Args:
        path: Absolute or relative path to a TOML config file.

    Returns:
        Parsed and validated :class:`BenchmarkConfig` instance.

    Raises:
        FileNotFoundError: If the config path does not exist.
        ValueError: If TOML parsing or schema validation fails.

    Example:
        >>> from pathlib import Path
        >>> config = load_benchmark_config(Path("configs/benchmark.toml"))
        >>> len(config.models)
        3
    """
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
        return BenchmarkConfig.model_validate(raw)
    except (tomllib.TOMLDecodeError, ValidationError, ValueError) as exc:
        raise ValueError(f"Invalid benchmark config at {path}: {exc}") from exc
