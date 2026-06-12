"""Local Ollama integration for grounded answer generation."""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from urllib import error, request

import orjson

from ask_my_docs.models import RetrievedChunk
from ask_my_docs.utils import tokenize

_SPLIT_RE = re.compile(r"\s{2,}")


@dataclass(frozen=True, slots=True)
class OllamaModelInfo:
    """One model row from `ollama list`."""

    name: str
    size: str

    @property
    def is_local(self) -> bool:
        """Return True when the model is local (not cloud-only)."""

        return self.size.strip() != "-"


@dataclass(frozen=True, slots=True)
class OllamaGeneration:
    """Structured generation output from Ollama."""

    model: str
    text: str
    prompt_tokens: int
    completion_tokens: int


def parse_ollama_list(output: str) -> list[OllamaModelInfo]:
    """Parse `ollama list` CLI output into typed rows."""

    lines = [line.rstrip() for line in output.splitlines() if line.strip()]
    if not lines:
        return []

    data_lines = lines[1:] if lines[0].strip().lower().startswith("name") else lines
    models: list[OllamaModelInfo] = []

    for line in data_lines:
        columns = _SPLIT_RE.split(line.strip())
        if not columns:
            continue
        name = columns[0]
        size = columns[2] if len(columns) >= 3 else ""
        models.append(OllamaModelInfo(name=name, size=size))

    return models


def resolve_ollama_model(
    configured_model: str | None,
    available_models: list[OllamaModelInfo],
) -> str:
    """Select model using configured model or first local entry.

    Args:
        configured_model: Optional explicit model name from settings.
        available_models: Parsed rows from `ollama list`.

    Returns:
        Selected model name.

    Raises:
        RuntimeError: If configured model is unavailable or list is empty.
    """

    if not available_models:
        raise RuntimeError("No models found in `ollama list`")

    if configured_model is not None:
        if any(model.name == configured_model for model in available_models):
            return configured_model
        available_names = ", ".join(model.name for model in available_models)
        raise RuntimeError(
            f"Configured model '{configured_model}' not found. Available: {available_names}"
        )

    for model in available_models:
        if model.is_local:
            return model.name

    return available_models[0].name


class OllamaGenerator:
    """Generate grounded answers through local Ollama API."""

    def __init__(
        self,
        base_url: str,
        timeout_seconds: float,
        configured_model: str | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._configured_model = configured_model
        self._model: str | None = None

    @property
    def model(self) -> str:
        """Return selected Ollama model name."""

        if self._model is None:
            self._model = resolve_ollama_model(
                configured_model=self._configured_model,
                available_models=self._list_models(),
            )
        return self._model

    def _list_models(self) -> list[OllamaModelInfo]:
        process = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            check=False,
        )
        if process.returncode != 0:
            raise RuntimeError(
                "Failed to run `ollama list`: "
                f"{process.stderr.strip() or process.stdout.strip() or 'unknown error'}"
            )

        models = parse_ollama_list(process.stdout)
        if not models:
            raise RuntimeError("`ollama list` returned no models")
        return models

    def generate(self, question: str, retrieved: list[RetrievedChunk]) -> OllamaGeneration:
        """Generate answer from retrieved context.

        Args:
            question: User question.
            retrieved: Ranked context chunks.

        Returns:
            OllamaGeneration object containing answer text and token stats.
        """

        context_blocks = [f"[{item.chunk.doc_id}] {item.chunk.text}" for item in retrieved]
        prompt = (
            "You are a grounded RAG assistant. "
            "Answer only from the provided context. "
            "If context is insufficient, say so. "
            "Cite evidence inline using [doc_id] markers.\n\n"
            f"Question: {question}\n\n"
            "Context:\n"
            f"{chr(10).join(context_blocks)}\n\n"
            "Answer:"
        )

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1},
        }

        endpoint = f"{self._base_url}/api/generate"
        http_request = request.Request(
            url=endpoint,
            data=orjson.dumps(payload),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with request.urlopen(http_request, timeout=self._timeout_seconds) as response:
                raw = response.read()
        except error.URLError as exc:
            raise RuntimeError(f"Ollama request failed: {exc}") from exc

        parsed = orjson.loads(raw)
        if not isinstance(parsed, dict):
            raise RuntimeError("Unexpected Ollama response format")

        text_value = parsed.get("response")
        text = str(text_value).strip() if text_value is not None else ""
        if not text:
            text = "I could not produce an answer from the provided context."

        prompt_eval_count = parsed.get("prompt_eval_count")
        eval_count = parsed.get("eval_count")

        prompt_tokens = (
            int(prompt_eval_count)
            if isinstance(prompt_eval_count, int)
            else len(tokenize(prompt))
        )
        completion_tokens = int(eval_count) if isinstance(eval_count, int) else len(tokenize(text))

        return OllamaGeneration(
            model=self.model,
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
