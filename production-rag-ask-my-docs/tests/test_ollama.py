"""Unit tests for Ollama model discovery utilities."""

import subprocess

import pytest

from ask_my_docs.llm.ollama import (
    OllamaGenerator,
    OllamaModelInfo,
    parse_ollama_list,
    resolve_ollama_model,
)


def test_parse_ollama_list_reads_rows() -> None:
    output = """
NAME                     ID              SIZE      MODIFIED
cloud-model:cloud        abc             -         2 days ago
local-model:latest       def             4.1 GB    1 hour ago
""".strip()

    models = parse_ollama_list(output)

    assert [model.name for model in models] == ["cloud-model:cloud", "local-model:latest"]
    assert models[0].size == "-"
    assert models[1].size == "4.1 GB"


def test_resolve_ollama_model_prefers_local_when_unset() -> None:
    selected = resolve_ollama_model(
        configured_model=None,
        available_models=[
            OllamaModelInfo(name="cloud-only:cloud", size="-"),
            OllamaModelInfo(name="local:7b", size="3.8 GB"),
        ],
    )

    assert selected == "local:7b"


def test_ollama_list_timeout_raises_runtime_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def _timeout(*args: object, **kwargs: object) -> object:
        raise subprocess.TimeoutExpired(cmd=["ollama", "list"], timeout=1.5)

    monkeypatch.setattr("ask_my_docs.llm.ollama.subprocess.run", _timeout)
    generator = OllamaGenerator(
        base_url="http://127.0.0.1:11434",
        timeout_seconds=30.0,
        list_timeout_seconds=1.5,
        configured_model=None,
    )

    with pytest.raises(RuntimeError, match="timed out"):
        generator._list_models()
