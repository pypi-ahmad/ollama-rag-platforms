"""CLI for offline Ollama RAG app."""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from collections.abc import Coroutine
from typing import Any

import typer

from offline_ollama_rag.config import get_settings
from offline_ollama_rag.logging_utils import configure_logging
from offline_ollama_rag.ollama_client import AsyncOllamaGateway
from offline_ollama_rag.pipeline import answer_question, build_index, evaluate, run_all

app = typer.Typer(help="Offline Ollama RAG app")


def _run_async[T](coro: Coroutine[Any, Any, T]) -> T:
    return asyncio.run(coro)


@app.callback()
def callback() -> None:
    settings = get_settings()
    configure_logging()
    settings.ensure_dirs()


@app.command("build-index")
def build_index_cmd() -> None:
    settings = get_settings()
    payload = _run_async(build_index(settings, gateway=AsyncOllamaGateway(settings.ollama_host)))
    typer.echo(json.dumps(payload, indent=2))


@app.command("ask")
def ask_cmd(question: str, rag: bool = True) -> None:
    settings = get_settings()
    payload = _run_async(answer_question(settings=settings, question=question, use_rag=rag))
    typer.echo(json.dumps(payload, indent=2))


@app.command("evaluate")
def evaluate_cmd() -> None:
    settings = get_settings()
    payload = _run_async(evaluate(settings))
    typer.echo(json.dumps(payload["summary"], indent=2))


@app.command("run-all")
def run_all_cmd() -> None:
    settings = get_settings()
    payload = _run_async(run_all(settings))
    typer.echo(json.dumps(payload["evaluation"]["summary"], indent=2))


@app.command("serve-app")
def serve_app_cmd(port: int = 8503) -> None:
    settings = get_settings()
    app_path = settings.resolve(settings.project_root / "app" / "streamlit_app.py")
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(app_path), "--server.port", str(port)],
        check=True,
    )


if __name__ == "__main__":
    app()
