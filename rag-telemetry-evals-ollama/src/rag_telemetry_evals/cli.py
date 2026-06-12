"""CLI for local RAG with telemetry and evals."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Coroutine
from typing import Any

import typer

from rag_telemetry_evals.config import get_settings
from rag_telemetry_evals.logging_utils import configure_logging
from rag_telemetry_evals.ollama_client import AsyncOllamaGateway
from rag_telemetry_evals.pipeline import answer_question, build_index, evaluate, run_all
from rag_telemetry_evals.telemetry.tracer import JsonlTelemetryTracer, summarize_traces

app = typer.Typer(help="RAG systems with telemetry and evals (local Ollama)")


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
    tracer = JsonlTelemetryTracer(settings.trace_file)
    payload = _run_async(
        build_index(
            settings,
            gateway=AsyncOllamaGateway(settings.ollama_host),
            tracer=tracer,
        )
    )
    typer.echo(json.dumps(payload, indent=2))


@app.command("ask")
def ask_cmd(question: str, rag: bool = typer.Option(True, "--rag/--baseline")) -> None:
    settings = get_settings()
    payload = _run_async(answer_question(settings=settings, question=question, use_rag=rag))
    typer.echo(json.dumps(payload, indent=2))


@app.command("evaluate")
def evaluate_cmd() -> None:
    settings = get_settings()
    payload = _run_async(evaluate(settings=settings))
    typer.echo(json.dumps(payload["summary"], indent=2))


@app.command("run-all")
def run_all_cmd() -> None:
    settings = get_settings()
    payload = _run_async(run_all(settings))
    typer.echo(json.dumps(payload["evaluation"]["summary"], indent=2))


@app.command("summarize-telemetry")
def summarize_telemetry_cmd() -> None:
    settings = get_settings()
    payload = summarize_traces(settings.trace_file, settings.telemetry_summary_file)
    typer.echo(json.dumps(payload, indent=2))


if __name__ == "__main__":
    app()
