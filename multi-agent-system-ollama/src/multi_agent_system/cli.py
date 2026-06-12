"""CLI for multi-agent system with local Ollama."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Coroutine
from typing import Any

import typer

from multi_agent_system.config import get_settings
from multi_agent_system.logging_utils import configure_logging
from multi_agent_system.ollama_client import AsyncOllamaGateway
from multi_agent_system.pipeline import run_all, run_demo, run_evaluation
from multi_agent_system.telemetry.tracer import summarize_traces

app = typer.Typer(help="Multi-agent system (local Ollama)")


def _run_async[T](coro: Coroutine[Any, Any, T]) -> T:
    return asyncio.run(coro)


@app.callback()
def callback() -> None:
    settings = get_settings()
    configure_logging()
    settings.ensure_dirs()


@app.command("check-models")
def check_models_cmd() -> None:
    settings = get_settings()
    gateway = AsyncOllamaGateway(settings.ollama_host)

    async def _run() -> dict[str, Any]:
        models = sorted(await gateway.list_model_names())
        return {
            "configured": [settings.planner_model, settings.reviewer_model],
            "available_subset": [
                m for m in [settings.planner_model, settings.reviewer_model] if m in models
            ],
            "n_local_models": len(models),
        }

    payload = _run_async(_run())
    typer.echo(json.dumps(payload, indent=2))


@app.command("run-demo")
def run_demo_cmd(question: list[str] | None = None) -> None:
    settings = get_settings()
    questions = question or [
        "When should emergency rollback be triggered for API incidents?",
        "What is the enterprise support first-response SLA?",
    ]
    payload = _run_async(run_demo(settings, questions))
    typer.echo(json.dumps(payload, indent=2))


@app.command("evaluate")
def evaluate_cmd() -> None:
    settings = get_settings()
    payload = _run_async(run_evaluation(settings))
    typer.echo(json.dumps(payload["summary"], indent=2))


@app.command("run-all")
def run_all_cmd() -> None:
    settings = get_settings()
    payload = _run_async(run_all(settings))
    typer.echo(json.dumps(payload["evaluation"]["summary"], indent=2))


@app.command("summarize-telemetry")
def summarize_telemetry_cmd() -> None:
    settings = get_settings()
    payload = summarize_traces(settings.traces_file, settings.telemetry_summary_file)
    typer.echo(json.dumps(payload, indent=2))


if __name__ == "__main__":
    app()
