"""End-to-end pipeline for demo runs, evaluation, and reporting."""

from __future__ import annotations

import json
from typing import Any

from multi_agent_system.agents.planner import PlannerAgent
from multi_agent_system.agents.retriever import RetrieverAgent
from multi_agent_system.agents.reviewer import ReviewerAgent
from multi_agent_system.agents.router import RouterAgent
from multi_agent_system.config import Settings
from multi_agent_system.eval.evaluator import evaluate, load_eval_tasks
from multi_agent_system.ollama_client import AsyncOllamaGateway
from multi_agent_system.orchestration.coordinator import MultiAgentCoordinator
from multi_agent_system.reporting.reporting import (
    render_report,
    save_demo_runs,
    save_predictions,
    save_summary,
)
from multi_agent_system.telemetry.tracer import JsonlTelemetryTracer, summarize_traces
from multi_agent_system.tools.knowledge_base import load_knowledge_docs


def _build_coordinator(settings: Settings, tracer: JsonlTelemetryTracer) -> MultiAgentCoordinator:
    docs = load_knowledge_docs(settings.resolved_knowledge_dir)
    gateway = AsyncOllamaGateway(settings.ollama_host)

    router = RouterAgent()
    retriever = RetrieverAgent(docs=docs, top_k=settings.retrieval_top_k)
    planner = PlannerAgent(settings=settings, gateway=gateway)
    reviewer = ReviewerAgent()

    return MultiAgentCoordinator(
        router=router,
        retriever=retriever,
        planner=planner,
        reviewer=reviewer,
        tracer=tracer,
    )


async def run_demo(settings: Settings, questions: list[str]) -> list[dict[str, Any]]:
    """Run multi-agent flow for ad-hoc demo questions."""
    tracer = JsonlTelemetryTracer(settings.traces_file)
    coordinator = _build_coordinator(settings, tracer)

    runs: list[dict[str, Any]] = []
    for idx, question in enumerate(questions, start=1):
        trace_id = f"demo-{idx:03d}"
        result = await coordinator.run(question=question, trace_id=trace_id)
        runs.append(json.loads(result.model_dump_json()))

    save_demo_runs(runs, settings.demo_runs_file)
    return runs


async def run_evaluation(settings: Settings, reset_traces: bool = True) -> dict[str, Any]:
    """Run full evaluation and persist artifacts."""
    if reset_traces and settings.traces_file.exists():
        settings.traces_file.unlink()

    tracer = JsonlTelemetryTracer(settings.traces_file)
    coordinator = _build_coordinator(settings, tracer)

    tasks = load_eval_tasks(settings.resolved_evaluation_file.as_posix())
    rows, summary = await evaluate(coordinator=coordinator, tasks=tasks)

    save_predictions(rows, settings.predictions_file)
    save_summary(summary, settings.summary_file)

    telemetry_summary = summarize_traces(settings.traces_file, settings.telemetry_summary_file)
    render_report(
        summary=summary,
        rows=rows,
        planner_model=settings.planner_model,
        reviewer_model=settings.reviewer_model,
        telemetry_summary=telemetry_summary,
        output_path=settings.report_file,
    )

    return {
        "summary": json.loads(summary.model_dump_json()),
        "predictions_path": settings.predictions_file.as_posix(),
        "summary_path": settings.summary_file.as_posix(),
        "report_path": settings.report_file.as_posix(),
        "trace_path": settings.traces_file.as_posix(),
        "telemetry_summary_path": settings.telemetry_summary_file.as_posix(),
    }


async def run_all(settings: Settings) -> dict[str, Any]:
    """Run demo + evaluation and save overall run summary."""
    demo_questions = [
        "When should emergency rollback be triggered for API incidents?",
        "What is the enterprise support first-response SLA?",
        "What canary percentage is used at release start?",
    ]

    if settings.traces_file.exists():
        settings.traces_file.unlink()

    demo_runs = await run_demo(settings, questions=demo_questions)
    eval_payload = await run_evaluation(settings, reset_traces=False)

    payload = {
        "planner_model": settings.planner_model,
        "reviewer_model": settings.reviewer_model,
        "demo_runs_path": settings.demo_runs_file.as_posix(),
        "demo_questions": demo_questions,
        "demo_count": len(demo_runs),
        "evaluation": eval_payload,
        "run_summary_path": settings.run_summary_file.as_posix(),
    }
    settings.run_summary_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
