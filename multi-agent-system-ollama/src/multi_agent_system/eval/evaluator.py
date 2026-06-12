"""Evaluation logic for baseline vs multi-agent outputs."""

from __future__ import annotations

import json

from multi_agent_system.orchestration.coordinator import MultiAgentCoordinator
from multi_agent_system.schemas import EvalPrediction, EvalSummary, TaskExample


def load_eval_tasks(path: str) -> list[TaskExample]:
    """Load task definitions from JSON file."""
    with open(path, encoding="utf-8") as handle:
        payload = json.loads(handle.read())
    return [TaskExample.model_validate(item) for item in payload]


def keyword_recall(answer: str, required_keywords: list[str]) -> float:
    """Compute keyword recall in [0, 1]."""
    if not required_keywords:
        return 0.0
    text = answer.lower()
    hits = sum(1 for kw in required_keywords if kw.lower() in text)
    return hits / len(required_keywords)


def _safe_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


async def evaluate(
    coordinator: MultiAgentCoordinator,
    tasks: list[TaskExample],
) -> tuple[list[EvalPrediction], EvalSummary]:
    """Run evaluation tasks and aggregate metrics."""
    rows: list[EvalPrediction] = []

    for idx, task in enumerate(tasks, start=1):
        trace_id = f"eval-{idx:03d}-{task.task_id}"
        run = await coordinator.run(question=task.question, trace_id=trace_id)

        baseline_score = keyword_recall(run.baseline_answer, task.required_keywords)
        final_score = keyword_recall(run.final_answer, task.required_keywords)

        retrieval_hit = any(hit["source"] == task.expected_source for hit in run.retrieved)
        source_cited = task.expected_source.lower() in run.final_answer.lower()

        rows.append(
            EvalPrediction(
                task_id=task.task_id,
                question=task.question,
                expected_source=task.expected_source,
                baseline_answer=run.baseline_answer,
                final_answer=run.final_answer,
                baseline_keyword_recall=baseline_score,
                final_keyword_recall=final_score,
                keyword_gain=final_score - baseline_score,
                retrieval_hit=retrieval_hit,
                source_cited_in_final=source_cited,
                planner_fallback_used=run.planner_fallback_used,
                reviewer_fallback_used=run.reviewer_fallback_used,
                total_latency_ms=run.total_latency_ms,
            )
        )

    summary = EvalSummary(
        n_tasks=len(rows),
        baseline_keyword_recall_mean=_safe_mean([row.baseline_keyword_recall for row in rows]),
        multi_agent_keyword_recall_mean=_safe_mean([row.final_keyword_recall for row in rows]),
        keyword_recall_gain=_safe_mean([row.final_keyword_recall for row in rows])
        - _safe_mean([row.baseline_keyword_recall for row in rows]),
        retrieval_hit_rate=_safe_mean([1.0 if row.retrieval_hit else 0.0 for row in rows]),
        source_citation_rate=_safe_mean([1.0 if row.source_cited_in_final else 0.0 for row in rows]),
        planner_fallback_rate=_safe_mean([1.0 if row.planner_fallback_used else 0.0 for row in rows]),
        reviewer_fallback_rate=_safe_mean([1.0 if row.reviewer_fallback_used else 0.0 for row in rows]),
        avg_total_latency_ms=_safe_mean([row.total_latency_ms for row in rows]),
    )

    return rows, summary
