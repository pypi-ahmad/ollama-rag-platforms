"""Artifact persistence and report rendering."""

from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from jinja2 import Template

from rag_telemetry_evals.schemas import EvalPrediction, EvalSummary

_REPORT_TEMPLATE = """# RAG Telemetry + Evaluation Report

Generated at: `{{ generated_at }}`

## Runtime Configuration

- Chat model: `{{ chat_model }}`
- Embedding model: `{{ embedding_model }}`
- Questions evaluated: **{{ n_questions }}**

## Core Metrics

- Retrieval hit rate: **{{ retrieval_hit_rate }}**
- RAG cites expected source rate: **{{ source_mention_rate }}**
- Baseline keyword recall mean: **{{ baseline_keyword }}**
- RAG keyword recall mean: **{{ rag_keyword }}**
- Keyword recall gain: **{{ keyword_gain }}**
- Baseline semantic similarity mean: **{{ baseline_semantic }}**
- RAG semantic similarity mean: **{{ rag_semantic }}**
- Semantic similarity gain: **{{ semantic_gain }}**
- Baseline latency mean (ms): **{{ baseline_latency }}**
- RAG latency mean (ms): **{{ rag_latency }}**

## Telemetry Summary

- Total spans: **{{ telemetry_spans }}**
- Unique traces: **{{ telemetry_traces }}**

| Span | Count | Mean ms | P95 ms | Errors |
|---|---:|---:|---:|---:|
{% for row in telemetry_rows %}| {{ row.span_name }} | {{ row.count }} | {{ row.latency_ms_mean }} | {{ row.latency_ms_p95 }} | {{ row.error_count }} |
{% endfor %}

## Per-question Metrics

| ID | Retrieval Hit | Baseline Keyword | RAG Keyword | Baseline Semantic | RAG Semantic |
|---|---:|---:|---:|---:|---:|
{% for row in rows %}| {{ row.question_id }} | {{ 1 if row.retrieval_hit else 0 }} | {{ "%.3f"|format(row.baseline_keyword_recall) }} | {{ "%.3f"|format(row.rag_keyword_recall) }} | {{ "%.3f"|format(row.baseline_semantic_similarity) }} | {{ "%.3f"|format(row.rag_semantic_similarity) }} |
{% endfor %}
"""


def save_predictions(rows: list[EvalPrediction], output_path: Path) -> None:
    """Save prediction rows as CSV."""
    records = [row.model_dump() for row in rows]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not records:
        output_path.write_text("", encoding="utf-8")
        return

    fieldnames = list(records[0].keys())
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def save_summary(summary: EvalSummary, output_path: Path) -> None:
    """Save summary metrics as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(summary.model_dump_json(indent=2), encoding="utf-8")


def render_report(
    summary: EvalSummary,
    rows: list[EvalPrediction],
    chat_model: str,
    embedding_model: str,
    telemetry_summary: dict[str, Any],
    output_path: Path,
) -> str:
    """Render markdown report and save it to disk."""
    text = Template(_REPORT_TEMPLATE).render(
        generated_at=datetime.now(tz=UTC).isoformat(timespec="seconds"),
        chat_model=chat_model,
        embedding_model=embedding_model,
        n_questions=summary.n_questions,
        retrieval_hit_rate=f"{summary.retrieval_hit_rate:.4f}",
        source_mention_rate=f"{summary.rag_mentions_expected_source_rate:.4f}",
        baseline_keyword=f"{summary.baseline_keyword_recall_mean:.4f}",
        rag_keyword=f"{summary.rag_keyword_recall_mean:.4f}",
        keyword_gain=f"{summary.keyword_recall_gain:.4f}",
        baseline_semantic=f"{summary.baseline_semantic_similarity_mean:.4f}",
        rag_semantic=f"{summary.rag_semantic_similarity_mean:.4f}",
        semantic_gain=f"{summary.semantic_similarity_gain:.4f}",
        baseline_latency=f"{summary.baseline_latency_ms_mean:.2f}",
        rag_latency=f"{summary.rag_latency_ms_mean:.2f}",
        telemetry_spans=telemetry_summary.get("n_spans", 0),
        telemetry_traces=telemetry_summary.get("n_unique_traces", 0),
        telemetry_rows=telemetry_summary.get("by_span", []),
        rows=[json.loads(row.model_dump_json()) for row in rows],
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    return text
