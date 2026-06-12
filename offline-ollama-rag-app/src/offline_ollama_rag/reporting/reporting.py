"""Artifact persistence and report rendering."""

from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path

from jinja2 import Template

from offline_ollama_rag.schemas import EvalPrediction, EvalSummary

_REPORT_TEMPLATE = """# Offline Ollama RAG Evaluation Report

Generated at: `{{ generated_at }}`

## Setup

- Chat model: `{{ chat_model }}`
- Embedding model: `{{ embedding_model }}`
- Questions: {{ n_questions }}

## Metrics

- Baseline keyword recall mean: **{{ baseline_mean }}**
- RAG keyword recall mean: **{{ rag_mean }}**
- Gain: **{{ gain }}**
- Baseline non-empty rate: **{{ baseline_nonempty }}**
- RAG non-empty rate: **{{ rag_nonempty }}**

## Per-question Scores

| Question ID | Baseline Recall | RAG Recall |
|---|---:|---:|
{% for row in rows %}| {{ row.question_id }} | {{ row.baseline_keyword_recall }} | {{ row.rag_keyword_recall }} |
{% endfor %}
"""


def save_predictions(rows: list[EvalPrediction], output_path: Path) -> None:
    """Save prediction rows as CSV."""
    records = [json.loads(row.model_dump_json()) for row in rows]
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
    output_path: Path,
) -> str:
    """Render markdown report and save it to disk."""
    text = Template(_REPORT_TEMPLATE).render(
        generated_at=datetime.now(tz=UTC).isoformat(timespec="seconds"),
        chat_model=chat_model,
        embedding_model=embedding_model,
        n_questions=summary.n_questions,
        baseline_mean=f"{summary.baseline_keyword_recall_mean:.4f}",
        rag_mean=f"{summary.rag_keyword_recall_mean:.4f}",
        gain=f"{summary.keyword_recall_gain:.4f}",
        baseline_nonempty=f"{summary.baseline_nonempty_rate:.4f}",
        rag_nonempty=f"{summary.rag_nonempty_rate:.4f}",
        rows=[json.loads(row.model_dump_json()) for row in rows],
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    return text
