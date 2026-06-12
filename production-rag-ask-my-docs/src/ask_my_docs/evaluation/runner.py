"""Evaluation runner for quality, latency, and cost metrics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import orjson
import polars as pl

from ask_my_docs.evaluation.metrics import exact_match_score, token_f1_score
from ask_my_docs.pipeline import RAGPipeline


@dataclass(slots=True)
class EvalExample:
    """One evaluation sample."""

    question: str
    reference_answer: str
    expected_doc_ids: set[str]


@dataclass(slots=True)
class EvalReport:
    """Aggregate and per-example evaluation results."""

    aggregate: dict[str, float]
    per_example: list[dict[str, object]]

    def to_payload(self) -> dict[str, object]:
        """Return JSON-serializable report payload."""

        return {
            "aggregate": self.aggregate,
            "per_example": self.per_example,
        }


def _coerce_float(value: object) -> float:
    if value is None:
        return 0.0
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float)):
        return float(value)
    return float(str(value))


def load_eval_examples(path: Path) -> list[EvalExample]:
    """Load JSONL evaluation examples from disk."""

    examples: list[EvalExample] = []
    for line in path.read_bytes().splitlines():
        payload = orjson.loads(line)
        examples.append(
            EvalExample(
                question=str(payload["question"]),
                reference_answer=str(payload["reference_answer"]),
                expected_doc_ids={str(item) for item in payload["expected_doc_ids"]},
            )
        )

    if not examples:
        raise ValueError(f"No eval samples found in: {path}")
    return examples


def run_evaluation(pipeline: RAGPipeline, examples: list[EvalExample], top_k: int) -> EvalReport:
    """Run evaluation across examples and return aggregate report."""

    rows: list[dict[str, object]] = []

    for example in examples:
        result = pipeline.answer(
            question=example.question,
            top_k=top_k,
            expected_doc_ids=example.expected_doc_ids,
        )

        answer_f1 = token_f1_score(result.answer, example.reference_answer)
        exact_match = exact_match_score(result.answer, example.reference_answer)
        pipeline.update_quality_metrics(
            request_id=result.request_id,
            answer_f1=answer_f1,
            exact_match=exact_match,
        )

        rows.append(
            {
                "question": example.question,
                "reference_answer": example.reference_answer,
                "predicted_answer": result.answer,
                "model_name": result.model_name,
                "citations": result.citations,
                "retrieved_doc_ids": result.retrieved_doc_ids,
                "retrieval_recall_at_k": result.retrieval_recall_at_k or 0.0,
                "answer_f1": answer_f1,
                "exact_match": exact_match,
                "latency_ms": result.latency_ms,
                "retrieval_latency_ms": result.retrieval_latency_ms,
                "generation_latency_ms": result.generation_latency_ms,
                "estimated_cost_usd": result.estimated_cost_usd,
                "total_tokens": result.total_tokens,
            }
        )

    frame = pl.DataFrame(rows)

    aggregate = {
        "num_examples": _coerce_float(len(rows)),
        "answer_f1_mean": _coerce_float(frame["answer_f1"].mean()),
        "exact_match_mean": _coerce_float(frame["exact_match"].mean()),
        "retrieval_recall_at_k_mean": _coerce_float(frame["retrieval_recall_at_k"].mean()),
        "latency_p50_ms": _coerce_float(
            frame["latency_ms"].quantile(0.50, interpolation="linear")
        ),
        "latency_p95_ms": _coerce_float(
            frame["latency_ms"].quantile(0.95, interpolation="linear")
        ),
        "avg_cost_usd": _coerce_float(frame["estimated_cost_usd"].mean()),
        "avg_tokens": _coerce_float(frame["total_tokens"].mean()),
    }

    return EvalReport(aggregate=aggregate, per_example=rows)


def save_eval_report(path: Path, report: EvalReport) -> None:
    """Persist evaluation report JSON to disk."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        orjson.dumps(
            report.to_payload(),
            option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS,
        )
    )
