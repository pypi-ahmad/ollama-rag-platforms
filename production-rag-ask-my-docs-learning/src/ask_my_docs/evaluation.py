"""Evaluation and CI gating for retrieval + citation quality."""

from __future__ import annotations

import json
import re
from pathlib import Path
from statistics import mean

import yaml
from pydantic import ValidationError

from ask_my_docs.pipeline import AskMyDocsEngine
from ask_my_docs.text import extract_citation_numbers
from ask_my_docs.types import EvalDataset

_CITED_LINE_PATTERN = re.compile(r"\[[0-9]+]")


def load_eval_dataset(dataset_path: Path) -> EvalDataset:
    """Load YAML evaluation rows into validated models."""
    payload = yaml.safe_load(dataset_path.read_text(encoding="utf-8"))
    try:
        return EvalDataset.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(f"Invalid evaluation dataset: {exc}") from exc


def load_thresholds(thresholds_path: Path) -> dict[str, float]:
    """Load metric thresholds for pass/fail CI gates."""
    payload = yaml.safe_load(thresholds_path.read_text(encoding="utf-8"))
    if "thresholds" not in payload:
        raise ValueError("Threshold file must contain a 'thresholds' mapping")
    return {str(k): float(v) for k, v in payload["thresholds"].items()}


def _reciprocal_rank(retrieved_ids: list[str], gold_ids: set[str], k: int) -> float:
    for idx, chunk_id in enumerate(retrieved_ids[:k], start=1):
        if chunk_id in gold_ids:
            return 1.0 / idx
    return 0.0


def _citation_coverage(answer: str) -> float:
    evidence_lines = [line for line in answer.splitlines() if line.strip().startswith("-")]
    if not evidence_lines:
        return 0.0
    cited = sum(1 for line in evidence_lines if _CITED_LINE_PATTERN.search(line) is not None)
    return cited / len(evidence_lines)


def _citation_validity(answer: str, num_contexts: int) -> float:
    citations = extract_citation_numbers(answer)
    if not citations:
        return 0.0
    return 1.0 if all(1 <= n <= num_contexts for n in citations) else 0.0


def _keyword_recall(answer: str, expected_keywords: list[str]) -> float:
    if not expected_keywords:
        return 1.0
    answer_lower = answer.lower()
    hits = sum(1 for keyword in expected_keywords if keyword.lower() in answer_lower)
    return hits / len(expected_keywords)


def evaluate_engine(
    engine: AskMyDocsEngine,
    dataset: EvalDataset,
    retrieval_k: int = 5,
) -> dict[str, object]:
    """Run retrieval + answer metrics for each evaluation query."""
    hybrid_hits: list[float] = []
    vector_hits: list[float] = []
    hybrid_rrs: list[float] = []
    citation_coverages: list[float] = []
    citation_validities: list[float] = []
    keyword_recalls: list[float] = []

    rows: list[dict[str, object]] = []
    for example in dataset.queries:
        gold = set(example.gold_chunk_ids)

        hybrid_contexts = engine.retriever.retrieve(example.question, top_k=retrieval_k)
        vector_contexts = engine.retriever.vector_only(example.question, top_k=retrieval_k)
        response = engine.ask(example.question, top_k=retrieval_k)

        hybrid_ids = [chunk.chunk_id for chunk in hybrid_contexts]
        vector_ids = [chunk.chunk_id for chunk in vector_contexts]

        hybrid_hit = 1.0 if any(chunk_id in gold for chunk_id in hybrid_ids[:retrieval_k]) else 0.0
        vector_hit = 1.0 if any(chunk_id in gold for chunk_id in vector_ids[:retrieval_k]) else 0.0
        rr = _reciprocal_rank(hybrid_ids, gold, retrieval_k)
        c_cov = _citation_coverage(response.answer)
        c_val = _citation_validity(response.answer, len(response.citations))
        kw_recall = _keyword_recall(response.answer, example.expected_keywords)

        hybrid_hits.append(hybrid_hit)
        vector_hits.append(vector_hit)
        hybrid_rrs.append(rr)
        citation_coverages.append(c_cov)
        citation_validities.append(c_val)
        keyword_recalls.append(kw_recall)

        rows.append(
            {
                "question": example.question,
                "gold_chunk_ids": sorted(gold),
                "hybrid_top_k": hybrid_ids,
                "vector_top_k": vector_ids,
                "hybrid_hit": hybrid_hit,
                "vector_hit": vector_hit,
                "hybrid_rr": rr,
                "citation_coverage": c_cov,
                "citation_validity": c_val,
                "keyword_recall": kw_recall,
            }
        )

    hybrid_recall = mean(hybrid_hits) if hybrid_hits else 0.0
    vector_recall = mean(vector_hits) if vector_hits else 0.0

    metrics = {
        "hybrid_recall_at_5": hybrid_recall,
        "vector_recall_at_5": vector_recall,
        "hybrid_vs_vector_recall_delta": hybrid_recall - vector_recall,
        "hybrid_mrr_at_5": mean(hybrid_rrs) if hybrid_rrs else 0.0,
        "citation_coverage": mean(citation_coverages) if citation_coverages else 0.0,
        "citation_validity": mean(citation_validities) if citation_validities else 0.0,
        "keyword_recall": mean(keyword_recalls) if keyword_recalls else 0.0,
    }
    return {"metrics": metrics, "rows": rows, "num_queries": len(dataset.queries)}


def apply_gate(metrics: dict[str, float], thresholds: dict[str, float]) -> tuple[bool, list[str]]:
    """Compare measured metrics against thresholds."""
    failures: list[str] = []
    for key, threshold in thresholds.items():
        value = metrics.get(key)
        if value is None:
            failures.append(f"missing_metric::{key}")
            continue
        if value < threshold:
            failures.append(f"{key}={value:.4f} < {threshold:.4f}")
    return len(failures) == 0, failures


def save_eval_report(report: dict[str, object], output_path: Path) -> None:
    """Persist evaluation report JSON artifact."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
