"""Evaluation logic for baseline-vs-RAG comparisons."""

from __future__ import annotations

import json
from pathlib import Path

from loguru import logger

from offline_ollama_rag.retrieval.rag_engine import OfflineRAGEngine
from offline_ollama_rag.schemas import EvalPrediction, EvalSummary, QAExample


def load_eval_examples(path: Path) -> list[QAExample]:
    """Load evaluation questions from JSON."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [QAExample.model_validate(item) for item in payload]


def keyword_recall(answer: str, required_keywords: list[str]) -> float:
    """Compute keyword recall score in [0, 1]."""
    if not required_keywords:
        return 0.0

    normalized = answer.lower()
    hits = sum(1 for keyword in required_keywords if keyword.lower() in normalized)
    return hits / len(required_keywords)


def nonempty_rate(answers: list[str]) -> float:
    """Compute fraction of non-empty answers."""
    if not answers:
        return 0.0
    nonempty = sum(1 for answer in answers if answer.strip())
    return nonempty / len(answers)


async def run_evaluation(
    engine: OfflineRAGEngine,
    examples: list[QAExample],
) -> tuple[list[EvalPrediction], EvalSummary]:
    """Run baseline and RAG predictions and aggregate metrics."""
    rows: list[EvalPrediction] = []

    for idx, example in enumerate(examples, start=1):
        logger.info("Evaluating question {}/{} ({})", idx, len(examples), example.question_id)
        baseline_answer = await engine.answer_without_rag(example.question)
        rag_answer, _ = await engine.answer_with_rag(example.question)

        baseline_score = keyword_recall(baseline_answer, example.required_keywords)
        rag_score = keyword_recall(rag_answer, example.required_keywords)

        rows.append(
            EvalPrediction(
                question_id=example.question_id,
                question=example.question,
                reference_answer=example.reference_answer,
                baseline_answer=baseline_answer,
                rag_answer=rag_answer,
                baseline_keyword_recall=baseline_score,
                rag_keyword_recall=rag_score,
            )
        )

    baseline_mean = sum(row.baseline_keyword_recall for row in rows) / len(rows)
    rag_mean = sum(row.rag_keyword_recall for row in rows) / len(rows)

    summary = EvalSummary(
        n_questions=len(rows),
        baseline_keyword_recall_mean=baseline_mean,
        rag_keyword_recall_mean=rag_mean,
        keyword_recall_gain=rag_mean - baseline_mean,
        baseline_nonempty_rate=nonempty_rate([row.baseline_answer for row in rows]),
        rag_nonempty_rate=nonempty_rate([row.rag_answer for row in rows]),
    )

    return rows, summary
