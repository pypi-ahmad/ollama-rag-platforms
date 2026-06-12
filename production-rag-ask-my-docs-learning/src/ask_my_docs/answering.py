"""Citation-enforced answer synthesis over retrieved chunks."""

from __future__ import annotations

import math

from ask_my_docs.text import ensure_line_citations, simple_tokenize, split_into_sentences
from ask_my_docs.types import RetrievedChunk


def _sentence_overlap_score(sentence: str, query: str) -> float:
    sentence_tokens = simple_tokenize(sentence)
    query_tokens = simple_tokenize(query)
    if not sentence_tokens or not query_tokens:
        return 0.0

    sentence_set = set(sentence_tokens)
    query_set = set(query_tokens)
    overlap = len(sentence_set & query_set)

    precision = overlap / max(len(sentence_set), 1)
    recall = overlap / max(len(query_set), 1)
    f1 = 0.0 if precision + recall == 0 else 2 * (precision * recall) / (precision + recall)

    length_penalty = math.exp(-abs(len(sentence_tokens) - 24) / 30)
    numeric_bonus = 0.15 if any(char.isdigit() for char in sentence) else 0.0
    query_coverage = overlap / max(len(query_set), 1)

    return (f1 * length_penalty) + (0.2 * query_coverage) + numeric_bonus


def build_cited_extractive_answer(
    question: str,
    retrieved: list[RetrievedChunk],
    max_points: int = 4,
) -> str:
    """Build an extractive answer where every evidence line is citation-backed.

    Example:
        >>> answer = build_cited_extractive_answer("What is the SLA?", retrieved)
        >>> "[1]" in answer
        True
    """
    if not retrieved:
        return "I could not find relevant evidence in the indexed documents."

    evidence_candidates: list[tuple[float, str, int]] = []
    for citation_idx, chunk in enumerate(retrieved[:3], start=1):
        for sentence in split_into_sentences(chunk.text):
            sentence_clean = sentence.strip()
            if len(sentence_clean) < 30:
                continue
            score = _sentence_overlap_score(sentence_clean, question)
            if score > 0:
                evidence_candidates.append((score, sentence_clean, citation_idx))

    if not evidence_candidates:
        evidence_lines = [f"- The strongest available context is from {retrieved[0].title}. [1]"]
    else:
        evidence_candidates.sort(key=lambda row: row[0], reverse=True)
        selected = evidence_candidates[:max_points]
        evidence_lines = [
            f"- {sentence} [{citation_idx}]" for _, sentence, citation_idx in selected
        ]

    evidence_lines = ensure_line_citations(evidence_lines, default_citation=1)

    source_lines = [
        f"[{idx}] {chunk.title} ({chunk.source_path}, chunk_id={chunk.chunk_id})"
        for idx, chunk in enumerate(retrieved, start=1)
    ]

    return "\n".join([
        "Answer (citation-enforced):",
        *evidence_lines,
        "",
        "Sources:",
        *source_lines,
    ])
