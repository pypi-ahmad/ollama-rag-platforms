"""Evaluation metrics for retrieval and generation quality."""

from __future__ import annotations

from collections import Counter

from ask_my_docs.utils import tokenize


def exact_match_score(prediction: str, reference: str) -> float:
    """Compute token-normalized exact-match score."""

    pred_text = " ".join(tokenize(prediction))
    ref_text = " ".join(tokenize(reference))
    return 1.0 if pred_text == ref_text else 0.0


def token_f1_score(prediction: str, reference: str) -> float:
    """Compute token-level F1 overlap between prediction and reference."""

    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(reference)

    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    overlap = Counter(pred_tokens) & Counter(ref_tokens)
    num_overlap = sum(overlap.values())
    if num_overlap == 0:
        return 0.0

    precision = num_overlap / len(pred_tokens)
    recall = num_overlap / len(ref_tokens)
    return 2.0 * precision * recall / (precision + recall)


def retrieval_recall_at_k(expected_doc_ids: set[str], retrieved_doc_ids: list[str]) -> float:
    """Compute recall@k over expected supporting document IDs."""

    if not expected_doc_ids:
        return 0.0
    intersection = expected_doc_ids.intersection(set(retrieved_doc_ids))
    return len(intersection) / len(expected_doc_ids)
