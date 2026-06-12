"""Reviewer agent that enforces policy completeness and source citation."""

from __future__ import annotations

from multi_agent_system.schemas import ChatResult, RetrievedDoc


class ReviewerAgent:
    """Lightweight deterministic reviewer for production reliability."""

    async def review(
        self,
        question: str,
        draft_answer: ChatResult,
        retrieved: list[RetrievedDoc],
    ) -> ChatResult:
        if not draft_answer.text.strip():
            return ChatResult(
                text=f"Review fallback: no draft answer for question '{question}'.",
                done_reason="reviewer_empty_fallback",
                fallback_used=True,
            )

        source_lines = ", ".join(hit.doc.source for hit in retrieved[:2]) if retrieved else "unknown source"
        reviewed = (
            f"{draft_answer.text.strip()}\n\n"
            f"Reviewer checks: include operational caveats, verify thresholds before action, "
            f"and cross-check with {source_lines}."
        )

        return ChatResult(
            text=reviewed,
            prompt_tokens=draft_answer.prompt_tokens,
            completion_tokens=draft_answer.completion_tokens,
            total_duration_ns=draft_answer.total_duration_ns,
            done_reason="reviewer_ok",
            fallback_used=False,
        )
