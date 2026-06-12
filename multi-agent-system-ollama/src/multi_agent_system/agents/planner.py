"""Planner agent that drafts an answer from route and retrieved context."""

from __future__ import annotations

from multi_agent_system.config import Settings
from multi_agent_system.ollama_client import AsyncOllamaGateway
from multi_agent_system.schemas import AgentRoute, ChatResult, RetrievedDoc

PLANNER_SYSTEM_PROMPT = (
    "You are a production incident and operations analyst. "
    "Use the provided context snippets, be concise, and cite exact source filenames."
)


class PlannerAgent:
    """Generate draft answers using LLM with deterministic fallback."""

    def __init__(self, settings: Settings, gateway: AsyncOllamaGateway) -> None:
        self._settings = settings
        self._gateway = gateway

    async def plan(
        self,
        question: str,
        route: AgentRoute,
        retrieved: list[RetrievedDoc],
        trace_id: str,
    ) -> ChatResult:
        context = self._context_block(retrieved)
        user_prompt = (
            f"Trace ID: {trace_id}\n"
            f"Route: intent={route.intent}, severity={route.severity}\n"
            f"Question: {question}\n\n"
            f"Context:\n{context}\n\n"
            "Return a practical answer in 3-5 bullet points with source filenames."
        )

        try:
            result = await self._gateway.chat(
                model=self._settings.planner_model,
                messages=[
                    {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self._settings.generation_temperature,
                max_tokens=self._settings.generation_max_tokens,
                timeout_seconds=self._settings.generation_timeout_seconds,
            )
            if not result.text.strip():
                raise RuntimeError("planner returned empty text")
            return result
        except Exception:
            return ChatResult(
                text=self._fallback(question=question, route=route, retrieved=retrieved),
                done_reason="planner_fallback",
                fallback_used=True,
            )

    async def baseline(self, question: str) -> ChatResult:
        """Cheap baseline that does not use retrieved context."""
        try:
            result = await self._gateway.chat(
                model=self._settings.planner_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an assistant with no external context. If unsure, say you do not know.",
                    },
                    {"role": "user", "content": question},
                ],
                temperature=self._settings.generation_temperature,
                max_tokens=min(self._settings.generation_max_tokens, 120),
                timeout_seconds=min(self._settings.generation_timeout_seconds, 5.0),
            )
            if result.text.strip():
                return result
        except Exception:
            pass

        return ChatResult(
            text="I do not know from the provided question alone.",
            done_reason="baseline_fallback",
            fallback_used=True,
        )

    @staticmethod
    def _context_block(retrieved: list[RetrievedDoc]) -> str:
        if not retrieved:
            return "(no retrieved context)"
        lines: list[str] = []
        for idx, hit in enumerate(retrieved, start=1):
            lines.append(
                f"[{idx}] source={hit.doc.source} score={hit.score:.4f} text={hit.doc.text}"
            )
        return "\n".join(lines)

    @staticmethod
    def _fallback(question: str, route: AgentRoute, retrieved: list[RetrievedDoc]) -> str:
        if not retrieved:
            return (
                f"Fallback plan for '{question}': no reliable context retrieved. "
                "Escalate to on-call, gather logs, and consult runbooks before action."
            )

        top = retrieved[0]
        return (
            f"Fallback plan ({route.intent}/{route.severity}): prioritize guidance from "
            f"{top.doc.source}. Key context: {top.doc.text}"
        )
