"""Coordinator that orchestrates multi-agent flow end to end."""

from __future__ import annotations

from time import perf_counter

from multi_agent_system.agents.planner import PlannerAgent
from multi_agent_system.agents.retriever import RetrieverAgent
from multi_agent_system.agents.reviewer import ReviewerAgent
from multi_agent_system.agents.router import RouterAgent
from multi_agent_system.schemas import AgentRunResult
from multi_agent_system.telemetry.tracer import JsonlTelemetryTracer


class MultiAgentCoordinator:
    """Coordinate router, retriever, planner, and reviewer agents."""

    def __init__(
        self,
        router: RouterAgent,
        retriever: RetrieverAgent,
        planner: PlannerAgent,
        reviewer: ReviewerAgent,
        tracer: JsonlTelemetryTracer,
    ) -> None:
        self._router = router
        self._retriever = retriever
        self._planner = planner
        self._reviewer = reviewer
        self._tracer = tracer

    async def run(self, question: str, trace_id: str) -> AgentRunResult:
        """Run one coordinated query through all agents."""
        start = perf_counter()

        with self._tracer.span(trace_id, "route", {"question_chars": len(question)}):
            route = self._router.route(question)

        with self._tracer.span(trace_id, "retrieve", {"top_k": self._retriever.top_k}):
            retrieved = self._retriever.retrieve(question)

        with self._tracer.span(trace_id, "baseline_answer"):
            baseline = await self._planner.baseline(question)

        with self._tracer.span(trace_id, "plan_answer", {"intent": route.intent}):
            plan = await self._planner.plan(question=question, route=route, retrieved=retrieved, trace_id=trace_id)

        with self._tracer.span(trace_id, "review_answer"):
            final = await self._reviewer.review(question=question, draft_answer=plan, retrieved=retrieved)

        total_ms = (perf_counter() - start) * 1000.0

        return AgentRunResult(
            trace_id=trace_id,
            question=question,
            route=route,
            retrieved=[
                {
                    "source": item.doc.source,
                    "title": item.doc.title,
                    "score": round(item.score, 4),
                    "text": item.doc.text,
                }
                for item in retrieved
            ],
            baseline_answer=baseline.text,
            plan_answer=plan.text,
            final_answer=final.text,
            planner_fallback_used=plan.fallback_used,
            reviewer_fallback_used=final.fallback_used,
            total_latency_ms=round(total_ms, 3),
        )
