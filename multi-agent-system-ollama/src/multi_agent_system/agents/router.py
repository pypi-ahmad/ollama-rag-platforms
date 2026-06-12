"""Routing agent for intent and severity classification."""

from __future__ import annotations

from multi_agent_system.schemas import AgentRoute


class RouterAgent:
    """Classify query into operational intent and severity."""

    def route(self, question: str) -> AgentRoute:
        q = question.lower()

        if any(token in q for token in ["incident", "outage", "latency", "rollback"]):
            if any(token in q for token in ["outage", "rollback"]):
                return AgentRoute(
                    intent="incident", severity="S1", reason="incident keywords matched"
                )
            return AgentRoute(intent="incident", severity="S2", reason="incident keywords matched")

        if any(token in q for token in ["release", "canary", "deploy"]):
            return AgentRoute(intent="release", severity="NA", reason="release keywords matched")

        if any(token in q for token in ["support", "sla", "ticket", "escalation"]):
            return AgentRoute(intent="support", severity="NA", reason="support keywords matched")

        if any(token in q for token in ["retention", "governance", "pii", "backup"]):
            return AgentRoute(intent="governance", severity="NA", reason="governance keywords matched")

        return AgentRoute(intent="general", severity="NA", reason="no strong route keywords")
