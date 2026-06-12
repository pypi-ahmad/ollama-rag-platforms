from __future__ import annotations

from multi_agent_system.agents.router import RouterAgent


def test_router_detects_incident() -> None:
    agent = RouterAgent()
    route = agent.route("We have outage and need rollback")

    assert route.intent == "incident"
    assert route.severity in {"S1", "S2"}
