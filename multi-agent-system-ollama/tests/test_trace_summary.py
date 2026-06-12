from __future__ import annotations

import json
from pathlib import Path

from multi_agent_system.telemetry.tracer import JsonlTelemetryTracer, summarize_traces


def test_trace_summary(tmp_path: Path) -> None:
    trace_file = tmp_path / "traces.jsonl"
    summary_file = tmp_path / "summary.json"

    tracer = JsonlTelemetryTracer(trace_file)
    with tracer.span("trace-1", "route", {"intent": "incident"}):
        pass
    with tracer.span("trace-1", "retrieve", {"top_k": 4}):
        pass

    payload = summarize_traces(trace_file, summary_file)

    assert payload["n_spans"] == 2
    loaded = json.loads(summary_file.read_text(encoding="utf-8"))
    assert loaded["n_unique_traces"] == 1
