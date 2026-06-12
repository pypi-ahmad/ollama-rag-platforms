from __future__ import annotations

import json
from pathlib import Path

from rag_telemetry_evals.telemetry.tracer import JsonlTelemetryTracer, summarize_traces


def test_trace_summary(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace.jsonl"
    summary_path = tmp_path / "summary.json"

    tracer = JsonlTelemetryTracer(trace_path)
    with tracer.span("t1", "retrieve", {"top_k": 4}):
        pass
    with tracer.span("t1", "generate_rag", {"model": "phi3.5:3.8b"}):
        pass

    payload = summarize_traces(trace_path, summary_path)

    assert payload["n_spans"] == 2
    assert payload["n_unique_traces"] == 1
    assert summary_path.exists()
    loaded = json.loads(summary_path.read_text(encoding="utf-8"))
    assert loaded["n_spans"] == 2
