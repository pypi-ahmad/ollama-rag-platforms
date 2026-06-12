"""Unit tests for DuckDB-backed metric summaries."""

from pathlib import Path

from ask_my_docs.observability.metrics_store import MetricsStore, RequestMetricRecord


def test_metrics_summary_reports_p50_and_p95(tmp_path: Path) -> None:
    store = MetricsStore(db_path=tmp_path / "metrics.duckdb")

    for idx, latency in enumerate([10.0, 20.0, 30.0, 40.0, 100.0], start=1):
        store.record(
            RequestMetricRecord(
                request_id=f"r-{idx}",
                trace_id=None,
                model_name="gemma4:12b",
                question="q",
                latency_ms=latency,
                retrieval_latency_ms=latency * 0.3,
                generation_latency_ms=latency * 0.7,
                prompt_tokens=100,
                completion_tokens=40,
                total_tokens=140,
                estimated_cost_usd=0.0001,
                retrieval_recall_at_k=1.0,
                answer_f1=0.8,
                exact_match=1.0,
                timestamp_utc=f"2026-01-01T00:00:0{idx}Z",
            )
        )

    summary = store.summarize()

    assert summary["request_count"] == 5.0
    assert 28.0 <= summary["latency_p50_ms"] <= 32.0
    assert summary["latency_p95_ms"] >= 80.0
