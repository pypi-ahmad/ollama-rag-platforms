"""JSONL telemetry tracer and aggregate summaries."""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from multi_agent_system.schemas import TraceSpanRecord

JsonScalar = str | int | float | bool | None


class JsonlTelemetryTracer:
    """Persist execution spans to a JSONL file."""

    def __init__(self, trace_path: Path) -> None:
        self._trace_path = trace_path
        self._trace_path.parent.mkdir(parents=True, exist_ok=True)
        if not self._trace_path.exists():
            self._trace_path.touch()

    @property
    def trace_path(self) -> Path:
        return self._trace_path

    @contextmanager
    def span(
        self,
        trace_id: str,
        span_name: str,
        attributes: dict[str, JsonScalar] | None = None,
    ) -> Iterator[dict[str, JsonScalar]]:
        """Context manager that records one span."""
        start_wall = datetime.now(tz=UTC)
        start_perf = perf_counter()
        attrs: dict[str, JsonScalar] = dict(attributes or {})
        status = "ok"

        try:
            yield attrs
        except Exception as exc:
            status = "error"
            attrs["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            end_wall = datetime.now(tz=UTC)
            latency_ms = (perf_counter() - start_perf) * 1000.0
            record = TraceSpanRecord(
                trace_id=trace_id,
                span_name=span_name,
                status="error" if status == "error" else "ok",
                start_time_utc=start_wall.isoformat(timespec="milliseconds"),
                end_time_utc=end_wall.isoformat(timespec="milliseconds"),
                latency_ms=round(latency_ms, 3),
                attributes={k: _sanitize(v) for k, v in attrs.items()},
            )
            self._append(record)

    def _append(self, record: TraceSpanRecord) -> None:
        with self._trace_path.open("a", encoding="utf-8") as handle:
            handle.write(record.model_dump_json())
            handle.write("\n")


def _sanitize(value: JsonScalar) -> JsonScalar:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _safe_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _safe_percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.quantile(np.asarray(values, dtype=np.float64), q=q))


def summarize_traces(trace_path: Path, output_path: Path) -> dict[str, Any]:
    """Aggregate spans into per-stage latency statistics."""
    records: list[TraceSpanRecord] = []
    if trace_path.exists():
        for line in trace_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            records.append(TraceSpanRecord.model_validate_json(line))

    by_span: dict[str, list[TraceSpanRecord]] = {}
    for record in records:
        by_span.setdefault(record.span_name, []).append(record)

    rows: list[dict[str, Any]] = []
    for span_name in sorted(by_span):
        bucket = by_span[span_name]
        latencies = [item.latency_ms for item in bucket]
        rows.append(
            {
                "span_name": span_name,
                "count": len(bucket),
                "error_count": sum(1 for item in bucket if item.status == "error"),
                "latency_ms_mean": round(_safe_mean(latencies), 3),
                "latency_ms_p50": round(_safe_percentile(latencies, 0.50), 3),
                "latency_ms_p95": round(_safe_percentile(latencies, 0.95), 3),
                "latency_ms_max": round(max(latencies) if latencies else 0.0, 3),
            }
        )

    payload: dict[str, Any] = {
        "generated_at_utc": datetime.now(tz=UTC).isoformat(timespec="seconds"),
        "trace_file": trace_path.as_posix(),
        "n_spans": len(records),
        "n_unique_traces": len({record.trace_id for record in records}),
        "by_span": rows,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
