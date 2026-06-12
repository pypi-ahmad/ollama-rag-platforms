"""OpenTelemetry tracing configuration and JSONL export."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from threading import Lock

import orjson
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter, SpanExportResult

from ask_my_docs.utils import ensure_dir


class JsonLineSpanExporter(SpanExporter):
    """Export spans to a local JSONL file for trace inspection."""

    def __init__(self, trace_path: Path) -> None:
        self._trace_path = trace_path
        ensure_dir(self._trace_path.parent)

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        """Serialize spans and append them to disk."""

        lines: list[bytes] = []
        for span in spans:
            start_time = span.start_time or 0
            end_time = span.end_time or start_time
            payload = {
                "name": span.name,
                "trace_id": format(span.context.trace_id, "032x"),
                "span_id": format(span.context.span_id, "016x"),
                "parent_span_id": (
                    format(span.parent.span_id, "016x") if span.parent is not None else None
                ),
                "start_time_unix_ns": start_time,
                "end_time_unix_ns": end_time,
                "duration_ms": round((end_time - start_time) / 1_000_000.0, 3),
                "status": span.status.status_code.name,
                "attributes": dict(span.attributes or {}),
            }
            lines.append(orjson.dumps(payload))

        with self._trace_path.open("ab") as file_obj:
            for line in lines:
                file_obj.write(line)
                file_obj.write(b"\n")

        return SpanExportResult.SUCCESS


_CONFIG_LOCK = Lock()
_CONFIGURED = False


def configure_tracing(service_name: str, trace_path: Path) -> trace.Tracer:
    """Configure tracing once and return a named tracer."""

    global _CONFIGURED

    with _CONFIG_LOCK:
        if not _CONFIGURED:
            provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
            provider.add_span_processor(BatchSpanProcessor(JsonLineSpanExporter(trace_path)))
            trace.set_tracer_provider(provider)
            _CONFIGURED = True

    return trace.get_tracer(service_name)
