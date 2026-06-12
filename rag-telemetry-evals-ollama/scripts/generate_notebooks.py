"""Generate tutorial notebooks for the project."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


def _write_notebook(path: Path, cells: list[nbf.NotebookNode]) -> None:
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, path)


def main() -> None:
    base = Path("notebooks")

    _write_notebook(
        base / "01_setup_and_model_check.ipynb",
        [
            nbf.v4.new_markdown_cell(
                "# 01 - Setup and Local Model Check\n"
                "This notebook verifies local Ollama connectivity and required models."
            ),
            nbf.v4.new_code_cell(
                "from rag_telemetry_evals.config import get_settings\n"
                "from rag_telemetry_evals.ollama_client import AsyncOllamaGateway\n"
                "settings = get_settings()\n"
                "settings"
            ),
            nbf.v4.new_code_cell(
                "gateway = AsyncOllamaGateway(settings.ollama_host)\n"
                "models = await gateway.list_model_names()\n"
                "required = {settings.chat_model, settings.embedding_model}\n"
                "{'required_models': sorted(required), 'available_subset': sorted([m for m in models if m in required])}"
            ),
            nbf.v4.new_code_cell(
                "await gateway.ensure_required_models(settings.chat_model, settings.embedding_model)\n"
                "'Model check passed'"
            ),
        ],
    )

    _write_notebook(
        base / "02_index_build_tutorial.ipynb",
        [
            nbf.v4.new_markdown_cell(
                "# 02 - Build Vector Index\n"
                "Load docs, chunk, embed, and persist index artifacts with telemetry spans."
            ),
            nbf.v4.new_code_cell(
                "from rag_telemetry_evals.config import get_settings\n"
                "from rag_telemetry_evals.ollama_client import AsyncOllamaGateway\n"
                "from rag_telemetry_evals.pipeline import build_index\n"
                "from rag_telemetry_evals.telemetry.tracer import JsonlTelemetryTracer\n"
                "settings = get_settings()\n"
                "gateway = AsyncOllamaGateway(settings.ollama_host)\n"
                "tracer = JsonlTelemetryTracer(settings.trace_file)"
            ),
            nbf.v4.new_code_cell("index_info = await build_index(settings, gateway, tracer)\nindex_info"),
        ],
    )

    _write_notebook(
        base / "03_question_answering_and_traces.ipynb",
        [
            nbf.v4.new_markdown_cell(
                "# 03 - Baseline vs RAG Q&A + Raw Traces\n"
                "Run one question in both modes and inspect the raw JSONL telemetry records."
            ),
            nbf.v4.new_code_cell(
                "from pathlib import Path\n"
                "from rag_telemetry_evals.config import get_settings\n"
                "from rag_telemetry_evals.pipeline import answer_question\n"
                "settings = get_settings()\n"
                "question = 'When must emergency rollback happen for API incidents?'"
            ),
            nbf.v4.new_code_cell(
                "baseline = await answer_question(settings, question=question, use_rag=False)\n"
                "baseline['answer']"
            ),
            nbf.v4.new_code_cell(
                "rag = await answer_question(settings, question=question, use_rag=True)\n"
                "{'answer': rag['answer'], 'retrieved_sources': [r['source'] for r in rag['retrieved']]}"
            ),
            nbf.v4.new_code_cell(
                "trace_lines = Path(settings.trace_file).read_text(encoding='utf-8').splitlines()\n"
                "trace_lines[-5:]"
            ),
        ],
    )

    _write_notebook(
        base / "04_evaluation_tutorial.ipynb",
        [
            nbf.v4.new_markdown_cell(
                "# 04 - Full Evaluation\n"
                "Run evaluation across all questions and inspect the metric summary."
            ),
            nbf.v4.new_code_cell(
                "from pathlib import Path\n"
                "import json\n"
                "from rag_telemetry_evals.config import get_settings\n"
                "from rag_telemetry_evals.pipeline import evaluate\n"
                "settings = get_settings()"
            ),
            nbf.v4.new_code_cell("eval_payload = await evaluate(settings)\neval_payload['summary']"),
            nbf.v4.new_code_cell(
                "summary = json.loads(Path(eval_payload['summary_path']).read_text(encoding='utf-8'))\n"
                "summary"
            ),
        ],
    )

    _write_notebook(
        base / "05_telemetry_analysis.ipynb",
        [
            nbf.v4.new_markdown_cell(
                "# 05 - Telemetry Analysis\n"
                "Aggregate traces into per-span latency stats and inspect the generated report."
            ),
            nbf.v4.new_code_cell(
                "from pathlib import Path\n"
                "import json\n"
                "from rag_telemetry_evals.config import get_settings\n"
                "from rag_telemetry_evals.telemetry.tracer import summarize_traces\n"
                "settings = get_settings()"
            ),
            nbf.v4.new_code_cell(
                "telemetry = summarize_traces(settings.trace_file, settings.telemetry_summary_file)\n"
                "telemetry"
            ),
            nbf.v4.new_code_cell(
                "report_text = Path(settings.report_file).read_text(encoding='utf-8')\n"
                "print(report_text[:1400])"
            ),
        ],
    )


if __name__ == "__main__":
    main()
