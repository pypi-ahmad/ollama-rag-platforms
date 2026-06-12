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
                "from offline_ollama_rag.config import get_settings\n"
                "from offline_ollama_rag.ollama_client import AsyncOllamaGateway\n"
                "settings = get_settings()\n"
                "settings"
            ),
            nbf.v4.new_code_cell(
                "gateway = AsyncOllamaGateway(settings.ollama_host)\n"
                "models = await gateway.list_model_names()\n"
                "required = {settings.chat_model, settings.embedding_model}\n"
                "{'required_models': required, 'available_subset': sorted([m for m in models if m in required])}"
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
                "This notebook loads local docs, chunks them, embeds chunks, and saves index artifacts."
            ),
            nbf.v4.new_code_cell(
                "from offline_ollama_rag.config import get_settings\n"
                "from offline_ollama_rag.ollama_client import AsyncOllamaGateway\n"
                "from offline_ollama_rag.pipeline import build_index\n"
                "settings = get_settings()\n"
                "gateway = AsyncOllamaGateway(settings.ollama_host)"
            ),
            nbf.v4.new_code_cell("index_info = await build_index(settings, gateway)\nindex_info"),
        ],
    )

    _write_notebook(
        base / "03_question_answering_tutorial.ipynb",
        [
            nbf.v4.new_markdown_cell(
                "# 03 - Baseline vs RAG Q&A\n"
                "Run one question in baseline mode and one in RAG mode."
            ),
            nbf.v4.new_code_cell(
                "from offline_ollama_rag.config import get_settings\n"
                "from offline_ollama_rag.pipeline import answer_question\n"
                "settings = get_settings()\n"
                "question = 'How long are raw event logs retained?'"
            ),
            nbf.v4.new_code_cell(
                "baseline = await answer_question(settings, question=question, use_rag=False)\n"
                "baseline"
            ),
            nbf.v4.new_code_cell("rag = await answer_question(settings, question=question, use_rag=True)\nrag"),
        ],
    )

    _write_notebook(
        base / "04_evaluation_and_report.ipynb",
        [
            nbf.v4.new_markdown_cell(
                "# 04 - Full Evaluation and Report\n"
                "Evaluate baseline vs RAG and inspect summary/report artifacts."
            ),
            nbf.v4.new_code_cell(
                "from pathlib import Path\n"
                "from offline_ollama_rag.config import get_settings\n"
                "from offline_ollama_rag.pipeline import evaluate\n"
                "settings = get_settings()"
            ),
            nbf.v4.new_code_cell("eval_payload = await evaluate(settings)\neval_payload['summary']"),
            nbf.v4.new_code_cell(
                "report_text = Path(eval_payload['report_path']).read_text(encoding='utf-8')\n"
                "print(report_text[:1200])"
            ),
        ],
    )


if __name__ == "__main__":
    main()
