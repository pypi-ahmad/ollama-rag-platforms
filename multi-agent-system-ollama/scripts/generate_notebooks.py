"""Generate notebook-first tutorial for Project 6."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


def _write(path: Path, cells: list[nbf.NotebookNode]) -> None:
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, path)


def main() -> None:
    base = Path("notebooks")

    _write(
        base / "01_setup_and_model_check.ipynb",
        [
            nbf.v4.new_markdown_cell(
                "# 01 - Setup and Local Model Check\n"
                "Validate environment settings and local Ollama model availability."
            ),
            nbf.v4.new_code_cell(
                "from multi_agent_system.config import get_settings\n"
                "from multi_agent_system.ollama_client import AsyncOllamaGateway\n"
                "settings = get_settings()\n"
                "settings"
            ),
            nbf.v4.new_code_cell(
                "gateway = AsyncOllamaGateway(settings.ollama_host)\n"
                "models = await gateway.list_model_names()\n"
                "required = [settings.planner_model, settings.reviewer_model]\n"
                "{'required_models': required, 'available_subset': [m for m in required if m in models], 'n_local_models': len(models)}"
            ),
        ],
    )

    _write(
        base / "02_agent_workflow_walkthrough.ipynb",
        [
            nbf.v4.new_markdown_cell(
                "# 02 - Multi-Agent Workflow Walkthrough\n"
                "Run a single query through router, retriever, planner, and reviewer."
            ),
            nbf.v4.new_code_cell(
                "from multi_agent_system.config import get_settings\n"
                "from multi_agent_system.pipeline import run_demo\n"
                "settings = get_settings()"
            ),
            nbf.v4.new_code_cell(
                "runs = await run_demo(settings, ['When should emergency rollback be triggered for API incidents?'])\n"
                "runs[0]"
            ),
        ],
    )

    _write(
        base / "03_batch_demo_runs.ipynb",
        [
            nbf.v4.new_markdown_cell(
                "# 03 - Batch Demo Runs\n"
                "Execute multiple practical questions and inspect generated answers."
            ),
            nbf.v4.new_code_cell(
                "from multi_agent_system.config import get_settings\n"
                "from multi_agent_system.pipeline import run_demo\n"
                "settings = get_settings()\n"
                "questions = [\n"
                "  'What canary percentage is used at release start?',\n"
                "  'Where should urgent support escalations be posted?',\n"
                "  'How long are raw event logs retained?'\n"
                "]"
            ),
            nbf.v4.new_code_cell("batch_runs = await run_demo(settings, questions)\nbatch_runs"),
        ],
    )

    _write(
        base / "04_evaluation.ipynb",
        [
            nbf.v4.new_markdown_cell(
                "# 04 - Baseline vs Multi-Agent Evaluation\n"
                "Run full task evaluation and inspect quantitative metrics."
            ),
            nbf.v4.new_code_cell(
                "from multi_agent_system.config import get_settings\n"
                "from multi_agent_system.pipeline import run_evaluation\n"
                "settings = get_settings()"
            ),
            nbf.v4.new_code_cell("eval_payload = await run_evaluation(settings)\neval_payload['summary']"),
        ],
    )

    _write(
        base / "05_telemetry_and_report.ipynb",
        [
            nbf.v4.new_markdown_cell(
                "# 05 - Telemetry and Report\n"
                "Summarize trace spans and preview the generated markdown report."
            ),
            nbf.v4.new_code_cell(
                "from pathlib import Path\n"
                "import json\n"
                "from multi_agent_system.config import get_settings\n"
                "settings = get_settings()"
            ),
            nbf.v4.new_code_cell(
                "telemetry = json.loads(Path(settings.telemetry_summary_file).read_text(encoding='utf-8'))\n"
                "telemetry"
            ),
            nbf.v4.new_code_cell(
                "report_text = Path(settings.report_file).read_text(encoding='utf-8')\n"
                "print(report_text[:1500])"
            ),
        ],
    )


if __name__ == "__main__":
    main()
