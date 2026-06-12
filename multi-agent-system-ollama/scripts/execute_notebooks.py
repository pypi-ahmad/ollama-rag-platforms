"""Execute all tutorial notebooks to persist real outputs."""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbclient import NotebookClient

NOTEBOOKS = [
    "01_setup_and_model_check.ipynb",
    "02_agent_workflow_walkthrough.ipynb",
    "03_batch_demo_runs.ipynb",
    "04_evaluation.ipynb",
    "05_telemetry_and_report.ipynb",
]


def execute(path: Path) -> None:
    nb = nbformat.read(path, as_version=4)
    client = NotebookClient(nb, timeout=1800, kernel_name="python3")
    client.execute()
    nbformat.write(nb, path)


def main() -> None:
    base = Path("notebooks")
    for notebook in NOTEBOOKS:
        execute(base / notebook)


if __name__ == "__main__":
    main()
