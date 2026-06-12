"""Execute tutorial notebooks to persist real outputs."""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbclient import NotebookClient

NOTEBOOKS = [
    "01_setup_and_model_check.ipynb",
    "02_index_build_tutorial.ipynb",
    "03_question_answering_tutorial.ipynb",
    "04_evaluation_and_report.ipynb",
]


def execute_notebook(path: Path) -> None:
    nb = nbformat.read(path, as_version=4)
    client = NotebookClient(nb, timeout=1200, kernel_name="python3")
    client.execute()
    nbformat.write(nb, path)


def main() -> None:
    base = Path("notebooks")
    for name in NOTEBOOKS:
        execute_notebook(base / name)


if __name__ == "__main__":
    main()
