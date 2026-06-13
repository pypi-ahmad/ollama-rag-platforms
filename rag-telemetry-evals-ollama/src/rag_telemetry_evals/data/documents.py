"""Knowledge document loading utilities."""

from __future__ import annotations

from pathlib import Path

from rag_telemetry_evals.schemas import Document

SUPPORTED_SUFFIXES = {".md", ".txt"}


def _extract_title(path: Path, text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip()
    return path.stem.replace("_", " ").title()


def load_documents(knowledge_dir: Path) -> list[Document]:
    """Load markdown/text documents under the knowledge directory."""
    docs: list[Document] = []
    for path in sorted(knowledge_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue

        text = path.read_text(encoding="utf-8")
        if not text.strip():
            continue

        docs.append(
            Document(
                source=path.relative_to(knowledge_dir).as_posix(),
                title=_extract_title(path, text),
                text=text.strip(),
            )
        )

    return docs
