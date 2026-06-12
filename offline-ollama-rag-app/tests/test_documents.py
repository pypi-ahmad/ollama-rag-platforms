from __future__ import annotations

from pathlib import Path

from offline_ollama_rag.data.documents import load_documents


def test_load_documents(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("# A\nhello", encoding="utf-8")
    (tmp_path / "b.txt").write_text("world", encoding="utf-8")
    (tmp_path / "c.json").write_text("{}", encoding="utf-8")

    docs = load_documents(tmp_path)

    assert len(docs) == 2
    assert docs[0].title
