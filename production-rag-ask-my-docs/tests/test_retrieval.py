"""Unit tests for hybrid retrieval artifacts and document loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from ask_my_docs.retrieval.hybrid import HybridRetriever, load_documents
from ask_my_docs.settings import RetrievalConfig


def test_load_documents_generates_unique_doc_ids_for_duplicate_stems(tmp_path: Path) -> None:
    docs_root = tmp_path / "docs"
    (docs_root / "a").mkdir(parents=True)
    (docs_root / "b").mkdir(parents=True)
    (docs_root / "a" / "policy.md").write_text("policy A", encoding="utf-8")
    (docs_root / "b" / "policy.md").write_text("policy B", encoding="utf-8")

    documents = load_documents(docs_root)
    doc_ids = [doc.doc_id for doc in documents]

    assert len(doc_ids) == len(set(doc_ids))
    assert set(doc_ids) == {"a.policy", "b.policy"}


def test_load_retriever_rejects_embedding_dimension_mismatch(tmp_path: Path) -> None:
    docs_root = tmp_path / "docs"
    docs_root.mkdir(parents=True)
    (docs_root / "one.md").write_text("hello world from retrieval", encoding="utf-8")

    documents = load_documents(docs_root)
    index_dir = tmp_path / "index"
    ingest_config = RetrievalConfig(embedding_dim=16, chunk_size_tokens=32, chunk_overlap_tokens=0)
    retriever = HybridRetriever.from_documents(documents=documents, config=ingest_config)
    retriever.save(index_dir)

    with pytest.raises(ValueError, match="Configured embedding_dim does not match"):
        HybridRetriever.load(
            index_dir=index_dir,
            config=RetrievalConfig(embedding_dim=8, chunk_size_tokens=32, chunk_overlap_tokens=0),
        )
