from __future__ import annotations

from rag_telemetry_evals.retrieval.chunking import chunk_document
from rag_telemetry_evals.schemas import Document


def test_chunk_document_overlap() -> None:
    doc = Document(
        source="x.md",
        title="X",
        text=" ".join(f"w{i}" for i in range(25)),
    )

    chunks = chunk_document(doc, chunk_size_words=10, chunk_overlap_words=3)

    assert len(chunks) == 4
    assert chunks[0].start_word == 0
    assert chunks[1].start_word == 7
    assert chunks[-1].end_word == 25
