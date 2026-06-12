from __future__ import annotations

import numpy as np

from rag_telemetry_evals.retrieval.vector_index import build_vector_index
from rag_telemetry_evals.schemas import DocumentChunk


def test_vector_search_returns_top_hit() -> None:
    chunks = [
        DocumentChunk(
            chunk_id="a",
            source="a.md",
            title="A",
            text="solar pager",
            start_word=0,
            end_word=2,
        ),
        DocumentChunk(
            chunk_id="b",
            source="b.md",
            title="B",
            text="release train",
            start_word=0,
            end_word=2,
        ),
    ]

    embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    index = build_vector_index(chunks, embeddings)

    hits = index.search(np.asarray([0.9, 0.1], dtype=np.float32), top_k=1)

    assert len(hits) == 1
    assert hits[0].chunk.chunk_id == "a"
