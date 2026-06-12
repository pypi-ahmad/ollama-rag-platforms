"""Simple in-memory vector index with cosine retrieval."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from offline_ollama_rag.schemas import DocumentChunk, RetrievedChunk


@dataclass
class VectorIndex:
    """Embeddings matrix aligned with chunk metadata."""

    chunks: list[DocumentChunk]
    embeddings: np.ndarray

    def search(self, query_embedding: np.ndarray, top_k: int) -> list[RetrievedChunk]:
        """Return top-k chunks by cosine similarity."""
        if self.embeddings.size == 0 or not self.chunks:
            return []

        query = query_embedding.reshape(-1).astype(np.float32)
        matrix = self.embeddings.astype(np.float32)

        query_norm = np.linalg.norm(query) + 1e-12
        matrix_norm = np.linalg.norm(matrix, axis=1) + 1e-12
        scores = (matrix @ query) / (matrix_norm * query_norm)

        k = min(top_k, len(self.chunks))
        top_indices = np.argsort(scores)[::-1][:k]

        return [
            RetrievedChunk(chunk=self.chunks[int(idx)], score=float(scores[int(idx)]))
            for idx in top_indices
        ]


def build_vector_index(chunks: list[DocumentChunk], embeddings: np.ndarray) -> VectorIndex:
    """Build a vector index and verify shape alignment."""
    if len(chunks) != len(embeddings):
        raise ValueError(
            f"chunks ({len(chunks)}) and embeddings ({len(embeddings)}) must have equal length"
        )

    if len(chunks) == 0:
        return VectorIndex(chunks=[], embeddings=np.zeros((0, 0), dtype=np.float32))

    return VectorIndex(chunks=chunks, embeddings=embeddings.astype(np.float32))


def save_vector_index(index: VectorIndex, embeddings_path: Path, chunks_path: Path) -> None:
    """Persist index artifacts to disk."""
    embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    chunks_path.parent.mkdir(parents=True, exist_ok=True)

    np.save(embeddings_path, index.embeddings)
    payload = [chunk.model_dump() for chunk in index.chunks]
    chunks_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_vector_index(embeddings_path: Path, chunks_path: Path) -> VectorIndex:
    """Load index artifacts from disk."""
    embeddings = np.load(embeddings_path)
    chunk_payload = json.loads(chunks_path.read_text(encoding="utf-8"))
    chunks = [DocumentChunk.model_validate(item) for item in chunk_payload]
    return build_vector_index(chunks=chunks, embeddings=embeddings)
