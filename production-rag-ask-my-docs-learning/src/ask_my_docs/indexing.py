"""BM25 + vector index building and persistence."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Protocol

import faiss
import numpy as np
import polars as pl
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from ask_my_docs.logging_config import logger
from ask_my_docs.text import simple_tokenize


class Embedder(Protocol):
    """Small interface for embedding text batches."""

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode text to float32 normalized vectors."""


class SentenceTransformerEmbedder:
    """SentenceTransformer-based embedder for retrieval vectors."""

    def __init__(self, model_name: str) -> None:
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return np.asarray(vectors, dtype=np.float32)


def _build_bm25_model(chunks_df: pl.DataFrame) -> tuple[BM25Okapi, list[str]]:
    texts = chunks_df.get_column("text").to_list()
    tokenized = [simple_tokenize(text) for text in texts]
    bm25 = BM25Okapi(tokenized)
    chunk_ids = chunks_df.get_column("chunk_id").to_list()
    return bm25, chunk_ids


def _build_faiss_index(vectors: np.ndarray) -> faiss.IndexFlatIP:
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


def save_indexes(
    *,
    chunks_df: pl.DataFrame,
    index_dir: Path,
    bm25: BM25Okapi,
    bm25_chunk_ids: list[str],
    faiss_index: faiss.IndexFlatIP,
    embedding_model: str,
    reranker_model: str,
) -> None:
    """Write all retrieval artifacts to disk."""
    index_dir.mkdir(parents=True, exist_ok=True)

    chunks_df.write_parquet(index_dir / "chunks.parquet")
    with (index_dir / "bm25.pkl").open("wb") as f:
        pickle.dump({"bm25": bm25, "chunk_ids": bm25_chunk_ids}, f)
    faiss.write_index(faiss_index, str(index_dir / "vectors.faiss"))

    metadata = {
        "embedding_model": embedding_model,
        "reranker_model": reranker_model,
        "num_chunks": chunks_df.height,
    }
    (index_dir / "meta.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.info("Saved index artifacts in {}", index_dir)


def build_and_save_indexes(
    chunks_df: pl.DataFrame,
    index_dir: Path,
    embedding_model: str,
    reranker_model: str,
    embedder: Embedder | None = None,
) -> None:
    """Build BM25/vector indexes and persist artifacts."""
    active_embedder = embedder or SentenceTransformerEmbedder(embedding_model)
    bm25, bm25_chunk_ids = _build_bm25_model(chunks_df)
    texts = chunks_df.get_column("text").to_list()
    vectors = active_embedder.encode(texts)
    faiss_index = _build_faiss_index(vectors)

    save_indexes(
        chunks_df=chunks_df,
        index_dir=index_dir,
        bm25=bm25,
        bm25_chunk_ids=bm25_chunk_ids,
        faiss_index=faiss_index,
        embedding_model=embedding_model,
        reranker_model=reranker_model,
    )
