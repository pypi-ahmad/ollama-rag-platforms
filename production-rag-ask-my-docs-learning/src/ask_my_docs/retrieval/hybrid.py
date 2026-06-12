"""Hybrid retriever: BM25 + dense vectors + cross-encoder reranking."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import cast

import faiss
import numpy as np
import polars as pl
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from ask_my_docs.text import simple_tokenize
from ask_my_docs.types import RetrievedChunk


def reciprocal_rank(rank: int | None, rrf_k: int) -> float:
    """RRF contribution for one rank list entry."""
    return 0.0 if rank is None else 1.0 / (rrf_k + rank)


def reciprocal_rank_fusion(
    *,
    bm25_rank: int | None,
    vector_rank: int | None,
    bm25_weight: float,
    vector_weight: float,
    rrf_k: int,
) -> float:
    """Weighted RRF fusion score."""
    return bm25_weight * reciprocal_rank(bm25_rank, rrf_k) + vector_weight * reciprocal_rank(
        vector_rank, rrf_k
    )


class HybridRetriever:
    """Retrieve and rerank chunks from persisted index artifacts."""

    def __init__(
        self,
        index_dir: Path,
        embedding_model: str | None = None,
        reranker_model: str | None = None,
        bm25_weight: float = 0.5,
        vector_weight: float = 0.5,
        rrf_k: int = 60,
        candidate_pool_size: int = 30,
    ) -> None:
        self.index_dir = index_dir
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight
        self.rrf_k = rrf_k
        self.candidate_pool_size = candidate_pool_size
        self._load_artifacts(embedding_model, reranker_model)

    def _load_artifacts(self, embedding_model: str | None, reranker_model: str | None) -> None:
        chunks_path = self.index_dir / "chunks.parquet"
        bm25_path = self.index_dir / "bm25.pkl"
        faiss_path = self.index_dir / "vectors.faiss"
        meta_path = self.index_dir / "meta.json"

        if not chunks_path.exists() or not bm25_path.exists() or not faiss_path.exists():
            raise FileNotFoundError(
                f"Index artifacts missing in {self.index_dir}. Run build-index first."
            )

        self.chunks_df = pl.read_parquet(chunks_path)
        with bm25_path.open("rb") as f:
            payload = pickle.load(f)
        self.bm25: BM25Okapi = payload["bm25"]
        self.bm25_chunk_ids: list[str] = payload["chunk_ids"]
        self.faiss_index = faiss.read_index(str(faiss_path))

        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}

        embed_model_name = embedding_model or str(
            meta.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
        )
        rerank_model_name = reranker_model or str(
            meta.get("reranker_model", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        )

        self.embedder = SentenceTransformer(embed_model_name)
        self.reranker = CrossEncoder(rerank_model_name)

        self.chunk_id_to_row_idx = {
            chunk_id: idx
            for idx, chunk_id in enumerate(self.chunks_df.get_column("chunk_id").to_list())
        }

    def _bm25_rankings(self, query: str, top_n: int) -> tuple[list[int], dict[int, float]]:
        scores = self.bm25.get_scores(simple_tokenize(query))
        best_idxs = np.argsort(scores)[::-1][:top_n]

        ranked_rows: list[int] = []
        score_map: dict[int, float] = {}
        for bm25_idx in best_idxs:
            chunk_id = self.bm25_chunk_ids[int(bm25_idx)]
            row_idx = self.chunk_id_to_row_idx[chunk_id]
            ranked_rows.append(row_idx)
            score_map[row_idx] = float(scores[int(bm25_idx)])
        return ranked_rows, score_map

    def _vector_rankings(self, query: str, top_n: int) -> tuple[list[int], dict[int, float]]:
        query_vec = self.embedder.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
        ).astype(np.float32)
        scores, idxs = self.faiss_index.search(query_vec, top_n)

        ranked_rows: list[int] = []
        score_map: dict[int, float] = {}
        for raw_idx, score in zip(idxs[0], scores[0], strict=True):
            if raw_idx < 0:
                continue
            row_idx = int(raw_idx)
            ranked_rows.append(row_idx)
            score_map[row_idx] = float(score)
        return ranked_rows, score_map

    def _row_to_chunk(
        self,
        row_idx: int,
        *,
        rank: int,
        bm25_score: float | None,
        vector_score: float | None,
        hybrid_score: float | None,
        rerank_score: float | None,
    ) -> RetrievedChunk:
        row = self.chunks_df.row(row_idx, named=True)
        return RetrievedChunk(
            chunk_id=str(row["chunk_id"]),
            doc_id=str(row["doc_id"]),
            title=str(row["title"]),
            source_path=str(row["source_path"]),
            text=str(row["text"]),
            token_count=int(row["token_count"]),
            rank=rank,
            bm25_score=bm25_score,
            vector_score=vector_score,
            hybrid_score=hybrid_score,
            rerank_score=rerank_score,
        )

    def vector_only(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """Dense retrieval baseline used in evaluation comparisons."""
        vector_rows, vector_scores = self._vector_rankings(query, top_k)
        return [
            self._row_to_chunk(
                row_idx,
                rank=rank,
                bm25_score=None,
                vector_score=vector_scores.get(row_idx),
                hybrid_score=None,
                rerank_score=None,
            )
            for rank, row_idx in enumerate(vector_rows[:top_k], start=1)
        ]

    def retrieve_with_traces(self, query: str, top_k: int) -> dict[str, object]:
        """Return intermediate scores useful for learning/debugging."""
        candidate_n = max(self.candidate_pool_size, top_k)

        bm25_rows, bm25_scores = self._bm25_rankings(query, candidate_n)
        vector_rows, vector_scores = self._vector_rankings(query, candidate_n)

        bm25_rank_map = {row_idx: rank for rank, row_idx in enumerate(bm25_rows, start=1)}
        vector_rank_map = {row_idx: rank for rank, row_idx in enumerate(vector_rows, start=1)}

        candidate_rows = list(set(bm25_rows) | set(vector_rows))

        fusion_rows: list[tuple[int, float]] = []
        for row_idx in candidate_rows:
            score = reciprocal_rank_fusion(
                bm25_rank=bm25_rank_map.get(row_idx),
                vector_rank=vector_rank_map.get(row_idx),
                bm25_weight=self.bm25_weight,
                vector_weight=self.vector_weight,
                rrf_k=self.rrf_k,
            )
            fusion_rows.append((row_idx, score))

        fusion_rows.sort(key=lambda item: item[1], reverse=True)
        shortlisted = fusion_rows[:candidate_n]

        pairs = [
            (query, str(self.chunks_df.row(row_idx, named=True)["text"]))
            for row_idx, _ in shortlisted
        ]
        rerank_scores = [
            float(score) for score in self.reranker.predict(pairs, show_progress_bar=False)
        ]

        reranked = []
        for (row_idx, hybrid_score), rerank_score in zip(shortlisted, rerank_scores, strict=True):
            reranked.append((row_idx, hybrid_score, rerank_score))

        reranked.sort(key=lambda item: item[2], reverse=True)

        top_chunks: list[RetrievedChunk] = []
        for rank, (row_idx, hybrid_score, rerank_score) in enumerate(reranked[:top_k], start=1):
            top_chunks.append(
                self._row_to_chunk(
                    row_idx,
                    rank=rank,
                    bm25_score=bm25_scores.get(row_idx),
                    vector_score=vector_scores.get(row_idx),
                    hybrid_score=hybrid_score,
                    rerank_score=rerank_score,
                )
            )

        return {
            "query": query,
            "top_chunks": top_chunks,
        }

    def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """Run hybrid retrieval + reranking and return only final top-k chunks."""
        traced = self.retrieve_with_traces(query, top_k=top_k)
        return cast(list[RetrievedChunk], traced["top_chunks"])
