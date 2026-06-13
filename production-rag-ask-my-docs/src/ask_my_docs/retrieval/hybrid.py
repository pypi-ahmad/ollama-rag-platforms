"""Hybrid retrieval implementation (BM25 + FAISS semantic search)."""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict
from pathlib import Path

import faiss
import numpy as np
import orjson
from rank_bm25 import BM25Okapi

from ask_my_docs.models import Chunk, Document, RetrievedChunk
from ask_my_docs.settings import RetrievalConfig
from ask_my_docs.utils import ensure_dir, tokenize

_DOC_ID_SANITIZE_RE = re.compile(r"[^A-Za-z0-9_.:-]+")


def _sanitize_doc_id(raw_id: str) -> str:
    sanitized = _DOC_ID_SANITIZE_RE.sub("_", raw_id).strip("._:-")
    return sanitized or "doc"


def load_documents(docs_dir: Path) -> list[Document]:
    """Load markdown/text documents recursively from disk."""

    if not docs_dir.exists():
        raise FileNotFoundError(f"Docs directory does not exist: {docs_dir}")

    documents: list[Document] = []
    used_doc_ids: set[str] = set()
    for path in sorted(docs_dir.rglob("*")):
        if path.suffix.lower() not in {".md", ".txt"} or not path.is_file():
            continue
        relative_path = path.relative_to(docs_dir)
        raw_id = ".".join(relative_path.with_suffix("").parts)
        doc_id = _sanitize_doc_id(raw_id)
        if doc_id in used_doc_ids:
            suffix = hashlib.blake2b(
                str(relative_path).encode("utf-8"),
                digest_size=4,
            ).hexdigest()
            doc_id = f"{doc_id}.{suffix}"
        if doc_id in used_doc_ids:
            raise ValueError(f"Duplicate document id generated for path: {path}")

        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        used_doc_ids.add(doc_id)
        documents.append(
            Document(
                doc_id=doc_id,
                text=text,
                metadata={"path": str(relative_path)},
            )
        )

    if not documents:
        raise ValueError(f"No source docs found under: {docs_dir}")
    return documents


class HashingEmbedder:
    """Deterministic hashing-based embedder for offline semantic retrieval."""

    def __init__(self, dim: int) -> None:
        self._dim = dim

    def embed(self, text: str) -> np.ndarray:
        """Return an L2-normalized dense embedding."""

        vector = np.zeros(self._dim, dtype=np.float32)
        for token in tokenize(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "little") % self._dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign

        norm = float(np.linalg.norm(vector))
        if norm > 0:
            vector /= norm
        return vector


def _chunk_document(document: Document, chunk_size: int, overlap: int) -> list[Chunk]:
    words = document.text.split()
    if not words:
        return []

    step = max(chunk_size - overlap, 1)
    chunks: list[Chunk] = []
    for start in range(0, len(words), step):
        chunk_words = words[start : start + chunk_size]
        if not chunk_words:
            continue
        chunk_id = f"{document.doc_id}::chunk-{len(chunks)}"
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                doc_id=document.doc_id,
                text=" ".join(chunk_words),
                metadata={"start_word": str(start), **document.metadata},
            )
        )
        if start + chunk_size >= len(words):
            break
    return chunks


def _normalize(scores: np.ndarray) -> np.ndarray:
    if scores.size == 0:
        return scores
    min_score = float(scores.min())
    max_score = float(scores.max())
    if abs(max_score - min_score) < 1e-9:
        return np.zeros_like(scores)
    return (scores - min_score) / (max_score - min_score)


class HybridRetriever:
    """Hybrid lexical + semantic retriever persisted to disk."""

    def __init__(
        self,
        chunks: list[Chunk],
        embeddings: np.ndarray,
        bm25: BM25Okapi,
        faiss_index: faiss.Index,
        embedder: HashingEmbedder,
        config: RetrievalConfig,
    ) -> None:
        if not chunks:
            raise ValueError("Retriever requires at least one chunk")
        if embeddings.ndim != 2:
            raise ValueError(f"Expected 2D embeddings array, got shape={embeddings.shape}")
        if embeddings.shape[0] != len(chunks):
            raise ValueError(
                "Chunk/embedding mismatch: "
                f"chunk_count={len(chunks)} embedding_rows={embeddings.shape[0]}"
            )
        if embeddings.shape[1] != config.embedding_dim:
            raise ValueError(
                "Embedding dimension mismatch: "
                f"config={config.embedding_dim} embeddings={embeddings.shape[1]}"
            )
        if faiss_index.d != embeddings.shape[1]:
            raise ValueError(
                "FAISS/embedding dimension mismatch: "
                f"faiss_dim={faiss_index.d} embeddings={embeddings.shape[1]}"
            )
        if faiss_index.ntotal != len(chunks):
            raise ValueError(
                "FAISS/chunk count mismatch: "
                f"faiss_count={faiss_index.ntotal} chunk_count={len(chunks)}"
            )

        self._chunks = chunks
        self._embeddings = embeddings
        self._bm25 = bm25
        self._faiss_index = faiss_index
        self._embedder = embedder
        self._config = config

    @property
    def chunk_count(self) -> int:
        """Return total number of indexed chunks."""

        return len(self._chunks)

    @classmethod
    def from_documents(
        cls,
        documents: list[Document],
        config: RetrievalConfig,
    ) -> HybridRetriever:
        """Build retriever from source documents."""

        chunks: list[Chunk] = []
        for document in documents:
            chunks.extend(
                _chunk_document(
                    document=document,
                    chunk_size=config.chunk_size_tokens,
                    overlap=config.chunk_overlap_tokens,
                )
            )

        if not chunks:
            raise ValueError("Chunking produced zero chunks")

        embedder = HashingEmbedder(dim=config.embedding_dim)
        embeddings = np.vstack([embedder.embed(chunk.text) for chunk in chunks]).astype(np.float32)

        tokenized_chunks = [tokenize(chunk.text) for chunk in chunks]
        bm25 = BM25Okapi(tokenized_chunks)

        faiss_index = faiss.IndexFlatIP(config.embedding_dim)
        faiss_index.add(embeddings)

        return cls(
            chunks=chunks,
            embeddings=embeddings,
            bm25=bm25,
            faiss_index=faiss_index,
            embedder=embedder,
            config=config,
        )

    @classmethod
    def load(cls, index_dir: Path, config: RetrievalConfig) -> HybridRetriever:
        """Load retriever artifacts from disk."""

        chunk_path = index_dir / "chunks.jsonl"
        embedding_path = index_dir / "embeddings.npy"
        faiss_path = index_dir / "semantic.index"
        manifest_path = index_dir / "manifest.json"

        if not chunk_path.exists() or not embedding_path.exists() or not faiss_path.exists():
            raise FileNotFoundError(
                "Index artifacts not found. Run `ask-my-docs ingest` before querying/evaluating."
            )

        chunks: list[Chunk] = []
        for line in chunk_path.read_bytes().splitlines():
            payload = orjson.loads(line)
            chunks.append(
                Chunk(
                    chunk_id=str(payload["chunk_id"]),
                    doc_id=str(payload["doc_id"]),
                    text=str(payload["text"]),
                    metadata={k: str(v) for k, v in dict(payload.get("metadata", {})).items()},
                )
            )
        if not chunks:
            raise ValueError(f"No chunks found in index file: {chunk_path}")

        embeddings = np.load(embedding_path).astype(np.float32)
        if embeddings.ndim != 2:
            raise ValueError(
                f"Embeddings artifact must be a 2D array, got shape={embeddings.shape}"
            )
        if embeddings.shape[0] != len(chunks):
            raise ValueError(
                "Chunk/embedding mismatch in artifacts: "
                f"chunk_count={len(chunks)} embedding_rows={embeddings.shape[0]}"
            )

        tokenized_chunks = [tokenize(chunk.text) for chunk in chunks]
        bm25 = BM25Okapi(tokenized_chunks)
        faiss_index = faiss.read_index(str(faiss_path))
        index_embedding_dim = int(embeddings.shape[1])

        if faiss_index.d != index_embedding_dim:
            raise ValueError(
                "FAISS/embedding dimension mismatch in artifacts: "
                f"faiss_dim={faiss_index.d} embedding_dim={index_embedding_dim}"
            )
        if faiss_index.ntotal != len(chunks):
            raise ValueError(
                "FAISS/chunk mismatch in artifacts: "
                f"faiss_count={faiss_index.ntotal} chunk_count={len(chunks)}"
            )

        if manifest_path.exists():
            manifest_payload = orjson.loads(manifest_path.read_bytes())
            if not isinstance(manifest_payload, dict):
                raise ValueError(f"Invalid manifest format at {manifest_path}")

            manifest_chunk_count = manifest_payload.get("chunk_count")
            if isinstance(manifest_chunk_count, int) and manifest_chunk_count != len(chunks):
                raise ValueError(
                    "Manifest chunk_count does not match chunk artifact: "
                    f"manifest={manifest_chunk_count} chunks={len(chunks)}"
                )

            manifest_embedding_dim = manifest_payload.get("embedding_dim")
            if (
                isinstance(manifest_embedding_dim, int)
                and manifest_embedding_dim != index_embedding_dim
            ):
                raise ValueError(
                    "Manifest embedding_dim does not match embedding artifact: "
                    f"manifest={manifest_embedding_dim} embeddings={index_embedding_dim}"
                )

        if config.embedding_dim != index_embedding_dim:
            raise ValueError(
                "Configured embedding_dim does not match indexed embeddings: "
                f"config={config.embedding_dim} index={index_embedding_dim}. "
                "Update retrieval config or rebuild the index."
            )

        embedder = HashingEmbedder(dim=index_embedding_dim)

        return cls(
            chunks=chunks,
            embeddings=embeddings,
            bm25=bm25,
            faiss_index=faiss_index,
            embedder=embedder,
            config=config,
        )

    def save(self, index_dir: Path) -> None:
        """Persist index artifacts to disk."""

        ensure_dir(index_dir)

        chunk_path = index_dir / "chunks.jsonl"
        with chunk_path.open("wb") as file_obj:
            for chunk in self._chunks:
                file_obj.write(orjson.dumps(asdict(chunk)))
                file_obj.write(b"\n")

        np.save(index_dir / "embeddings.npy", self._embeddings)
        faiss.write_index(self._faiss_index, str(index_dir / "semantic.index"))

        manifest = {
            "chunk_count": len(self._chunks),
            "embedding_dim": self._config.embedding_dim,
            "lexical_weight": self._config.lexical_weight,
            "semantic_weight": self._config.semantic_weight,
        }
        (index_dir / "manifest.json").write_bytes(
            orjson.dumps(manifest, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS)
        )

    def search(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        """Perform weighted hybrid retrieval and return top-k chunks."""

        k = top_k or self._config.top_k
        if k <= 0:
            raise ValueError("top_k must be greater than zero")
        k = min(k, len(self._chunks))

        lexical_scores = np.asarray(self._bm25.get_scores(tokenize(query)), dtype=np.float32)

        query_embedding = self._embedder.embed(query)
        semantic_scores = np.zeros(shape=(len(self._chunks),), dtype=np.float32)
        search_k = min(len(self._chunks), max(k * 4, k))
        top_scores, top_indices = self._faiss_index.search(query_embedding[None, :], search_k)
        for score, index in zip(top_scores[0], top_indices[0], strict=False):
            if index >= 0:
                semantic_scores[index] = score

        lexical_norm = _normalize(lexical_scores)
        semantic_norm = _normalize(semantic_scores)
        combined = (
            self._config.lexical_weight * lexical_norm
            + self._config.semantic_weight * semantic_norm
        ).astype(np.float32)

        if k == len(self._chunks):
            candidate_indices = np.arange(len(self._chunks), dtype=np.int64)
        else:
            candidate_indices = np.argpartition(combined, -k)[-k:]

        ranked_indices = sorted(
            (int(idx) for idx in candidate_indices),
            key=lambda idx: (
                float(combined[idx]),
                float(lexical_scores[idx]),
                float(semantic_scores[idx]),
            ),
            reverse=True,
        )

        results: list[RetrievedChunk] = []
        for rank, index in enumerate(ranked_indices[:k], start=1):
            results.append(
                RetrievedChunk(
                    chunk=self._chunks[index],
                    score=float(combined[index]),
                    lexical_score=float(lexical_scores[index]),
                    semantic_score=float(semantic_scores[index]),
                    rank=rank,
                )
            )
        return results
