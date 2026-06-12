"""End-to-end RAG orchestration."""

from __future__ import annotations

from pathlib import Path

from ask_my_docs.answering import build_cited_extractive_answer
from ask_my_docs.retrieval.hybrid import HybridRetriever
from ask_my_docs.types import AskResponse, RetrievedChunk


class AskMyDocsEngine:
    """Coordinates retrieval and citation-enforced answer synthesis."""

    def __init__(self, retriever: HybridRetriever, default_top_k: int = 5) -> None:
        self.retriever = retriever
        self.default_top_k = default_top_k

    @classmethod
    def from_index(
        cls,
        *,
        index_dir: Path,
        embedding_model: str | None = None,
        reranker_model: str | None = None,
        bm25_weight: float = 0.5,
        vector_weight: float = 0.5,
        rrf_k: int = 60,
        candidate_pool_size: int = 30,
        default_top_k: int = 5,
    ) -> AskMyDocsEngine:
        retriever = HybridRetriever(
            index_dir=index_dir,
            embedding_model=embedding_model,
            reranker_model=reranker_model,
            bm25_weight=bm25_weight,
            vector_weight=vector_weight,
            rrf_k=rrf_k,
            candidate_pool_size=candidate_pool_size,
        )
        return cls(retriever=retriever, default_top_k=default_top_k)

    def retrieve(self, question: str, top_k: int | None = None) -> list[RetrievedChunk]:
        return self.retriever.retrieve(question, top_k=top_k or self.default_top_k)

    def ask(self, question: str, top_k: int | None = None) -> AskResponse:
        contexts = self.retrieve(question=question, top_k=top_k)
        answer = build_cited_extractive_answer(question=question, retrieved=contexts)
        return AskResponse(question=question, answer=answer, citations=contexts)
