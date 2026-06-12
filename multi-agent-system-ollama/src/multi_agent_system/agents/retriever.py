"""Retriever agent over local knowledge base."""

from __future__ import annotations

from multi_agent_system.schemas import KnowledgeDoc, RetrievedDoc
from multi_agent_system.tools.knowledge_base import retrieve_docs


class RetrieverAgent:
    """Retrieve top relevant docs for a question."""

    def __init__(self, docs: list[KnowledgeDoc], top_k: int) -> None:
        self._docs = docs
        self._top_k = top_k

    @property
    def top_k(self) -> int:
        return self._top_k

    def retrieve(self, question: str) -> list[RetrievedDoc]:
        return retrieve_docs(query=question, docs=self._docs, top_k=self._top_k)
