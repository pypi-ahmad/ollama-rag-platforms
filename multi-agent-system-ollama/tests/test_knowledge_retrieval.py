from __future__ import annotations

from multi_agent_system.schemas import KnowledgeDoc
from multi_agent_system.tools.knowledge_base import retrieve_docs


def test_retrieve_docs_ranks_expected_source() -> None:
    docs = [
        KnowledgeDoc(source="a.md", title="A", text="rollback at 1200 ms"),
        KnowledgeDoc(source="b.md", title="B", text="release tuesday 16:00 utc"),
    ]

    hits = retrieve_docs("When rollback at 1200 ms?", docs, top_k=1)

    assert len(hits) == 1
    assert hits[0].doc.source == "a.md"
