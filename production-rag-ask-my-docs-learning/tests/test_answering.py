from ask_my_docs.answering import build_cited_extractive_answer
from ask_my_docs.types import RetrievedChunk


def test_answer_includes_citations() -> None:
    chunks = [
        RetrievedChunk(
            chunk_id="doc::chunk-000",
            doc_id="doc",
            title="Doc",
            source_path="doc.md",
            text=(
                "Invoice disputes are acknowledged in 1 business day "
                "and resolved in 3 business days."
            ),
            token_count=14,
            rank=1,
            bm25_score=1.0,
            vector_score=1.0,
            hybrid_score=1.0,
            rerank_score=1.0,
        )
    ]

    answer = build_cited_extractive_answer("What is the invoice SLA?", chunks)
    assert "[1]" in answer
    assert "Sources:" in answer
