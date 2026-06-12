from ask_my_docs.retrieval.hybrid import reciprocal_rank, reciprocal_rank_fusion


def test_reciprocal_rank_none() -> None:
    assert reciprocal_rank(None, rrf_k=60) == 0.0


def test_reciprocal_rank_fusion_prefers_better_rank() -> None:
    better = reciprocal_rank_fusion(
        bm25_rank=1,
        vector_rank=5,
        bm25_weight=0.5,
        vector_weight=0.5,
        rrf_k=60,
    )
    worse = reciprocal_rank_fusion(
        bm25_rank=8,
        vector_rank=5,
        bm25_weight=0.5,
        vector_weight=0.5,
        rrf_k=60,
    )
    assert better > worse
