"""FastAPI service for Ask My Docs."""

from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ask_my_docs.logging_config import configure_logging
from ask_my_docs.pipeline import AskMyDocsEngine
from ask_my_docs.settings import SETTINGS
from ask_my_docs.types import AskResponse


class AskRequest(BaseModel):
    """API request payload for question answering."""

    question: str = Field(min_length=3)
    top_k: int | None = Field(default=None, ge=1, le=20)


@lru_cache(maxsize=1)
def _engine() -> AskMyDocsEngine:
    return AskMyDocsEngine.from_index(
        index_dir=SETTINGS.index_dir,
        embedding_model=SETTINGS.embedding_model,
        reranker_model=SETTINGS.reranker_model,
        bm25_weight=SETTINGS.bm25_weight,
        vector_weight=SETTINGS.vector_weight,
        rrf_k=SETTINGS.hybrid_rrf_k,
        candidate_pool_size=SETTINGS.candidate_pool_size,
        default_top_k=SETTINGS.default_top_k,
    )


def create_app() -> FastAPI:
    """Application factory to keep app import side effects minimal."""
    configure_logging()
    app = FastAPI(title="Ask My Docs", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/ask", response_model=AskResponse)
    def ask(payload: AskRequest) -> AskResponse:
        try:
            return _engine().ask(question=payload.question, top_k=payload.top_k)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app


app = create_app()
