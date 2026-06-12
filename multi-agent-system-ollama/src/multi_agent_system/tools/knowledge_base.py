"""Knowledge base loading and lexical retrieval tools."""

from __future__ import annotations

from pathlib import Path

from multi_agent_system.schemas import KnowledgeDoc, RetrievedDoc

SUPPORTED_SUFFIXES = {".md", ".txt"}


def _extract_title(path: Path, text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip()
    return path.stem.replace("_", " ").title()


def _tokenize(text: str) -> set[str]:
    tokens: set[str] = set()
    for token in text.split():
        normalized = token.strip(".,:;!?()[]{}\"'").lower()
        if normalized:
            tokens.add(normalized)
    return tokens


def load_knowledge_docs(knowledge_dir: Path) -> list[KnowledgeDoc]:
    """Load local markdown/text docs into memory."""
    docs: list[KnowledgeDoc] = []
    for path in sorted(knowledge_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue

        text = path.read_text(encoding="utf-8")
        if not text.strip():
            continue

        docs.append(
            KnowledgeDoc(
                source=path.relative_to(knowledge_dir).as_posix(),
                title=_extract_title(path, text),
                text=text.strip(),
            )
        )

    return docs


def retrieve_docs(query: str, docs: list[KnowledgeDoc], top_k: int) -> list[RetrievedDoc]:
    """Lexical-overlap retrieval over local docs."""
    query_terms = _tokenize(query)
    if not query_terms:
        return []

    scored: list[RetrievedDoc] = []
    for doc in docs:
        doc_terms = _tokenize(doc.text)
        overlap = len(query_terms & doc_terms)
        score = overlap / max(1, len(query_terms))
        scored.append(RetrievedDoc(doc=doc, score=float(score)))

    scored.sort(key=lambda item: item.score, reverse=True)
    return scored[: min(top_k, len(scored))]
