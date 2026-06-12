"""Document ingestion and chunk generation."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from ask_my_docs.logging_config import logger
from ask_my_docs.text import simple_tokenize, sliding_word_chunks
from ask_my_docs.types import Chunk


def _read_document(path: Path) -> str:
    """Read a UTF-8 document as plain text."""
    return path.read_text(encoding="utf-8")


def discover_documents(docs_dir: Path) -> list[Path]:
    """Find supported source files under a root folder."""
    candidates = [*docs_dir.rglob("*.md"), *docs_dir.rglob("*.txt")]
    return sorted({path.resolve() for path in candidates})


def build_chunks_from_docs(
    docs_dir: Path,
    chunk_size_words: int,
    chunk_overlap_words: int,
) -> list[Chunk]:
    """Load documents and split into deterministic chunk records.

    Args:
        docs_dir: Root directory containing markdown/text docs.
        chunk_size_words: Chunk size measured in words.
        chunk_overlap_words: Overlap between consecutive chunks.

    Returns:
        List of chunk models ready for indexing.

    Example:
        >>> chunks = build_chunks_from_docs(Path("data/docs"), 180, 40)
        >>> chunks[0].chunk_id
        'billing_disputes::chunk-000'
    """
    docs_root = docs_dir.resolve()
    doc_paths = discover_documents(docs_root)
    if not doc_paths:
        raise FileNotFoundError(f"No .md or .txt documents found in {docs_root}")

    chunks: list[Chunk] = []
    for path in doc_paths:
        text = _read_document(path).strip()
        if not text:
            logger.warning("Skipping empty document: {}", path)
            continue

        doc_id = path.stem.lower().replace(" ", "_")
        title = path.stem.replace("_", " ").title()
        relative_path = str(path.relative_to(docs_root))

        windowed = sliding_word_chunks(text, chunk_size_words, chunk_overlap_words)
        for idx, chunk_text in enumerate(windowed):
            chunk_id = f"{doc_id}::chunk-{idx:03d}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    title=title,
                    source_path=relative_path,
                    text=chunk_text,
                    token_count=len(simple_tokenize(chunk_text)),
                )
            )

    logger.info("Ingested {} chunks from {} documents", len(chunks), len(doc_paths))
    return chunks


def chunks_to_dataframe(chunks: list[Chunk]) -> pl.DataFrame:
    """Convert chunks to a Polars DataFrame."""
    return pl.DataFrame([chunk.model_dump() for chunk in chunks])


def chunks_to_parquet(chunks: list[Chunk], output_path: Path) -> Path:
    """Persist chunk records to parquet for fast downstream loading."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    chunks_to_dataframe(chunks).write_parquet(output_path)
    logger.info("Wrote chunk parquet: {}", output_path)
    return output_path


def load_chunks_parquet(chunks_path: Path) -> pl.DataFrame:
    """Load chunk records from parquet."""
    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunk parquet not found: {chunks_path}")
    return pl.read_parquet(chunks_path)
