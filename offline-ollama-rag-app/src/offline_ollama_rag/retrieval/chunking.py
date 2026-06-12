"""Text chunking for RAG retrieval."""

from __future__ import annotations

from offline_ollama_rag.schemas import Document, DocumentChunk


def chunk_document(document: Document, chunk_size_words: int, chunk_overlap_words: int) -> list[DocumentChunk]:
    """Split a document into overlapping word chunks."""
    words = document.text.split()
    if not words:
        return []

    step = max(1, chunk_size_words - chunk_overlap_words)
    chunks: list[DocumentChunk] = []

    for index, start in enumerate(range(0, len(words), step)):
        end = min(len(words), start + chunk_size_words)
        window = words[start:end]
        if not window:
            continue

        chunks.append(
            DocumentChunk(
                chunk_id=f"{document.source}::chunk-{index}",
                source=document.source,
                title=document.title,
                text=" ".join(window),
                start_word=start,
                end_word=end,
            )
        )

        if end >= len(words):
            break

    return chunks


def chunk_documents(
    documents: list[Document],
    chunk_size_words: int,
    chunk_overlap_words: int,
) -> list[DocumentChunk]:
    """Chunk all documents."""
    all_chunks: list[DocumentChunk] = []
    for document in documents:
        all_chunks.extend(chunk_document(document, chunk_size_words, chunk_overlap_words))
    return all_chunks
