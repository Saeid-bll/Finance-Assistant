"""Document chunking utilities for RAG retrieval."""

from __future__ import annotations

from typing import Iterable, List, Union

from rag.types import DocumentChunk, KnowledgeDocument, normalize_document


def _split_words(text: str) -> List[str]:
    return [word for word in text.replace("\n", " ").split(" ") if word]


def _words_to_text(words: List[str]) -> str:
    return " ".join(words).strip()


def _overlap_words(words: List[str], overlap_chars: int) -> List[str]:
    if overlap_chars <= 0:
        return []

    kept: List[str] = []
    current_length = 0
    for word in reversed(words):
        projected = current_length + len(word) + (1 if kept else 0)
        if projected > overlap_chars:
            break
        kept.insert(0, word)
        current_length = projected
    return kept


def chunk_document(
    document: KnowledgeDocument,
    *,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> List[DocumentChunk]:
    """Split one document into word-aware chunks while preserving metadata."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    words = _split_words(document.content)
    if not words:
        return []

    chunks: List[DocumentChunk] = []
    current_words: List[str] = []

    for word in words:
        projected = _words_to_text([*current_words, word])
        if current_words and len(projected) > chunk_size:
            chunk_text = _words_to_text(current_words)
            chunks.append(
                DocumentChunk(
                    chunk_id=f"{document.source_id}:{len(chunks)}",
                    source_id=document.source_id,
                    title=document.title,
                    url=document.url,
                    path=document.path,
                    content=chunk_text,
                    chunk_index=len(chunks),
                    metadata=dict(document.metadata),
                )
            )
            current_words = [*_overlap_words(current_words, chunk_overlap), word]
        else:
            current_words.append(word)

    if current_words:
        chunks.append(
            DocumentChunk(
                chunk_id=f"{document.source_id}:{len(chunks)}",
                source_id=document.source_id,
                title=document.title,
                url=document.url,
                path=document.path,
                content=_words_to_text(current_words),
                chunk_index=len(chunks),
                metadata=dict(document.metadata),
            )
        )

    return chunks


def chunk_documents(
    documents: Iterable[Union[KnowledgeDocument, dict]],
    *,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> List[DocumentChunk]:
    """Split multiple documents into chunks."""

    chunks: List[DocumentChunk] = []
    for document in documents:
        chunks.extend(
            chunk_document(
                normalize_document(document),
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        )
    return chunks
