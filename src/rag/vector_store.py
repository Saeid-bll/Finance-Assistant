"""Local vector store for RAG retrieval."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Union

from rag.chunker import chunk_documents
from rag.embeddings import EmbeddingVector, LocalEmbeddingModel, cosine_similarity
from rag.types import DocumentChunk, RetrievalResult, resolve_path


class VectorStore:
    """A deterministic in-memory vector store with JSON persistence."""

    INDEX_FILENAME = "vector_store.json"

    def __init__(
        self,
        chunks: Optional[Sequence[DocumentChunk]] = None,
        *,
        embedding_model: LocalEmbeddingModel | None = None,
    ) -> None:
        self.embedding_model = embedding_model or LocalEmbeddingModel()
        self.chunks = list(chunks or [])
        self.embeddings: List[EmbeddingVector] = self.embedding_model.embed_many(
            chunk.content for chunk in self.chunks
        )

    @classmethod
    def from_documents(
        cls,
        documents: Iterable[dict],
        *,
        chunk_size: int = 800,
        chunk_overlap: int = 120,
    ) -> "VectorStore":
        return cls(
            chunk_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        )

    @classmethod
    def from_chunks(cls, chunks: Sequence[DocumentChunk]) -> "VectorStore":
        return cls(chunks)

    def search(self, query: str, *, top_k: int = 4, min_score: float = 0.0) -> List[RetrievalResult]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if not query.strip() or not self.chunks:
            return []

        query_embedding = self.embedding_model.embed(query)
        scored_results = []
        for chunk, embedding in zip(self.chunks, self.embeddings):
            score = cosine_similarity(query_embedding, embedding)
            if score > min_score:
                scored_results.append(
                    RetrievalResult(
                        chunk_id=chunk.chunk_id,
                        source_id=chunk.source_id,
                        title=chunk.title,
                        url=chunk.url,
                        path=chunk.path,
                        content=chunk.content,
                        score=score,
                        metadata=dict(chunk.metadata),
                    )
                )

        return sorted(scored_results, key=lambda result: result.score, reverse=True)[:top_k]

    def save(self, path: Union[Path, str]) -> None:
        index_path = resolve_path(path)
        index_path.mkdir(parents=True, exist_ok=True)
        payload = {"chunks": [chunk.to_dict() for chunk in self.chunks]}
        (index_path / self.INDEX_FILENAME).write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Union[Path, str]) -> "VectorStore":
        index_path = resolve_path(path) / cls.INDEX_FILENAME
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        chunks = [DocumentChunk(**chunk) for chunk in payload.get("chunks", [])]
        return cls(chunks)
