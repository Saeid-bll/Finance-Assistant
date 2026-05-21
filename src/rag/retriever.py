"""High-level RAG retriever."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Union

from rag.loader import load_knowledge_base
from rag.types import RetrievalResult
from rag.vector_store import VectorStore


class Retriever:
    """Query interface over the local vector store."""

    def __init__(self, vector_store: VectorStore) -> None:
        self.vector_store = vector_store

    @classmethod
    def from_documents(
        cls,
        documents: Iterable[dict],
        *,
        chunk_size: int = 800,
        chunk_overlap: int = 120,
    ) -> "Retriever":
        return cls(
            VectorStore.from_documents(
                documents,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        )

    @classmethod
    def from_knowledge_base(
        cls,
        directory: Union[Path, str],
        *,
        chunk_size: int = 800,
        chunk_overlap: int = 120,
    ) -> "Retriever":
        return cls.from_documents(
            load_knowledge_base(directory),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 4,
        min_score: float = 0.0,
    ) -> List[RetrievalResult]:
        return self.vector_store.search(query, top_k=top_k, min_score=min_score)
