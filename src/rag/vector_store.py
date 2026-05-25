"""LangChain FAISS vector store helpers for RAG retrieval."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Optional, Union

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from core.tracing import traceable_span
from rag.chunker import chunk_documents
from rag.embeddings import create_embedding_model
from rag.types import resolve_path


DocumentLike = Union[Document, Mapping[str, object]]


@traceable_span(name="rag.create_vector_store", run_type="tool", tags=["rag", "vector_store"])
def create_vector_store(
    documents: Iterable[DocumentLike],
    *,
    embedding_model: Optional[Embeddings] = None,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> Optional[FAISS]:
    """Create a LangChain FAISS vector store from source documents."""

    chunks = chunk_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return create_vector_store_from_chunks(chunks, embedding_model=embedding_model)


@traceable_span(name="rag.create_vector_store_from_chunks", run_type="tool", tags=["rag", "vector_store"])
def create_vector_store_from_chunks(
    chunks: Iterable[Document],
    *,
    embedding_model: Optional[Embeddings] = None,
) -> Optional[FAISS]:
    """Create a LangChain FAISS vector store from already-split Documents."""

    chunk_list = list(chunks)
    if not chunk_list:
        return None

    embeddings = embedding_model or create_embedding_model()
    return FAISS.from_documents(chunk_list, embeddings)


@traceable_span(name="rag.save_vector_store", run_type="tool", tags=["rag", "vector_store"])
def save_vector_store(vector_store: FAISS, path: Union[Path, str]) -> None:
    """Persist a FAISS vector store using LangChain's built-in save method."""

    vector_store.save_local(str(resolve_path(path)))


@traceable_span(name="rag.load_vector_store", run_type="tool", tags=["rag", "vector_store"])
def load_vector_store(
    path: Union[Path, str],
    *,
    embedding_model: Optional[Embeddings] = None,
    allow_dangerous_deserialization: bool = True,
) -> FAISS:
    """Load a FAISS vector store using LangChain's built-in load method."""

    return FAISS.load_local(
        str(resolve_path(path)),
        embedding_model or create_embedding_model(),
        allow_dangerous_deserialization=allow_dangerous_deserialization,
    )
