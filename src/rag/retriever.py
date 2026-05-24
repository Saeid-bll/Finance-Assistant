"""LangChain-native RAG retriever factories."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Union

from langchain_community.vectorstores import FAISS
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever

from rag.chunker import chunk_documents
from rag.loader import load_knowledge_base
from rag.vector_store import create_vector_store


DocumentLike = Union[Document, Mapping[str, object]]


class EmptyRetriever(BaseRetriever):
    """LangChain retriever that returns no documents for an empty knowledge base."""

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        return []


def create_retriever(
    vector_store: Optional[FAISS],
    *,
    top_k: int = 4,
    search_type: str = "similarity",
    search_kwargs: Optional[dict[str, Any]] = None,
) -> BaseRetriever:
    """Create a LangChain retriever from a FAISS vector store."""

    if top_k <= 0:
        raise ValueError("top_k must be positive")

    if vector_store is None:
        return EmptyRetriever()

    kwargs = dict(search_kwargs or {})
    kwargs.setdefault("k", top_k)
    return vector_store.as_retriever(search_type=search_type, search_kwargs=kwargs)


def create_retriever_from_documents(
    documents: Iterable[DocumentLike],
    *,
    embedding_model: Optional[Embeddings] = None,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
    top_k: int = 4,
    search_type: str = "similarity",
    search_kwargs: Optional[dict[str, Any]] = None,
) -> BaseRetriever:
    """Create a LangChain retriever from source documents."""

    if top_k <= 0:
        raise ValueError("top_k must be positive")

    vector_store = create_vector_store(
        documents,
        embedding_model=embedding_model,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return create_retriever(
        vector_store,
        top_k=top_k,
        search_type=search_type,
        search_kwargs=search_kwargs,
    )


def create_retriever_from_knowledge_base(
    directory: Union[Path, str],
    *,
    embedding_model: Optional[Embeddings] = None,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
    top_k: int = 4,
    search_type: str = "similarity",
    search_kwargs: Optional[dict[str, Any]] = None,
) -> BaseRetriever:
    """Create a LangChain retriever from a markdown knowledge-base directory."""

    return create_retriever_from_documents(
        load_knowledge_base(directory),
        embedding_model=embedding_model,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        top_k=top_k,
        search_type=search_type,
        search_kwargs=search_kwargs,
    )


def split_documents_for_retrieval(
    documents: Iterable[DocumentLike],
    *,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> list[Document]:
    """Expose the exact chunking path used before creating a retriever."""

    return chunk_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
