"""Document chunking utilities for RAG retrieval."""

from __future__ import annotations

from typing import Iterable, List, Mapping, Union

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownTextSplitter

from core.tracing import traceable_span
from rag.types import CHUNK_ID_KEY, CHUNK_INDEX_KEY, SOURCE_ID_KEY, normalize_document


DocumentLike = Union[Document, Mapping[str, object]]


@traceable_span(name="rag.chunk_document", run_type="tool", tags=["rag", "chunker"])
def chunk_document(
    document: DocumentLike,
    *,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> List[Document]:
    """Split one document into LangChain Document chunks with preserved metadata."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    source_document = normalize_document(document)
    if not source_document.page_content.strip():
        return []

    splitter = MarkdownTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    split_documents = splitter.split_documents([source_document])
    source_id = str(source_document.metadata[SOURCE_ID_KEY])

    chunks: List[Document] = []
    for chunk_index, chunk in enumerate(split_documents):
        chunk_id = f"{source_id}:{chunk_index}"
        metadata = dict(chunk.metadata)
        metadata[CHUNK_INDEX_KEY] = chunk_index
        metadata[CHUNK_ID_KEY] = chunk_id
        chunks.append(
            Document(
                page_content=chunk.page_content.strip(),
                metadata=metadata,
                id=chunk_id,
            )
        )

    return chunks


@traceable_span(name="rag.chunk_documents", run_type="tool", tags=["rag", "chunker"])
def chunk_documents(
    documents: Iterable[DocumentLike],
    *,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> List[Document]:
    """Split multiple documents into chunks."""

    chunks: List[Document] = []
    for document in documents:
        chunks.extend(
            chunk_document(
                document,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        )
    return chunks
