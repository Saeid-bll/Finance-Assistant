"""Shared RAG data structures."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union


@dataclass(frozen=True)
class KnowledgeDocument:
    """A source document loaded from the financial knowledge base."""

    source_id: str
    title: str
    content: str
    url: Optional[str] = None
    path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DocumentChunk:
    """A chunk of a source document with source metadata preserved."""

    chunk_id: str
    source_id: str
    title: str
    content: str
    chunk_index: int
    url: Optional[str] = None
    path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RetrievalResult:
    """A ranked retrieval result returned by the vector store."""

    chunk_id: str
    source_id: str
    title: str
    content: str
    score: float
    url: Optional[str] = None
    path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def excerpt(self) -> str:
        return self.content


def slugify(value: str) -> str:
    cleaned = "".join(character.lower() if character.isalnum() else "-" for character in value)
    parts = [part for part in cleaned.split("-") if part]
    return "-".join(parts) or "document"


def normalize_document(document: Union[KnowledgeDocument, Mapping[str, Any]]) -> KnowledgeDocument:
    """Normalize dict-like fixtures and loaded files into a KnowledgeDocument."""

    if isinstance(document, KnowledgeDocument):
        return document

    source_id = str(document.get("source_id") or slugify(str(document.get("title", "document"))))
    title = str(document.get("title") or source_id.replace("-", " ").title())
    content = str(document.get("content") or "")
    url = document.get("url")
    path = document.get("path")
    metadata = dict(document.get("metadata") or {})

    if not content.strip():
        raise ValueError(f"Document {source_id} has empty content")

    return KnowledgeDocument(
        source_id=source_id,
        title=title,
        content=content.strip(),
        url=str(url) if url else None,
        path=str(path) if path else None,
        metadata=metadata,
    )


def resolve_path(value: Union[Path, str]) -> Path:
    return Path(value).expanduser().resolve()
