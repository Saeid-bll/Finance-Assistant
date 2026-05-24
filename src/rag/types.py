"""Shared RAG helpers built around LangChain documents."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Union

from langchain_core.documents import Document


SOURCE_ID_KEY = "source_id"
TITLE_KEY = "title"
URL_KEY = "url"
PATH_KEY = "path"
SOURCE_KEY = "source"
CHUNK_INDEX_KEY = "chunk_index"
CHUNK_ID_KEY = "chunk_id"


def slugify(value: str) -> str:
    cleaned = "".join(character.lower() if character.isalnum() else "-" for character in value)
    parts = [part for part in cleaned.split("-") if part]
    return "-".join(parts) or "document"


def _metadata_value(metadata: Mapping[str, Any], key: str) -> str | None:
    value = metadata.get(key)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def normalize_document(document: Union[Document, Mapping[str, Any]]) -> Document:
    """Normalize project fixtures and loaded content into a LangChain Document."""

    if isinstance(document, Document):
        content = document.page_content.strip()
        metadata = dict(document.metadata or {})
        source_id = (
            _metadata_value(metadata, SOURCE_ID_KEY)
            or _metadata_value(metadata, SOURCE_KEY)
            or str(document.id or "").strip()
            or slugify(_metadata_value(metadata, TITLE_KEY) or "document")
        )
        title = _metadata_value(metadata, TITLE_KEY) or source_id.replace("-", " ").title()
        metadata[SOURCE_ID_KEY] = source_id
        metadata[TITLE_KEY] = title
        if document.id:
            doc_id = str(document.id)
        else:
            doc_id = str(metadata.get(CHUNK_ID_KEY) or source_id)
    else:
        raw_metadata = document.get("metadata") or {}
        if not isinstance(raw_metadata, Mapping):
            raise ValueError("Document metadata must be a mapping")

        metadata = dict(raw_metadata)
        content = str(document.get("page_content") or document.get("content") or "").strip()
        source_id = str(
            document.get(SOURCE_ID_KEY)
            or metadata.get(SOURCE_ID_KEY)
            or document.get(SOURCE_KEY)
            or metadata.get(SOURCE_KEY)
            or slugify(str(document.get(TITLE_KEY) or metadata.get(TITLE_KEY) or "document"))
        ).strip()
        title = str(
            document.get(TITLE_KEY)
            or metadata.get(TITLE_KEY)
            or source_id.replace("-", " ").title()
        ).strip()
        metadata.update(
            {
                SOURCE_ID_KEY: source_id,
                TITLE_KEY: title,
            }
        )
        for key in (URL_KEY, PATH_KEY, SOURCE_KEY):
            value = document.get(key) or metadata.get(key)
            if value:
                metadata[key] = str(value)
        doc_id = str(document.get("id") or metadata.get(CHUNK_ID_KEY) or source_id)

    if not content.strip():
        raise ValueError(f"Document {source_id} has empty content")

    return Document(page_content=content, metadata=metadata, id=doc_id)


def resolve_path(value: Union[Path, str]) -> Path:
    return Path(value).expanduser().resolve()
