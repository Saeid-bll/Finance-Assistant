"""Knowledge base loading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import yaml

from rag.types import KnowledgeDocument, resolve_path, slugify


def _split_front_matter(text: str) -> Tuple[Dict[str, Any], str]:
    if not text.startswith("---\n"):
        return {}, text

    _, remainder = text.split("---\n", 1)
    if "\n---\n" not in remainder:
        return {}, text

    raw_metadata, body = remainder.split("\n---\n", 1)
    metadata = yaml.safe_load(raw_metadata) or {}
    if not isinstance(metadata, dict):
        raise ValueError("Markdown front matter must be a mapping")
    return metadata, body


def _extract_title(body: str, fallback: str) -> str:
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()
    return fallback


def load_markdown_document(path: Union[Path, str]) -> KnowledgeDocument:
    """Load one markdown file as a KnowledgeDocument."""

    file_path = resolve_path(path)
    text = file_path.read_text(encoding="utf-8")
    metadata, body = _split_front_matter(text)
    fallback_title = file_path.stem.replace("-", " ").replace("_", " ").title()
    title = str(metadata.get("title") or _extract_title(body, fallback_title)).strip()
    source_id = str(metadata.get("source_id") or slugify(title)).strip()
    url = metadata.get("url")

    content = body.strip()
    if not content:
        raise ValueError(f"Knowledge base document is empty: {file_path}")

    return KnowledgeDocument(
        source_id=source_id,
        title=title,
        url=str(url).strip() if url else None,
        path=str(file_path),
        content=content,
        metadata={key: value for key, value in metadata.items() if key not in {"source_id", "title", "url"}},
    )


def load_knowledge_base(directory: Union[Path, str]) -> List[KnowledgeDocument]:
    """Load all markdown files from a knowledge base directory."""

    base_path = resolve_path(directory)
    if not base_path.exists():
        raise FileNotFoundError(f"Knowledge base directory not found: {base_path}")
    if not base_path.is_dir():
        raise NotADirectoryError(f"Knowledge base path must be a directory: {base_path}")

    documents = [
        load_markdown_document(path)
        for path in sorted(base_path.rglob("*.md"))
        if not path.name.startswith(".")
    ]
    return documents
