"""Deterministic local embeddings for development and tests."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, Iterable, Optional, Set


EmbeddingVector = Dict[str, float]


DEFAULT_STOPWORDS: Set[str] = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "what",
    "with",
}


def tokenize(text: str, *, stopwords: Optional[Set[str]] = None) -> list[str]:
    """Tokenize text into normalized terms."""

    ignored = DEFAULT_STOPWORDS if stopwords is None else stopwords
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9_]*", text.lower())
    return [token for token in tokens if token not in ignored]


class LocalEmbeddingModel:
    """Simple term-frequency embedding model suitable for local tests."""

    def embed(self, text: str) -> EmbeddingVector:
        tokens = tokenize(text)
        if not tokens:
            return {}

        counts = Counter(tokens)
        length = math.sqrt(sum(count * count for count in counts.values()))
        return {token: count / length for token, count in counts.items()}

    def embed_many(self, texts: Iterable[str]) -> list[EmbeddingVector]:
        return [self.embed(text) for text in texts]


def cosine_similarity(left: EmbeddingVector, right: EmbeddingVector) -> float:
    """Calculate cosine similarity for sparse normalized vectors."""

    if not left or not right:
        return 0.0

    if len(left) > len(right):
        left, right = right, left
    return sum(value * right.get(token, 0.0) for token, value in left.items())
