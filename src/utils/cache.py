"""Small TTL cache helpers."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class _CacheEntry:
    value: Any
    expires_at: float


class TTLCache:
    """In-memory cache with monotonic-time expiration."""

    def __init__(self, *, ttl_seconds: float = 1800) -> None:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        self.ttl_seconds = float(ttl_seconds)
        self._items: dict[str, _CacheEntry] = {}

    def get(self, key: str, default: Optional[Any] = None) -> Any:
        entry = self._items.get(key)
        if entry is None:
            return default
        if entry.expires_at <= time.monotonic():
            self._items.pop(key, None)
            return default
        return entry.value

    def set(self, key: str, value: Any) -> None:
        self._items[key] = _CacheEntry(
            value=value,
            expires_at=time.monotonic() + self.ttl_seconds,
        )

    def clear(self) -> None:
        self._items.clear()
