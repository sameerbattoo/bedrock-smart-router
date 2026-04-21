"""Response cache — exact-match in-memory LRU cache.

Caches Bedrock Converse responses keyed by a hash of the model ID,
messages, and inference parameters.  Cache hits bypass model selection
and Bedrock invocation entirely, returning at zero API cost.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CacheConfig:
    """Cache configuration."""

    enabled: bool = True
    ttl_seconds: float = 3600.0
    max_entries: int = 10_000


@dataclass
class _CacheEntry:
    response: dict[str, Any]
    model_id: str
    created_at: float


def _make_cache_key(
    model_id: str,
    messages: list[dict[str, Any]],
    system: list[dict[str, Any]] | None = None,
    inference_config: dict[str, Any] | None = None,
) -> str:
    """Deterministic hash of the request parameters."""
    payload = json.dumps(
        {
            "model": model_id,
            "messages": messages,
            "system": system or [],
            "config": inference_config or {},
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


class ResponseCache:
    """In-memory LRU response cache with TTL expiry."""

    def __init__(self, config: CacheConfig | None = None) -> None:
        self.config = config or CacheConfig()
        self._cache: OrderedDict[str, _CacheEntry] = OrderedDict()
        self._hits = 0
        self._misses = 0

    def get(
        self,
        model_id: str,
        messages: list[dict[str, Any]],
        system: list[dict[str, Any]] | None = None,
        inference_config: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Look up a cached response.  Returns *None* on miss."""
        if not self.config.enabled:
            return None

        key = _make_cache_key(model_id, messages, system, inference_config)
        entry = self._cache.get(key)
        if entry is None:
            self._misses += 1
            return None

        # Check TTL
        if time.monotonic() - entry.created_at > self.config.ttl_seconds:
            del self._cache[key]
            self._misses += 1
            return None

        # Move to end (most recently used)
        self._cache.move_to_end(key)
        self._hits += 1
        logger.debug("Cache HIT for %s (key=%s…)", model_id, key[:12])
        return entry.response

    def put(
        self,
        model_id: str,
        messages: list[dict[str, Any]],
        response: dict[str, Any],
        system: list[dict[str, Any]] | None = None,
        inference_config: dict[str, Any] | None = None,
    ) -> None:
        """Store a response in the cache."""
        if not self.config.enabled:
            return

        key = _make_cache_key(model_id, messages, system, inference_config)
        self._cache[key] = _CacheEntry(
            response=response,
            model_id=model_id,
            created_at=time.monotonic(),
        )
        self._cache.move_to_end(key)

        # Evict oldest if over capacity
        while len(self._cache) > self.config.max_entries:
            self._cache.popitem(last=False)

    def invalidate(self, model_id: str | None = None) -> int:
        """Remove entries.  If *model_id* given, only that model's entries."""
        if model_id is None:
            count = len(self._cache)
            self._cache.clear()
            return count
        keys_to_remove = [
            k for k, v in self._cache.items() if v.model_id == model_id
        ]
        for k in keys_to_remove:
            del self._cache[k]
        return len(keys_to_remove)

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self.hit_rate, 4),
            "size": len(self._cache),
            "max_entries": self.config.max_entries,
        }
