"""Response cache — exact-match in-memory LRU cache.

Caches Bedrock Converse responses keyed by a hash of the user's
request (messages, system prompt, inference config).  The cache key
does NOT include the model ID — this means a cached response is
returned regardless of which model the router would have selected,
which is correct because:

1. If the request is identical, the response quality is acceptable
   (it was good enough the first time).
2. Fallbacks don't break the cache — if request A fell back from
   model X to model Y, request A repeated still gets the cached
   response from model Y.
3. The routing decision on a cache hit still reflects which model
   *would have been* selected, for observability.
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
    model_id: str  # Which model produced this response (for invalidation)
    created_at: float


def _make_cache_key(
    messages: list[dict[str, Any]],
    system: list[dict[str, Any]] | None = None,
    inference_config: dict[str, Any] | None = None,
) -> str:
    """Deterministic hash of the user's request — model-independent."""
    payload = json.dumps(
        {
            "messages": messages,
            "system": system or [],
            "config": inference_config or {},
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


class ResponseCache:
    """In-memory LRU response cache with TTL expiry.

    Keys are based on the user's request (messages + system + config),
    NOT the model.  This means identical requests always hit the cache
    even if the router would have picked a different model.
    """

    def __init__(self, config: CacheConfig | None = None) -> None:
        self.config = config or CacheConfig()
        self._cache: OrderedDict[str, _CacheEntry] = OrderedDict()
        self._hits = 0
        self._misses = 0

    def get(
        self,
        messages: list[dict[str, Any]],
        system: list[dict[str, Any]] | None = None,
        inference_config: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Look up a cached response.  Returns *None* on miss."""
        if not self.config.enabled:
            return None

        key = _make_cache_key(messages, system, inference_config)
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
        logger.debug("Cache HIT (key=%s…)", key[:12])
        return entry.response

    def put(
        self,
        messages: list[dict[str, Any]],
        response: dict[str, Any],
        model_id: str = "",
        system: list[dict[str, Any]] | None = None,
        inference_config: dict[str, Any] | None = None,
    ) -> None:
        """Store a response in the cache."""
        if not self.config.enabled:
            return

        key = _make_cache_key(messages, system, inference_config)
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
