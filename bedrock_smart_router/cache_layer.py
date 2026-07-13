# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Response cache — pluggable backends for caching Bedrock responses.

Backends:
  - ``memory`` (default): In-memory LRU with TTL. Single-process only.
  - ``redis``: Redis/ElastiCache. Shared across instances, survives restarts.

Cache keys are based on the user's request (messages + system + config),
NOT the model ID.  Identical requests always hit the cache even if the
router would have picked a different model.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CacheConfig:
    """Cache configuration."""

    enabled: bool = True
    backend: str = "memory"  # "memory" | "redis" | "valkey"
    ttl_seconds: float = 3600.0
    max_entries: int = 10_000
    # Redis/Valkey/ElastiCache connection
    redis_url: str = ""  # Also accepts Valkey and ElastiCache endpoints
    key_prefix: str = "bsr:"


def _make_cache_key(
    messages: list[dict[str, Any]],
    system: list[dict[str, Any]] | None = None,
    inference_config: dict[str, Any] | None = None,
    routing_key: str | None = None,
) -> str:
    """Deterministic hash of the user's request.

    Includes routing_key (strategy name) so different strategies
    don't share cache entries for the same prompt.
    """
    payload = json.dumps(
        {
            "messages": messages,
            "system": system or [],
            "config": inference_config or {},
            "routing": routing_key or "",
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


# ── Abstract interface ──────────────────────────────────────────────

class ResponseCache(ABC):
    """Abstract cache interface.  All backends implement this."""

    def __init__(self, config: CacheConfig | None = None) -> None:
        self.config = config or CacheConfig()
        self._hits = 0
        self._misses = 0

    @abstractmethod
    def get(
        self,
        messages: list[dict[str, Any]],
        system: list[dict[str, Any]] | None = None,
        inference_config: dict[str, Any] | None = None,
        routing_key: str | None = None,
    ) -> dict[str, Any] | None:
        ...

    @abstractmethod
    def put(
        self,
        messages: list[dict[str, Any]],
        response: dict[str, Any],
        model_id: str = "",
        system: list[dict[str, Any]] | None = None,
        inference_config: dict[str, Any] | None = None,
        routing_key: str | None = None,
    ) -> None:
        ...

    @abstractmethod
    def invalidate(self, model_id: str | None = None) -> int:
        ...

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    @property
    @abstractmethod
    def stats(self) -> dict[str, Any]:
        ...


# ── In-memory backend ───────────────────────────────────────────────

@dataclass
class _MemoryCacheEntry:
    response: dict[str, Any]
    model_id: str
    created_at: float


class InMemoryCache(ResponseCache):
    """In-memory LRU cache with TTL.  Single-process only."""

    def __init__(self, config: CacheConfig | None = None) -> None:
        super().__init__(config)
        self._cache: OrderedDict[str, _MemoryCacheEntry] = OrderedDict()
        self._lock = threading.Lock()

    def get(
        self,
        messages: list[dict[str, Any]],
        system: list[dict[str, Any]] | None = None,
        inference_config: dict[str, Any] | None = None,
        routing_key: str | None = None,
    ) -> dict[str, Any] | None:
        if not self.config.enabled:
            return None

        key = _make_cache_key(messages, system, inference_config, routing_key)
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                self._misses += 1
                return None

            if time.monotonic() - entry.created_at > self.config.ttl_seconds:
                del self._cache[key]
                self._misses += 1
                return None

            self._cache.move_to_end(key)
            self._hits += 1
        logger.debug("Memory cache HIT (key=%s…)", key[:12])
        return entry.response

    def put(
        self,
        messages: list[dict[str, Any]],
        response: dict[str, Any],
        model_id: str = "",
        system: list[dict[str, Any]] | None = None,
        inference_config: dict[str, Any] | None = None,
        routing_key: str | None = None,
    ) -> None:
        if not self.config.enabled:
            return

        key = _make_cache_key(messages, system, inference_config, routing_key)
        with self._lock:
            self._cache[key] = _MemoryCacheEntry(
                response=response, model_id=model_id, created_at=time.monotonic(),
            )
            self._cache.move_to_end(key)
            while len(self._cache) > self.config.max_entries:
                self._cache.popitem(last=False)

    def invalidate(self, model_id: str | None = None) -> int:
        with self._lock:
            if model_id is None:
                count = len(self._cache)
                self._cache.clear()
                return count
            keys = [k for k, v in self._cache.items() if v.model_id == model_id]
            for k in keys:
                del self._cache[k]
            return len(keys)

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "backend": "memory",
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self.hit_rate, 4),
            "size": len(self._cache),
            "max_entries": self.config.max_entries,
        }


# ── Factory ─────────────────────────────────────────────────────────

def build_cache(config: CacheConfig | None = None) -> ResponseCache:
    """Build the appropriate cache backend from config.

    Returns an ``InMemoryCache`` by default, or a ``RedisCache`` when
    ``config.backend`` is ``"redis"`` or ``"valkey"`` (both use the
    same Redis-protocol client — works with Redis, Valkey, and
    ElastiCache Serverless).
    """
    config = config or CacheConfig()
    if not config.enabled:
        return InMemoryCache(config)

    if config.backend in ("redis", "valkey"):
        from bedrock_smart_router.redis_cache import RedisCache
        return RedisCache(config)

    return InMemoryCache(config)
