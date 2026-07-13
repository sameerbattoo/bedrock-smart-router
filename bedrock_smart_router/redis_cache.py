# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Redis/Valkey-backed response cache.

Shared across all instances, survives restarts.  Uses the Redis
protocol (``SET``/``GET`` with ``EX`` for TTL).  Compatible with:

  - **Redis** (open source or managed)
  - **Valkey** (Redis-compatible fork)
  - **Amazon ElastiCache** (Redis or Valkey engine)
  - **Amazon ElastiCache Serverless**
  - **Amazon MemoryDB**

Install the optional dependency::

    pip install bedrock-smart-router[redis]

Configuration::

    # Redis
    cache:
      backend: redis
      redis_url: "redis://localhost:6379"

    # ElastiCache Valkey with TLS
    cache:
      backend: valkey
      redis_url: "rediss://my-cluster.abc123.use1.cache.amazonaws.com:6379"

    # ElastiCache Serverless
    cache:
      backend: redis
      redis_url: "rediss://my-serverless.abc123.serverless.use1.cache.amazonaws.com:6379"
"""

from __future__ import annotations

import json
import logging
from typing import Any

from bedrock_smart_router.cache_layer import (
    CacheConfig,
    ResponseCache,
    _make_cache_key,
)

logger = logging.getLogger(__name__)


class RedisCache(ResponseCache):
    """Redis-backed response cache.

    Requires the ``redis`` package.  Connects lazily on first use.
    """

    def __init__(self, config: CacheConfig | None = None) -> None:
        super().__init__(config)
        self._client: Any | None = None
        self._connected = False

    def _get_client(self) -> Any:
        """Lazy-connect to Redis."""
        if self._client is not None:
            return self._client

        try:
            import redis
        except ImportError:
            raise ImportError(
                "Redis cache requires the 'redis' package. "
                "Install with: pip install bedrock-smart-router[redis]"
            )

        url = self.config.redis_url
        if not url:
            raise ValueError(
                "redis_url is required for Redis cache. "
                "Set cache.redis_url in your config."
            )

        self._client = redis.Redis.from_url(
            url, decode_responses=True, socket_timeout=5,
        )
        # Test connection
        self._client.ping()
        self._connected = True
        logger.info("Connected to Redis at %s", url.split("@")[-1])
        return self._client

    def _key(self, cache_hash: str) -> str:
        return f"{self.config.key_prefix}{cache_hash}"

    def _model_key(self, cache_hash: str) -> str:
        return f"{self.config.key_prefix}model:{cache_hash}"

    def get(
        self,
        messages: list[dict[str, Any]],
        system: list[dict[str, Any]] | None = None,
        inference_config: dict[str, Any] | None = None,
        routing_key: str | None = None,
    ) -> dict[str, Any] | None:
        if not self.config.enabled:
            return None

        cache_hash = _make_cache_key(messages, system, inference_config, routing_key)
        try:
            client = self._get_client()
            raw = client.get(self._key(cache_hash))
        except Exception as exc:
            logger.warning("Redis GET failed: %s", exc)
            self._misses += 1
            return None

        if raw is None:
            self._misses += 1
            return None

        self._hits += 1
        logger.debug("Redis cache HIT (key=%s…)", cache_hash[:12])
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            self._misses += 1
            return None

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

        cache_hash = _make_cache_key(messages, system, inference_config, routing_key)
        ttl = int(self.config.ttl_seconds)

        try:
            client = self._get_client()
            serialized = json.dumps(response, default=str)
            pipe = client.pipeline()
            pipe.set(self._key(cache_hash), serialized, ex=ttl)
            if model_id:
                pipe.set(self._model_key(cache_hash), model_id, ex=ttl)
            pipe.execute()
        except Exception as exc:
            logger.warning("Redis SET failed: %s", exc)

    def invalidate(self, model_id: str | None = None) -> int:
        """Invalidate cache entries.

        When *model_id* is None, flushes all keys with our prefix.
        When *model_id* is given, scans for matching model keys and
        deletes the corresponding response keys.

        Note: ``SCAN`` is used instead of ``KEYS`` to avoid blocking
        Redis on large datasets.
        """
        try:
            client = self._get_client()
        except Exception:
            return 0

        if model_id is None:
            # Delete all keys with our prefix
            count = 0
            cursor = 0
            while True:
                cursor, keys = client.scan(
                    cursor, match=f"{self.config.key_prefix}*", count=100
                )
                if keys:
                    count += client.delete(*keys)
                if cursor == 0:
                    break
            return count

        # Delete entries for a specific model
        count = 0
        cursor = 0
        while True:
            cursor, keys = client.scan(
                cursor, match=f"{self.config.key_prefix}model:*", count=100
            )
            for model_key in keys:
                stored_model = client.get(model_key)
                if stored_model == model_id:
                    # Extract the hash from "bsr:model:{hash}"
                    cache_hash = model_key.split(":")[-1]
                    pipe = client.pipeline()
                    pipe.delete(self._key(cache_hash))
                    pipe.delete(model_key)
                    pipe.execute()
                    count += 1
            if cursor == 0:
                break
        return count

    @property
    def stats(self) -> dict[str, Any]:
        info: dict[str, Any] = {
            "backend": "redis",
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self.hit_rate, 4),
            "connected": self._connected,
            "redis_url": (
                self.config.redis_url.split("@")[-1]
                if self.config.redis_url else ""
            ),
        }
        # Try to get Redis-side stats
        try:
            client = self._get_client()
            db_info = client.info("keyspace")
            for db, db_stats in db_info.items():
                if isinstance(db_stats, dict):
                    info["redis_keys"] = db_stats.get("keys", 0)
                    break
        except Exception:
            pass
        return info
