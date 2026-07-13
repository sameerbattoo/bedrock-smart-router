# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the response cache (in-memory and factory)."""

import time

from bedrock_smart_router.cache_layer import (
    CacheConfig,
    InMemoryCache,
    build_cache,
)


def _msgs(text: str) -> list[dict]:
    return [{"role": "user", "content": [{"text": text}]}]


class TestInMemoryCache:
    def setup_method(self):
        self.cache = InMemoryCache(CacheConfig(enabled=True, ttl_seconds=10))

    def test_miss_on_empty(self):
        assert self.cache.get(_msgs("hello")) is None

    def test_put_and_hit(self):
        resp = {"output": {"message": {"content": [{"text": "Hi!"}]}}}
        self.cache.put(_msgs("hello"), resp, model_id="model-a")
        hit = self.cache.get(_msgs("hello"))
        assert hit is not None
        assert hit["output"]["message"]["content"][0]["text"] == "Hi!"

    def test_same_request_different_model_still_hits(self):
        resp = {"output": "Hi!"}
        self.cache.put(_msgs("hello"), resp, model_id="model-a")
        hit = self.cache.get(_msgs("hello"))
        assert hit is not None

    def test_different_messages_miss(self):
        self.cache.put(_msgs("hello"), {"output": "Hi!"})
        assert self.cache.get(_msgs("goodbye")) is None

    def test_ttl_expiry(self):
        cache = InMemoryCache(CacheConfig(enabled=True, ttl_seconds=0.1))
        cache.put(_msgs("hello"), {"output": "Hi!"})
        assert cache.get(_msgs("hello")) is not None
        time.sleep(0.15)
        assert cache.get(_msgs("hello")) is None

    def test_lru_eviction(self):
        cache = InMemoryCache(CacheConfig(enabled=True, max_entries=2))
        cache.put(_msgs("a"), {"r": "1"})
        cache.put(_msgs("b"), {"r": "2"})
        cache.put(_msgs("c"), {"r": "3"})
        assert cache.get(_msgs("a")) is None
        assert cache.get(_msgs("b")) is not None
        assert cache.get(_msgs("c")) is not None

    def test_disabled_cache(self):
        cache = InMemoryCache(CacheConfig(enabled=False))
        cache.put(_msgs("hello"), {"r": "1"})
        assert cache.get(_msgs("hello")) is None

    def test_hit_rate(self):
        self.cache.put(_msgs("a"), {"r": "1"})
        self.cache.get(_msgs("a"))  # hit
        self.cache.get(_msgs("b"))  # miss
        assert self.cache.hit_rate == 0.5

    def test_invalidate_all(self):
        self.cache.put(_msgs("a"), {"r": "1"})
        self.cache.put(_msgs("b"), {"r": "2"})
        count = self.cache.invalidate()
        assert count == 2
        assert self.cache.stats["size"] == 0

    def test_invalidate_by_model(self):
        self.cache.put(_msgs("x"), {"r": "1"}, model_id="model-a")
        self.cache.put(_msgs("y"), {"r": "2"}, model_id="model-b")
        count = self.cache.invalidate("model-a")
        assert count == 1
        assert self.cache.get(_msgs("y")) is not None

    def test_system_prompt_affects_key(self):
        self.cache.put(_msgs("hi"), {"r": "1"}, system=[{"text": "Be a pirate"}])
        assert self.cache.get(_msgs("hi"), system=[{"text": "Be a doctor"}]) is None
        assert self.cache.get(_msgs("hi"), system=[{"text": "Be a pirate"}]) is not None

    def test_stats_backend_field(self):
        assert self.cache.stats["backend"] == "memory"


class TestBuildCache:
    def test_default_builds_memory(self):
        cache = build_cache()
        assert isinstance(cache, InMemoryCache)

    def test_explicit_memory(self):
        cache = build_cache(CacheConfig(backend="memory"))
        assert isinstance(cache, InMemoryCache)

    def test_disabled_builds_memory(self):
        cache = build_cache(CacheConfig(enabled=False))
        assert isinstance(cache, InMemoryCache)
        cache.put(_msgs("x"), {"r": "1"})
        assert cache.get(_msgs("x")) is None

    def test_redis_without_url_raises(self):
        """Redis with empty URL should fail gracefully on get (logged warning)."""
        cache = build_cache(CacheConfig(backend="redis", redis_url=""))
        # get() catches the error and returns None (fail-open)
        result = cache.get(_msgs("test"))
        assert result is None


class TestValkeyAlias:
    def test_valkey_backend_builds_redis_cache(self):
        """backend='valkey' should create a RedisCache (same protocol)."""
        from bedrock_smart_router.redis_cache import RedisCache
        cache = build_cache(CacheConfig(
            backend="valkey",
            redis_url="redis://localhost:6379",
        ))
        assert isinstance(cache, RedisCache)

    def test_valkey_config_example(self):
        """ElastiCache Valkey TLS URL should be accepted."""
        from bedrock_smart_router.redis_cache import RedisCache
        cache = build_cache(CacheConfig(
            backend="valkey",
            redis_url="rediss://my-cluster.abc123.use1.cache.amazonaws.com:6379",
        ))
        assert isinstance(cache, RedisCache)
        assert cache.config.redis_url.startswith("rediss://")
