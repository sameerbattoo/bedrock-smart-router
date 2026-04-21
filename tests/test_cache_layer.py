"""Tests for the response cache."""

import time

from bedrock_smart_router.cache_layer import CacheConfig, ResponseCache


def _msgs(text: str) -> list[dict]:
    return [{"role": "user", "content": [{"text": text}]}]


class TestResponseCache:
    def setup_method(self):
        self.cache = ResponseCache(CacheConfig(enabled=True, ttl_seconds=10))

    def test_miss_on_empty(self):
        assert self.cache.get("model-a", _msgs("hello")) is None

    def test_put_and_hit(self):
        resp = {"output": {"message": {"content": [{"text": "Hi!"}]}}}
        self.cache.put("model-a", _msgs("hello"), resp)
        hit = self.cache.get("model-a", _msgs("hello"))
        assert hit is not None
        assert hit["output"]["message"]["content"][0]["text"] == "Hi!"

    def test_different_messages_miss(self):
        resp = {"output": "Hi!"}
        self.cache.put("model-a", _msgs("hello"), resp)
        assert self.cache.get("model-a", _msgs("goodbye")) is None

    def test_different_model_miss(self):
        resp = {"output": "Hi!"}
        self.cache.put("model-a", _msgs("hello"), resp)
        assert self.cache.get("model-b", _msgs("hello")) is None

    def test_ttl_expiry(self):
        cache = ResponseCache(CacheConfig(enabled=True, ttl_seconds=0.1))
        cache.put("model-a", _msgs("hello"), {"output": "Hi!"})
        assert cache.get("model-a", _msgs("hello")) is not None
        time.sleep(0.15)
        assert cache.get("model-a", _msgs("hello")) is None

    def test_lru_eviction(self):
        cache = ResponseCache(CacheConfig(enabled=True, max_entries=2))
        cache.put("m", _msgs("a"), {"r": "1"})
        cache.put("m", _msgs("b"), {"r": "2"})
        cache.put("m", _msgs("c"), {"r": "3"})  # Evicts "a"
        assert cache.get("m", _msgs("a")) is None
        assert cache.get("m", _msgs("b")) is not None
        assert cache.get("m", _msgs("c")) is not None

    def test_disabled_cache(self):
        cache = ResponseCache(CacheConfig(enabled=False))
        cache.put("m", _msgs("hello"), {"r": "1"})
        assert cache.get("m", _msgs("hello")) is None

    def test_hit_rate(self):
        self.cache.put("m", _msgs("a"), {"r": "1"})
        self.cache.get("m", _msgs("a"))  # hit
        self.cache.get("m", _msgs("b"))  # miss
        assert self.cache.hit_rate == 0.5

    def test_invalidate_all(self):
        self.cache.put("m", _msgs("a"), {"r": "1"})
        self.cache.put("m", _msgs("b"), {"r": "2"})
        count = self.cache.invalidate()
        assert count == 2
        assert self.cache.stats["size"] == 0

    def test_invalidate_by_model(self):
        self.cache.put("model-a", _msgs("x"), {"r": "1"})
        self.cache.put("model-b", _msgs("x"), {"r": "2"})
        count = self.cache.invalidate("model-a")
        assert count == 1
        assert self.cache.get("model-b", _msgs("x")) is not None
