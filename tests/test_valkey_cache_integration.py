"""Integration test — Redis/Valkey cache against real ElastiCache.

Run with:
    INTEGRATION_TEST=1 .venv/bin/python -m pytest tests/test_valkey_cache_integration.py -v -s

Uses the ElastiCache Valkey cluster:
    arn:aws:elasticache:us-west-2:175918693907:replicationgroup:capstone-sqlagent-valkey-cache

Requires network access to the VPC where ElastiCache is running.
"""

from __future__ import annotations

import os
import uuid

import pytest

from bedrock_smart_router.cache_layer import CacheConfig, build_cache

SKIP_REASON = "Set INTEGRATION_TEST=1 and VALKEY_URL=rediss://... to run (requires VPC access)"
VALKEY_URL = os.environ.get(
    "VALKEY_URL",
    "rediss://master.capstone-sqlagent-valkey-cache.8ot617.usw2.cache.amazonaws.com:6379",
)

# Note: ElastiCache is VPC-only. This test requires network access to
# the VPC (run from EC2, Lambda, or VPN-connected machine).
# Set VALKEY_URL env var to override the endpoint.


def _msgs(text: str) -> list[dict]:
    return [{"role": "user", "content": [{"text": text}]}]


@pytest.fixture
def valkey_cache():
    """Create a cache connected to the real Valkey cluster."""
    url = os.environ.get("VALKEY_URL", VALKEY_URL)
    prefix = f"bsr-test-{uuid.uuid4().hex[:8]}:"
    config = CacheConfig(
        enabled=True,
        backend="valkey",
        redis_url=url,
        ttl_seconds=60,
        key_prefix=prefix,
    )
    cache = build_cache(config)

    yield cache

    # Cleanup: delete all test keys
    try:
        cache.invalidate()
        print(f"\n  Cleaned up keys with prefix '{prefix}'")
    except Exception as exc:
        print(f"\n  Warning: cleanup failed: {exc}")


@pytest.mark.skipif(
    os.environ.get("INTEGRATION_TEST") != "1" or not os.environ.get("VALKEY_URL"),
    reason=SKIP_REASON,
)
class TestValkeyCacheIntegration:

    def test_connection(self, valkey_cache):
        """Should connect to the Valkey cluster."""
        stats = valkey_cache.stats
        assert stats["connected"] is True
        assert stats["backend"] == "redis"
        print(f"\n  Connected: {stats}")

    def test_put_and_get(self, valkey_cache):
        """Store a response and retrieve it."""
        resp = {"output": {"message": {"content": [{"text": "Hello from Valkey!"}]}}}
        valkey_cache.put(_msgs("test-put-get"), resp, model_id="model-a")

        hit = valkey_cache.get(_msgs("test-put-get"))
        assert hit is not None
        assert hit["output"]["message"]["content"][0]["text"] == "Hello from Valkey!"
        print(f"\n  Cache hit: {hit['output']['message']['content'][0]['text']}")

    def test_cache_miss(self, valkey_cache):
        """Unknown key should return None."""
        result = valkey_cache.get(_msgs("never-stored-this"))
        assert result is None
        print("\n  Cache miss as expected")

    def test_different_messages_different_keys(self, valkey_cache):
        """Different messages should not collide."""
        valkey_cache.put(_msgs("question-a"), {"answer": "A"})
        valkey_cache.put(_msgs("question-b"), {"answer": "B"})

        assert valkey_cache.get(_msgs("question-a"))["answer"] == "A"
        assert valkey_cache.get(_msgs("question-b"))["answer"] == "B"
        print("\n  Different messages stored independently")

    def test_system_prompt_affects_key(self, valkey_cache):
        """Same user message with different system prompts = different keys."""
        valkey_cache.put(
            _msgs("hi"), {"r": "pirate"},
            system=[{"text": "Be a pirate"}],
        )
        valkey_cache.put(
            _msgs("hi"), {"r": "doctor"},
            system=[{"text": "Be a doctor"}],
        )

        pirate = valkey_cache.get(_msgs("hi"), system=[{"text": "Be a pirate"}])
        doctor = valkey_cache.get(_msgs("hi"), system=[{"text": "Be a doctor"}])
        assert pirate["r"] == "pirate"
        assert doctor["r"] == "doctor"
        print("\n  System prompt correctly differentiates cache keys")

    def test_hit_rate_tracking(self, valkey_cache):
        """Hit rate should be tracked client-side."""
        valkey_cache.put(_msgs("tracked"), {"r": "1"})
        valkey_cache.get(_msgs("tracked"))  # hit
        valkey_cache.get(_msgs("unknown"))  # miss

        assert valkey_cache.hit_rate == pytest.approx(0.5)
        stats = valkey_cache.stats
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        print(f"\n  Hit rate: {stats['hit_rate']}")

    def test_invalidate_all(self, valkey_cache):
        """Invalidate should remove all keys with our prefix."""
        for i in range(5):
            valkey_cache.put(_msgs(f"inv-{i}"), {"r": str(i)})

        count = valkey_cache.invalidate()
        assert count >= 5
        print(f"\n  Invalidated {count} keys")

        # Verify they're gone
        for i in range(5):
            assert valkey_cache.get(_msgs(f"inv-{i}")) is None

    def test_invalidate_by_model(self, valkey_cache):
        """Invalidate by model_id should only remove that model's entries."""
        valkey_cache.put(_msgs("model-a-q"), {"r": "a"}, model_id="model-a")
        valkey_cache.put(_msgs("model-b-q"), {"r": "b"}, model_id="model-b")

        count = valkey_cache.invalidate("model-a")
        assert count == 1

        assert valkey_cache.get(_msgs("model-a-q")) is None
        assert valkey_cache.get(_msgs("model-b-q")) is not None
        print(f"\n  Invalidated {count} entries for model-a, model-b still cached")

    def test_large_response(self, valkey_cache):
        """Should handle large responses (typical Bedrock output)."""
        large_text = "x" * 50_000  # 50KB response
        resp = {"output": {"message": {"content": [{"text": large_text}]}}}
        valkey_cache.put(_msgs("large"), resp)

        hit = valkey_cache.get(_msgs("large"))
        assert hit is not None
        assert len(hit["output"]["message"]["content"][0]["text"]) == 50_000
        print(f"\n  Large response (50KB) cached and retrieved OK")
