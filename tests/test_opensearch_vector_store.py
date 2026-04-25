"""Integration tests for OpenSearch Serverless vector store.

Gated behind the OPENSEARCH_ENDPOINT environment variable.
AOSS has ~60s eventual consistency, so this test batches all writes
upfront, waits once, then runs all reads.

Run with:
    OPENSEARCH_ENDPOINT=https://abc123.us-west-2.aoss.amazonaws.com \
    python -m pytest tests/test_opensearch_vector_store.py -v -s
"""

from __future__ import annotations

import os
import time
import uuid

import pytest

OPENSEARCH_ENDPOINT = os.environ.get("OPENSEARCH_ENDPOINT", "")
SKIP_REASON = "Set OPENSEARCH_ENDPOINT to run OpenSearch integration tests"

pytestmark = pytest.mark.skipif(not OPENSEARCH_ENDPOINT, reason=SKIP_REASON)


@pytest.fixture(scope="module")
def store():
    """Create an OpenSearch vector store with a unique test index."""
    from bedrock_smart_router.opensearch_vector_store import OpenSearchVectorStore

    index_name = f"bsr-test-{uuid.uuid4().hex[:8]}"
    s = OpenSearchVectorStore(
        endpoint=OPENSEARCH_ENDPOINT,
        index_name=index_name,
        dimension=4,
        region="us-west-2",
    )
    yield s

    # Cleanup: delete the test index
    try:
        s._client.indices.delete(index=index_name)
    except Exception:
        pass


class TestOpenSearchVectorStore:
    """All tests in one class — batch writes, wait once, then read."""

    def test_full_lifecycle(self, store):
        """Test add, search, delete, clear in a single test to minimize waits."""
        # --- Write phase ---
        store.add("doc1", [0.9, 0.1, 0.1, 0.1], {"topic": "password"})
        store.add("doc2", [0.1, 0.9, 0.1, 0.1], {"topic": "s3"})
        store.add("doc3", [0.1, 0.1, 0.9, 0.1], {"topic": "vpc"})
        store.add("doc4", [0.5, 0.5, 0.5, 0.5], {
            "query": "test", "var_hash": "abc123", "created_at": 1234567890.0,
        })

        # --- Wait for AOSS eventual consistency ---
        print("\n  Waiting 65s for AOSS to propagate...")
        time.sleep(65)

        # --- Count ---
        count = store.count()
        print(f"  Count: {count}")
        assert count == 4, f"Expected 4 docs, got {count}"

        # --- Search: closest match ---
        results = store.search([0.9, 0.1, 0.1, 0.1], top_k=2)
        assert len(results) >= 1
        assert results[0].payload["topic"] == "password"
        print(f"  Search for password: score={results[0].score:.3f} ✅")

        # --- Search: threshold filtering ---
        results = store.search([0.0, 0.0, 0.0, 1.0], top_k=1, threshold=0.99)
        assert len(results) == 0
        print(f"  Threshold filter: 0 results ✅")

        # --- Payload preserved ---
        results = store.search([0.5, 0.5, 0.5, 0.5], top_k=1)
        assert len(results) >= 1
        assert results[0].payload["query"] == "test"
        assert results[0].payload["var_hash"] == "abc123"
        print(f"  Payload preserved ✅")

        # --- Delete ---
        deleted = store.delete("doc1")
        assert deleted is True
        print(f"  Delete doc1: {deleted} ✅")

        deleted = store.delete("nonexistent-id")
        assert deleted is False
        print(f"  Delete nonexistent: {deleted} ✅")

        # --- Clear ---
        cleared = store.clear()
        assert cleared >= 3  # doc1 already deleted
        print(f"  Clear: removed {cleared} docs ✅")

        # Wait for clear to propagate
        time.sleep(10)
        assert store.count() == 0
        print(f"  Count after clear: 0 ✅")

    def test_search_empty_index(self, store):
        """Search on a fresh empty index should return empty list."""
        from bedrock_smart_router.opensearch_vector_store import OpenSearchVectorStore
        empty_store = OpenSearchVectorStore(
            endpoint=OPENSEARCH_ENDPOINT,
            index_name=f"bsr-empty-{uuid.uuid4().hex[:8]}",
            dimension=4,
            region="us-west-2",
        )
        results = empty_store.search([0.1, 0.2, 0.3, 0.4], top_k=5)
        assert results == []
        # Cleanup
        try:
            empty_store._client.indices.delete(index=empty_store._index_name)
        except Exception:
            pass


class TestOpenSearchWithSemanticCache:
    """Test the full semantic cache pipeline with OpenSearch backend."""

    def test_semantic_cache_put_and_get(self, store):
        from bedrock_smart_router.semantic_cache import SemanticCache, SemanticCacheConfig

        cache = SemanticCache(
            config=SemanticCacheConfig(
                threshold=0.80,
                vector_store_backend="opensearch",
                opensearch_endpoint=OPENSEARCH_ENDPOINT,
                opensearch_index_name=store._index_name,
            ),
            region="us-west-2",
            vector_store=store,
        )

        def mock_embed(text):
            if "password" in text.lower() or "forgot" in text.lower():
                return [0.9, 0.1, 0.1, 0.1]
            return [0.1, 0.1, 0.9, 0.1]

        cache._get_embedding = mock_embed

        cache.put("How do I reset my password?", {"answer": "Go to settings"})
        print("\n  Waiting 65s for AOSS to propagate...")
        time.sleep(65)

        hit = cache.get("I forgot my password")
        assert hit is not None
        assert hit["answer"] == "Go to settings"
        print(f"  Semantic cache HIT ✅")

        miss = cache.get("What is Amazon S3?")
        assert miss is None
        print(f"  Semantic cache MISS ✅")
