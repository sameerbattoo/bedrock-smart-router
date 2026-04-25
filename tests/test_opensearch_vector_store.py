"""Integration tests for OpenSearch Serverless vector store.

Gated behind the OPENSEARCH_ENDPOINT environment variable.
AOSS has ~60-90s eventual consistency for new data.

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
    def test_add_search_and_payload(self, store):
        """Test add, search, and payload preservation in one test."""
        store.add("doc1", [0.9, 0.1, 0.1, 0.1], {"topic": "password"})
        store.add("doc2", [0.1, 0.9, 0.1, 0.1], {"topic": "s3"})
        store.add("doc3", [0.5, 0.5, 0.5, 0.5], {
            "query": "test", "var_hash": "abc123",
        })

        print("\n  Waiting 90s for AOSS to propagate...")
        time.sleep(90)

        # Count
        count = store.count()
        print(f"  Count: {count}")
        assert count >= 3

        # Search: closest match
        results = store.search([0.9, 0.1, 0.1, 0.1], top_k=2)
        assert len(results) >= 1
        assert results[0].payload["topic"] == "password"
        print(f"  Search for password: score={results[0].score:.3f} ✅")

        # Payload preserved
        results = store.search([0.5, 0.5, 0.5, 0.5], top_k=1)
        assert len(results) >= 1
        assert results[0].payload["var_hash"] == "abc123"
        print(f"  Payload preserved ✅")

    def test_search_empty_index(self):
        """Search on a fresh empty index returns empty list."""
        from bedrock_smart_router.opensearch_vector_store import OpenSearchVectorStore
        s = OpenSearchVectorStore(
            endpoint=OPENSEARCH_ENDPOINT,
            index_name=f"bsr-empty-{uuid.uuid4().hex[:8]}",
            dimension=4,
            region="us-west-2",
        )
        results = s.search([0.1, 0.2, 0.3, 0.4], top_k=5)
        assert results == []
        try:
            s._client.indices.delete(index=s._index_name)
        except Exception:
            pass


class TestOpenSearchWithSemanticCache:
    def test_semantic_cache_put_and_get(self):
        """Full semantic cache pipeline with OpenSearch backend."""
        from bedrock_smart_router.opensearch_vector_store import OpenSearchVectorStore
        from bedrock_smart_router.semantic_cache import SemanticCache, SemanticCacheConfig

        index_name = f"bsr-cache-{uuid.uuid4().hex[:8]}"
        vs = OpenSearchVectorStore(
            endpoint=OPENSEARCH_ENDPOINT,
            index_name=index_name,
            dimension=4,
            region="us-west-2",
        )

        cache = SemanticCache(
            config=SemanticCacheConfig(threshold=0.80),
            region="us-west-2",
            vector_store=vs,
        )

        def mock_embed(text):
            if "password" in text.lower() or "forgot" in text.lower():
                return [0.9, 0.1, 0.1, 0.1]
            return [0.1, 0.1, 0.9, 0.1]

        cache._get_embedding = mock_embed

        cache.put("How do I reset my password?", {"answer": "Go to settings"})
        print("\n  Waiting 90s for AOSS to propagate...")
        time.sleep(90)

        hit = cache.get("I forgot my password")
        assert hit is not None
        assert hit["answer"] == "Go to settings"
        print(f"  Semantic cache HIT ✅")

        miss = cache.get("What is Amazon S3?")
        assert miss is None
        print(f"  Semantic cache MISS ✅")

        # Cleanup
        try:
            vs._client.indices.delete(index=index_name)
        except Exception:
            pass
