"""Tests for the semantic cache with pluggable vector stores.

Uses mocked embeddings to test cache logic without calling Bedrock.
Tests in-memory and FAISS backends.
"""

from __future__ import annotations

import hashlib
import time
from unittest.mock import MagicMock

import pytest

from bedrock_smart_router.semantic_cache import SemanticCache, SemanticCacheConfig
from bedrock_smart_router.vector_store import InMemoryVectorStore


def _mock_embedding(text: str) -> list[float]:
    """Deterministic mock embedding — similar texts get similar vectors."""
    text_lower = text.lower()
    vec = [0.1, 0.1, 0.1, 0.1]
    if "password" in text_lower or "forgot" in text_lower or "reset" in text_lower:
        vec[0] = 0.9
    if "s3" in text_lower or "bucket" in text_lower or "storage" in text_lower:
        vec[1] = 0.9
    if "vpc" in text_lower or "network" in text_lower:
        vec[2] = 0.9
    if "weather" in text_lower or "temperature" in text_lower:
        vec[3] = 0.9
    return vec


@pytest.fixture
def cache():
    """SemanticCache with mocked embeddings and in-memory store."""
    c = SemanticCache(
        config=SemanticCacheConfig(
            enabled=True,
            threshold=0.80,
            max_entries=100,
            ttl_seconds=60,
        ),
    )
    c._get_embedding = _mock_embedding
    return c


@pytest.fixture
def cache_faiss():
    """SemanticCache with FAISS backend and mocked embeddings."""
    c = SemanticCache(
        config=SemanticCacheConfig(
            enabled=True,
            threshold=0.80,
            vector_store_backend="faiss",
            embedding_dimension=4,
        ),
    )
    c._get_embedding = _mock_embedding
    return c


class TestSemanticCacheMemory:
    """Tests with in-memory vector store."""

    def test_put_and_get_exact(self, cache):
        cache.put("How do I reset my password?", {"answer": "Go to settings"})
        hit = cache.get("How do I reset my password?")
        assert hit is not None
        assert hit["answer"] == "Go to settings"

    def test_semantic_match(self, cache):
        cache.put("How do I reset my password?", {"answer": "Go to settings"})
        hit = cache.get("I forgot my password, help")
        assert hit is not None  # Same topic → similar embedding → HIT

    def test_different_topic_miss(self, cache):
        cache.put("How do I reset my password?", {"answer": "Go to settings"})
        hit = cache.get("What's the weather today?")
        assert hit is None  # Different topic → MISS

    def test_disabled_cache(self):
        c = SemanticCache(config=SemanticCacheConfig(enabled=False))
        c._get_embedding = _mock_embedding
        c.put("test", {"r": "1"})
        assert c.get("test") is None

    def test_hit_rate(self, cache):
        cache.put("password reset", {"r": "1"})
        cache.get("forgot password")  # HIT
        cache.get("weather today")    # MISS
        assert cache.hit_rate == pytest.approx(0.5)

    def test_stats(self, cache):
        cache.put("test", {"r": "1"})
        cache.get("test")
        stats = cache.stats
        assert stats["hits"] >= 0
        assert stats["backend"] == "memory"
        assert stats["entries"] == 1

    def test_invalidate(self, cache):
        cache.put("a", {"r": "1"})
        cache.put("b", {"r": "2"})
        count = cache.invalidate()
        assert count == 2
        assert cache.stats["entries"] == 0


class TestSemanticCacheVariables:
    """Tests for variable-aware caching."""

    def test_same_variables_hit(self, cache):
        cache.put("top users for Electronics 2024", {"users": ["a", "b"]},
                  variables={"category": "Electronics", "year": "2024"})
        hit = cache.get("show top users in Electronics 2024",
                       variables={"category": "Electronics", "year": "2024"})
        assert hit is not None

    def test_different_variables_miss(self, cache):
        cache.put("top users for Electronics 2024", {"users": ["a", "b"]},
                  variables={"category": "Electronics", "year": "2024"})
        hit = cache.get("top users for Clothing 2025",
                       variables={"category": "Clothing", "year": "2025"})
        assert hit is None

    def test_no_variables_matches_any(self, cache):
        """Entries stored without variables should match get() without variables."""
        cache.put("How do I reset my password?", {"answer": "settings"})
        hit = cache.get("I forgot my password")
        assert hit is not None

    def test_variable_hash_deterministic(self):
        from bedrock_smart_router.semantic_cache import SemanticCache
        h1 = SemanticCache._hash_variables({"a": "1", "b": "2"})
        h2 = SemanticCache._hash_variables({"b": "2", "a": "1"})
        assert h1 == h2  # Order-independent

    def test_empty_variables_same_as_none(self):
        from bedrock_smart_router.semantic_cache import SemanticCache
        assert SemanticCache._hash_variables({}) == ""


class TestSemanticCacheFAISS:
    """Tests with FAISS vector store backend."""

    def test_put_and_get(self, cache_faiss):
        cache_faiss.put("How do I reset my password?", {"answer": "settings"})
        hit = cache_faiss.get("I forgot my password")
        assert hit is not None

    def test_different_topic_miss(self, cache_faiss):
        cache_faiss.put("How do I reset my password?", {"answer": "settings"})
        hit = cache_faiss.get("What's the weather?")
        assert hit is None

    def test_variables_with_faiss(self, cache_faiss):
        cache_faiss.put("top users Electronics", {"users": ["a"]},
                       variables={"cat": "Electronics"})
        hit = cache_faiss.get("top users Electronics",
                             variables={"cat": "Electronics"})
        assert hit is not None

        hit = cache_faiss.get("top users Clothing",
                             variables={"cat": "Clothing"})
        assert hit is None

    def test_many_entries(self, cache_faiss):
        """FAISS should handle many entries efficiently."""
        for i in range(100):
            # Create varied embeddings by using different keywords
            topics = ["password", "s3", "vpc", "weather"]
            topic = topics[i % 4]
            cache_faiss.put(f"{topic} question {i}", {"i": i})
        assert cache_faiss.stats["entries"] == 100

    def test_stats_backend(self, cache_faiss):
        assert cache_faiss.stats["backend"] == "faiss"
