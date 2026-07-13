# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the semantic cache with pluggable vector stores.

Uses mocked embeddings to test cache logic without calling Bedrock.
Tests in-memory and FAISS backends, plus auto-extraction and multi-turn.
"""

from __future__ import annotations

import hashlib
import time
from unittest.mock import MagicMock, patch

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
    # Intent-based embeddings for auto-extract tests
    if "users" in text_lower and "geography" in text_lower:
        vec = [0.8, 0.2, 0.1, 0.1]
    if "top" in text_lower and "product" in text_lower:
        vec = [0.1, 0.8, 0.2, 0.1]
    return vec


@pytest.fixture
def cache():
    """SemanticCache with mocked embeddings and in-memory store."""
    c = SemanticCache(
        config=SemanticCacheConfig(
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
    """Tests for manual variable-aware caching."""

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
        h1 = SemanticCache._hash_variables({"a": "1", "b": "2"})
        h2 = SemanticCache._hash_variables({"b": "2", "a": "1"})
        assert h1 == h2  # Order-independent

    def test_empty_variables_same_as_none(self):
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
            topics = ["password", "s3", "vpc", "weather"]
            topic = topics[i % 4]
            cache_faiss.put(f"{topic} question {i}", {"i": i})
        assert cache_faiss.stats["entries"] == 100

    def test_stats_backend(self, cache_faiss):
        assert cache_faiss.stats["backend"] == "faiss"


class TestAutoExtraction:
    """Tests for auto_extract mode with mocked IntentExtractor."""

    @pytest.fixture
    def auto_cache(self):
        """SemanticCache with auto_extract enabled and mocked extractor."""
        from bedrock_smart_router.intent_extractor import ExtractionResult

        c = SemanticCache(
            config=SemanticCacheConfig(
                threshold=0.80,
                max_entries=100,
                ttl_seconds=60,
                auto_extract=True,
            ),
        )
        c._get_embedding = _mock_embedding

        # Mock the extractor
        mock_extractor = MagicMock()
        mock_extractor.extract.return_value = ExtractionResult(
            intent="Count users by geography for a year with sales above a threshold",
            variables={"year": "2026", "sales_threshold": "200"},
            raw_query="Count users by geo for 2026 with sales > $200",
            source="single-turn",
        )
        mock_extractor.extract_from_messages.return_value = ExtractionResult(
            intent="Count users by geography for a year with sales above a threshold",
            variables={"year": "2026", "sales_threshold": "200"},
            raw_query="multi-turn",
            source="multi-turn",
        )
        c._extractor = mock_extractor
        return c

    def test_auto_extract_single_turn_put_and_get(self, auto_cache):
        """Auto-extract should extract intent+variables and use them for matching."""
        auto_cache.put("Count users by geo for 2026 with sales > $200",
                       {"result": "42 users"})
        hit = auto_cache.get("Show user distribution by geography, year 2026, sales over $200")
        assert hit is not None
        assert hit["result"] == "42 users"

    def test_auto_extract_different_variables_miss(self, auto_cache):
        """Different extracted variables should cause a cache miss."""
        from bedrock_smart_router.intent_extractor import ExtractionResult

        auto_cache.put("Count users by geo for 2026 with sales > $200",
                       {"result": "42 users"})

        # Change the extractor to return different variables for the get
        auto_cache._extractor.extract.return_value = ExtractionResult(
            intent="Count users by geography for a year with sales above a threshold",
            variables={"year": "2025", "sales_threshold": "100"},
        )
        hit = auto_cache.get("Count users by geo for 2025 with sales > $100")
        assert hit is None

    def test_auto_extract_multi_turn_matches_single_turn(self, auto_cache):
        """Multi-turn conversation should resolve to same intent as single-turn."""
        # Store from single-turn
        auto_cache.put("Count users by geo for 2026 with sales > $200",
                       {"result": "42 users"})

        # Lookup from multi-turn conversation
        messages = [
            {"role": "user", "content": [{"text": "show me users by geo"}]},
            {"role": "assistant", "content": [{"text": "Here are users..."}]},
            {"role": "user", "content": [{"text": "now for 2026 with sales > $200"}]},
        ]
        hit = auto_cache.get(messages=messages)
        assert hit is not None
        assert hit["result"] == "42 users"

    def test_auto_extract_no_variables_query(self, auto_cache):
        """Queries with no variables should still work."""
        from bedrock_smart_router.intent_extractor import ExtractionResult

        auto_cache._extractor.extract.return_value = ExtractionResult(
            intent="What is Amazon S3",
            variables={},
        )
        auto_cache.put("What is Amazon S3?", {"answer": "Object storage"})

        # Same intent, no variables
        auto_cache._extractor.extract.return_value = ExtractionResult(
            intent="What is Amazon S3",
            variables={},
        )
        hit = auto_cache.get("Tell me about S3")
        assert hit is not None

    def test_auto_extract_overrides_manual_variables(self, auto_cache):
        """When auto_extract is on, manual variables should be ignored."""
        auto_cache.put("Count users for 2026",
                       {"result": "42"},
                       variables={"year": "9999"})  # Manual var — should be ignored

        # The extractor returns year=2026, not 9999
        hit = auto_cache.get("Count users for 2026",
                            variables={"year": "9999"})  # Also ignored
        assert hit is not None  # Matches on extracted variables, not manual

    def test_stats_include_auto_extract_fields(self, auto_cache):
        stats = auto_cache.stats
        assert stats["auto_extract"] is True


class TestMultiTurnWithoutAutoExtract:
    """Test the messages parameter in manual mode."""

    def test_messages_uses_last_user_text(self, cache):
        """Without auto_extract, messages should use the last user message text."""
        cache.put("How do I reset my password?", {"answer": "settings"})
        messages = [
            {"role": "user", "content": [{"text": "Hello"}]},
            {"role": "assistant", "content": [{"text": "Hi"}]},
            {"role": "user", "content": [{"text": "I forgot my password"}]},
        ]
        hit = cache.get(messages=messages)
        assert hit is not None  # Last user message matches semantically


class TestEmbeddingRetry:
    """Test retry logic on embedding calls."""

    def test_embedding_retries_on_throttle(self):
        from botocore.exceptions import ClientError

        c = SemanticCache(
            config=SemanticCacheConfig(threshold=0.80),
        )

        mock_client = MagicMock()
        error_resp = {"Error": {"Code": "ThrottlingException", "Message": "Rate exceeded"}}

        # First call fails, second succeeds
        mock_body = MagicMock()
        mock_body.read.return_value = b'{"embedding": [0.1, 0.2, 0.3]}'
        mock_client.invoke_model.side_effect = [
            ClientError(error_resp, "InvokeModel"),
            {"body": mock_body},
        ]

        mock_session = MagicMock()
        mock_session.client.return_value = mock_client
        c._session = mock_session

        with patch("bedrock_smart_router.semantic_cache.time.sleep"):
            embedding = c._get_embedding("test")
        assert embedding == [0.1, 0.2, 0.3]
        assert mock_client.invoke_model.call_count == 2

    def test_embedding_no_retry_on_validation(self):
        from botocore.exceptions import ClientError

        c = SemanticCache(
            config=SemanticCacheConfig(threshold=0.80),
        )

        mock_client = MagicMock()
        error_resp = {"Error": {"Code": "ValidationException", "Message": "Bad"}}
        mock_client.invoke_model.side_effect = ClientError(error_resp, "InvokeModel")

        mock_session = MagicMock()
        mock_session.client.return_value = mock_client
        c._session = mock_session

        with pytest.raises(ClientError):
            c._get_embedding("test")
        assert mock_client.invoke_model.call_count == 1
