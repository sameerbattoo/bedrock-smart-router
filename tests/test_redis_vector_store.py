"""Tests for RedisVectorStore — validates imports and core logic.

These tests mock the Redis connection so they run without a live Redis
instance.  They verify that:
1. The redis-py imports resolve correctly (caught the indexDefinition →
   index_definition rename in redis-py 7.x)
2. The vector store logic (add, search, delete, clear) works correctly
   against a mocked Redis client
"""

import json
import struct
from unittest.mock import MagicMock, patch

import pytest


class TestRedisImports:
    """Verify that redis-py module imports used by RedisVectorStore resolve."""

    def test_index_definition_import(self):
        """The index_definition module should be importable (redis >= 7.x)."""
        from redis.commands.search.index_definition import IndexDefinition, IndexType
        assert IndexDefinition is not None
        assert IndexType is not None

    def test_search_field_import(self):
        """TagField and VectorField should be importable."""
        from redis.commands.search.field import TagField, VectorField
        assert TagField is not None
        assert VectorField is not None

    def test_query_import(self):
        """Query class should be importable."""
        from redis.commands.search.query import Query
        assert Query is not None


class TestRedisVectorStoreUnit:
    """Unit tests for RedisVectorStore with mocked Redis client."""

    @pytest.fixture
    def mock_redis(self):
        """Create a mock Redis client and patch Redis.from_url."""
        mock_client = MagicMock()
        # Make ft().info() raise so _ensure_index tries to create
        mock_ft = MagicMock()
        mock_ft.info.side_effect = Exception("Index not found")
        mock_ft.create_index.return_value = True
        mock_client.ft.return_value = mock_ft
        return mock_client

    @pytest.fixture
    def store(self, mock_redis):
        """Create a RedisVectorStore with mocked connection."""
        with patch("redis.Redis.from_url", return_value=mock_redis):
            from bedrock_smart_router.redis_vector_store import RedisVectorStore
            store = RedisVectorStore(
                redis_url="redis://localhost:6379",
                key_prefix="test:vec:",
                dimension=3,
            )
        return store

    def test_init_creates_index(self, store, mock_redis):
        """Store should attempt to create the RediSearch index on init."""
        assert store._index_created is True
        mock_redis.ft.return_value.create_index.assert_called_once()

    def test_add(self, store, mock_redis):
        """add() should store a hash with entry_id, embedding, and payload."""
        store.add("doc1", [1.0, 0.0, 0.0], {"text": "hello"})
        mock_redis.hset.assert_called_once()
        call_args = mock_redis.hset.call_args
        key = call_args[0][0]
        mapping = call_args[1]["mapping"]
        assert key == "test:vec:doc1"
        assert mapping["entry_id"] == b"doc1"
        assert mapping["payload"] == json.dumps({"text": "hello"}).encode()
        # Verify embedding is packed as float32
        expected_bytes = struct.pack("3f", 1.0, 0.0, 0.0)
        assert mapping["embedding"] == expected_bytes

    def test_delete(self, store, mock_redis):
        """delete() should call Redis DELETE on the prefixed key."""
        mock_redis.delete.return_value = 1
        assert store.delete("doc1") is True
        mock_redis.delete.assert_called_with("test:vec:doc1")

    def test_delete_nonexistent(self, store, mock_redis):
        """delete() should return False when key doesn't exist."""
        mock_redis.delete.return_value = 0
        assert store.delete("nope") is False

    def test_clear(self, store, mock_redis):
        """clear() should scan and delete all prefixed keys."""
        # Simulate scan returning some keys then finishing
        mock_redis.scan.side_effect = [
            (0, [b"test:vec:a", b"test:vec:b"]),
        ]
        mock_redis.delete.return_value = 2
        count = store.clear()
        assert count == 2

    def test_count(self, store, mock_redis):
        """count() should scan and count all prefixed keys."""
        mock_redis.scan.side_effect = [
            (42, [b"test:vec:a", b"test:vec:b"]),
            (0, [b"test:vec:c"]),
        ]
        assert store.count() == 3

    def test_search_when_index_not_created(self, mock_redis):
        """search() should return empty list if index creation failed."""
        with patch("redis.Redis.from_url", return_value=mock_redis):
            from bedrock_smart_router.redis_vector_store import RedisVectorStore
            # Make both info() and create_index() fail
            mock_ft = MagicMock()
            mock_ft.info.side_effect = Exception("no index")
            mock_ft.create_index.side_effect = Exception("no RediSearch module")
            mock_redis.ft.return_value = mock_ft

            store = RedisVectorStore(
                redis_url="redis://localhost:6379",
                dimension=3,
            )
            assert store._index_created is False
            results = store.search([1.0, 0.0, 0.0], top_k=5, threshold=0.0)
            assert results == []

    def test_vec_to_bytes(self):
        """_vec_to_bytes should pack floats as little-endian float32."""
        from bedrock_smart_router.redis_vector_store import RedisVectorStore
        result = RedisVectorStore._vec_to_bytes([1.0, 2.0, 3.0])
        expected = struct.pack("3f", 1.0, 2.0, 3.0)
        assert result == expected
