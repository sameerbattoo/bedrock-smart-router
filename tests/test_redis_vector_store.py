# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for RedisVectorStore — validates raw FT command approach.

These tests mock the Redis connection so they run without a live Redis
instance.  They verify that:
1. The store uses raw execute_command (FT.CREATE, FT.SEARCH, FT.INFO)
   instead of the redis.commands.search Python wrapper — avoiding
   import compatibility issues across redis-py versions
2. The vector store logic (add, search, delete, clear) works correctly
   against a mocked Redis client
3. FT.SEARCH response parsing handles the raw array format correctly
"""

import json
import struct
from unittest.mock import MagicMock, patch, call

import pytest


class TestRedisVectorStoreUnit:
    """Unit tests for RedisVectorStore with mocked Redis client."""

    @pytest.fixture
    def mock_redis(self):
        """Create a mock Redis client."""
        mock_client = MagicMock()
        # FT.INFO raises → index doesn't exist → FT.CREATE succeeds
        def execute_side_effect(*args, **kwargs):
            if args[0] == "FT.INFO":
                raise Exception("Unknown index name")
            if args[0] == "FT.CREATE":
                return "OK"
            return None
        mock_client.execute_command.side_effect = execute_side_effect
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

    def test_init_uses_execute_command(self, store, mock_redis):
        """Store should use raw FT.INFO and FT.CREATE, not ft() wrapper."""
        calls = mock_redis.execute_command.call_args_list
        # First call: FT.INFO to check if index exists
        assert calls[0] == call("FT.INFO", "bsr_vectors")
        # Second call: FT.CREATE to create the index
        assert calls[1][0][0] == "FT.CREATE"
        assert calls[1][0][1] == "bsr_vectors"
        assert store._index_created is True

    def test_init_existing_index(self):
        """If FT.INFO succeeds, FT.CREATE should not be called."""
        mock_client = MagicMock()
        mock_client.execute_command.return_value = ["some", "info"]  # FT.INFO succeeds
        with patch("redis.Redis.from_url", return_value=mock_client):
            from bedrock_smart_router.redis_vector_store import RedisVectorStore
            store = RedisVectorStore(
                redis_url="redis://localhost:6379",
                dimension=3,
            )
        assert store._index_created is True
        # Only FT.INFO should have been called, not FT.CREATE
        calls = mock_client.execute_command.call_args_list
        assert len(calls) == 1
        assert calls[0] == call("FT.INFO", "bsr_vectors")

    def test_ft_create_includes_dimension(self, mock_redis):
        """FT.CREATE should include the configured dimension."""
        with patch("redis.Redis.from_url", return_value=mock_redis):
            from bedrock_smart_router.redis_vector_store import RedisVectorStore
            RedisVectorStore(
                redis_url="redis://localhost:6379",
                dimension=768,
            )
        create_call = mock_redis.execute_command.call_args_list[1]
        args = create_call[0]
        # Find DIM in the args
        dim_idx = list(args).index("DIM")
        assert args[dim_idx + 1] == "768"

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
        expected_bytes = struct.pack("3f", 1.0, 0.0, 0.0)
        assert mapping["embedding"] == expected_bytes

    def test_search_uses_execute_command(self, store, mock_redis):
        """search() should use raw FT.SEARCH, not ft().search()."""
        # Mock FT.SEARCH response: [count, key, [fields...]]
        mock_redis.execute_command.side_effect = None
        mock_redis.execute_command.return_value = [
            1,  # total count
            b"test:vec:doc1",
            [b"entry_id", b"doc1", b"payload", b'{"text":"hello"}', b"score", b"0.1"],
        ]
        results = store.search([1.0, 0.0, 0.0], top_k=5, threshold=0.0)

        # Verify FT.SEARCH was called via execute_command
        ft_search_call = mock_redis.execute_command.call_args
        assert ft_search_call[0][0] == "FT.SEARCH"
        assert ft_search_call[0][1] == "bsr_vectors"

        assert len(results) == 1
        assert results[0].id == "doc1"
        assert results[0].payload == {"text": "hello"}
        # distance 0.1 → similarity = 1 - (0.1 / 2) = 0.95
        assert results[0].score == pytest.approx(0.95)

    def test_search_threshold_filtering(self, store, mock_redis):
        """search() should filter results below the similarity threshold."""
        mock_redis.execute_command.side_effect = None
        mock_redis.execute_command.return_value = [
            2,
            b"test:vec:a", [b"entry_id", b"a", b"payload", b"{}", b"score", b"0.1"],
            b"test:vec:b", [b"entry_id", b"b", b"payload", b"{}", b"score", b"1.8"],
        ]
        # threshold 0.5 → only doc "a" (similarity 0.95) passes, "b" (similarity 0.1) doesn't
        results = store.search([1.0, 0.0, 0.0], top_k=5, threshold=0.5)
        assert len(results) == 1
        assert results[0].id == "a"

    def test_search_empty_results(self, store, mock_redis):
        """search() should return empty list when no results."""
        mock_redis.execute_command.side_effect = None
        mock_redis.execute_command.return_value = [0]
        results = store.search([1.0, 0.0, 0.0], top_k=5, threshold=0.0)
        assert results == []

    def test_search_when_index_not_created(self):
        """search() should return empty list if index creation failed."""
        mock_client = MagicMock()
        mock_client.execute_command.side_effect = Exception("no module")
        with patch("redis.Redis.from_url", return_value=mock_client):
            from bedrock_smart_router.redis_vector_store import RedisVectorStore
            store = RedisVectorStore(
                redis_url="redis://localhost:6379",
                dimension=3,
            )
            assert store._index_created is False
            results = store.search([1.0, 0.0, 0.0], top_k=5, threshold=0.0)
            assert results == []

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

    def test_vec_to_bytes(self):
        """_vec_to_bytes should pack floats as little-endian float32."""
        from bedrock_smart_router.redis_vector_store import RedisVectorStore
        result = RedisVectorStore._vec_to_bytes([1.0, 2.0, 3.0])
        expected = struct.pack("3f", 1.0, 2.0, 3.0)
        assert result == expected

    def test_no_redis_search_imports(self):
        """The module should NOT import from redis.commands.search at all."""
        import ast
        from pathlib import Path
        source = Path("bedrock_smart_router/redis_vector_store.py").read_text()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                module = getattr(node, "module", "") or ""
                assert "redis.commands.search" not in module, (
                    f"Found import from redis.commands.search: {module}. "
                    "The store should use raw execute_command for Valkey compatibility."
                )
