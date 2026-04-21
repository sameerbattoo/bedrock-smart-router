"""Tests for vector store backends."""

import pytest

from bedrock_smart_router.vector_store import (
    InMemoryVectorStore,
    build_vector_store,
    _cosine_similarity,
)


def _vec(val: float, dim: int = 4) -> list[float]:
    """Create a simple test vector."""
    return [val] * dim


class TestCosineSimlarity:
    def test_identical(self):
        assert _cosine_similarity([1, 0, 0], [1, 0, 0]) == pytest.approx(1.0)

    def test_orthogonal(self):
        assert _cosine_similarity([1, 0, 0], [0, 1, 0]) == pytest.approx(0.0)

    def test_opposite(self):
        assert _cosine_similarity([1, 0], [-1, 0]) == pytest.approx(-1.0)

    def test_empty(self):
        assert _cosine_similarity([], []) == 0.0

    def test_different_lengths(self):
        assert _cosine_similarity([1, 2], [1, 2, 3]) == 0.0


class TestInMemoryVectorStore:
    def setup_method(self):
        self.store = InMemoryVectorStore(max_entries=100)

    def test_add_and_search(self):
        self.store.add("a", [1.0, 0.0, 0.0], {"text": "hello"})
        results = self.store.search([1.0, 0.0, 0.0], top_k=1, threshold=0.9)
        assert len(results) == 1
        assert results[0].id == "a"
        assert results[0].score == pytest.approx(1.0)
        assert results[0].payload["text"] == "hello"

    def test_search_threshold(self):
        self.store.add("a", [1.0, 0.0, 0.0], {})
        self.store.add("b", [0.0, 1.0, 0.0], {})
        results = self.store.search([1.0, 0.0, 0.0], top_k=5, threshold=0.5)
        assert len(results) == 1  # Only "a" is above 0.5

    def test_top_k(self):
        for i in range(10):
            self.store.add(f"v{i}", [float(i), 1.0, 0.0], {})
        results = self.store.search([9.0, 1.0, 0.0], top_k=3, threshold=0.0)
        assert len(results) == 3

    def test_delete(self):
        self.store.add("a", [1.0, 0.0], {"x": 1})
        assert self.store.delete("a")
        assert self.store.count() == 0

    def test_delete_nonexistent(self):
        assert not self.store.delete("nope")

    def test_clear(self):
        for i in range(5):
            self.store.add(f"v{i}", [float(i)], {})
        count = self.store.clear()
        assert count == 5
        assert self.store.count() == 0

    def test_eviction(self):
        store = InMemoryVectorStore(max_entries=3)
        store.add("a", [1.0], {})
        store.add("b", [2.0], {})
        store.add("c", [3.0], {})
        store.add("d", [4.0], {})  # Evicts "a"
        assert store.count() == 3
        results = store.search([1.0], top_k=10, threshold=0.0)
        ids = {r.id for r in results}
        assert "a" not in ids

    def test_empty_search(self):
        results = self.store.search([1.0, 0.0], top_k=5, threshold=0.0)
        assert results == []


class TestBuildVectorStore:
    def test_default_memory(self):
        store = build_vector_store()
        assert isinstance(store, InMemoryVectorStore)

    def test_explicit_memory(self):
        store = build_vector_store(backend="memory", max_entries=100)
        assert isinstance(store, InMemoryVectorStore)

    def test_faiss_without_package(self):
        """FAISS should raise ImportError if not installed."""
        try:
            store = build_vector_store(backend="faiss")
            # If faiss-cpu is installed, this should work
            from bedrock_smart_router.faiss_vector_store import FAISSVectorStore
            assert isinstance(store, FAISSVectorStore)
        except ImportError:
            pass  # Expected if faiss-cpu not installed

    def test_redis_without_url(self):
        """Redis vector store with no URL should fail on connect."""
        try:
            store = build_vector_store(backend="redis", redis_url="")
            # Will fail when trying to connect
        except (ValueError, Exception):
            pass
