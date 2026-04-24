"""Tests for the FAISS vector store backend."""

import pytest

from bedrock_smart_router.faiss_vector_store import FAISSVectorStore
from bedrock_smart_router.vector_store import build_vector_store


class TestFAISSVectorStore:
    def setup_method(self):
        self.store = FAISSVectorStore(dimension=4)

    def test_add_and_search(self):
        self.store.add("a", [1.0, 0.0, 0.0, 0.0], {"text": "hello"})
        results = self.store.search([1.0, 0.0, 0.0, 0.0], top_k=1, threshold=0.9)
        assert len(results) == 1
        assert results[0].id == "a"
        assert results[0].score >= 0.99
        assert results[0].payload["text"] == "hello"

    def test_similar_vectors_match(self):
        self.store.add("a", [1.0, 0.1, 0.0, 0.0], {"text": "a"})
        self.store.add("b", [0.0, 0.0, 1.0, 0.0], {"text": "b"})
        results = self.store.search([1.0, 0.0, 0.0, 0.0], top_k=1, threshold=0.5)
        assert len(results) == 1
        assert results[0].id == "a"  # More similar to query

    def test_threshold_filters(self):
        self.store.add("a", [1.0, 0.0, 0.0, 0.0], {})
        self.store.add("b", [0.0, 1.0, 0.0, 0.0], {})
        results = self.store.search([1.0, 0.0, 0.0, 0.0], top_k=5, threshold=0.9)
        assert len(results) == 1  # Only "a" above 0.9

    def test_top_k(self):
        for i in range(10):
            vec = [0.0] * 4
            vec[i % 4] = 1.0
            self.store.add(f"v{i}", vec, {"i": i})
        results = self.store.search([1.0, 0.0, 0.0, 0.0], top_k=3, threshold=0.0)
        assert len(results) == 3

    def test_empty_search(self):
        results = self.store.search([1.0, 0.0, 0.0, 0.0], top_k=5)
        assert results == []

    def test_count(self):
        assert self.store.count() == 0
        self.store.add("a", [1.0, 0.0, 0.0, 0.0], {})
        self.store.add("b", [0.0, 1.0, 0.0, 0.0], {})
        assert self.store.count() == 2

    def test_delete(self):
        self.store.add("a", [1.0, 0.0, 0.0, 0.0], {"x": 1})
        assert self.store.delete("a")
        assert self.store.count() == 0

    def test_delete_nonexistent(self):
        assert not self.store.delete("nope")

    def test_clear(self):
        for i in range(5):
            self.store.add(f"v{i}", [float(i), 0.0, 0.0, 0.0], {})
        count = self.store.clear()
        assert count == 5
        assert self.store.count() == 0

    def test_many_vectors(self):
        """FAISS should handle hundreds of vectors efficiently."""
        import random
        for i in range(500):
            vec = [random.random() for _ in range(4)]
            self.store.add(f"v{i}", vec, {"i": i})
        assert self.store.count() == 500
        results = self.store.search([1.0, 0.0, 0.0, 0.0], top_k=5, threshold=0.0)
        assert len(results) == 5

    def test_normalized_cosine(self):
        """Inner product after normalization should equal cosine similarity."""
        self.store.add("a", [3.0, 4.0, 0.0, 0.0], {})  # norm = 5
        results = self.store.search([6.0, 8.0, 0.0, 0.0], top_k=1)  # norm = 10, same direction
        assert len(results) == 1
        assert results[0].score == pytest.approx(1.0, abs=0.01)


class TestBuildFAISS:
    def test_build_faiss(self):
        store = build_vector_store(backend="faiss", dimension=8)
        assert isinstance(store, FAISSVectorStore)
        store.add("test", [1.0] * 8, {"x": 1})
        assert store.count() == 1
