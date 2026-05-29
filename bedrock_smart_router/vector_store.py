"""Pluggable vector store backends for semantic cache and routing.

Backends:
  - ``memory``: In-process list with brute-force cosine similarity.
    Good for dev and small caches (~500 entries).
  - ``faiss``: Facebook AI Similarity Search. Fast in-process ANN.
    Good for single-instance with up to ~100K entries.
    Requires: ``pip install faiss-cpu``
  - ``redis``: Redis 7+ with RediSearch module, or ElastiCache Valkey 8.2+.
    Shared across instances.
    Requires: ``pip install redis``
  - ``opensearch``: Amazon OpenSearch Serverless or managed.
    Scales to billions of vectors.
    Requires: ``pip install opensearch-py requests-aws4auth``
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class VectorEntry:
    """A stored vector with its payload."""

    id: str
    embedding: list[float]
    payload: dict[str, Any]
    created_at: float = 0.0


@dataclass
class SearchResult:
    """A vector search result."""

    id: str
    score: float
    payload: dict[str, Any]


class VectorStore(ABC):
    """Abstract vector store interface."""

    @abstractmethod
    def add(self, id: str, embedding: list[float], payload: dict[str, Any]) -> None:
        """Store a vector with its payload."""
        ...

    @abstractmethod
    def search(
        self, embedding: list[float], top_k: int = 1, threshold: float = 0.0
    ) -> list[SearchResult]:
        """Find the nearest vectors above the similarity threshold."""
        ...

    @abstractmethod
    def delete(self, id: str) -> bool:
        """Delete a vector by ID."""
        ...

    @abstractmethod
    def clear(self) -> int:
        """Delete all vectors. Returns count deleted."""
        ...

    @abstractmethod
    def count(self) -> int:
        """Return the number of stored vectors."""
        ...


# ── In-Memory Backend ───────────────────────────────────────────────

from bedrock_smart_router.utils import cosine_similarity as _cosine_similarity


class InMemoryVectorStore(VectorStore):
    """Brute-force cosine similarity over a Python list.

    Good for development and small caches (~500 entries).
    No external dependencies.
    """

    def __init__(self, max_entries: int = 5000) -> None:
        self._entries: dict[str, VectorEntry] = {}
        self._max = max_entries

    def add(self, id: str, embedding: list[float], payload: dict[str, Any]) -> None:
        self._entries[id] = VectorEntry(
            id=id, embedding=embedding, payload=payload, created_at=time.time(),
        )
        # Evict oldest if over capacity
        while len(self._entries) > self._max:
            oldest_key = next(iter(self._entries))
            del self._entries[oldest_key]

    def search(
        self, embedding: list[float], top_k: int = 1, threshold: float = 0.0
    ) -> list[SearchResult]:
        results: list[tuple[float, VectorEntry]] = []
        for entry in self._entries.values():
            score = _cosine_similarity(embedding, entry.embedding)
            if score >= threshold:
                results.append((score, entry))
        results.sort(key=lambda x: x[0], reverse=True)
        return [
            SearchResult(id=e.id, score=s, payload=e.payload)
            for s, e in results[:top_k]
        ]

    def delete(self, id: str) -> bool:
        return self._entries.pop(id, None) is not None

    def clear(self) -> int:
        count = len(self._entries)
        self._entries.clear()
        return count

    def count(self) -> int:
        return len(self._entries)


# ── Factory ─────────────────────────────────────────────────────────

def build_vector_store(
    backend: str = "memory",
    dimension: int = 1024,
    max_entries: int = 5000,
    redis_url: str = "",
    key_prefix: str = "bsr:vec:",
    opensearch_endpoint: str = "",
    opensearch_index_name: str = "bsr-semantic-cache",
    opensearch_region: str = "us-west-2",
) -> VectorStore:
    """Build the appropriate vector store backend.

    Args:
        backend: ``"memory"``, ``"faiss"``, ``"redis"``, or ``"opensearch"``.
        dimension: Embedding vector dimension (default 1024 for Titan v2).
        max_entries: Max entries for in-memory backend.
        redis_url: Redis connection URL (for redis backend).
        key_prefix: Key prefix for Redis backend.
        opensearch_endpoint: OpenSearch Serverless endpoint URL.
        opensearch_index_name: Index name for OpenSearch backend.
        opensearch_region: AWS region for OpenSearch Serverless.
    """
    if backend == "faiss":
        from bedrock_smart_router.faiss_vector_store import FAISSVectorStore
        return FAISSVectorStore(dimension=dimension, max_entries=max_entries)

    if backend in ("redis", "valkey"):
        from bedrock_smart_router.redis_vector_store import RedisVectorStore
        return RedisVectorStore(
            redis_url=redis_url, key_prefix=key_prefix, dimension=dimension,
        )

    if backend == "opensearch":
        from bedrock_smart_router.opensearch_vector_store import OpenSearchVectorStore
        return OpenSearchVectorStore(
            endpoint=opensearch_endpoint,
            index_name=opensearch_index_name,
            dimension=dimension,
            region=opensearch_region,
        )

    return InMemoryVectorStore(max_entries=max_entries)
