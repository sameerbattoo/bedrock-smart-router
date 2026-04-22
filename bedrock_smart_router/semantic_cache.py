"""Semantic response cache — embedding-based similarity matching.

Unlike the exact-match cache, the semantic cache uses embedding
similarity to match requests that are phrased differently but have
the same intent.  This dramatically increases cache hit rates for
workloads like customer support and FAQ bots.

**Variable-aware caching:**

Queries with variables (e.g. "Find top 5 users for @category in @year")
are semantically identical but have different correct answers depending
on the variable values.  Pass ``variables`` to ``get()`` and ``put()``
to ensure the cache only hits when both the intent AND the variable
values match::

    cache.put("Find top 5 users for Electronics in 2024", response,
              variables={"category": "Electronics", "year": "2024"})

    # Same intent, same variables → HIT
    cache.get("Show me the top 5 users in Electronics for 2024",
              variables={"category": "Electronics", "year": "2024"})

    # Same intent, different variables → MISS
    cache.get("Find top 5 users for Clothing in 2025",
              variables={"category": "Clothing", "year": "2025"})

Vector store backends:
  - ``memory`` (default): In-process brute-force. Good for dev.
  - ``faiss``: Fast in-process ANN. ``pip install bedrock-smart-router[faiss]``
  - ``redis``: Shared via RediSearch. ``pip install bedrock-smart-router[redis]``
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from typing import Any

from bedrock_smart_router.vector_store import VectorStore, build_vector_store

logger = logging.getLogger(__name__)


@dataclass
class SemanticCacheConfig:
    """Semantic cache configuration."""

    enabled: bool = False
    threshold: float = 0.92  # Cosine similarity threshold (0.0–1.0)
    embedding_model: str = "amazon.titan-embed-text-v2:0"
    embedding_dimension: int = 1024  # Titan v2 default
    max_entries: int = 5000
    ttl_seconds: float = 3600.0
    # Vector store backend
    vector_store_backend: str = "memory"  # "memory" | "faiss" | "redis"
    redis_url: str = ""
    redis_key_prefix: str = "bsr:semcache:"


class SemanticCache:
    """Embedding-based semantic response cache.

    Stores responses keyed by embedding vectors.  On lookup, computes
    the embedding of the query and finds the nearest cached entry
    above the similarity threshold.
    """

    def __init__(
        self,
        config: SemanticCacheConfig | None = None,
        boto_session: Any | None = None,
        region: str = "us-west-2",
        vector_store: VectorStore | None = None,
    ) -> None:
        self.config = config or SemanticCacheConfig()
        self._session = boto_session
        self._region = region
        self._hits = 0
        self._misses = 0
        # LRU cache for embeddings — avoids redundant Bedrock API calls
        self._embedding_cache: dict[str, list[float]] = {}
        self._embedding_cache_max = 500

        # Use provided vector store or build from config
        self._store = vector_store or build_vector_store(
            backend=self.config.vector_store_backend,
            dimension=self.config.embedding_dimension,
            max_entries=self.config.max_entries,
            redis_url=self.config.redis_url,
            key_prefix=self.config.redis_key_prefix,
        )

    def _get_embedding(self, text: str) -> list[float]:
        """Compute an embedding vector using Bedrock, with local caching."""
        # Check local embedding cache first
        cache_key = hashlib.sha256(text.encode()).hexdigest()[:16]
        cached = self._embedding_cache.get(cache_key)
        if cached is not None:
            return cached

        if self._session is None:
            import boto3
            self._session = boto3.Session(region_name=self._region)
        client = self._session.client("bedrock-runtime", region_name=self._region)
        resp = client.invoke_model(
            modelId=self.config.embedding_model,
            body=json.dumps({"inputText": text}),
        )
        body = json.loads(resp["body"].read())
        embedding = body.get("embedding", [])

        # Store in local cache (evict oldest if full)
        if len(self._embedding_cache) >= self._embedding_cache_max:
            oldest = next(iter(self._embedding_cache))
            del self._embedding_cache[oldest]
        self._embedding_cache[cache_key] = embedding

        return embedding

    def get(
        self,
        query_text: str,
        variables: dict[str, str] | None = None,
    ) -> dict[str, Any] | None:
        """Look up a semantically similar cached response.

        Args:
            query_text: The user's query.
            variables: Optional variable values.  When provided, the
                cache only hits if both the semantic similarity is
                above threshold AND the variable values match exactly.
        """
        if not self.config.enabled:
            self._misses += 1
            return None

        if self._store.count() == 0:
            self._misses += 1
            return None

        query_emb = self._get_embedding(query_text)
        # Fetch more candidates when variables are set, since we need
        # to filter by variable match after semantic search
        top_k = 10 if variables else 1
        results = self._store.search(
            query_emb, top_k=top_k, threshold=self.config.threshold,
        )

        var_hash = self._hash_variables(variables) if variables else None

        for result in results:
            payload = result.payload

            # Check TTL
            created_at = payload.get("created_at", 0)
            if time.time() - created_at > self.config.ttl_seconds:
                self._store.delete(result.id)
                continue

            # Check variable match (if variables provided)
            if var_hash is not None:
                stored_var_hash = payload.get("var_hash", "")
                if stored_var_hash != var_hash:
                    continue  # Same intent but different variables — skip

            self._hits += 1
            logger.debug(
                "Semantic cache HIT (score=%.3f, query='%s', vars=%s)",
                result.score, query_text[:50], variables,
            )
            return payload.get("response")

        self._misses += 1
        return None

    def put(
        self,
        query_text: str,
        response: dict[str, Any],
        variables: dict[str, str] | None = None,
    ) -> None:
        """Store a response with its embedding.

        Args:
            query_text: The user's query.
            response: The Bedrock response to cache.
            variables: Optional variable values.  When provided, the
                cached entry will only match queries with the same
                variable values.
        """
        if not self.config.enabled:
            return

        embedding = self._get_embedding(query_text)
        var_hash = self._hash_variables(variables) if variables else ""
        # Include var_hash in the entry ID so different variables get
        # different entries even for semantically identical queries
        entry_id = hashlib.sha256(
            f"{query_text}|{var_hash}".encode()
        ).hexdigest()[:16]

        self._store.add(
            id=entry_id,
            embedding=embedding,
            payload={
                "response": response,
                "query": query_text,
                "variables": variables or {},
                "var_hash": var_hash,
                "created_at": time.time(),
            },
        )

    @staticmethod
    def _hash_variables(variables: dict[str, str]) -> str:
        """Deterministic hash of variable key-value pairs."""
        if not variables:
            return ""
        sorted_pairs = "|".join(
            f"{k}={v}" for k, v in sorted(variables.items())
        )
        return hashlib.sha256(sorted_pairs.encode()).hexdigest()[:12]

    def invalidate(self) -> int:
        """Clear all cached entries."""
        return self._store.clear()

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self.hit_rate, 4),
            "entries": self._store.count(),
            "backend": self.config.vector_store_backend,
            "threshold": self.config.threshold,
            "embedding_model": self.config.embedding_model,
        }
