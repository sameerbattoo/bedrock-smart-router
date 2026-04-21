"""Semantic response cache (optional — requires embeddings extra).

Unlike the exact-match cache, the semantic cache uses embedding
similarity to match requests that are phrased differently but have
the same intent.  This dramatically increases cache hit rates for
workloads like customer support and FAQ bots.

Install with::

    pip install bedrock-smart-router[embeddings]

Usage::

    router = BedrockRouter.create({
        "cache": {"type": "semantic", "threshold": 0.95},
    })
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class SemanticCacheConfig:
    """Semantic cache configuration."""

    enabled: bool = False
    threshold: float = 0.95  # Cosine similarity threshold
    embedding_model: str = "amazon.titan-embed-text-v2:0"
    max_entries: int = 5000
    ttl_seconds: float = 3600.0


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
    ) -> None:
        self.config = config or SemanticCacheConfig()
        self._session = boto_session
        self._region = region
        self._entries: list[dict[str, Any]] = []
        self._hits = 0
        self._misses = 0

    def _get_embedding(self, text: str) -> list[float]:
        """Compute an embedding vector using Bedrock Titan Embeddings."""
        if self._session is None:
            import boto3
            self._session = boto3.Session(region_name=self._region)
        client = self._session.client("bedrock-runtime", region_name=self._region)
        import json
        resp = client.invoke_model(
            modelId=self.config.embedding_model,
            body=json.dumps({"inputText": text}),
        )
        body = json.loads(resp["body"].read())
        return body.get("embedding", [])

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def get(self, query_text: str) -> dict[str, Any] | None:
        """Look up a semantically similar cached response."""
        if not self.config.enabled or not self._entries:
            self._misses += 1
            return None

        query_emb = self._get_embedding(query_text)
        now = time.monotonic()

        best_score = 0.0
        best_entry: dict[str, Any] | None = None

        for entry in self._entries:
            if now - entry["created_at"] > self.config.ttl_seconds:
                continue
            score = self._cosine_similarity(query_emb, entry["embedding"])
            if score > best_score:
                best_score = score
                best_entry = entry

        if best_entry and best_score >= self.config.threshold:
            self._hits += 1
            return best_entry["response"]

        self._misses += 1
        return None

    def put(self, query_text: str, response: dict[str, Any]) -> None:
        """Store a response with its embedding."""
        if not self.config.enabled:
            return
        embedding = self._get_embedding(query_text)
        self._entries.append({
            "embedding": embedding,
            "response": response,
            "query": query_text,
            "created_at": time.monotonic(),
        })
        # Evict oldest if over capacity
        while len(self._entries) > self.config.max_entries:
            self._entries.pop(0)

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0
