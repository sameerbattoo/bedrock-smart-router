"""Redis vector store — shared vector search via RediSearch.

Uses Redis with the RediSearch module for vector similarity search.
Shared across all instances, works with ElastiCache and Valkey.

Requires Redis 7+ with the RediSearch module, or ElastiCache with
vector search enabled.

Install::

    pip install bedrock-smart-router[redis]

Configuration::

    semantic_cache:
      vector_store: redis
      redis_url: "redis://localhost:6379"
      key_prefix: "bsr:vec:"
"""

from __future__ import annotations

import json
import logging
import struct
from typing import Any

from bedrock_smart_router.vector_store import SearchResult, VectorStore

logger = logging.getLogger(__name__)

_INDEX_NAME = "bsr_vectors"


class RedisVectorStore(VectorStore):
    """Redis-backed vector store using RediSearch vector similarity.

    Each vector is stored as a Redis Hash with a VECTOR field.
    Search uses RediSearch's KNN query.
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        key_prefix: str = "bsr:vec:",
        dimension: int = 1024,
    ) -> None:
        try:
            import redis
        except ImportError:
            raise ImportError(
                "Redis vector store requires the 'redis' package. "
                "Install with: pip install bedrock-smart-router[redis]"
            )

        self._prefix = key_prefix
        self._dimension = dimension
        self._client = redis.Redis.from_url(
            redis_url, decode_responses=False, socket_timeout=5,
        )
        self._index_created = False
        self._ensure_index()

    def _ensure_index(self) -> None:
        """Create the RediSearch index if it doesn't exist."""
        try:
            from redis.commands.search.field import TagField, VectorField
            from redis.commands.search.index_definition import IndexDefinition, IndexType

            self._client.ft(_INDEX_NAME).info()
            self._index_created = True
            logger.debug("RediSearch index '%s' already exists", _INDEX_NAME)
        except Exception:
            try:
                from redis.commands.search.field import TagField, VectorField
                from redis.commands.search.index_definition import IndexDefinition, IndexType

                schema = (
                    TagField("entry_id"),
                    VectorField(
                        "embedding",
                        "FLAT",
                        {
                            "TYPE": "FLOAT32",
                            "DIM": self._dimension,
                            "DISTANCE_METRIC": "COSINE",
                        },
                    ),
                )
                self._client.ft(_INDEX_NAME).create_index(
                    schema,
                    definition=IndexDefinition(
                        prefix=[self._prefix],
                        index_type=IndexType.HASH,
                    ),
                )
                self._index_created = True
                logger.info("Created RediSearch index '%s'", _INDEX_NAME)
            except Exception as exc:
                logger.warning(
                    "Could not create RediSearch index: %s. "
                    "Vector search may not work. Ensure Redis has the "
                    "RediSearch module loaded.",
                    exc,
                )

    @staticmethod
    def _vec_to_bytes(vec: list[float]) -> bytes:
        """Convert a float list to bytes for Redis VECTOR field."""
        return struct.pack(f"{len(vec)}f", *vec)

    def add(self, id: str, embedding: list[float], payload: dict[str, Any]) -> None:
        key = f"{self._prefix}{id}"
        self._client.hset(key, mapping={
            "entry_id": id.encode(),
            "embedding": self._vec_to_bytes(embedding),
            "payload": json.dumps(payload).encode(),
        })

    def search(
        self, embedding: list[float], top_k: int = 1, threshold: float = 0.0
    ) -> list[SearchResult]:
        if not self._index_created:
            return []

        try:
            from redis.commands.search.query import Query

            query_vec = self._vec_to_bytes(embedding)
            q = (
                Query(f"*=>[KNN {top_k} @embedding $vec AS score]")
                .sort_by("score")
                .return_fields("entry_id", "payload", "score")
                .dialect(2)
            )
            results = self._client.ft(_INDEX_NAME).search(
                q, query_params={"vec": query_vec}
            )

            out: list[SearchResult] = []
            for doc in results.docs:
                # RediSearch COSINE distance: 0 = identical, 2 = opposite
                # Convert to similarity: 1 - (distance / 2)
                distance = float(doc.score)
                similarity = 1.0 - (distance / 2.0)
                if similarity < threshold:
                    continue
                payload_raw = getattr(doc, "payload", b"{}")
                if isinstance(payload_raw, bytes):
                    payload_raw = payload_raw.decode()
                out.append(SearchResult(
                    id=getattr(doc, "entry_id", b"").decode() if isinstance(getattr(doc, "entry_id", ""), bytes) else str(getattr(doc, "entry_id", "")),
                    score=round(similarity, 4),
                    payload=json.loads(payload_raw) if payload_raw else {},
                ))
            return out

        except Exception as exc:
            logger.warning("Redis vector search failed: %s", exc)
            return []

    def delete(self, id: str) -> bool:
        key = f"{self._prefix}{id}"
        return self._client.delete(key) > 0

    def clear(self) -> int:
        count = 0
        cursor = 0
        while True:
            cursor, keys = self._client.scan(
                cursor, match=f"{self._prefix}*", count=100
            )
            if keys:
                count += self._client.delete(*keys)
            if cursor == 0:
                break
        return count

    def count(self) -> int:
        count = 0
        cursor = 0
        while True:
            cursor, keys = self._client.scan(
                cursor, match=f"{self._prefix}*", count=100
            )
            count += len(keys)
            if cursor == 0:
                break
        return count
