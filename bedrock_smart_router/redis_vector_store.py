"""Redis/Valkey vector store — shared vector search via FT commands.

Uses raw FT.CREATE / FT.SEARCH execute_command calls for compatibility
with both Redis (RediSearch module) and Amazon ElastiCache Valkey
(native vector search in Valkey 8.2+).

Install::

    pip install bedrock-smart-router[redis]

Configuration::

    semantic_cache:
      vector_store: redis
      redis_url: "rediss://your-valkey-endpoint:6379"
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
    """Redis/Valkey-backed vector store using FT vector similarity search.

    Uses raw execute_command calls for FT.CREATE and FT.SEARCH to ensure
    compatibility with both Redis+RediSearch and ElastiCache Valkey.
    Each vector is stored as a Redis Hash with a VECTOR field.
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
        """Create the vector search index if it doesn't exist.

        Uses raw FT.CREATE via execute_command for Valkey compatibility.
        """
        try:
            # Check if index already exists
            self._client.execute_command("FT.INFO", _INDEX_NAME)
            self._index_created = True
            logger.debug("Vector index '%s' already exists", _INDEX_NAME)
        except Exception:
            # Index doesn't exist — create it
            try:
                self._client.execute_command(
                    "FT.CREATE", _INDEX_NAME,
                    "ON", "HASH",
                    "PREFIX", "1", self._prefix,
                    "SCHEMA",
                    "entry_id", "TAG",
                    "embedding", "VECTOR", "FLAT", "6",
                    "TYPE", "FLOAT32",
                    "DIM", str(self._dimension),
                    "DISTANCE_METRIC", "COSINE",
                )
                self._index_created = True
                logger.info("Created vector index '%s' (dim=%d)", _INDEX_NAME, self._dimension)
            except Exception as exc:
                logger.warning(
                    "Could not create vector index '%s': %s. "
                    "Vector search may not work.",
                    _INDEX_NAME, exc,
                )

    @staticmethod
    def _vec_to_bytes(vec: list[float]) -> bytes:
        """Convert a float list to bytes for the VECTOR field."""
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
            query_vec = self._vec_to_bytes(embedding)
            query = f"*=>[KNN {top_k} @embedding $vec AS score]"

            result = self._client.execute_command(
                "FT.SEARCH", _INDEX_NAME,
                query,
                "PARAMS", "2", "vec", query_vec,
                "RETURN", "3", "entry_id", "payload", "score",
                "DIALECT", "2",
            )

            # Parse FT.SEARCH response: [count, key1, [field, val, ...], key2, ...]
            if not result or result[0] == 0:
                return []

            out: list[SearchResult] = []
            i = 1
            while i < len(result):
                _key = result[i]
                i += 1
                if i >= len(result):
                    break
                fields_raw = result[i]
                i += 1

                # Parse field pairs into a dict
                doc: dict[str, Any] = {}
                for j in range(0, len(fields_raw), 2):
                    fname = fields_raw[j]
                    fval = fields_raw[j + 1] if j + 1 < len(fields_raw) else None
                    if isinstance(fname, bytes):
                        fname = fname.decode()
                    doc[fname] = fval

                # Extract score and convert to similarity
                score_raw = doc.get("score")
                if score_raw is None:
                    continue
                distance = float(score_raw.decode() if isinstance(score_raw, bytes) else score_raw)
                # COSINE distance: 0 = identical, 2 = opposite
                similarity = 1.0 - (distance / 2.0)

                if similarity < threshold:
                    continue

                # Extract entry_id
                entry_id_raw = doc.get("entry_id", b"")
                entry_id = entry_id_raw.decode() if isinstance(entry_id_raw, bytes) else str(entry_id_raw)

                # Extract payload
                payload_raw = doc.get("payload", b"{}")
                if isinstance(payload_raw, bytes):
                    payload_raw = payload_raw.decode()

                out.append(SearchResult(
                    id=entry_id,
                    score=round(similarity, 4),
                    payload=json.loads(payload_raw) if payload_raw else {},
                ))

            return out

        except Exception as exc:
            logger.warning("Vector search failed: %s", exc)
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
