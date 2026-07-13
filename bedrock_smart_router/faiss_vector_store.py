# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""FAISS vector store — fast in-process approximate nearest neighbor.

Uses Facebook AI Similarity Search for sub-millisecond vector lookups.
Good for single-instance deployments with up to ~100K entries.

Memory safety:
  - Bounded by ``max_entries`` (default 50,000). Oldest entries are
    evicted when the limit is reached.
  - Soft-deleted entries are compacted automatically when garbage
    exceeds 20% of the index, preventing unbounded growth of the
    internal ID map and deleted-set.

Install::

    pip install bedrock-smart-router[faiss]

Or directly::

    pip install faiss-cpu
"""

from __future__ import annotations

import logging
import time
from collections import OrderedDict
from typing import Any

import numpy as np

from bedrock_smart_router.vector_store import SearchResult, VectorStore

logger = logging.getLogger(__name__)

# Compact the index when soft-deleted entries exceed this fraction
_COMPACTION_THRESHOLD = 0.20


class FAISSVectorStore(VectorStore):
    """FAISS-backed vector store using IndexFlatIP (inner product).

    Vectors are L2-normalized before indexing so inner product equals
    cosine similarity.

    Memory is bounded by ``max_entries``.  When the limit is reached,
    the oldest entries (by insertion time) are evicted and the index
    is rebuilt.  Soft-deleted entries are compacted automatically when
    they exceed 20% of the index size.
    """

    def __init__(self, dimension: int = 1024, max_entries: int = 50_000) -> None:
        """
        Args:
            dimension: Embedding vector dimension.  Titan Embed v2
                produces 1024-dim vectors by default.
            max_entries: Maximum number of live entries before eviction.
                When exceeded, the oldest 10% of entries are evicted
                and the index is rebuilt.
        """
        try:
            import faiss
        except ImportError:
            raise ImportError(
                "FAISS vector store requires the 'faiss-cpu' package. "
                "Install with: pip install bedrock-smart-router[faiss]"
            )

        self._dimension = dimension
        self._max_entries = max_entries
        self._index = faiss.IndexFlatIP(dimension)  # Inner product (cosine after normalization)
        self._id_map: list[str] = []  # Position → ID
        # OrderedDict preserves insertion order for LRU eviction.
        # Values hold the raw (pre-normalized) embedding for rebuild.
        self._entries: OrderedDict[str, _FAISSEntry] = OrderedDict()
        self._deleted_ids: set[str] = set()  # Soft-deleted IDs (FAISS doesn't support removal)
        logger.info(
            "FAISS vector store initialized (dim=%d, max_entries=%d)",
            dimension, max_entries,
        )

    @staticmethod
    def _normalize(vec: list[float]) -> np.ndarray:
        """L2-normalize a vector so inner product = cosine similarity."""
        arr = np.array(vec, dtype=np.float32).reshape(1, -1)
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr = arr / norm
        return arr

    def add(self, id: str, embedding: list[float], payload: dict[str, Any]) -> None:
        # If this ID already exists, treat as an upsert — soft-delete the old one
        if id in self._entries:
            self._deleted_ids.add(id)

        vec = self._normalize(embedding)
        self._index.add(vec)
        self._id_map.append(id)
        self._entries[id] = _FAISSEntry(
            embedding=embedding, payload=payload, created_at=time.time(),
        )
        # Move to end (most recent) for LRU tracking
        self._entries.move_to_end(id)

        # Evict if over capacity
        if len(self._entries) > self._max_entries:
            self._evict()
        elif self._should_compact():
            self._compact()

    def search(
        self, embedding: list[float], top_k: int = 1, threshold: float = 0.0
    ) -> list[SearchResult]:
        if self._index.ntotal == 0:
            return []

        vec = self._normalize(embedding)
        # Fetch extra candidates to account for soft-deleted entries
        k = min(top_k + len(self._deleted_ids), self._index.ntotal)
        scores, indices = self._index.search(vec, k)

        results: list[SearchResult] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._id_map):
                continue
            entry_id = self._id_map[idx]
            if entry_id in self._deleted_ids:
                continue  # Skip soft-deleted entries
            if entry_id not in self._entries:
                continue  # Evicted entry still in index
            if score < threshold:
                continue
            results.append(SearchResult(
                id=entry_id,
                score=float(score),
                payload=self._entries[entry_id].payload,
            ))
            if len(results) >= top_k:
                break
        return results

    def delete(self, id: str) -> bool:
        """Soft-delete an entry. Triggers compaction if garbage ratio is high."""
        if id in self._entries:
            del self._entries[id]
            self._deleted_ids.add(id)
            if self._should_compact():
                self._compact()
            return True
        return False

    def clear(self) -> int:
        import faiss
        count = len(self._entries)
        self._index = faiss.IndexFlatIP(self._dimension)
        self._id_map.clear()
        self._entries.clear()
        self._deleted_ids.clear()
        return count

    def count(self) -> int:
        return len(self._entries)

    # ── Internal: eviction and compaction ───────────────────────────

    def _evict(self) -> None:
        """Evict the oldest 10% of entries and rebuild the index."""
        evict_count = max(1, self._max_entries // 10)
        evicted = 0
        while self._entries and evicted < evict_count:
            oldest_id, _ = self._entries.popitem(last=False)
            self._deleted_ids.add(oldest_id)
            evicted += 1
        logger.debug("Evicted %d oldest entries (live=%d)", evicted, len(self._entries))
        self._compact()

    def _should_compact(self) -> bool:
        """Check if garbage ratio warrants a full index rebuild."""
        total_in_index = self._index.ntotal
        if total_in_index == 0:
            return False
        garbage = len(self._deleted_ids)
        return (garbage / total_in_index) > _COMPACTION_THRESHOLD

    def _compact(self) -> None:
        """Rebuild the FAISS index from live entries only.

        This reclaims memory from soft-deleted and evicted vectors.
        """
        import faiss

        live_ids = list(self._entries.keys())
        if not live_ids:
            self._index = faiss.IndexFlatIP(self._dimension)
            self._id_map.clear()
            self._deleted_ids.clear()
            return

        # Rebuild index from live entries
        new_index = faiss.IndexFlatIP(self._dimension)
        vectors = np.vstack([
            self._normalize(self._entries[eid].embedding)
            for eid in live_ids
        ])
        new_index.add(vectors)

        self._index = new_index
        self._id_map = live_ids
        self._deleted_ids.clear()
        logger.debug("Compacted FAISS index (entries=%d)", len(live_ids))


class _FAISSEntry:
    """Internal entry holding embedding + payload + insertion time."""

    __slots__ = ("embedding", "payload", "created_at")

    def __init__(
        self, embedding: list[float], payload: dict[str, Any], created_at: float
    ) -> None:
        self.embedding = embedding
        self.payload = payload
        self.created_at = created_at
