"""FAISS vector store — fast in-process approximate nearest neighbor.

Uses Facebook AI Similarity Search for sub-millisecond vector lookups.
Good for single-instance deployments with up to ~100K entries.

Install::

    pip install bedrock-smart-router[faiss]

Or directly::

    pip install faiss-cpu
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from bedrock_smart_router.vector_store import SearchResult, VectorStore

logger = logging.getLogger(__name__)


class FAISSVectorStore(VectorStore):
    """FAISS-backed vector store using IndexFlatIP (inner product).

    Vectors are L2-normalized before indexing so inner product equals
    cosine similarity.
    """

    def __init__(self, dimension: int = 1024) -> None:
        """
        Args:
            dimension: Embedding vector dimension.  Titan Embed v2
                produces 1024-dim vectors by default.
        """
        try:
            import faiss
        except ImportError:
            raise ImportError(
                "FAISS vector store requires the 'faiss-cpu' package. "
                "Install with: pip install bedrock-smart-router[faiss]"
            )

        self._dimension = dimension
        self._index = faiss.IndexFlatIP(dimension)  # Inner product (cosine after normalization)
        self._id_map: list[str] = []  # Position → ID
        self._payloads: dict[str, dict[str, Any]] = {}  # ID → payload
        self._deleted_ids: set[str] = set()  # Soft-deleted IDs (FAISS doesn't support removal)
        logger.info("FAISS vector store initialized (dim=%d)", dimension)

    @staticmethod
    def _normalize(vec: list[float]) -> np.ndarray:
        """L2-normalize a vector so inner product = cosine similarity."""
        arr = np.array(vec, dtype=np.float32).reshape(1, -1)
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr = arr / norm
        return arr

    def add(self, id: str, embedding: list[float], payload: dict[str, Any]) -> None:
        vec = self._normalize(embedding)
        self._index.add(vec)
        self._id_map.append(id)
        self._payloads[id] = payload

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
            if score < threshold:
                continue
            results.append(SearchResult(
                id=entry_id,
                score=float(score),
                payload=self._payloads.get(entry_id, {}),
            ))
            if len(results) >= top_k:
                break
        return results

    def delete(self, id: str) -> bool:
        # FAISS IndexFlatIP doesn't support vector removal.
        # Soft-delete: remove payload and mark ID so search() skips it.
        if id in self._payloads:
            del self._payloads[id]
            self._deleted_ids.add(id)
            return True
        return False

    def clear(self) -> int:
        import faiss
        count = len(self._payloads)
        self._index = faiss.IndexFlatIP(self._dimension)
        self._id_map.clear()
        self._payloads.clear()
        self._deleted_ids.clear()
        return count

    def count(self) -> int:
        return len(self._payloads)
