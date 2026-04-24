"""Semantic response cache — embedding-based similarity matching.

Unlike the exact-match cache, the semantic cache uses embedding
similarity to match requests that are phrased differently but have
the same intent.  This dramatically increases cache hit rates for
workloads like customer support and FAQ bots.

**Variable-aware caching (manual):**

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

**Auto-extraction (optional):**

When ``auto_extract=True``, the cache uses a cheap Bedrock model to
automatically extract the canonical intent and variables from each
query.  No manual variable passing needed::

    cache.put("Count users by geo for 2026 with sales > $200", response)
    cache.get("Show user distribution by geography, year 2026, sales over $200")
    # → HIT (intent matches, variables {year: 2026, sales_threshold: 200} match)

**Multi-turn resolution (optional):**

When ``multi_turn_resolution=True``, the cache can resolve a multi-turn
conversation into a single self-contained query before extraction::

    cache.get(messages=[
        {"role": "user", "content": [{"text": "show me users by geo"}]},
        {"role": "assistant", "content": [{"text": "Here are users..."}]},
        {"role": "user", "content": [{"text": "now for 2026 with sales > $200"}]},
    ])
    # → Resolves to "Count users by geography for 2026 with sales > $200"
    # → Matches cached single-turn query with same intent + variables

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

# Retry defaults for embedding calls
_EMBED_MAX_RETRIES = 2
_EMBED_BACKOFF_BASE = 0.3
_EMBED_BACKOFF_MULTIPLIER = 2.0
_EMBED_BACKOFF_MAX = 4.0
_EMBED_RETRYABLE_ERRORS = frozenset({
    "ThrottlingException",
    "ServiceUnavailableException",
    "InternalServerException",
    "ModelTimeoutException",
})


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
    # Auto-extraction (optional)
    auto_extract: bool = False
    extraction_model: str = "us.amazon.nova-micro-v1:0"
    multi_turn_resolution: bool = False


class SemanticCache:
    """Embedding-based semantic response cache.

    Stores responses keyed by embedding vectors.  On lookup, computes
    the embedding of the query and finds the nearest cached entry
    above the similarity threshold.

    When ``auto_extract`` is enabled, the cache automatically extracts
    the canonical intent and variables from each query using a cheap
    Bedrock model, so the caller doesn't need to pass variables manually.
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

        # Lazy-init intent extractor (only when auto_extract is enabled)
        self._extractor: Any | None = None

    def _get_extractor(self) -> Any:
        """Lazy-initialise the IntentExtractor."""
        if self._extractor is None:
            from bedrock_smart_router.intent_extractor import (
                IntentExtractor,
                IntentExtractorConfig,
            )
            self._extractor = IntentExtractor(
                config=IntentExtractorConfig(
                    model_id=self.config.extraction_model,
                ),
                boto_session=self._session,
                region=self._region,
            )
        return self._extractor

    def _get_embedding(self, text: str) -> list[float]:
        """Compute an embedding vector using Bedrock, with local caching and retries."""
        # Check local embedding cache first
        cache_key = hashlib.sha256(text.encode()).hexdigest()[:16]
        cached = self._embedding_cache.get(cache_key)
        if cached is not None:
            return cached

        if self._session is None:
            import boto3
            self._session = boto3.Session(region_name=self._region)
        client = self._session.client("bedrock-runtime", region_name=self._region)

        # Retry with exponential backoff
        last_exc: Exception | None = None
        for attempt in range(1 + _EMBED_MAX_RETRIES):
            try:
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
            except Exception as exc:
                last_exc = exc
                error_code = _get_error_code(exc)
                if error_code not in _EMBED_RETRYABLE_ERRORS:
                    raise
                if attempt >= _EMBED_MAX_RETRIES:
                    raise
                delay = min(
                    _EMBED_BACKOFF_BASE * (_EMBED_BACKOFF_MULTIPLIER ** attempt),
                    _EMBED_BACKOFF_MAX,
                )
                logger.info(
                    "Embedding retry %d/%d after %s, backoff %.2fs",
                    attempt + 1, _EMBED_MAX_RETRIES, error_code, delay,
                )
                time.sleep(delay)

        raise last_exc  # type: ignore[misc]

    def get(
        self,
        query_text: str | None = None,
        variables: dict[str, str] | None = None,
        *,
        messages: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any] | None:
        """Look up a semantically similar cached response.

        Args:
            query_text: The user's query (single-turn).
            variables: Optional variable values (manual mode).  When
                provided, the cache only hits if both the semantic
                similarity is above threshold AND the variable values
                match exactly.  Ignored when ``auto_extract`` is enabled.
            messages: Full conversation history (multi-turn).  When
                provided with ``auto_extract=True`` and
                ``multi_turn_resolution=True``, the conversation is
                resolved into a single query before lookup.

        Returns:
            The cached response dict, or None on miss.
        """
        if not self.config.enabled:
            self._misses += 1
            return None

        if self._store.count() == 0:
            self._misses += 1
            return None

        # Resolve the lookup text and variables
        lookup_text, lookup_vars = self._resolve_lookup(
            query_text, variables, messages,
        )

        query_emb = self._get_embedding(lookup_text)
        # Fetch more candidates when variables are set, since we need
        # to filter by variable match after semantic search
        top_k = 10 if lookup_vars else 1
        results = self._store.search(
            query_emb, top_k=top_k, threshold=self.config.threshold,
        )

        var_hash = self._hash_variables(lookup_vars) if lookup_vars else None

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
                result.score, lookup_text[:50], lookup_vars,
            )
            return payload.get("response")

        self._misses += 1
        return None

    def put(
        self,
        query_text: str | None = None,
        response: dict[str, Any] | None = None,
        variables: dict[str, str] | None = None,
        *,
        messages: list[dict[str, Any]] | None = None,
    ) -> None:
        """Store a response with its embedding.

        Args:
            query_text: The user's query (single-turn).
            response: The Bedrock response to cache.
            variables: Optional variable values (manual mode).  Ignored
                when ``auto_extract`` is enabled.
            messages: Full conversation history (multi-turn).
        """
        if not self.config.enabled:
            return
        if response is None:
            return

        # Resolve the storage text and variables
        store_text, store_vars = self._resolve_lookup(
            query_text, variables, messages,
        )

        embedding = self._get_embedding(store_text)
        var_hash = self._hash_variables(store_vars) if store_vars else ""
        # Include var_hash in the entry ID so different variables get
        # different entries even for semantically identical queries
        entry_id = hashlib.sha256(
            f"{store_text}|{var_hash}".encode()
        ).hexdigest()[:16]

        self._store.add(
            id=entry_id,
            embedding=embedding,
            payload={
                "response": response,
                "query": store_text,
                "variables": store_vars or {},
                "var_hash": var_hash,
                "created_at": time.time(),
            },
        )

    def _resolve_lookup(
        self,
        query_text: str | None,
        variables: dict[str, str] | None,
        messages: list[dict[str, Any]] | None,
    ) -> tuple[str, dict[str, str] | None]:
        """Resolve the lookup text and variables based on mode.

        Returns:
            (text_to_embed, variables_dict_or_none)
        """
        # Auto-extract mode
        if self.config.auto_extract:
            extractor = self._get_extractor()

            # Multi-turn with messages
            if (
                messages is not None
                and self.config.multi_turn_resolution
                and self._count_user_messages(messages) >= 2
            ):
                result = extractor.extract_from_messages(messages)
                return result.intent, result.variables or None

            # Single-turn with query_text
            if query_text:
                result = extractor.extract(query_text)
                return result.intent, result.variables or None

            # Fallback: extract from last user message in messages
            if messages:
                last_text = self._last_user_text(messages)
                if last_text:
                    result = extractor.extract(last_text)
                    return result.intent, result.variables or None

        # Manual mode (existing behavior)
        text = query_text or ""
        if not text and messages:
            text = self._last_user_text(messages) or ""
        return text, variables

    @staticmethod
    def _count_user_messages(messages: list[dict[str, Any]]) -> int:
        return sum(1 for m in messages if m.get("role") == "user")

    @staticmethod
    def _last_user_text(messages: list[dict[str, Any]]) -> str | None:
        for msg in reversed(messages):
            if msg.get("role") == "user":
                for block in msg.get("content", []):
                    if isinstance(block, dict) and "text" in block:
                        return block["text"]
        return None

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
            "auto_extract": self.config.auto_extract,
            "multi_turn_resolution": self.config.multi_turn_resolution,
        }


def _get_error_code(exc: Exception) -> str:
    """Extract the Bedrock/botocore error code from an exception."""
    if hasattr(exc, "response"):
        return exc.response.get("Error", {}).get("Code", type(exc).__name__)
    return type(exc).__name__
