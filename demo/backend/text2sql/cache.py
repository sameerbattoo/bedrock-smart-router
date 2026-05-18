"""Semantic cache with filesystem-backed response storage.

Architecture:
- FAISS in-memory: stores question vectors + variable hashes + file path reference
- Local filesystem: stores actual response payloads as JSON files

This keeps memory usage low (vectors are small) while supporting
large responses (SQL results, chart paths) on disk.

Uses the core library's SemanticCache for:
- Auto-extract: LLM extracts intent + key variables
- Variable hashing: different variables = different cache entries
- Multi-turn intent resolution: follow-ups resolved against conversation
- FAISS vector similarity search
"""
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

from bedrock_smart_router.semantic_cache import SemanticCache, SemanticCacheConfig

logger = logging.getLogger(__name__)

RESPONSE_CACHE_DIR = Path("/tmp/text2sql_responses")
RESPONSE_CACHE_DIR.mkdir(exist_ok=True)


class FilesystemSemanticCache:
    """Wraps the core SemanticCache with filesystem response storage.

    The core cache stores only a file path reference in its response field.
    The actual response payload (SQL results, chart paths, etc.) lives on disk.
    """

    def __init__(self, region: str = "us-west-2"):
        self._cache = SemanticCache(
            config=SemanticCacheConfig(
                enabled=True,
                threshold=0.70,
                embedding_model="amazon.titan-embed-text-v2:0",
                embedding_dimension=1024,
                max_entries=5000,
                ttl_seconds=3600,
                vector_store_backend="faiss",
                auto_extract=True,
                extraction_model="us.amazon.nova-micro-v1:0",
            ),
            region=region,
        )
        logger.info(
            "FilesystemSemanticCache created — FAISS in-memory + filesystem responses at %s",
            RESPONSE_CACHE_DIR,
        )

    def get(self, query_text: str, messages: list[dict] | None = None) -> dict[str, Any] | None:
        """Look up a cached response.

        Returns the full response payload from disk, or None on miss.
        """
        # Log the lookup details from the core cache for debugging
        lookup_text, lookup_vars = self._cache._resolve_lookup(query_text, None, messages)
        print(f"[CACHE-DEBUG] Query: '{query_text}'", flush=True)
        print(f"[CACHE-DEBUG]   Intent: '{lookup_text}'", flush=True)
        print(f"[CACHE-DEBUG]   Vars: {lookup_vars}", flush=True)
        print(f"[CACHE-DEBUG]   Entries: {self._cache._store.count()}", flush=True)

        # Core cache returns the stored response (which contains a file path)
        cached = self._cache.get(query_text=query_text, messages=messages)
        if cached is None:
            print(f"[CACHE-DEBUG]   Result: MISS", flush=True)
            return None

        # Extract file path from cached response
        file_path = cached.get("_response_file")
        if not file_path:
            return None

        # Load actual response from filesystem
        path = Path(file_path)
        if not path.exists():
            logger.warning("Cache hit but response file missing: %s", file_path)
            return None

        try:
            payload = json.loads(path.read_text())
            logger.info("Cache HIT (file=%s): %s", path.name, query_text[:60])
            return payload
        except Exception as exc:
            logger.warning("Failed to load cached response: %s", exc)
            return None

    def put(
        self, query_text: str, response: dict[str, Any], messages: list[dict] | None = None,
    ) -> None:
        """Store a response in the cache."""
        # Log what's being stored
        lookup_text, lookup_vars = self._cache._resolve_lookup(query_text, None, messages)
        print(f"[CACHE-DEBUG] PUT: '{query_text}'", flush=True)
        print(f"[CACHE-DEBUG]   Stored Intent: '{lookup_text}'", flush=True)
        print(f"[CACHE-DEBUG]   Stored Vars: {lookup_vars}", flush=True)

        # Generate a unique filename
        ts = int(time.time() * 1000)
        key_hash = hashlib.sha256(f"{query_text}:{ts}".encode()).hexdigest()[:12]
        filename = f"resp_{key_hash}.json"
        file_path = RESPONSE_CACHE_DIR / filename

        # Save response payload to filesystem
        try:
            file_path.write_text(json.dumps(response, default=str))
        except Exception as exc:
            logger.warning("Failed to save response to file: %s", exc)
            return

        # Store only the file path reference in the core cache
        self._cache.put(
            query_text=query_text,
            response={"_response_file": str(file_path)},
            messages=messages,
        )
        logger.debug("Cache PUT: %s → %s", query_text[:60], filename)

    def stats(self) -> dict[str, Any]:
        """Return cache statistics."""
        if hasattr(self._cache, 'stats'):
            s = self._cache.stats
            core_stats = s() if callable(s) else s
        else:
            core_stats = {}
        # Count response files on disk
        file_count = len(list(RESPONSE_CACHE_DIR.glob("resp_*.json")))
        return {
            **(core_stats if isinstance(core_stats, dict) else {}),
            "response_files_on_disk": file_count,
            "response_dir": str(RESPONSE_CACHE_DIR),
        }
