# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Semantic cache configuration for the Text2SQL demo.

Uses the core library's SemanticCache directly with:
- FAISS vector store (fast in-process similarity search)
- FilesystemResponseStore (large SQL results stored on disk)
- Auto-extract (LLM extracts intent + variables automatically)
- Cache filter (only cache successful queries with results)

No custom wrapper needed — the core library handles everything.
"""
import logging

from bedrock_smart_router.semantic_cache import SemanticCache, SemanticCacheConfig
from bedrock_smart_router.semantic_response_store import FilesystemResponseStore

logger = logging.getLogger(__name__)

RESPONSE_CACHE_DIR = "/tmp/text2sql_responses"

# Global shared cache singleton (persists across all sessions)
_shared_cache: SemanticCache | None = None


def get_shared_cache(region: str = "us-west-2") -> SemanticCache:
    """Get or create the shared semantic cache instance."""
    global _shared_cache
    if _shared_cache is None:
        _shared_cache = SemanticCache(
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
            response_store=FilesystemResponseStore(path=RESPONSE_CACHE_DIR),
            cache_filter=lambda query, response: (
                response.get("row_count", 0) > 0
                and not response.get("error")
                and bool(response.get("results"))
            ),
        )
        logger.info(
            "Semantic cache created — FAISS + FilesystemResponseStore at %s",
            RESPONSE_CACHE_DIR,
        )
    return _shared_cache
