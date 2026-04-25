"""Semantic Cache — cache responses by meaning, not exact text.

Unlike the exact-match cache (which only hits on identical requests),
the semantic cache uses embedding similarity to match queries that
are phrased differently but have the same intent.

"How do I reset my password?" → cache miss (first time)
"I forgot my password, help"  → cache HIT (same meaning)

Cost: ~$0.0001 per embedding call, ~50-100ms latency overhead.
Best for: customer support, FAQ bots, repetitive query workloads.

Demonstrates:
  - Basic semantic cache (store and retrieve by meaning)
  - Variable-aware caching (intent + variables must match)
  - Combining semantic cache with the smart router
"""

from bedrock_smart_router import BedrockRouter
from bedrock_smart_router.semantic_cache import SemanticCache, SemanticCacheConfig

router = BedrockRouter.create()


# ═══════════════════════════════════════════════════════════════════
# Example 1: Basic semantic cache — match by meaning
# ═══════════════════════════════════════════════════════════════════

semantic_cache = SemanticCache(
    config=SemanticCacheConfig(
        threshold=0.92,
        embedding_model="amazon.titan-embed-text-v2:0",
        max_entries=5000,
        ttl_seconds=3600,
    ),
    region="us-west-2",
)

# First request — cache miss, call the router
query1 = "How do I reset my password?"
cached = semantic_cache.get(query1)
if cached is None:
    response = router.converse(
        messages=[{"role": "user", "content": [{"text": query1}]}],
    )
    semantic_cache.put(query1, response)
    print(f"Query 1: cache MISS → called Bedrock")

# Second request — different words, same meaning → HIT
query2 = "I forgot my password and need to change it"
cached = semantic_cache.get(query2)
print(f"Query 2: cache {'HIT' if cached else 'MISS'}")


# ═══════════════════════════════════════════════════════════════════
# Example 2: Variable-aware caching
# ═══════════════════════════════════════════════════════════════════
# "Find top 5 users for Electronics in 2024" and
# "Find top 5 users for Clothing in 2025" are semantically identical
# but have different correct answers.  Pass variables to distinguish.

semantic_cache.put(
    "Find top 5 users for Electronics in 2024",
    {"result": ["alice", "bob", "charlie"]},
    variables={"category": "Electronics", "year": "2024"},
)

# Same intent + same variables → HIT
hit = semantic_cache.get(
    "Show me the top 5 users in Electronics for 2024",
    variables={"category": "Electronics", "year": "2024"},
)
print(f"\nSame variables:      {'HIT' if hit else 'MISS'}")

# Same intent + different variables → MISS
hit = semantic_cache.get(
    "Find top 5 users for Clothing in 2025",
    variables={"category": "Clothing", "year": "2025"},
)
print(f"Different variables: {'HIT' if hit else 'MISS'}")


# ═══════════════════════════════════════════════════════════════════
# Example 3: Semantic cache + smart router — full pipeline
# ═══════════════════════════════════════════════════════════════════

def smart_converse(query: str) -> dict:
    """Semantic cache → smart router → cache store."""
    cached = semantic_cache.get(query)
    if cached is not None:
        print(f"  Semantic cache HIT: '{query[:50]}'")
        return cached

    response = router.converse(
        messages=[{"role": "user", "content": [{"text": query}]}],
    )
    semantic_cache.put(query, response)
    print(f"  Semantic cache MISS → stored: '{query[:50]}'")
    return response


print(f"\nFull pipeline:")
smart_converse("How do I create an S3 bucket?")
smart_converse("What are the steps to make a new S3 bucket?")  # Semantic hit
smart_converse("Tell me about creating buckets in S3")          # Semantic hit
