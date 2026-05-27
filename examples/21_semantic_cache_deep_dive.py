"""Semantic Cache Deep Dive — vector stores, variables, and configuration.

The semantic cache matches queries by meaning, not exact text.
This example covers all the configuration options, vector store
backends, and the variable-aware caching feature.

Demonstrates:
  - In-memory vector store (default, no dependencies)
  - FAISS vector store (fast in-process, pip install bedrock-smart-router[faiss])
  - Redis/Valkey vector store (shared across instances)
  - OpenSearch Serverless vector store (AWS managed)
  - FAISS with auto-extract (automatic intent + variable extraction)
  - Caching without variables (intent-only matching)
  - Caching with variables (intent + variable values must match)
  - Response store backends (inline, filesystem, S3, DynamoDB)
  - Cache filter (selective caching — app decides what to cache)
  - Threshold tuning
  - Cache stats and invalidation
"""

from bedrock_smart_router.semantic_cache import SemanticCache, SemanticCacheConfig


# ═══════════════════════════════════════════════════════════════════
# Example 1: In-Memory Vector Store (default)
# ═══════════════════════════════════════════════════════════════════
# No extra dependencies.  Good for development, Lambda, and small
# caches (~500 entries).  Brute-force cosine similarity — O(n) per
# lookup but fast enough for small N.

cache_memory = SemanticCache(
    config=SemanticCacheConfig(
        enabled=True,
        threshold=0.90,
        embedding_model="amazon.titan-embed-text-v2:0",
        vector_store_backend="memory",  # Default
        max_entries=1000,
        ttl_seconds=3600,
    ),
    region="us-west-2",
)

# Store a response
cache_memory.put(
    "How do I create an S3 bucket?",
    {"answer": "Use the AWS Console or aws s3 mb command..."},
)

# Different phrasing, same meaning → HIT
hit = cache_memory.get("What are the steps to make a new S3 bucket?")
print(f"Memory backend:")
print(f"  Same meaning, different words: {'HIT' if hit else 'MISS'}")

# Completely different question → MISS
hit = cache_memory.get("What is the capital of France?")
print(f"  Different question: {'HIT' if hit else 'MISS'}")
print(f"  Stats: {cache_memory.stats}")


# ═══════════════════════════════════════════════════════════════════
# Example 2: FAISS Vector Store
# ═══════════════════════════════════════════════════════════════════
# Fast approximate nearest neighbor search.  Sub-millisecond lookups
# even with 100K entries.  Single-process only (not shared).
#
# pip install bedrock-smart-router[faiss]

cache_faiss = SemanticCache(
    config=SemanticCacheConfig(
        enabled=True,
        threshold=0.90,
        vector_store_backend="faiss",
        embedding_dimension=1024,  # Must match your embedding model
    ),
    region="us-west-2",
)

cache_faiss.put("How do I configure a VPC?", {"answer": "Go to VPC console..."})
hit = cache_faiss.get("Steps to set up a Virtual Private Cloud")
print(f"\nFAISS backend:")
print(f"  VPC question: {'HIT' if hit else 'MISS'}")
print(f"  Stats: {cache_faiss.stats}")


# ═══════════════════════════════════════════════════════════════════
# Example 3: Redis / Valkey Vector Store
# ═══════════════════════════════════════════════════════════════════
# Shared across all instances.  Works with:
#   - Redis 7+ with the RediSearch module
#   - Amazon ElastiCache Valkey 8.2+ (native vector search)
#
# pip install bedrock-smart-router[redis]

# cache_redis = SemanticCache(
#     config=SemanticCacheConfig(
#         enabled=True,
#         threshold=0.90,
#         vector_store_backend="redis",
#         redis_url="redis://localhost:6379",
#         redis_key_prefix="bsr:semcache:",
#         embedding_dimension=1024,
#     ),
#     region="us-west-2",
# )
#
# # Shared across all Lambda invocations / ECS tasks
# cache_redis.put("How do I reset my password?", {"answer": "Go to settings..."})
# hit = cache_redis.get("I forgot my password, help")
print(f"\nRedis backend: (uncomment to run with a real Redis instance)")


# ═══════════════════════════════════════════════════════════════════
# Example 3b: OpenSearch Serverless Vector Store
# ═══════════════════════════════════════════════════════════════════
# Shared across all instances via Amazon OpenSearch Serverless (AOSS).
# Uses SigV4 authentication — no passwords, just IAM roles.
# Scales to millions of entries with k-NN vector search.
#
# pip install bedrock-smart-router[opensearch]
#
# Requires an active AOSS collection with VECTORSEARCH type.
# The index is auto-created on first use.
# NOTE: AOSS has eventual consistency — data takes ~60-90s to be searchable.

# import time
# AOSS_ENDPOINT = "https://your-collection-id.us-west-2.aoss.amazonaws.com"
#
# cache_opensearch = SemanticCache(
#     config=SemanticCacheConfig(
#         threshold=0.85,
#         vector_store_backend="opensearch",
#         opensearch_endpoint=AOSS_ENDPOINT,
#         opensearch_index_name="bsr-example-21",
#         embedding_dimension=1024,
#         auto_extract=True,
#         extraction_model="us.amazon.nova-micro-v1:0",
#     ),
#     region="us-west-2",
# )
#
# cache_opensearch.put(
#     "Count users by geography for 2026 with sales > $200",
#     {"result": "42 users across 5 regions"},
# )
#
# # AOSS has eventual consistency — wait for propagation
# time.sleep(90)
#
# hit = cache_opensearch.get("Show user distribution by geo, year 2026, sales over $200")
# print(f"  Same intent+vars: {'HIT' if hit else 'MISS'}")
#
# cache_opensearch.invalidate()

print(f"\nOpenSearch Serverless backend: (commented out — requires live AOSS endpoint + 90s wait)")


# ═══════════════════════════════════════════════════════════════════
# Example 3c: FAISS with Auto-Extract
# ═══════════════════════════════════════════════════════════════════
# Combine FAISS (fast vector search) with auto-extract (automatic
# intent + variable extraction).  No manual variable passing needed.

cache_faiss_auto = SemanticCache(
    config=SemanticCacheConfig(
        threshold=0.85,
        vector_store_backend="faiss",
        embedding_dimension=1024,
        auto_extract=True,
        extraction_model="us.amazon.nova-micro-v1:0",
    ),
    region="us-west-2",
)

# Store — variables extracted automatically by Nova Micro
cache_faiss_auto.put(
    "Count users by geography for 2026 with sales > $200",
    {"result": "42 users across 5 regions"},
)

# Same intent + same variables → HIT (auto-extracted)
hit = cache_faiss_auto.get("Show user distribution by geo, year 2026, sales over $200")
print(f"\nFAISS + auto-extract:")
print(f"  Same intent+vars:    {'HIT' if hit else 'MISS'}")

# Different variables → MISS (auto-extracted variables differ)
hit = cache_faiss_auto.get("Count users by geography for 2025 with sales > $100")
print(f"  Different variables: {'HIT' if hit else 'MISS'}")

# Multi-turn resolution (auto when messages with 2+ user turns are passed)
hit = cache_faiss_auto.get(messages=[
    {"role": "user", "content": [{"text": "show me users by geo"}]},
    {"role": "assistant", "content": [{"text": "Here are users by geography..."}]},
    {"role": "user", "content": [{"text": "now for 2026 with sales > $200"}]},
])
print(f"  Multi-turn match:    {'HIT' if hit else 'MISS'}")
print(f"  Backend: {cache_faiss_auto.stats['backend']}, auto_extract: {cache_faiss_auto.stats['auto_extract']}")


# ═══════════════════════════════════════════════════════════════════
# Example 4: Caching WITHOUT Variables (intent-only)
# ═══════════════════════════════════════════════════════════════════
# When no variables are passed, the cache matches purely on semantic
# similarity.  Good for FAQ, general knowledge, how-to questions.

cache = SemanticCache(
    config=SemanticCacheConfig(enabled=True, threshold=0.90),
    region="us-west-2",
)

# All of these are the same intent — second and third should HIT
cache.put("What is Amazon S3?", {"answer": "S3 is object storage..."})

queries = [
    "What is Amazon S3?",                    # Exact match → HIT
    "Explain S3 to me",                      # Same intent → HIT
    "Tell me about Amazon Simple Storage",   # Same intent → HIT
    "How do I configure a VPC?",             # Different intent → MISS
]

print(f"\nWithout variables (intent-only):")
for q in queries:
    hit = cache.get(q)
    print(f"  '{q[:45]:45s}' → {'HIT' if hit else 'MISS'}")


# ═══════════════════════════════════════════════════════════════════
# Example 5: Caching WITH Variables
# ═══════════════════════════════════════════════════════════════════
# When queries contain parameters that change the answer, pass them
# as variables.  The cache only hits when BOTH the intent AND the
# variable values match.

cache_vars = SemanticCache(
    config=SemanticCacheConfig(enabled=True, threshold=0.85),
    region="us-west-2",
)

# Store responses for different variable combinations
cache_vars.put(
    "Find top 5 users for Electronics in 2024",
    {"users": ["alice", "bob", "charlie", "dave", "eve"]},
    variables={"category": "Electronics", "year": "2024"},
)

cache_vars.put(
    "Find top 5 users for Clothing in 2025",
    {"users": ["frank", "grace", "heidi"]},
    variables={"category": "Clothing", "year": "2025"},
)

print(f"\nWith variables:")

# Same intent + same variables → HIT (returns Electronics 2024 data)
hit = cache_vars.get(
    "Show me the top 5 users in Electronics for 2024",
    variables={"category": "Electronics", "year": "2024"},
)
print(f"  Electronics 2024: {'HIT' if hit else 'MISS'} → {hit.get('users', [])[:3] if hit else 'N/A'}...")

# Same intent + different variables → MISS (Clothing 2025 ≠ Electronics 2024)
hit = cache_vars.get(
    "Show me the top 5 users in Clothing for 2025",
    variables={"category": "Clothing", "year": "2025"},
)
print(f"  Clothing 2025:    {'HIT' if hit else 'MISS'} → {hit.get('users', [])[:3] if hit else 'N/A'}...")

# Same intent + new variables → MISS (never cached this combination)
hit = cache_vars.get(
    "Find top 5 users for Books in 2023",
    variables={"category": "Books", "year": "2023"},
)
print(f"  Books 2023:       {'HIT' if hit else 'MISS'} (never cached)")


# ═══════════════════════════════════════════════════════════════════
# Example 6: Mixed — some queries have variables, some don't
# ═══════════════════════════════════════════════════════════════════
# You can mix variable-aware and variable-free entries in the same
# cache.  Entries stored without variables match any get() call
# (backward compatible).

cache_mixed = SemanticCache(
    config=SemanticCacheConfig(enabled=True, threshold=0.88),
    region="us-west-2",
)

# FAQ entry — no variables (matches any similar question)
cache_mixed.put("What is your return policy?", {"answer": "30-day returns..."})

# Parameterized entry — with variables
cache_mixed.put(
    "What is the price of the Pro plan?",
    {"price": "$49/month"},
    variables={"plan": "Pro"},
)

print(f"\nMixed cache:")

# FAQ hit — no variables needed
hit = cache_mixed.get("How do returns work?")
print(f"  Return policy (no vars): {'HIT' if hit else 'MISS'}")

# Parameterized hit — correct variables
hit = cache_mixed.get(
    "How much does the Pro plan cost?",
    variables={"plan": "Pro"},
)
print(f"  Pro plan price (vars match): {'HIT' if hit else 'MISS'}")

# Parameterized miss — wrong variables
hit = cache_mixed.get(
    "How much does the Enterprise plan cost?",
    variables={"plan": "Enterprise"},
)
print(f"  Enterprise plan (vars differ): {'HIT' if hit else 'MISS'}")


# ═══════════════════════════════════════════════════════════════════
# Example 7: Threshold Tuning
# ═══════════════════════════════════════════════════════════════════
# The threshold controls how similar queries must be to match.
#
# Higher threshold (0.95+): fewer false positives, more cache misses
# Lower threshold (0.80):   more cache hits, risk of wrong matches
#
# Recommended starting points:
#   FAQ / customer support: 0.88–0.92
#   General knowledge:      0.90–0.95
#   Code questions:          0.93–0.97 (code is more specific)

print(f"\nThreshold comparison:")
for threshold in [0.80, 0.88, 0.92, 0.95]:
    c = SemanticCache(
        config=SemanticCacheConfig(enabled=True, threshold=threshold),
        region="us-west-2",
    )
    c.put("How do I create an S3 bucket?", {"answer": "..."})
    hit = c.get("Steps to make a new bucket in S3")
    print(f"  threshold={threshold}: {'HIT' if hit else 'MISS'}")


# ═══════════════════════════════════════════════════════════════════
# Example 8: Cache Stats and Invalidation
# ═══════════════════════════════════════════════════════════════════

cache = SemanticCache(
    config=SemanticCacheConfig(enabled=True, threshold=0.90),
    region="us-west-2",
)

cache.put("Question 1", {"a": "1"})
cache.put("Question 2", {"a": "2"})
cache.put("Question 3", {"a": "3"})

cache.get("Question 1")  # HIT
cache.get("Unknown")     # MISS

print(f"\nCache stats: {cache.stats}")
# {hits: 1, misses: 1, hit_rate: 0.5, entries: 3, backend: memory, ...}

# Clear all entries
count = cache.invalidate()
print(f"Invalidated {count} entries")
print(f"After invalidation: {cache.stats['entries']} entries")

# ═══════════════════════════════════════════════════════════════════
# Example 9: Response Store Backends
# ═══════════════════════════════════════════════════════════════════
# By default, responses are stored "inline" in the vector store payload.
# For large responses (SQL results, charts, full LLM outputs), use an
# external response store to keep the vector store lean.
#
# Available backends:
#   - "inline"     — store in vector payload (default, good for <100KB)
#   - "filesystem" — store on disk (dev/testing, Lambda /tmp, EFS)
#   - "s3"         — store in S3 (production, large payloads, durability)
#   - "dynamodb"   — store in DynamoDB (serverless, low-latency, auto-expiry)
#
# You can also pass a custom ResponseStore instance for full control.

from bedrock_smart_router.semantic_response_store import (
    FilesystemResponseStore,
    S3ResponseStore,
    DynamoDBResponseStore,
)

# Option A: Configure via SemanticCacheConfig
cache_fs_config = SemanticCache(
    config=SemanticCacheConfig(
        threshold=0.90,
        vector_store_backend="faiss",
        response_store_backend="filesystem",
        response_store_path="/tmp/my_cache_responses",
    ),
    region="us-west-2",
)

# Option B: Pass a ResponseStore instance directly (takes precedence over config)
cache_fs_instance = SemanticCache(
    config=SemanticCacheConfig(threshold=0.90, vector_store_backend="faiss"),
    response_store=FilesystemResponseStore(path="/tmp/my_cache_responses"),
    region="us-west-2",
)

# Option C: S3 for production (large responses, durability, lifecycle rules)
# cache_s3 = SemanticCache(
#     config=SemanticCacheConfig(threshold=0.90, vector_store_backend="faiss"),
#     response_store=S3ResponseStore(
#         bucket="my-cache-bucket",
#         prefix="semantic_cache/",
#         region="us-west-2",
#     ),
# )

# Option D: DynamoDB for serverless (low-latency, auto-expiry via TTL)
# cache_ddb = SemanticCache(
#     config=SemanticCacheConfig(threshold=0.90, vector_store_backend="faiss"),
#     response_store=DynamoDBResponseStore(
#         table_name="cache-responses",
#         ttl_seconds=3600,
#         region="us-west-2",
#     ),
# )

# Option E: Custom response store (subclass ResponseStore)
# from bedrock_smart_router.semantic_response_store import ResponseStore
#
# class MyCustomStore(ResponseStore):
#     def save(self, key, response): ...
#     def load(self, reference): ...
#     def delete(self, reference): ...
#
# cache_custom = SemanticCache(
#     config=SemanticCacheConfig(threshold=0.90),
#     response_store=MyCustomStore(),
# )

print(f"\nResponse store backends:")
print(f"  Filesystem: {cache_fs_config.stats['response_store']}")
print(f"  Instance:   configured via FilesystemResponseStore")

# Store and retrieve — works the same regardless of backend
cache_fs_config.put("What is DynamoDB?", {"answer": "A serverless NoSQL database..."})
hit = cache_fs_config.get("Tell me about DynamoDB")
print(f"  Filesystem store: {'HIT' if hit else 'MISS'}")


# ═══════════════════════════════════════════════════════════════════
# Example 10: Cache Filter — Selective Caching
# ═══════════════════════════════════════════════════════════════════
# Not all responses should be cached.  Use a cache_filter to let the
# app decide which responses are worth storing.
#
# The filter is a callable: (query_text, response) -> bool
# - Return True to cache the response
# - Return False to skip caching
# - Exceptions are caught and treated as False (safe default)

# Only cache responses that have actual data (skip errors/empty)
cache_filtered = SemanticCache(
    config=SemanticCacheConfig(threshold=0.90, vector_store_backend="memory"),
    cache_filter=lambda query, response: (
        response.get("row_count", 0) > 0
        and not response.get("error")
    ),
    region="us-west-2",
)

# This gets cached (has results)
cache_filtered.put("Top products", {"row_count": 5, "results": [{"name": "Widget"}]})

# This gets FILTERED (error response)
cache_filtered.put("Bad query", {"error": "SQL syntax error", "row_count": 0})

# This gets FILTERED (empty results)
cache_filtered.put("No data", {"row_count": 0, "results": []})

print(f"\nCache filter:")
print(f"  Entries stored: {cache_filtered.stats['entries']}")
print(f"  Filtered (skipped): {cache_filtered.stats['filtered']}")

# More filter examples:
# Only cache if response is large enough to be worth caching
# cache = SemanticCache(
#     cache_filter=lambda q, r: len(str(r)) > 500,
# )

# Only cache successful responses with high confidence
# cache = SemanticCache(
#     cache_filter=lambda q, r: r.get("status") == "success" and r.get("confidence", 0) > 0.8,
# )
