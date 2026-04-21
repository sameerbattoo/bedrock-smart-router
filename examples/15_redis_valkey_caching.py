"""Redis / Valkey / ElastiCache Caching — shared cache across instances.

Demonstrates:
  - Redis cache for cross-instance response sharing
  - ElastiCache Valkey with TLS
  - Cache stats and invalidation via Redis
  - Combining Redis cache with the full router

Compatible with:
  - Redis (open source or managed)
  - Valkey (Redis-compatible fork)
  - Amazon ElastiCache (Redis or Valkey engine)
  - Amazon ElastiCache Serverless
  - Amazon MemoryDB

Install:
  pip install bedrock-smart-router[redis]
"""

from bedrock_smart_router import BedrockRouter, RoutingConfig

# ── Example 1: Local Redis ───────────────────────────────────────────
# Good for development and single-host deployments.

router = BedrockRouter.create({
    "cache": {
        "backend": "redis",
        "redis_url": "redis://localhost:6379",
        "ttl_seconds": 1800,
        "key_prefix": "myapp:",
    },
})

# First call — cache miss, calls Bedrock
response = router.converse(
    messages=[{"role": "user", "content": [{"text": "What is S3?"}]}],
)
print(f"First call:  cache_hit={response['routing_decision'].cache_hit}")

# Second call — cache hit from Redis, zero Bedrock cost
response = router.converse(
    messages=[{"role": "user", "content": [{"text": "What is S3?"}]}],
)
print(f"Second call: cache_hit={response['routing_decision'].cache_hit}")

print(f"Cache stats: {router.cache.stats}")


# ── Example 2: ElastiCache Valkey with TLS ───────────────────────────
# Production setup — shared across all Lambda/ECS instances.
# Use "rediss://" (double s) for TLS, which ElastiCache requires.

router = BedrockRouter.create({
    "cache": {
        "backend": "valkey",  # "valkey" and "redis" are interchangeable
        "redis_url": "rediss://master.my-cluster.abc123.usw2.cache.amazonaws.com:6379",
        "ttl_seconds": 3600,
        "key_prefix": "bsr:prod:",
    },
    "strategy": "balanced",
})


# ── Example 3: ElastiCache Serverless ────────────────────────────────
# Zero capacity management — scales automatically.

router = BedrockRouter.create({
    "cache": {
        "backend": "redis",
        "redis_url": "rediss://my-serverless.abc123.serverless.usw2.cache.amazonaws.com:6379",
        "ttl_seconds": 1800,
        "key_prefix": "bsr:",
    },
})


# ── Example 4: Cache invalidation ────────────────────────────────────
# Useful when a model is updated or misbehaving.

router = BedrockRouter.create({
    "cache": {
        "backend": "redis",
        "redis_url": "redis://localhost:6379",
        "key_prefix": "myapp:",
    },
})

# Invalidate all cached responses from a specific model
count = router.cache.invalidate("us.anthropic.claude-sonnet-4-6")
print(f"\nInvalidated {count} entries for Sonnet 4.6")

# Invalidate everything
count = router.cache.invalidate()
print(f"Invalidated all {count} entries")


# ── Example 5: Full production config with Valkey + DynamoDB ─────────
# Shared cache + persistent metrics — the recommended production setup.

router = BedrockRouter.create({
    "region": "us-west-2",
    "strategy": "balanced",
    "weights": {"cost": 0.4, "latency": 0.3, "quality": 0.3},
    "cache": {
        "backend": "valkey",
        "redis_url": "rediss://master.my-cluster.abc123.usw2.cache.amazonaws.com:6379",
        "ttl_seconds": 1800,
        "key_prefix": "bsr:prod:",
    },
    "metrics": {
        "backend": "dynamodb",
        "table_name": "BedrockRouterMetrics",
        "ttl_hours": 168,
    },
    "observability": {
        "cloudwatch_enabled": True,
        "cloudwatch_namespace": "MyApp/BedrockRouter",
    },
})

# Every converse() call now:
# 1. Checks Valkey cache first (shared across all instances)
# 2. On miss: routes to optimal model, calls Bedrock
# 3. Caches the response in Valkey for other instances
# 4. Records metrics to DynamoDB (survives cold starts)
# 5. Publishes CloudWatch metrics for dashboards
