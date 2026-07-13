# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Response Caching — avoid redundant Bedrock calls.

Demonstrates:
  - Cache hit on repeated request
  - Cache TTL configuration
  - Cache stats and invalidation
  - Cache is request-based (model-independent)
"""

from bedrock_smart_router import BedrockRouter

# ── Example 1: Cache hit on repeated request ─────────────────────────

router = BedrockRouter.create({
    "cache": {"enabled": True, "ttl_seconds": 300, "max_entries": 1000},
})

msgs = [{"role": "user", "content": [{"text": "What is EC2?"}]}]

r1 = router.converse(messages=msgs)
print(f"First call:  cache_hit={r1['routing_decision'].cache_hit}, "
      f"cost=${r1['routing_decision'].actual_cost:.6f}")

r2 = router.converse(messages=msgs)
print(f"Second call: cache_hit={r2['routing_decision'].cache_hit}, "
      f"cost=${r2['routing_decision'].actual_cost:.6f}")


# ── Example 2: Cache stats ───────────────────────────────────────────

print(f"\nCache stats: {router.cache.stats}")
# {"hits": 1, "misses": 1, "hit_rate": 0.5, "size": 1, "max_entries": 1000}


# ── Example 3: Invalidate cache for a specific model ─────────────────
# Useful when you know a model has been updated or is misbehaving.

count = router.cache.invalidate("us.amazon.nova-micro-v1:0")
print(f"\nInvalidated {count} entries for nova-micro")

# Or invalidate everything:
count = router.cache.invalidate()
print(f"Invalidated all {count} entries")
