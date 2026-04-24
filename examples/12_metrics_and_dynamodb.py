"""Historical Metrics — in-memory and DynamoDB backends.

Demonstrates:
  - In-memory metrics for Lambda / single-instance
  - DynamoDB metrics for cross-instance persistence
  - Querying historical metrics for a model
  - How quality-optimized strategy uses metrics
"""

from bedrock_smart_router import BedrockRouter, RoutingConfig

# ── Example 1: In-memory metrics (default) ───────────────────────────
# Good for Lambda — resets on cold start, warms up quickly.

router = BedrockRouter.create({"metrics": {"backend": "memory"}})

# Make some requests to build up metrics
for prompt in ["Hello", "Explain S3", "Write a function"]:
    router.converse(messages=[{"role": "user", "content": [{"text": prompt}]}])

# Query metrics for a specific model
model_id = router.last_routing_decision().selected_model
m = router.metrics.get_metrics(model_id, window_seconds=300)
print(f"Metrics for {model_id}:")
print(f"  Samples:     {m.sample_count}")
print(f"  Avg latency: {m.avg_latency_ms:.0f}ms")
print(f"  P95 latency: {m.p95_latency_ms:.0f}ms")
print(f"  Error rate:  {m.error_rate:.1%}")
print(f"  Avg cost:    ${m.avg_cost_per_request:.6f}")

# Query all models
all_metrics = router.metrics.get_all_metrics(window_seconds=300)
print(f"\nAll tracked models: {list(all_metrics.keys())}")


# ── Example 2: DynamoDB metrics (persistent, cross-instance) ─────────
# Survives Lambda cold starts. Shared across all instances.

router_ddb = BedrockRouter.create({
    "metrics": {
        "backend": "dynamodb",
        "table_name": "MyRouterMetrics",
        "ttl_hours": 168,          # 7 days retention
        "auto_create_table": True,  # Creates table on first use
    },
})

# Every converse() call now persists metrics to DynamoDB.
# router_ddb.converse(messages=[...])


# ── Example 3: Quality strategy uses historical metrics ──────────────
# After enough data, quality-optimized routing uses YOUR quality scores
# instead of generic tier heuristics.

router = BedrockRouter.create({"strategy": "quality-optimized"})

# First few requests use tier heuristics (micro=0.55, mid=0.82, etc.)
# After 20+ requests with quality scores, the router trusts historical data.
# Quality scores come from judge evaluations recorded via metrics_store.record().
