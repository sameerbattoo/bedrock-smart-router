# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Observability — logging, callbacks, cost tracking, CloudWatch.

Demonstrates:
  - Custom callback for every routing decision
  - Cost tracking with breakdowns
  - CloudWatch metrics publishing
  - Routing savings calculation
"""

from bedrock_smart_router import BedrockRouter, RoutingEvent

# ── Example 1: Custom callback — send to your analytics pipeline ─────

events_log = []

def my_callback(event: RoutingEvent):
    """Called on every routing decision."""
    events_log.append({
        "model": event.decision.selected_model,
        "cost": event.decision.actual_cost,
        "latency": event.decision.latency_ms,
        "complexity": event.decision.complexity_detected,
        "cache_hit": event.cache_hit,
    })

router = BedrockRouter.create(
    {"strategy": "balanced"},
    callbacks=[my_callback],
)

router.converse(messages=[{"role": "user", "content": [{"text": "Hello"}]}])
router.converse(messages=[{"role": "user", "content": [{"text": "Explain VPCs"}]}])

print("Events logged:")
for e in events_log:
    print(f"  {e}")


# ── Example 2: Cost tracking ─────────────────────────────────────────

stats = router.observability.cost_tracker.stats
print(f"\nCost tracking:")
print(f"  Total cost:           ${stats['total_cost']:.6f}")
print(f"  Total requests:       {stats['total_requests']}")
print(f"  Avg cost/request:     ${stats['avg_cost_per_request']:.6f}")
print(f"  Saved by routing:     ${stats['cost_saved_by_routing']:.6f}")
print(f"  Saved by cache:       ${stats['cost_saved_by_cache']:.6f}")
print(f"  Cost by model:        {stats['cost_by_model']}")


# ── Example 3: CloudWatch metrics ────────────────────────────────────
# Publishes RoutingDecisions, Latency, Cost, CacheHits, FallbacksUsed,
# CircuitBreakerSkips, CostSavings to CloudWatch.

router_with_cw = BedrockRouter.create({
    "observability": {
        "log_decisions": True,
        "cloudwatch_enabled": True,
        "cloudwatch_namespace": "MyApp/BedrockRouter",
    },
})

router_with_cw.converse(
    messages=[{"role": "user", "content": [{"text": "Test CloudWatch"}]}],
)
# Metrics are batched and flushed in background threads.
# Check CloudWatch console under namespace "MyApp/BedrockRouter".
