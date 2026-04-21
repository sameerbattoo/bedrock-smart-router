"""Routing Strategies — cost, latency, quality, balanced, budget.

Demonstrates:
  - Each built-in strategy
  - Custom weights for balanced strategy
  - Budget-constrained routing
"""

from bedrock_smart_router import BedrockRouter, RoutingConfig

router = BedrockRouter.create()

prompt = "Explain how DNS works in 2 sentences."
msgs = [{"role": "user", "content": [{"text": prompt}]}]


# ── Example 1: Cost-optimized — cheapest model that fits ─────────────

response = router.converse(messages=msgs, routing=RoutingConfig(strategy="cost-optimized"))
d = response["routing_decision"]
print(f"Cost-opt    → {d.selected_model}, ${d.actual_cost:.6f}")


# ── Example 2: Latency-optimized — fastest model ────────────────────

response = router.converse(messages=msgs, routing=RoutingConfig(strategy="latency-optimized"))
d = response["routing_decision"]
print(f"Latency-opt → {d.selected_model}, {d.latency_ms:.0f}ms")


# ── Example 3: Quality-optimized — best quality from historical data ─

response = router.converse(messages=msgs, routing=RoutingConfig(strategy="quality-optimized"))
d = response["routing_decision"]
print(f"Quality-opt → {d.selected_model}")


# ── Example 4: Balanced with custom weights ──────────────────────────
# 70% cost, 20% latency, 10% quality — aggressive cost savings.

response = router.converse(
    messages=msgs,
    routing=RoutingConfig(
        strategy="balanced",
        weights={"cost": 0.7, "latency": 0.2, "quality": 0.1},
    ),
)
d = response["routing_decision"]
print(f"Balanced(70/20/10) → {d.selected_model}, ${d.actual_cost:.6f}")


# ── Example 5: Budget-constrained — max $0.001 per request ──────────

response = router.converse(
    messages=msgs,
    routing=RoutingConfig(
        strategy="balanced",
        max_cost_per_request=0.001,
    ),
)
d = response["routing_decision"]
print(f"Budget $0.001 → {d.selected_model}, ${d.actual_cost:.6f}")
