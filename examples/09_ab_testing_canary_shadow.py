"""A/B Testing, Canary Deployments, and Shadow Mode.

Demonstrates:
  - A/B testing between two models with sticky user assignment
  - Canary rollout of a new model with auto-rollback
  - Shadow mode to mirror traffic for offline evaluation
"""

from bedrock_smart_router import BedrockRouter, RoutingConfig

# ── Example 1: A/B test — Sonnet 4.6 vs Nova Pro ────────────────────
# 50/50 split with sticky user assignment.

router = BedrockRouter.create({
    "ab_test": {
        "enabled": True,
        "name": "sonnet-vs-nova",
        "variants": {
            "control": {"model": "us.anthropic.claude-sonnet-4-6", "weight": 0.5},
            "treatment": {"model": "us.amazon.nova-pro-v1:0", "weight": 0.5},
        },
        "sticky": True,
    },
})

# Same user always gets the same variant
for i in range(3):
    response = router.converse(
        messages=[{"role": "user", "content": [{"text": f"Request {i}"}]}],
        routing=RoutingConfig(metadata={"user_id": "user-alice"}),
    )
    d = response["routing_decision"]
    print(f"  Alice request {i}: {d.selected_model} (variant={d.metadata.get('ab_variant')})")

print(f"\nA/B stats: {router.ab_test.stats}")


# ── Example 2: Canary — roll out Opus 4.7 at 10% traffic ────────────

router = BedrockRouter.create({
    "canary": {
        "enabled": True,
        "baseline": "us.anthropic.claude-sonnet-4-6",
        "canary_model": "us.anthropic.claude-opus-4-7",
        "canary_percentage": 10,
        "auto_rollback": {"max_error_rate": 0.10, "max_latency_p95_ms": 5000},
        "auto_promote": {"min_requests": 50, "max_error_rate": 0.02},
    },
})

for i in range(5):
    response = router.converse(
        messages=[{"role": "user", "content": [{"text": f"Canary test {i}"}]}],
    )
    d = response["routing_decision"]
    is_canary = d.metadata.get("is_canary", False)
    print(f"  Request {i}: {d.selected_model} {'(CANARY)' if is_canary else ''}")

print(f"\nCanary stats: {router.canary.stats}")


# ── Example 3: Shadow mode — mirror 20% of traffic to Nova Pro ───────

router = BedrockRouter.create({
    "shadow": {
        "enabled": True,
        "shadow_model": "us.amazon.nova-pro-v1:0",
        "sample_rate": 0.2,
    },
})

response = router.converse(
    messages=[{"role": "user", "content": [{"text": "Shadow test"}]}],
)
print(f"\nPrimary: {response['routing_decision'].selected_model}")

import time
time.sleep(0.5)  # Wait for background shadow thread
print(f"Shadow stats: {router.shadow.stats}")
# Shadow responses are logged but never returned to the caller.
