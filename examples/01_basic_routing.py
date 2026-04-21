"""Basic Routing — simplest usage of the Bedrock Smart Router.

Demonstrates:
  - Creating a router with defaults
  - Creating a router from a YAML config
  - Inspecting routing decisions
"""

import yaml
from bedrock_smart_router import BedrockRouter, RoutingConfig

# ── Example 1: Zero-config defaults ─────────────────────────────────
# Uses balanced strategy, in-memory metrics, all models eligible.

router = BedrockRouter.create()

response = router.converse(
    messages=[{"role": "user", "content": [{"text": "What is Amazon S3?"}]}],
)

d = response["routing_decision"]
print(f"Model:      {d.selected_model}")
print(f"Strategy:   {d.strategy_used}")
print(f"Complexity: {d.complexity_detected}")
print(f"Cost:       ${d.actual_cost:.6f}")
print(f"Latency:    {d.latency_ms:.0f}ms")


# ── Example 2: Config from a YAML file ──────────────────────────────
# Application code never changes — just edit the YAML.

CONFIG = """
region: us-west-2
strategy: cost-optimized
cache:
  ttl_seconds: 1800
  max_entries: 5000
fallback:
  max_depth: 3
circuit_breaker:
  failure_threshold: 10
excluded_models:
  - "us.meta.*"
"""

router = BedrockRouter.create(yaml.safe_load(CONFIG))

response = router.converse(
    messages=[{"role": "user", "content": [{"text": "Summarize cloud computing."}]}],
)
print(f"\nYAML config → Model: {response['routing_decision'].selected_model}")


# ── Example 3: Config from a dict (e.g. loaded from DynamoDB/S3) ────

router = BedrockRouter.create({
    "region": "us-west-2",
    "strategy": "balanced",
    "weights": {"cost": 0.6, "latency": 0.2, "quality": 0.2},
})

response = router.converse(
    messages=[{"role": "user", "content": [{"text": "Hello!"}]}],
)
print(f"\nDict config → Model: {response['routing_decision'].selected_model}")
