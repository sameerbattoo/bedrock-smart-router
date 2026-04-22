"""Basic Routing — simplest usage of the Bedrock Smart Router.

Demonstrates:
  1. Zero-config (defaults to us-west-2, balanced strategy)
  2. Explicit region
  3. Pinning a specific model with preferred_model
  4. Config from a YAML string
  5. Config from a dict
"""

import yaml
from bedrock_smart_router import BedrockRouter, RoutingConfig

# ── Example 1: Zero-config (defaults to us-west-2) ──────────────────
# No config needed — region defaults to us-west-2, strategy to balanced.

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
print(f"Stop reason:       {d.stop_reason}")
print(f"Bedrock latency:   {d.bedrock_latency_ms}ms")
print(f"Service tier:      {d.actual_service_tier}")
print(f"Prompt cache read: {d.prompt_cache_read_tokens} tokens")
print(f"Prompt cache write:{d.prompt_cache_write_tokens} tokens")
print(f"Prompt cache hit rate: {d.prompt_cache_hit_rate:.0%}")
print(f"Total input tokens:   {d.total_input_tokens}")
print(f"Network overhead:     {d.network_overhead_ms}ms")


# ── Example 2: Explicit region ───────────────────────────────────────
# Pass the region when your Bedrock endpoint is in a different region.

router = BedrockRouter.create({"region": "us-east-1"})

response = router.converse(
    messages=[{"role": "user", "content": [{"text": "List 3 benefits of Lambda."}]}],
)
print(f"\nus-east-1 → Model: {response['routing_decision'].selected_model}")


# ── Example 3: Pin a specific model ─────────────────────────────────
# Use preferred_model to always route to a specific model.
# The router still provides fallbacks, circuit breakers, and metrics —
# you get reliability without giving up model choice.

router = BedrockRouter.create({"region": "us-west-2"})

response = router.converse(
    messages=[{"role": "user", "content": [{"text": "Explain VPC peering in AWS."}]}],
    routing=RoutingConfig(preferred_model="us.amazon.nova-pro-v1:0"),
)

d = response["routing_decision"]
print(f"\nPinned model → {d.selected_model}")
print(f"Cost:    ${d.actual_cost:.6f}")
print(f"Latency: {d.latency_ms:.0f}ms")
print(f"Tokens:  {d.input_tokens} in / {d.output_tokens} out")
# If Nova Pro were down, the router would fall back automatically:
print(f"Fallback chain: {d.fallback_chain}")


# ── Example 4: Config from a YAML file ──────────────────────────────
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


# ── Example 5: Config from a dict (e.g. loaded from DynamoDB/S3) ────

router = BedrockRouter.create({
    "region": "us-west-2",
    "strategy": "balanced",
    "weights": {"cost": 0.6, "latency": 0.2, "quality": 0.2},
})

response = router.converse(
    messages=[{"role": "user", "content": [{"text": "Hello!"}]}],
)
print(f"\nDict config → Model: {response['routing_decision'].selected_model}")
