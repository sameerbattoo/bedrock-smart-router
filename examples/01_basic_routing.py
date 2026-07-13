# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Basic Routing — simplest usage of the Bedrock Smart Router.

Demonstrates:
  1. Zero-config (defaults to us-west-2, balanced strategy)
  2. Explicit region
  3. Boto3 drop-in: modelId parameter
  4. Boto3 drop-in: camelCase parameters (inferenceConfig, toolConfig)
  5. Pin a model with preferred_model (RoutingConfig)
  6. Let the router pick (no modelId) — adds routing intelligence
  7. Config from a YAML string
  8. Config from a dict
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


# ═══════════════════════════════════════════════════════════════════
# Boto3 Drop-In Compatibility
#
# The router accepts the same parameter names as boto3's
# bedrock-runtime.converse(). You can migrate by changing one line:
#
#   BEFORE:  response = client.converse(modelId="...", messages=[...])
#   AFTER:   response = router.converse(modelId="...", messages=[...])
#
# Both snake_case and camelCase parameter names are accepted.
# ═══════════════════════════════════════════════════════════════════


# ── Example 3: Boto3 drop-in with modelId ────────────────────────────
# Pass modelId just like you would with boto3. The router uses that
# exact model but still provides fallbacks, circuit breakers, and metrics.

router = BedrockRouter.create({"region": "us-west-2"})

response = router.converse(
    modelId="us.amazon.nova-pro-v1:0",
    messages=[{"role": "user", "content": [{"text": "Explain VPC peering."}]}],
)
d = response["routing_decision"]
print(f"\nBoto3 modelId → {d.selected_model}")
print(f"  Cost:    ${d.actual_cost:.6f}")
print(f"  Latency: {d.latency_ms:.0f}ms")
# The router still builds a fallback chain in case the model fails:
print(f"  Fallback chain: {d.fallback_chain}")


# ── Example 4: Boto3 camelCase parameters ────────────────────────────
# inferenceConfig, toolConfig — both camelCase (boto3) and snake_case work.

response = router.converse(
    modelId="us.amazon.nova-pro-v1:0",
    messages=[{"role": "user", "content": [{"text": "Write a haiku about clouds."}]}],
    inferenceConfig={"maxTokens": 100, "temperature": 0.9},
)
d = response["routing_decision"]
print(f"\ncamelCase params → {d.selected_model}, {d.output_tokens} tokens")

# Same thing with snake_case — both work:
response = router.converse(
    model_id="us.amazon.nova-pro-v1:0",
    messages=[{"role": "user", "content": [{"text": "Write a haiku about rain."}]}],
    inference_config={"maxTokens": 100, "temperature": 0.9},
)
d = response["routing_decision"]
print(f"snake_case params → {d.selected_model}, {d.output_tokens} tokens")


# ── Example 5: Pin a model with RoutingConfig ────────────────────────
# For advanced routing control, use RoutingConfig. preferred_model pins
# the model while letting you also set strategy, tags, metadata, etc.

response = router.converse(
    messages=[{"role": "user", "content": [{"text": "Explain VPC peering in AWS."}]}],
    routing=RoutingConfig(
        preferred_model="us.amazon.nova-pro-v1:0",
        metadata={"user_id": "u-123", "team": "networking"},
    ),
)
d = response["routing_decision"]
print(f"\nRoutingConfig → {d.selected_model}")
print(f"  Tokens: {d.input_tokens} in / {d.output_tokens} out")


# ── Example 6: Let the router pick (smart routing) ──────────────────
# Omit modelId entirely. The router analyzes the request complexity
# and picks the optimal model based on cost, latency, and quality.

# Simple question → cheap model (Nova Micro or similar)
response = router.converse(
    messages=[{"role": "user", "content": [{"text": "What is EC2?"}]}],
)
print(f"\nSimple question → {response['routing_decision'].selected_model}")

# Complex question → capable model (Sonnet, Nova Pro, or similar)
response = router.converse(
    messages=[{"role": "user", "content": [{"text": """
        Analyze the trade-offs between microservices and monolithic architecture.
        Consider scalability, team structure, and operational overhead.
    """}]}],
)
print(f"Complex question → {response['routing_decision'].selected_model}")


# ── Example 7: Config from a YAML file ──────────────────────────────
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


# ── Example 8: Config from a dict (e.g. loaded from DynamoDB/S3) ────

router = BedrockRouter.create({
    "region": "us-west-2",
    "strategy": "balanced",
    "weights": {"cost": 0.6, "latency": 0.2, "quality": 0.2},
})

response = router.converse(
    messages=[{"role": "user", "content": [{"text": "Hello!"}]}],
)
print(f"\nDict config → Model: {response['routing_decision'].selected_model}")
