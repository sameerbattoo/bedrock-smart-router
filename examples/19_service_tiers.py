"""Example 19: Service Tier Routing (Flex / Priority)

Amazon Bedrock offers multiple service tiers for inference:
- **Standard** (default): Pay-per-token, no commitment
- **Priority**: Higher throughput, ~25% better latency, premium pricing (~1.75×)
- **Flex**: Lower cost (~0.50×), latency-tolerant (may take minutes), 1-hour timeout

The smart router can filter models to only those supporting a specific
service tier, preventing ValidationException errors when a model doesn't
support the requested tier.

Not all models support Flex/Priority. The router knows which models support
which tiers (from the `supported_service_tiers` field in the model catalog)
and pre-filters candidates accordingly.

Usage:
    python examples/19_service_tiers.py
"""

from bedrock_smart_router import BedrockRouter
from bedrock_smart_router.config import RoutingConfig

router = BedrockRouter.create({"region": "us-west-2"})

# ═══════════════════════════════════════════════════════════════════════
# Scenario 1: Flex tier — cheap batch processing
# The router selects only from models that support Flex tier.
# ═══════════════════════════════════════════════════════════════════════

print("=" * 60)
print("Scenario 1: Flex tier (cost-optimized, latency-tolerant)")
print("=" * 60)

response = router.converse(
    messages=[{"role": "user", "content": [{"text": "Summarize the history of cloud computing in 3 sentences."}]}],
    routing=RoutingConfig(
        strategy="cost-optimized",
        service_tier="flex",  # Only models supporting Flex are candidates
    ),
    serviceTier={"type": "flex"},  # Passed to Bedrock API
)

d = response["routing_decision"]
print(f"Model selected: {d.selected_model}")
print(f"Strategy: {d.strategy_used}")
print(f"Complexity: {d.complexity_detected}")
print(f"Service tier used: {d.actual_service_tier}")
print(f"Cost: ${d.actual_cost:.6f}")
print()

# ═══════════════════════════════════════════════════════════════════════
# Scenario 2: Priority tier — mission-critical, low latency
# The router selects only from Priority-capable models and picks
# the best one based on quality strategy.
# ═══════════════════════════════════════════════════════════════════════

print("=" * 60)
print("Scenario 2: Priority tier (low-latency, mission-critical)")
print("=" * 60)

response = router.converse(
    messages=[{"role": "user", "content": [{"text": "What is Amazon S3?"}]}],
    routing=RoutingConfig(
        strategy="quality-optimized",
        service_tier="priority",  # Only Priority-capable models
    ),
    serviceTier={"type": "priority"},  # Passed to Bedrock API
)

d = response["routing_decision"]
print(f"Model selected: {d.selected_model}")
print(f"Strategy: {d.strategy_used}")
print(f"Complexity: {d.complexity_detected}")
print(f"Service tier used: {d.actual_service_tier}")
print(f"Cost: ${d.actual_cost:.6f}")
print()

# ═══════════════════════════════════════════════════════════════════════
# Scenario 3: Standard tier (default — no filtering)
# All models are eligible. No serviceTier param needed.
# ═══════════════════════════════════════════════════════════════════════

print("=" * 60)
print("Scenario 3: Standard tier (default, all models eligible)")
print("=" * 60)

response = router.converse(
    messages=[{"role": "user", "content": [{"text": "Hello!"}]}],
    routing=RoutingConfig(strategy="balanced"),
    # No serviceTier — uses standard (default)
)

d = response["routing_decision"]
print(f"Model selected: {d.selected_model}")
print(f"Strategy: {d.strategy_used}")
print(f"Complexity: {d.complexity_detected}")
print(f"Cost: ${d.actual_cost:.6f}")
print()

# ═══════════════════════════════════════════════════════════════════════
# Scenario 4: Flex + preferred family
# Combine service_tier with preferred_family to narrow further.
# ═══════════════════════════════════════════════════════════════════════

print("=" * 60)
print("Scenario 4: Flex tier + preferred family (mistral)")
print("=" * 60)

response = router.converse(
    messages=[{"role": "user", "content": [{"text": "Write a Python function to reverse a string."}]}],
    routing=RoutingConfig(
        strategy="balanced",
        service_tier="flex",
        preferred_family="mistral",
    ),
    serviceTier={"type": "flex"},
)

d = response["routing_decision"]
print(f"Model selected: {d.selected_model}")
print(f"Family: {d.selected_model.split('.')[0]}")
print(f"Complexity: {d.complexity_detected}")
print(f"Service tier used: {d.actual_service_tier}")
print(f"Cost: ${d.actual_cost:.6f}")
