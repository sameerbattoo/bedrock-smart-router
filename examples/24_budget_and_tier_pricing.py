# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Budget Enforcement & Inference Tier Pricing — control costs at every level.

Demonstrates:
  1. Per-request cost ceiling (max_cost_per_request)
  2. Inference tier pricing multipliers (Standard/Priority/Flex)
  3. Tier-aware cost estimation before the call
  4. How the router auto-selects tiers based on complexity and budget
  5. Rolling budget tracking per user/team
  6. Budget exceeded behavior (downgrade vs reject)
"""

from bedrock_smart_router import BedrockRouter, RoutingConfig, TIER_PRICING_MULTIPLIER
from bedrock_smart_router.models import ModelPricing
from bedrock_smart_router.budget_strategy import BudgetRule, BudgetTracker

# ── Example 1: Per-request cost ceiling ─────────────────────────────
# The simplest budget control: cap the cost of any single request.
# The router excludes models whose estimated cost exceeds the ceiling.

router = BedrockRouter.create({"region": "us-west-2"})

response = router.converse(
    messages=[{"role": "user", "content": [{"text": "What is EC2?"}]}],
    routing=RoutingConfig(max_cost_per_request=0.001),  # Max $0.001
)
d = response["routing_decision"]
print(f"Max $0.001 → {d.selected_model}  (actual: ${d.actual_cost:.6f})")

# Tighter budget forces cheaper models
response = router.converse(
    messages=[{"role": "user", "content": [{"text": "What is EC2?"}]}],
    routing=RoutingConfig(max_cost_per_request=0.0001),  # Max $0.0001
)
d = response["routing_decision"]
print(f"Max $0.0001 → {d.selected_model}  (actual: ${d.actual_cost:.6f})")


# ── Example 2: Inference tier pricing multipliers ───────────────────
# Bedrock charges different rates per tier:
#   Standard = 1.0× (base price)
#   Priority = ~1.75× (faster, premium)
#   Flex     = ~0.50× (slower, cheaper)
#
# The multipliers are available as a constant:

print(f"\nTier multipliers: {TIER_PRICING_MULTIPLIER}")
# {'standard': 1.0, 'priority': 1.75, 'flex': 0.5}


# ── Example 3: Tier-aware cost estimation ───────────────────────────
# Use estimate_cost(tier=...) to see what a request would cost on each tier.

model = router.registry.get("us.amazon.nova-pro-v1:0")
if model:
    input_tokens = 1000
    output_tokens = 500

    cost_standard = model.pricing.estimate_cost(input_tokens, output_tokens)
    cost_priority = model.pricing.estimate_cost(input_tokens, output_tokens, tier="optimized")
    cost_flex = model.pricing.estimate_cost(input_tokens, output_tokens, tier="standard")

    print(f"\nNova Pro cost for {input_tokens} in / {output_tokens} out:")
    print(f"  Standard: ${cost_standard:.6f}")
    print(f"  Priority: ${cost_priority:.6f}  ({cost_priority/cost_standard:.1f}× standard)")
    print(f"  Flex:     ${cost_flex:.6f}  ({cost_flex/cost_standard:.1f}× standard)")


# ── Example 4: Auto tier selection based on complexity ──────────────
# The router picks the tier automatically:
#   - Simple/moderate + tight budget → Flex (if model supports it)
#   - Complex/reasoning → Priority (if model supports it)
#   - Everything else → Standard
#
# IMPORTANT: Not all models support all tiers. Currently:
#   - Anthropic Claude: Standard + Priority
#   - Amazon Nova, Meta Llama, Mistral, DeepSeek: Standard only
# So Priority tier only kicks in when the strategy picks a Claude model.

router = BedrockRouter.create({
    "region": "us-west-2",
    "inference_tier": {
        "allow_priority": True,
        "allow_flex": True,
        "flex_for_batch": True,
        "priority_for_complex": True,
    },
})

# Simple question → Standard tier (Flex only triggers with tight budget)
response = router.converse(
    messages=[{"role": "user", "content": [{"text": "Hi"}]}],
    routing=RoutingConfig(max_cost_per_request=0.0001),
)
d = response["routing_decision"]
print(f"\nSimple + tight budget → tier={d.inference_tier}  model={d.selected_model}")

# Complex question with Anthropic → Priority tier (Claude supports it)
response = router.converse(
    messages=[{"role": "user", "content": [{"text": """
        Analyze the trade-offs between microservices and monolithic architecture.
        Consider scalability, team structure, deployment complexity, data consistency,
        and operational overhead. Provide a decision framework with specific criteria.
    """}]}],
    routing=RoutingConfig(preferred_family="anthropic"),
)
d = response["routing_decision"]
print(f"Complex + Anthropic → tier={d.inference_tier}  model={d.selected_model}")
# tier=priority because Claude supports it and the request is complex

# Same complex question with Nova → Standard tier (Nova doesn't support Priority)
response = router.converse(
    messages=[{"role": "user", "content": [{"text": """
        Analyze the trade-offs between microservices and monolithic architecture.
        Consider scalability, team structure, deployment complexity, data consistency,
        and operational overhead. Provide a decision framework with specific criteria.
    """}]}],
    routing=RoutingConfig(preferred_model="us.amazon.nova-pro-v1:0"),
)
d = response["routing_decision"]
print(f"Complex + Nova Pro  → tier={d.inference_tier}  model={d.selected_model}")
# tier=standard because Nova Pro only supports Standard


# ── Example 5: Disable specific tiers ───────────────────────────────
# Force Standard-only (no Priority premium, no Flex latency risk)

router_standard_only = BedrockRouter.create({
    "region": "us-west-2",
    "inference_tier": {
        "allow_priority": False,
        "allow_flex": False,
    },
})

response = router_standard_only.converse(
    messages=[{"role": "user", "content": [{"text": "Complex analysis..."}]}],
)
d = response["routing_decision"]
print(f"\nStandard-only → tier={d.inference_tier}")


# ── Example 6: Rolling budget tracking ──────────────────────────────
# Track spend per user/team with hourly and daily limits.
# When exceeded, the strategy either downgrades to a cheaper model
# or rejects the request.

tracker = BudgetTracker()

# Define budget rules
enterprise_rule = BudgetRule(
    max_cost_per_request=0.05,    # Max $0.05 per request
    max_hourly_spend=1.00,        # Max $1.00/hour
    max_daily_spend=10.00,        # Max $10.00/day
    on_exceeded="downgrade",      # Downgrade to cheaper model (vs "reject")
    downgrade_to_tier="lite",     # Downgrade target
)

free_rule = BudgetRule(
    max_cost_per_request=0.001,
    max_hourly_spend=0.10,
    max_daily_spend=0.50,
    on_exceeded="reject",         # Hard reject when budget exceeded
)

print(f"\nEnterprise budget: ${enterprise_rule.max_daily_spend}/day, "
      f"${enterprise_rule.max_hourly_spend}/hour")
print(f"Free budget:       ${free_rule.max_daily_spend}/day, "
      f"${free_rule.max_hourly_spend}/hour")

# Simulate spend tracking
import time
tracker.record_spend("user-enterprise-001", 0.02)
tracker.record_spend("user-enterprise-001", 0.03)

hourly_spend = tracker.get_spend("user-enterprise-001", 3600)
exceeded = tracker.check_budget("user-enterprise-001", enterprise_rule)
print(f"\nEnterprise user after $0.05 spend:")
print(f"  Hourly spend so far: ${hourly_spend:.4f}")
print(f"  Budget exceeded?     {exceeded or 'No — within limits'}")


# ── Example 7: Economy preset = built-in budget control ─────────────
# The "economy" preset sets max_cost_per_request=$0.002 automatically.
# This is the simplest way to enforce cost control.

router = BedrockRouter.create({"region": "us-west-2"})

response = router.converse(
    messages=[{"role": "user", "content": [{"text": "Classify: is this spam?"}]}],
    routing=RoutingConfig(preset="economy"),
)
d = response["routing_decision"]
print(f"\nEconomy preset → {d.selected_model}  ${d.actual_cost:.6f}")
print(f"  (economy preset enforces max_cost_per_request=$0.002)")
