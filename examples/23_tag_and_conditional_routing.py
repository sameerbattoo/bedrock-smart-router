# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tag-Based and Conditional Routing — route by metadata, not just content.

Demonstrates:
  1. Tag-based routing: free/paid tiers with different model pools
  2. Tag-based routing: team-specific model access
  3. Conditional routing: metadata-driven strategy selection
  4. Conditional routing: region-based model restriction
  5. Combining tags + conditions in a single request
"""

from bedrock_smart_router import BedrockRouter, RoutingConfig

# ── Example 1: Free vs Paid tier routing ────────────────────────────
# Free-tier users get cheap models. Paid-tier users get the full catalog.
# Tags are passed per-request via RoutingConfig.tags.

router = BedrockRouter.create({
    "region": "us-west-2",
    "strategy": "balanced",
})

# Free-tier user — restrict to cheap models via exclude
response = router.converse(
    messages=[{"role": "user", "content": [{"text": "What is S3?"}]}],
    routing=RoutingConfig(
        tags=["free-tier"],
        exclude_models=["us.anthropic.*", "us.deepseek.*"],  # Only Nova + Meta
        max_cost_per_request=0.001,
    ),
)
d = response["routing_decision"]
print(f"Free tier → {d.selected_model}  (${d.actual_cost:.6f})")

# Paid-tier user — full access, quality-optimized
response = router.converse(
    messages=[{"role": "user", "content": [{"text": "Analyze this contract clause..."}]}],
    routing=RoutingConfig(
        tags=["paid-tier"],
        strategy="quality-optimized",
    ),
)
d = response["routing_decision"]
print(f"Paid tier → {d.selected_model}  (${d.actual_cost:.6f})")


# ── Example 2: Team-specific model access ───────────────────────────
# Engineering team gets code-capable models. Marketing gets creative models.

# Engineering: prefer Anthropic (strong at code)
response = router.converse(
    messages=[{"role": "user", "content": [{"text": "Write a Python decorator for caching"}]}],
    routing=RoutingConfig(
        tags=["team-engineering"],
        preferred_family="anthropic",
    ),
)
d = response["routing_decision"]
print(f"\nEngineering → {d.selected_model}")

# Marketing: prefer cheap models, economy preset
response = router.converse(
    messages=[{"role": "user", "content": [{"text": "Write a tagline for our product"}]}],
    routing=RoutingConfig(
        tags=["team-marketing"],
        preset="economy",
    ),
)
d = response["routing_decision"]
print(f"Marketing   → {d.selected_model}")


# ── Example 3: Conditional routing by user metadata ─────────────────
# Route based on metadata fields: user tier, region, environment.
# The metadata dict is passed per-request and can drive routing decisions.

# Enterprise user → quality strategy
response = router.converse(
    messages=[{"role": "user", "content": [{"text": "Summarize quarterly earnings"}]}],
    routing=RoutingConfig(
        strategy="quality-optimized",
        metadata={"user_tier": "enterprise", "user_id": "u-001"},
    ),
)
d = response["routing_decision"]
print(f"\nEnterprise user → {d.selected_model}  strategy={d.strategy_used}")

# Internal/dev user → economy strategy
response = router.converse(
    messages=[{"role": "user", "content": [{"text": "Test prompt"}]}],
    routing=RoutingConfig(
        preset="economy",
        metadata={"user_tier": "internal", "user_id": "dev-test"},
    ),
)
d = response["routing_decision"]
print(f"Internal user   → {d.selected_model}  strategy={d.strategy_used}")


# ── Example 4: Region-based model restriction ───────────────────────
# EU users → Anthropic only (GDPR compliance, data stays in EU via CRIS)
# US users → any model

router_eu = BedrockRouter.create({
    "region": "eu-west-1",
    "cris": {"preferred_geography": "eu"},
})

response = router_eu.converse(
    messages=[{"role": "user", "content": [{"text": "Explain GDPR Article 17"}]}],
    routing=RoutingConfig(
        preferred_family="anthropic",
        metadata={"region": "eu", "compliance": "gdpr"},
    ),
)
d = response["routing_decision"]
print(f"\nEU user → {d.selected_model}  cris={d.cris_profile}")


# ── Example 5: Combining tags + metadata + presets ──────────────────
# Real-world: paid enterprise user in EU, needs quality routing

router = BedrockRouter.create({"region": "us-west-2"})

response = router.converse(
    messages=[{"role": "user", "content": [{"text": "Draft a legal brief on IP law"}]}],
    routing=RoutingConfig(
        preset="quality",
        tags=["paid-tier", "team-legal"],
        preferred_family="anthropic",
        metadata={
            "user_id": "u-legal-042",
            "user_tier": "enterprise",
            "tenant": "acme-corp",
            "region": "us",
        },
    ),
)
d = response["routing_decision"]
print(f"\nCombined → {d.selected_model}")
print(f"  Strategy:   {d.strategy_used}")
print(f"  Complexity: {d.complexity_detected}")
print(f"  Cost:       ${d.actual_cost:.6f}")
print(f"  Tier:       {d.inference_tier}")
print(f"  Metadata:   {d.metadata}")
