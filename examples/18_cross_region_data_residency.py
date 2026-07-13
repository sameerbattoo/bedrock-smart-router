# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cross-Region Inference & Data Residency — CRIS profiles explained.

Amazon Bedrock Cross-Region Inference (CRIS) automatically routes
requests across AWS regions for higher throughput and availability.
Geography-specific profiles ensure data residency compliance.

How it works:
  Your boto3 client connects to ONE region (e.g. us-west-2).
  But the model ID you send determines WHERE inference actually runs:

  us.anthropic.claude-sonnet-4-6      → US regions only
  eu.anthropic.claude-sonnet-4-6      → EU regions only
  global.anthropic.claude-sonnet-4-6  → Any commercial region

  The router's CRISManager selects the right profile automatically
  based on your config. You never manage regions manually.

Important distinctions:
  - preferred_geography is a SOFT preference — falls back if unavailable
  - blocked_prefixes is a HARD constraint — blocked profiles are never used
  - allow_global controls whether global.* profiles are permitted
  - The 'region' config determines which endpoint you connect to (and where
    Mantle-only models run, since Mantle has no CRIS)

Demonstrates:
  - US-only routing with hard enforcement (blocked_prefixes)
  - EU-only routing (GDPR compliance)
  - Global routing (maximum throughput)
  - Soft vs hard geography constraints
  - Inspecting which CRIS profile was selected
  - Mantle-only models and their regional behavior
"""

from bedrock_smart_router import BedrockRouter, RoutingConfig


# ═══════════════════════════════════════════════════════════════════
# Example 1: US-Only (HARD) — data guaranteed to stay in the US
# ═══════════════════════════════════════════════════════════════════
# Use case: US financial services, ITAR, FedRAMP workloads.
#
# Strategy: Use a US region endpoint + only allow us.* CRIS profiles.
# This guarantees inference runs in the US because:
#   1. The bedrock-runtime endpoint is in us-west-2 (US)
#   2. Only us.* CRIS profiles are permitted (allowlist)
#   3. Mantle-only models run on bedrock-mantle.us-west-2 (also US)

router_us_hard = BedrockRouter.create({
    "region": "us-west-2",
    "cris": {
        "allowed_prefixes": ["us"],  # HARD: only us.* profiles permitted
    },
})

response = router_us_hard.converse(
    messages=[{"role": "user", "content": [{"text": "Explain S3 encryption."}]}],
)
d = response["routing_decision"]
print(f"US-Only (hard enforcement):")
print(f"  Model:        {d.selected_model}")
print(f"  CRIS profile: {d.cris_profile}")
print(f"  Region:       us-west-2 (endpoint)")
# CRIS profile will be "us.anthropic.claude-..." or direct model ID.
# Either way, inference stays in the US.


# ═══════════════════════════════════════════════════════════════════
# Example 2: US Preferred (SOFT) — prefer US, allow global fallback
# ═══════════════════════════════════════════════════════════════════
# Use case: US-based app, prefer low latency (US), but allow global
# if US capacity is saturated. NOT suitable for strict data residency.
#
# Note: preferred_geography is a SOFT preference. If a model doesn't
# have a us.* profile, the router falls back to global.* or direct.

router_us_soft = BedrockRouter.create({
    "region": "us-west-2",
    "cris": {
        "preferred_geography": "us",
        "allow_global": True,  # Fall back to global if no us.* profile
    },
})

response = router_us_soft.converse(
    messages=[{"role": "user", "content": [{"text": "Hello"}]}],
)
d = response["routing_decision"]
print(f"\nUS-preferred (soft, with global fallback):")
print(f"  CRIS profile: {d.cris_profile}")
# May be us.*, global.*, or direct — depending on model availability.


# ═══════════════════════════════════════════════════════════════════
# Example 3: EU-Only — GDPR compliance, data stays in Europe
# ═══════════════════════════════════════════════════════════════════
# Use case: European banks, healthcare, GDPR-regulated workloads.
#
# Strategy: EU region endpoint + block non-EU profiles.
# Inference runs in eu-central-1, eu-west-1, eu-north-1, etc.

router_eu = BedrockRouter.create({
    "region": "eu-west-1",  # Client connects to Ireland
    "cris": {
        "allowed_prefixes": ["eu"],  # HARD: only eu.* profiles permitted
    },
})

# This request's inference will run in an EU region
# response = router_eu.converse(
#     messages=[{"role": "user", "content": [{"text": "Explain GDPR Article 17."}]}],
# )
# d = response["routing_decision"]
# print(f"  CRIS profile: {d.cris_profile}")  # eu.anthropic.claude-...
print("\nEU-Only: would use eu.* CRIS profiles (requires EU region credentials)")


# ═══════════════════════════════════════════════════════════════════
# Example 4: Global — maximum throughput, no residency requirement
# ═══════════════════════════════════════════════════════════════════
# Use case: Internal tools, non-regulated workloads, batch processing.
# Bedrock picks the region with the most available capacity worldwide.
# Best throughput and lowest queue times.

router_global = BedrockRouter.create({
    "region": "us-west-2",
    "cris": {
        "allow_global": True,
        # No preferred_geography — Bedrock picks optimal region
        # No blocked_prefixes — everything is allowed
    },
})

response = router_global.converse(
    messages=[{"role": "user", "content": [{"text": "What is EC2?"}]}],
)
d = response["routing_decision"]
print(f"\nGlobal routing (max throughput):")
print(f"  Model:        {d.selected_model}")
print(f"  CRIS profile: {d.cris_profile}")
# Profile will be "global.*" for maximum capacity, or "us.*" if global
# isn't available for the selected model.


# ═══════════════════════════════════════════════════════════════════
# Example 5: Inspect CRIS profiles available for each model
# ═══════════════════════════════════════════════════════════════════
# See which models have which geography profiles in your region.

router = BedrockRouter.create()
print(f"\nCRIS profiles per model (sample):")
count = 0
for model in router._registry.all_models:
    # Extract available geographies from the regions data
    geos = set()
    for r in model.regions:
        for profile in r.get("cris_profiles", []):
            geos.add(profile)
    if geos:
        print(f"  {model.display_name:35s} → {', '.join(sorted(geos))}")
        count += 1
    elif "converse" in model.api_support:
        # Converse model without CRIS = direct invocation only
        regions = [r["name"] for r in model.regions if r.get("direct")]
        if count < 5:  # Just show a few
            print(f"  {model.display_name:35s} → direct only ({', '.join(regions[:3])})")
    else:
        # Mantle-only model — no CRIS, runs in endpoint region
        if count < 5:
            print(f"  {model.display_name:35s} → Mantle-only (no CRIS)")
    count += 1
    if count >= 15:
        print(f"  ... ({len(router._registry.all_models) - 15} more)")
        break


# ═══════════════════════════════════════════════════════════════════
# Example 6: Mantle-only models and data residency
# ═══════════════════════════════════════════════════════════════════
# Mantle-only models (e.g., DeepSeek V3.1, Voxtral, GPT-5.4) have
# NO CRIS support. They run on the bedrock-mantle.<region> endpoint
# in whichever region you configure.
#
# The router automatically filters Mantle-only models to only those
# available in your configured region. If you're in us-west-2, a
# Mantle model only available in us-east-2 won't be selected.

router_mantle = BedrockRouter.create({"region": "us-west-2"})

# Force a Mantle-only model to see how it's handled
response = router_mantle.converse(
    messages=[{"role": "user", "content": [{"text": "Hi"}]}],
    routing=RoutingConfig(preferred_model="deepseek.v3.1"),
    inferenceConfig={"maxTokens": 10},
)
d = response["routing_decision"]
print(f"\nMantle-only model (DeepSeek V3.1):")
print(f"  Model:        {d.selected_model}")
print(f"  CRIS profile: {d.cris_profile}")
# CRIS profile will be the model ID itself (no us.*/eu.* prefix)
# because Mantle doesn't support CRIS.


# ═══════════════════════════════════════════════════════════════════
# Example 7: Multiple compliance zones in one application
# ═══════════════════════════════════════════════════════════════════
# Create separate routers for different compliance requirements.

router_regulated = BedrockRouter.create({
    "region": "eu-west-1",
    "cris": {
        "allowed_prefixes": ["eu"],  # HARD: only EU
    },
})
router_internal = BedrockRouter.create({
    "region": "us-west-2",
    "cris": {"allow_global": True},
})

# Customer-facing (GDPR) → EU only
# response = router_regulated.converse(messages=[...])

# Internal analytics → global for speed
response = router_internal.converse(
    messages=[{"role": "user", "content": [{"text": "Summarize this data"}]}],
)
print(f"\nInternal (global): {response['routing_decision'].cris_profile}")


# ═══════════════════════════════════════════════════════════════════
# Summary: CRIS Geography Profiles
# ═══════════════════════════════════════════════════════════════════
#
# | Profile Prefix | Routes To              | Data Residency        |
# |----------------|------------------------|-----------------------|
# | us.*           | US regions only        | Data stays in US      |
# | eu.*           | EU regions only        | GDPR compliant        |
# | ap.*           | Asia-Pacific only      | APAC data residency   |
# | global.*       | Any commercial region  | No restriction        |
# | (no prefix)    | Configured region only | Single-region locked  |
#
# Config options:
# | Setting              | Type | Effect                                 |
# |----------------------|------|----------------------------------------|
# | preferred_geography  | Soft | Prefers this geo, falls back if needed |
# | allow_global         | Hard | Blocks/allows global.* profiles        |
# | blocked_prefixes     | Hard | Blocks specific geo prefixes entirely  |
# | allowed_prefixes     | Hard | ONLY these prefixes permitted (safest) |
#
# For strict data residency (recommended):
#   1. Set region to a region in your required geography
#   2. Set allowed_prefixes to ONLY your geography (e.g., ["us"])
#   That's it — clean and simple.
#
# Alternative (blocklist approach):
#   1. Set region to your geography
#   2. Set allow_global: False
#   3. Set blocked_prefixes for all OTHER geographies
#
# The router handles CRIS for bedrock-runtime models automatically.
# Mantle-only models always run in the configured region (no CRIS).
