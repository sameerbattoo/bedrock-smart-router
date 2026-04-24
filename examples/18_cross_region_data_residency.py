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
  based on your config.  You never manage regions manually.

Demonstrates:
  - US-only routing (data stays in US)
  - EU-only routing (GDPR compliance)
  - Global routing (maximum throughput)
  - Inspecting which CRIS profile was selected
  - Restricting to geography with no global fallback
"""

from bedrock_smart_router import BedrockRouter, RoutingConfig


# ═══════════════════════════════════════════════════════════════════
# Example 1: US-Only — data never leaves the United States
# ═══════════════════════════════════════════════════════════════════
# Use case: US financial services, ITAR, FedRAMP workloads.
# Bedrock routes across us-east-1, us-west-2, etc. for capacity,
# but inference always runs in a US region.

router_us = BedrockRouter.create({
    "region": "us-west-2",
    "cris": {
        "preferred_geography": "us",
        "allow_global": False,  # Never fall back to global profiles
    },
})

response = router_us.converse(
    messages=[{"role": "user", "content": [{"text": "Explain S3 encryption."}]}],
)
d = response["routing_decision"]
print(f"US-Only routing:")
print(f"  Model:        {d.selected_model}")
print(f"  CRIS profile: {d.cris_profile}")
print(f"  Profile starts with 'us.': {d.cris_profile.startswith('us.')}")
# CRIS profile will be like "us.anthropic.claude-sonnet-4-6"
# Bedrock guarantees inference runs in a US region.


# ═══════════════════════════════════════════════════════════════════
# Example 2: EU-Only — GDPR compliance, data stays in Europe
# ═══════════════════════════════════════════════════════════════════
# Use case: European banks, healthcare, any GDPR-regulated workload.
# Even though your client connects to eu-west-1, inference could
# run in eu-central-1 or eu-north-1 — but always within the EU.

router_eu = BedrockRouter.create({
    "region": "eu-west-1",  # Client connects to Ireland
    "cris": {
        "preferred_geography": "eu",
        "allow_global": False,  # Strict — EU only, no exceptions
    },
})

# This request's inference will run in an EU region
# response = router_eu.converse(
#     messages=[{"role": "user", "content": [{"text": "Explain GDPR Article 17."}]}],
# )
# d = response["routing_decision"]
# print(f"\nEU-Only routing:")
# print(f"  CRIS profile: {d.cris_profile}")  # eu.anthropic.claude-...
print("\nEU-Only: would use eu.* CRIS profiles (requires EU region client)")


# ═══════════════════════════════════════════════════════════════════
# Example 3: Global — maximum throughput, no residency requirement
# ═══════════════════════════════════════════════════════════════════
# Use case: Internal tools, non-regulated workloads, batch processing.
# Bedrock picks the region with the most available capacity worldwide.
# Best throughput and lowest queue times.

router_global = BedrockRouter.create({
    "region": "us-west-2",
    "cris": {
        "allow_global": True,  # Prefer global.* profiles
        # No preferred_geography — let Bedrock pick the best region
    },
})

response = router_global.converse(
    messages=[{"role": "user", "content": [{"text": "What is EC2?"}]}],
)
d = response["routing_decision"]
print(f"\nGlobal routing:")
print(f"  Model:        {d.selected_model}")
print(f"  CRIS profile: {d.cris_profile}")
# Profile will be "global.anthropic.claude-..." or "us.amazon.nova-..."
# depending on which model was selected and what profiles are available.


# ═══════════════════════════════════════════════════════════════════
# Example 4: US preferred, global fallback
# ═══════════════════════════════════════════════════════════════════
# Use case: Prefer US for latency, but allow global if US is at capacity.
# This is the recommended default for US-based applications without
# strict data residency requirements.

router_us_fallback = BedrockRouter.create({
    "region": "us-west-2",
    "cris": {
        "preferred_geography": "us",
        "allow_global": True,  # Fall back to global if no US profile
    },
})

response = router_us_fallback.converse(
    messages=[{"role": "user", "content": [{"text": "Hello"}]}],
)
d = response["routing_decision"]
print(f"\nUS-preferred with global fallback:")
print(f"  CRIS profile: {d.cris_profile}")


# ═══════════════════════════════════════════════════════════════════
# Example 5: Inspect CRIS profiles for all models in the catalog
# ═══════════════════════════════════════════════════════════════════
# See which models have which geography profiles available.

router = BedrockRouter.create()
print(f"\nCRIS profiles per model:")
for model in router.registry.list_models():
    profiles = model.cris_profiles
    if profiles:
        geos = set()
        for p in profiles:
            for prefix in ("us.", "eu.", "ap.", "global."):
                if p.startswith(prefix):
                    geos.add(prefix.rstrip("."))
        print(f"  {model.display_name:30s} → {', '.join(sorted(geos))}")
    else:
        print(f"  {model.display_name:30s} → no CRIS (region-locked)")


# ═══════════════════════════════════════════════════════════════════
# Example 6: Per-request geography override
# ═══════════════════════════════════════════════════════════════════
# The CRIS config is global, but you can use different routers for
# different compliance zones in the same application.

router_regulated = BedrockRouter.create({
    "cris": {"preferred_geography": "eu", "allow_global": False},
})
router_internal = BedrockRouter.create({
    "cris": {"allow_global": True},
})

# Customer-facing (regulated) → EU only
# response = router_regulated.converse(messages=[...])

# Internal analytics (not regulated) → global for speed
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
# The router selects the profile automatically based on your config.
# You never need to manage region endpoints or create multiple clients.
