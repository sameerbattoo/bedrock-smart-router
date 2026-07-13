# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Multi-Tenant Support — Application Inference Profiles for cost tracking.

Demonstrates:
  - Auto-creating AIPs per tenant
  - Per-tenant cost attribution via tags
  - Passing tenant metadata in routing config
"""

from bedrock_smart_router import BedrockRouter, RoutingConfig

# ── Example 1: Enable multi-tenant with auto-create AIPs ─────────────

router = BedrockRouter.create({
    "aip": {
        "enabled": True,
        "auto_create": True,
        "tag_keys": ["tenant", "team", "environment"],
    },
})

# Each tenant's requests are tagged for Cost Explorer attribution
response = router.converse(
    messages=[{"role": "user", "content": [{"text": "Hello from Acme Corp"}]}],
    routing=RoutingConfig(
        metadata={"tenant": "acme-corp", "team": "engineering", "environment": "prod"},
    ),
)
print(f"Acme Corp → {response['routing_decision'].selected_model}")


# ── Example 2: Different tenant, same model — separate cost tracking ─

response = router.converse(
    messages=[{"role": "user", "content": [{"text": "Hello from Globex"}]}],
    routing=RoutingConfig(
        metadata={"tenant": "globex", "team": "research", "environment": "prod"},
    ),
)
print(f"Globex → {response['routing_decision'].selected_model}")

# In Cost Explorer, you'll see separate cost lines for acme-corp and globex.


# ── Example 3: Inspect cached AIP profiles ───────────────────────────

for key, entry in router.aip.cached_profiles.items():
    print(f"\n  AIP: {entry.profile_name}")
    print(f"  ARN: {entry.profile_arn}")
    print(f"  Model: {entry.model_id}")
    print(f"  Tags: {entry.tags}")
