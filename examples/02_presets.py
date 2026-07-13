# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Named Presets — one-word shortcuts for common routing profiles.

Demonstrates:
  - Using built-in presets (economy, speed, balanced, quality)
  - Overriding preset defaults with explicit fields
  - Comparing costs across presets
"""

from bedrock_smart_router import BedrockRouter, RoutingConfig

router = BedrockRouter.create()


# ── Example 1: Economy preset — cheapest model ──────────────────────
# Great for classification, extraction, simple Q&A.

response = router.converse(
    messages=[{"role": "user", "content": [{"text": "Is this email spam? Yes or no."}]}],
    routing=RoutingConfig(preset="economy"),
)
d = response["routing_decision"]
print(f"Economy → {d.selected_model}, cost=${d.actual_cost:.6f}")


# ── Example 2: Quality preset — best model regardless of cost ───────
# Great for complex reasoning, analysis, code generation.

response = router.converse(
    messages=[{"role": "user", "content": [
        {"text": "Analyze the trade-offs between microservices and monolithic "
                 "architecture. Compare scalability, maintainability, and "
                 "deployment complexity step by step."}
    ]}],
    routing=RoutingConfig(preset="quality"),
)
d = response["routing_decision"]
print(f"Quality → {d.selected_model}, cost=${d.actual_cost:.6f}")


# ── Example 3: Preset with override ─────────────────────────────────
# Use economy preset but restrict to Anthropic models only.

response = router.converse(
    messages=[{"role": "user", "content": [{"text": "Translate: Hello world"}]}],
    routing=RoutingConfig(preset="economy", preferred_family="anthropic"),
)
d = response["routing_decision"]
print(f"Economy+Anthropic → {d.selected_model}, cost=${d.actual_cost:.6f}")
