"""Bedrock-Native Features — CRIS, inference tiers, prompt caching, guardrails.

Demonstrates:
  - CRIS profile selection by geography
  - Inference tier auto-selection (Standard/Priority/Flex)
  - Prompt cache-aware routing
  - Pre-route and post-route guardrails
"""

from bedrock_smart_router import BedrockRouter, RoutingConfig

# ── Example 1: CRIS — prefer US regions ──────────────────────────────

router = BedrockRouter.create({
    "cris": {"preferred_geography": "us", "allow_global": True},
})

response = router.converse(
    messages=[{"role": "user", "content": [{"text": "Hello"}]}],
)
d = response["routing_decision"]
print(f"CRIS profile: {d.cris_profile}")
# e.g. "us.anthropic.claude-sonnet-4-6" (US region profile)


# ── Example 2: Inference tier — auto-selects Priority for complex ────

router = BedrockRouter.create({
    "inference_tier": {
        "allow_priority": True,
        "allow_flex": True,
        "priority_for_complex": True,
        "flex_for_batch": True,
    },
})

# Simple request → Standard tier
r1 = router.converse(messages=[{"role": "user", "content": [{"text": "Hi"}]}])
print(f"\nSimple → tier={r1['routing_decision'].inference_tier}")
print(f"  Actual tier served: {r1['routing_decision'].actual_service_tier}")
print(f"  Stop reason: {r1['routing_decision'].stop_reason}")
print(f"  Bedrock latency: {r1['routing_decision'].bedrock_latency_ms}ms")

# Complex request → Priority tier (if model supports it)
r2 = router.converse(messages=[{"role": "user", "content": [
    {"text": "Analyze the trade-offs between eventual consistency and strong "
             "consistency in distributed databases. Compare step by step."}
]}])
print(f"Complex → tier={r2['routing_decision'].inference_tier}")


# ── Example 3: Guardrails — block investment advice ──────────────────

router = BedrockRouter.create({
    "guardrails": {
        "pre_route": {
            "guardrail_id": "your-guardrail-id",  # Replace with your guardrail ID
            "guardrail_version": "DRAFT",
            "action_on_block": "reject",
        },
    },
})

from bedrock_smart_router.guardrails_integration import GuardrailBlockedError

try:
    router.converse(messages=[{"role": "user", "content": [
        {"text": "What stocks should I invest in?"}
    ]}])
except GuardrailBlockedError as e:
    print(f"\nGuardrail blocked: {e}")

# Safe request passes through
response = router.converse(messages=[{"role": "user", "content": [
    {"text": "What is the weather like?"}
]}])
print(f"Safe request → {response['routing_decision'].selected_model}")
