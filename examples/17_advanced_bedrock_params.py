# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Advanced Bedrock Parameters — passthrough via **kwargs.

The router is a 100% drop-in replacement for bedrock-runtime.converse()
and converse_stream().  Every Bedrock Converse API parameter is
supported — either as a first-class parameter (messages, system,
toolConfig, inferenceConfig) or via **kwargs passthrough.  Every
response field is captured in the routing decision.  Nothing is lost
by using the router instead of calling Bedrock directly.

Demonstrates:
  - additionalModelRequestFields (top_k, extended thinking)
  - additionalModelResponseFieldPaths
  - guardrailConfig (native Bedrock guardrail on the call)
  - promptVariables (Prompt Management integration)
  - outputConfig (structured JSON output)
  - performanceConfig (latency-optimized inference)
  - requestMetadata (CloudWatch invocation log filtering)
  - Inspecting all captured response fields
"""

from bedrock_smart_router import BedrockRouter, RoutingConfig

router = BedrockRouter.create()


# ── Example 1: Model-specific parameters (top_k) ────────────────────
# additionalModelRequestFields passes model-specific inference params
# that aren't part of the standard inferenceConfig.

response = router.converse(
    messages=[{"role": "user", "content": [{"text": "Write a creative story opening."}]}],
    inference_config={"maxTokens": 200, "temperature": 0.9},
    additionalModelRequestFields={"top_k": 50},
)
print(f"top_k → {response['routing_decision'].selected_model}")


# ── Example 2: Extended thinking (Anthropic) ─────────────────────────
# Enable Claude's extended thinking mode with a token budget.

response = router.converse(
    messages=[{"role": "user", "content": [
        {"text": "Solve this step by step: If a train leaves at 3pm going 60mph..."}
    ]}],
    routing=RoutingConfig(preferred_family="anthropic"),
    additionalModelRequestFields={
        "thinking": {
            "type": "enabled",
            "budget_tokens": 5000,
        }
    },
)
d = response["routing_decision"]
print(f"\nExtended thinking → {d.selected_model}, {d.output_tokens} tokens")


# ── Example 3: Request extra response fields ─────────────────────────
# additionalModelResponseFieldPaths asks Bedrock to include extra
# model-specific fields in the response.

response = router.converse(
    messages=[{"role": "user", "content": [{"text": "Hello"}]}],
    additionalModelResponseFieldPaths=["/stop_sequence"],
)
# The extra fields appear in response["additionalModelResponseFields"]
extra = response.get("additionalModelResponseFields", {})
print(f"\nExtra response fields: {extra}")


# ── Example 4: Native Bedrock guardrail on the Converse call ─────────
# This is different from our pre/post-route guardrails (ApplyGuardrail).
# This attaches a guardrail directly to the Converse API call.

response = router.converse(
    messages=[{"role": "user", "content": [{"text": "What is the weather?"}]}],
    guardrailConfig={
        "guardrailIdentifier": "your-guardrail-id",  # Replace with your guardrail ID
        "guardrailVersion": "DRAFT",
        "trace": "enabled",
    },
)
d = response["routing_decision"]
print(f"\nNative guardrail → stop_reason={d.stop_reason}")
print(f"  Guardrail trace: {d.guardrail_trace}")


# ── Example 5: Latency-optimized inference ───────────────────────────
# performanceConfig requests Bedrock's latency optimization mode.

response = router.converse(
    messages=[{"role": "user", "content": [{"text": "Quick answer: 2+2?"}]}],
    performanceConfig={"latency": "optimized"},
)
d = response["routing_decision"]
print(f"\nPerformance config → bedrock_latency={d.bedrock_latency_ms}ms")
print(f"  Response performance_config: {d.performance_config}")


# ── Example 6: Request metadata for CloudWatch log filtering ─────────
# requestMetadata is automatically forwarded from routing.metadata.
# These key-value pairs appear in CloudWatch invocation logs.

response = router.converse(
    messages=[{"role": "user", "content": [{"text": "Hello"}]}],
    routing=RoutingConfig(
        metadata={
            "tenant": "acme-corp",
            "team": "engineering",
            "environment": "production",
            "request_source": "api-gateway",
        },
    ),
)
print(f"\nMetadata forwarded to Bedrock for CloudWatch filtering")


# ── Example 7: Structured JSON output ────────────────────────────────
# outputConfig requests the model to return structured output.

response = router.converse(
    messages=[{"role": "user", "content": [{"text": "List 3 programming languages"}]}],
    outputConfig={
        "textFormat": {
            "type": "json",
            "structure": {
                "json_schema": {
                    "type": "object",
                    "properties": {
                        "languages": {
                            "type": "array",
                            "items": {"type": "string"},
                        }
                    },
                }
            },
        }
    },
)
text = response["output"]["message"]["content"][0]["text"]
print(f"\nStructured output: {text[:200]}")


# ── Example 8: Inspect all captured response metrics ─────────────────
# Every Bedrock response field is captured in the routing decision.

response = router.converse(
    messages=[{"role": "user", "content": [{"text": "Explain S3 in one sentence."}]}],
    system=[{"text": "You are a helpful AWS expert."}],
)
d = response["routing_decision"]

print(f"\n{'='*60}")
print(f"Full routing decision metrics:")
print(f"{'='*60}")
print(f"  Model:              {d.selected_model}")
print(f"  Strategy:           {d.strategy_used}")
print(f"  Complexity:         {d.complexity_detected} (score={d.complexity_score:.2f})")
print(f"  Stop reason:        {d.stop_reason}")
print(f"  ")
print(f"  Latency:")
print(f"    Wall-clock:       {d.latency_ms:.0f}ms")
print(f"    Bedrock server:   {d.bedrock_latency_ms}ms")
print(f"    Network overhead: {d.network_overhead_ms}ms")
print(f"    TTFT:             {d.ttft_ms}ms")
print(f"  ")
print(f"  Tokens:")
print(f"    Input (billed):   {d.input_tokens}")
print(f"    Output:           {d.output_tokens}")
print(f"    Total:            {d.total_tokens}")
print(f"    Cache read:       {d.prompt_cache_read_tokens}")
print(f"    Cache write:      {d.prompt_cache_write_tokens}")
print(f"    Total input:      {d.total_input_tokens}")
print(f"    Cache hit rate:   {d.prompt_cache_hit_rate:.0%}")
print(f"  ")
print(f"  Cost:")
print(f"    Estimated:        ${d.estimated_cost:.6f}")
print(f"    Actual:           ${d.actual_cost:.6f}")
print(f"    Cache savings:    ${d.prompt_cache_savings:.6f}")
print(f"  ")
print(f"  Infrastructure:")
print(f"    Inference tier:   {d.inference_tier}")
print(f"    Actual tier:      {d.actual_service_tier}")
print(f"    CRIS profile:     {d.cris_profile}")
print(f"    Fallback used:    {d.fallback_used}")
print(f"    Guardrail checked:{d.guardrail_checked}")
print(f"  ")
print(f"  Cache details:      {d.cache_details}")
print(f"  Performance config: {d.performance_config}")
print(f"  Guardrail trace:    {len(d.guardrail_trace)} keys")
