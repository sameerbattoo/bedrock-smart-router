# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Streaming — token-by-token responses via converse_stream().

Demonstrates:
  - Basic streaming with routing
  - Streaming with presets
  - Accessing the routing decision after stream completes
  - Async streaming
"""

from bedrock_smart_router import BedrockRouter, RoutingConfig

router = BedrockRouter.create()


# ── Example 1: Basic streaming ───────────────────────────────────────
# Tokens arrive one by one. The routing decision comes as the final event.

print("Streaming response:")
for event in router.converse_stream(
    messages=[{"role": "user", "content": [{"text": "Write a haiku about clouds."}]}],
):
    if "contentBlockDelta" in event:
        delta = event["contentBlockDelta"]["delta"]
        if "text" in delta:
            print(delta["text"], end="", flush=True)
    elif "routing_decision" in event:
        d = event["routing_decision"]
        print(f"\n\n[Model: {d.selected_model}, Cost: ${d.actual_cost:.6f}, "
              f"TTFT: {d.ttft_ms:.0f}ms, Total: {d.latency_ms:.0f}ms]")


# ── Example 2: Streaming with a preset ───────────────────────────────

print("\n\nEconomy streaming:")
for event in router.converse_stream(
    messages=[{"role": "user", "content": [{"text": "What is Lambda?"}]}],
    routing=RoutingConfig(preset="economy"),
):
    if "contentBlockDelta" in event:
        delta = event["contentBlockDelta"]["delta"]
        if "text" in delta:
            print(delta["text"], end="", flush=True)
    elif "routing_decision" in event:
        d = event["routing_decision"]
        print(f"\n[{d.selected_model}, {d.complexity_detected}]")


# ── Example 3: Collect full response + routing decision ──────────────

full_text = []
decision = None

for event in router.converse_stream(
    messages=[{"role": "user", "content": [{"text": "Explain VPCs in one sentence."}]}],
    routing=RoutingConfig(preset="speed"),
):
    if "contentBlockDelta" in event:
        delta = event["contentBlockDelta"]["delta"]
        if "text" in delta:
            full_text.append(delta["text"])
    elif "routing_decision" in event:
        decision = event["routing_decision"]

print(f"\n\nFull response: {''.join(full_text)}")
print(f"Model: {decision.selected_model}")
print(f"Latency: {decision.latency_ms:.0f}ms (TTFT: {decision.ttft_ms:.0f}ms)")
print(f"Tokens: {decision.input_tokens} in, {decision.output_tokens} out")
print(f"Stop reason: {decision.stop_reason}")
print(f"Bedrock latency: {decision.bedrock_latency_ms}ms")
print(f"Service tier: {decision.actual_service_tier}")
print(f"Prompt cache: {decision.prompt_cache_read_tokens} read, {decision.prompt_cache_write_tokens} write")
print(f"Prompt cache hit rate: {decision.prompt_cache_hit_rate:.0%}")
print(f"Network overhead: {decision.network_overhead_ms}ms")


# ── Example 4: Async streaming (for FastAPI, aiohttp, etc.) ──────────

"""
import asyncio
from bedrock_smart_router.async_router import AsyncBedrockRouter

async def stream_example():
    router = AsyncBedrockRouter.create()
    async for event in router.converse_stream(
        messages=[{"role": "user", "content": [{"text": "Hello!"}]}],
    ):
        if "contentBlockDelta" in event:
            print(event["contentBlockDelta"]["delta"].get("text", ""), end="")
        elif "routing_decision" in event:
            print(f"\\n[{event['routing_decision'].selected_model}]")

asyncio.run(stream_example())
"""
