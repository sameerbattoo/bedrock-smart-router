# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Async Usage — AsyncBedrockRouter for async/await applications.

Demonstrates:
  - Creating an async router
  - Using it in an async function
  - Parallel routing with asyncio.gather
"""

import asyncio
from bedrock_smart_router.async_router import AsyncBedrockRouter
from bedrock_smart_router import RoutingConfig


async def main():
    # ── Example 1: Basic async usage ─────────────────────────────
    router = AsyncBedrockRouter.create({"strategy": "balanced"})

    response = await router.converse(
        messages=[{"role": "user", "content": [{"text": "What is Lambda?"}]}],
    )
    d = response["routing_decision"]
    print(f"Async → {d.selected_model}, ${d.actual_cost:.6f}")

    # ── Example 2: Parallel requests ─────────────────────────────
    # Route 3 requests concurrently.

    prompts = [
        "What is S3?",
        "Explain VPCs briefly.",
        "Write a hello world in Python.",
    ]

    tasks = [
        router.converse(
            messages=[{"role": "user", "content": [{"text": p}]}],
        )
        for p in prompts
    ]

    results = await asyncio.gather(*tasks)
    for prompt, result in zip(prompts, results):
        d = result["routing_decision"]
        print(f"  '{prompt[:30]}' → {d.selected_model}")

    # ── Example 3: Async with presets ────────────────────────────

    response = await router.converse(
        messages=[{"role": "user", "content": [{"text": "Quick question"}]}],
        routing=RoutingConfig(preset="speed"),
    )
    print(f"\nAsync+preset → {response['routing_decision'].selected_model}")


if __name__ == "__main__":
    asyncio.run(main())
