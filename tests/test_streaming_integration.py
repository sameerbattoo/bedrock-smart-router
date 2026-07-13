# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration test — streaming against real Bedrock.

Run with:
    INTEGRATION_TEST=1 .venv/bin/python -m pytest tests/test_streaming_integration.py -v -s
"""

from __future__ import annotations

import os

import boto3
import pytest

from bedrock_smart_router.config import RouterConfig, RoutingConfig
from bedrock_smart_router.router import BedrockRouter

SKIP_REASON = "Set INTEGRATION_TEST=1 to run against real AWS"
REGION = "us-west-2"


def _msgs(text: str) -> list[dict]:
    return [{"role": "user", "content": [{"text": text}]}]


@pytest.fixture
def router():
    session = boto3.Session(region_name=REGION)
    cfg = RouterConfig.from_dict({
        "region": REGION,
        "strategy": "cost-optimized",
    })
    return BedrockRouter(cfg, boto_session=session)


@pytest.fixture
def quality_router():
    session = boto3.Session(region_name=REGION)
    cfg = RouterConfig.from_dict({
        "region": REGION,
        "strategy": "balanced",
        "weights": {"cost": 0.3, "latency": 0.3, "quality": 0.4},
    })
    return BedrockRouter(cfg, boto_session=session)


@pytest.mark.skipif(
    os.environ.get("INTEGRATION_TEST") != "1",
    reason=SKIP_REASON,
)
class TestStreamingIntegration:

    def test_basic_stream(self, router):
        """Stream should yield text tokens and a routing decision."""
        texts = []
        decision = None
        for event in router.converse_stream(messages=_msgs("Say hello in one word.")):
            if "contentBlockDelta" in event:
                delta = event["contentBlockDelta"]["delta"]
                if "text" in delta:
                    texts.append(delta["text"])
            elif "routing_decision" in event:
                decision = event["routing_decision"]

        full_text = "".join(texts)
        assert len(full_text) > 0
        assert decision is not None
        print(f"\n  Response: {full_text[:100]}")
        print(f"  Model: {decision.selected_model}")
        print(f"  TTFT: {decision.ttft_ms:.0f}ms")
        print(f"  Total latency: {decision.latency_ms:.0f}ms")
        print(f"  Tokens: {decision.input_tokens} in, {decision.output_tokens} out")

    def test_ttft_less_than_total(self, router):
        """TTFT should be significantly less than total latency."""
        decision = None
        for event in router.converse_stream(
            messages=_msgs("Write a short paragraph about clouds."),
        ):
            if "routing_decision" in event:
                decision = event["routing_decision"]

        assert decision.ttft_ms is not None
        assert decision.ttft_ms > 0
        assert decision.ttft_ms <= decision.latency_ms
        print(f"\n  TTFT: {decision.ttft_ms:.0f}ms vs Total: {decision.latency_ms:.0f}ms")
        print(f"  TTFT is {decision.ttft_ms / decision.latency_ms * 100:.0f}% of total")

    def test_stream_with_anthropic(self, quality_router):
        """Stream with Anthropic model via preferred_family."""
        texts = []
        decision = None
        for event in quality_router.converse_stream(
            messages=_msgs("What is EC2? One sentence."),
            routing=RoutingConfig(preferred_family="anthropic"),
        ):
            if "contentBlockDelta" in event:
                delta = event["contentBlockDelta"]["delta"]
                if "text" in delta:
                    texts.append(delta["text"])
            elif "routing_decision" in event:
                decision = event["routing_decision"]

        assert "anthropic" in decision.selected_model
        print(f"\n  Anthropic stream: {''.join(texts)[:150]}")
        print(f"  Model: {decision.selected_model}")
        print(f"  TTFT: {decision.ttft_ms:.0f}ms, Cost: ${decision.actual_cost:.6f}")

    def test_stream_with_system_prompt(self, router):
        """System prompt should work with streaming."""
        texts = []
        for event in router.converse_stream(
            messages=_msgs("What are you?"),
            system=[{"text": "You are a pirate. Respond in one sentence."}],
        ):
            if "contentBlockDelta" in event:
                delta = event["contentBlockDelta"]["delta"]
                if "text" in delta:
                    texts.append(delta["text"])

        response = "".join(texts)
        print(f"\n  Pirate stream: {response[:150]}")

    def test_stream_metrics_recorded(self, router):
        """Metrics should be recorded after stream completes."""
        decision = None
        for event in router.converse_stream(messages=_msgs("Test metrics")):
            if "routing_decision" in event:
                decision = event["routing_decision"]

        m = router.metrics.get_metrics(decision.selected_model, window_seconds=60)
        assert m.sample_count >= 1
        assert m.avg_ttft_ms >= 0
        print(f"\n  Metrics: latency={m.avg_latency_ms:.0f}ms, ttft={m.avg_ttft_ms:.0f}ms")

    def test_stream_cost_comparison(self, router, quality_router):
        """Compare streaming cost between economy and quality routing."""
        # Economy
        d_econ = None
        for event in router.converse_stream(
            messages=_msgs("What is cloud computing?"),
            routing=RoutingConfig(preset="economy"),
        ):
            if "routing_decision" in event:
                d_econ = event["routing_decision"]

        # Quality with Anthropic
        d_qual = None
        for event in quality_router.converse_stream(
            messages=_msgs("What is cloud computing?"),
            routing=RoutingConfig(preferred_family="anthropic"),
        ):
            if "routing_decision" in event:
                d_qual = event["routing_decision"]

        print(f"\n  Economy:  {d_econ.selected_model}, ${d_econ.actual_cost:.6f}, TTFT={d_econ.ttft_ms:.0f}ms")
        print(f"  Quality:  {d_qual.selected_model}, ${d_qual.actual_cost:.6f}, TTFT={d_qual.ttft_ms:.0f}ms")
