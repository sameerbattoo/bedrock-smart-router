"""Tests for converse_stream() — streaming responses with TTFT tracking."""

from unittest.mock import MagicMock, patch
import time

import boto3
import pytest

from bedrock_smart_router.config import RouterConfig, RoutingConfig
from bedrock_smart_router.router import BedrockRouter


def _msgs(text: str) -> list[dict]:
    return [{"role": "user", "content": [{"text": text}]}]


def _mock_stream_events():
    """Simulate a Bedrock ConverseStream response."""
    return [
        {"contentBlockStart": {"contentBlockIndex": 0, "start": {"text": ""}}},
        {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "Hello"}}},
        {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": " world"}}},
        {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "!"}}},
        {"contentBlockStop": {"contentBlockIndex": 0}},
        {"metadata": {
            "usage": {"inputTokens": 10, "outputTokens": 5},
            "metrics": {"latencyMs": 200},
        }},
        {"messageStop": {"stopReason": "end_turn"}},
    ]


@pytest.fixture
def mock_router():
    """Create a router with a mocked Bedrock client for streaming."""
    session = MagicMock()
    mock_client = MagicMock()

    # Mock converse_stream to return an iterable stream
    mock_client.converse_stream.return_value = {
        "stream": iter(_mock_stream_events()),
    }
    # Mock converse for fallback (shouldn't be called in happy path)
    mock_client.converse.return_value = {
        "output": {"message": {"content": [{"text": "fallback"}]}},
        "usage": {"inputTokens": 5, "outputTokens": 3},
    }

    session.client.return_value = mock_client
    cfg = RouterConfig.from_dict({"strategy": "cost-optimized"})
    router = BedrockRouter(cfg, boto_session=session)
    return router, mock_client


class TestConverseStream:
    def test_yields_content_events(self, mock_router):
        """Stream should yield content delta events."""
        router, _ = mock_router
        texts = []
        for event in router.converse_stream(messages=_msgs("Hello")):
            if "contentBlockDelta" in event:
                texts.append(event["contentBlockDelta"]["delta"]["text"])
        assert texts == ["Hello", " world", "!"]

    def test_yields_routing_decision_last(self, mock_router):
        """The final event should contain the routing decision."""
        router, _ = mock_router
        events = list(router.converse_stream(messages=_msgs("Hello")))
        last = events[-1]
        assert "routing_decision" in last
        d = last["routing_decision"]
        assert d.selected_model is not None
        assert d.strategy_used == "cost-optimized"

    def test_ttft_is_captured(self, mock_router):
        """TTFT should be measured from stream start to first content delta."""
        router, _ = mock_router
        decision = None
        for event in router.converse_stream(messages=_msgs("Hello")):
            if "routing_decision" in event:
                decision = event["routing_decision"]
        assert decision is not None
        assert decision.ttft_ms is not None
        assert decision.ttft_ms >= 0
        # TTFT should be less than total latency
        assert decision.ttft_ms <= decision.latency_ms

    def test_latency_is_total_time(self, mock_router):
        """latency_ms should be the total time from start to stream end."""
        router, _ = mock_router
        decision = None
        for event in router.converse_stream(messages=_msgs("Hello")):
            if "routing_decision" in event:
                decision = event["routing_decision"]
        assert decision.latency_ms > 0

    def test_tokens_from_metadata(self, mock_router):
        """Input/output tokens should come from the stream metadata event."""
        router, _ = mock_router
        decision = None
        for event in router.converse_stream(messages=_msgs("Hello")):
            if "routing_decision" in event:
                decision = event["routing_decision"]
        assert decision.input_tokens == 10
        assert decision.output_tokens == 5

    def test_cost_calculated(self, mock_router):
        """Cost should be calculated from actual token counts."""
        router, _ = mock_router
        decision = None
        for event in router.converse_stream(messages=_msgs("Hello")):
            if "routing_decision" in event:
                decision = event["routing_decision"]
        assert decision.actual_cost > 0

    def test_metrics_recorded(self, mock_router):
        """Metrics store should have data after streaming."""
        router, _ = mock_router
        decision = None
        for event in router.converse_stream(messages=_msgs("Hello")):
            if "routing_decision" in event:
                decision = event["routing_decision"]
        m = router.metrics.get_metrics(decision.selected_model, window_seconds=60)
        assert m.sample_count >= 1
        assert m.avg_ttft_ms >= 0

    def test_last_routing_decision_updated(self, mock_router):
        """last_routing_decision() should reflect the stream result."""
        router, _ = mock_router
        for _ in router.converse_stream(messages=_msgs("Hello")):
            pass
        d = router.last_routing_decision()
        assert d is not None
        assert d.ttft_ms is not None

    def test_preset_works_with_stream(self, mock_router):
        """Presets should work with converse_stream."""
        router, _ = mock_router
        decision = None
        for event in router.converse_stream(
            messages=_msgs("Hello"),
            routing=RoutingConfig(preset="economy"),
        ):
            if "routing_decision" in event:
                decision = event["routing_decision"]
        assert decision.strategy_used == "cost-optimized"

    def test_complexity_detected(self, mock_router):
        """Complexity should be detected for streaming requests."""
        router, _ = mock_router
        decision = None
        for event in router.converse_stream(messages=_msgs("Hi")):
            if "routing_decision" in event:
                decision = event["routing_decision"]
        assert decision.complexity_detected in ("simple", "moderate", "complex", "reasoning")

    def test_cris_and_tier_populated(self, mock_router):
        """CRIS profile and inference tier should be in the decision."""
        router, _ = mock_router
        decision = None
        for event in router.converse_stream(messages=_msgs("Hello")):
            if "routing_decision" in event:
                decision = event["routing_decision"]
        assert decision.cris_profile is not None
        assert decision.inference_tier in ("standard", "priority", "flex")
