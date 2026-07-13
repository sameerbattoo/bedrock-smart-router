# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Strands Agents SDK integration (SmartRouterModel)."""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from bedrock_smart_router.config import RouterConfig, RoutingConfig
from bedrock_smart_router.models import RoutingDecision


# ── Helpers ──────────────────────────────────────────────────────────

def _run(coro):
    """Run an async coroutine synchronously."""
    return asyncio.run(coro)


# ── Helpers ──────────────────────────────────────────────────────────

def _make_routing_decision(**overrides: Any) -> RoutingDecision:
    defaults = dict(
        selected_model="anthropic.claude-sonnet-4-20250514-v1:0",
        strategy_used="balanced",
        complexity_detected="moderate",
        complexity_score=0.5,
        candidates_evaluated=3,
        actual_cost=0.002,
        latency_ms=450.0,
        input_tokens=100,
        output_tokens=200,
        stop_reason="end_turn",
    )
    defaults.update(overrides)
    return RoutingDecision(**defaults)


def _make_stream_events(text: str = "Hello!", stop_reason: str = "end_turn") -> list[dict]:
    """Build a realistic Bedrock converse_stream event sequence."""
    return [
        {"messageStart": {"role": "assistant"}},
        {"contentBlockStart": {"start": {}}},
        {"contentBlockDelta": {"delta": {"text": text}}},
        {"contentBlockStop": {}},
        {"messageStop": {"stopReason": stop_reason}},
        {
            "metadata": {
                "usage": {"inputTokens": 100, "outputTokens": 50, "totalTokens": 150},
                "metrics": {"latencyMs": 300},
            }
        },
    ]


def _make_converse_response(text: str = "Hello!", stop_reason: str = "end_turn") -> dict:
    """Build a realistic Bedrock converse (non-streaming) response."""
    return {
        "output": {
            "message": {
                "role": "assistant",
                "content": [{"text": text}],
            }
        },
        "stopReason": stop_reason,
        "usage": {"inputTokens": 100, "outputTokens": 50, "totalTokens": 150},
        "metrics": {"latencyMs": 300},
    }


def _make_tool_use_response() -> dict:
    """Build a converse response with a tool use block."""
    return {
        "output": {
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "toolUse": {
                            "toolUseId": "tool_123",
                            "name": "get_weather",
                            "input": {"city": "Seattle"},
                        }
                    }
                ],
            }
        },
        "stopReason": "tool_use",
        "usage": {"inputTokens": 80, "outputTokens": 40, "totalTokens": 120},
        "metrics": {"latencyMs": 200},
    }


# ── Fixtures ─────────────────────────────────────────────────────────

@pytest.fixture
def mock_router():
    """A mocked BedrockRouter."""
    router = MagicMock()
    router.converse_stream.return_value = iter(
        _make_stream_events() + [{"routing_decision": _make_routing_decision()}]
    )
    router.converse.return_value = {
        **_make_converse_response(),
        "routing_decision": _make_routing_decision(),
    }
    return router


@pytest.fixture
def model(mock_router):
    """SmartRouterModel backed by a mock router."""
    from bedrock_smart_router.strands_model import SmartRouterModel
    return SmartRouterModel(router=mock_router)


# ── Tests ────────────────────────────────────────────────────────────

class TestSmartRouterModelConfig:
    """Configuration and lifecycle tests."""

    def test_default_config(self, model):
        cfg = model.get_config()
        assert cfg["streaming"] is True

    def test_update_config(self, model):
        model.update_config(routing_preset="economy", max_cost_per_request=0.001)
        cfg = model.get_config()
        assert cfg["routing_preset"] == "economy"
        assert cfg["max_cost_per_request"] == 0.001

    def test_init_with_config(self, mock_router):
        from bedrock_smart_router.strands_model import SmartRouterModel
        m = SmartRouterModel(
            router=mock_router,
            routing_preset="quality",
            preferred_family="anthropic",
            streaming=False,
        )
        cfg = m.get_config()
        assert cfg["routing_preset"] == "quality"
        assert cfg["preferred_family"] == "anthropic"
        assert cfg["streaming"] is False

    def test_router_property(self, model, mock_router):
        assert model.router is mock_router

    def test_last_decision_initially_none(self, model):
        assert model.last_routing_decision is None


class TestStreaming:
    """Streaming mode tests."""

    def test_stream_yields_bedrock_events(self, model, mock_router):
        """stream() should yield all Bedrock events except routing_decision."""
        messages = [{"role": "user", "content": [{"text": "Hello"}]}]

        events = _run(
            _collect_stream(model, messages)
        )

        event_types = [list(e.keys())[0] for e in events]
        assert "messageStart" in event_types
        assert "contentBlockDelta" in event_types
        assert "messageStop" in event_types
        assert "metadata" in event_types
        # routing_decision should NOT be in the yielded events.
        assert "routing_decision" not in event_types

    def test_stream_captures_routing_decision(self, model, mock_router):
        """The routing decision should be stored on the model."""
        messages = [{"role": "user", "content": [{"text": "Hello"}]}]

        _run(
            _collect_stream(model, messages)
        )

        assert model.last_routing_decision is not None
        assert model.last_routing_decision.selected_model == "anthropic.claude-sonnet-4-20250514-v1:0"
        assert model.last_routing_decision.strategy_used == "balanced"

    def test_stream_passes_system_prompt(self, model, mock_router):
        """System prompt should be converted to Bedrock format."""
        messages = [{"role": "user", "content": [{"text": "Hi"}]}]

        _run(
            _collect_stream(model, messages, system_prompt="You are helpful.")
        )

        call_kwargs = mock_router.converse_stream.call_args
        assert call_kwargs.kwargs["system"] == [{"text": "You are helpful."}]

    def test_stream_passes_tool_specs(self, model, mock_router):
        """Tool specs should be wrapped in Bedrock toolConfig format."""
        tool_specs = [
            {
                "name": "get_weather",
                "description": "Get weather",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                    }
                },
            }
        ]
        messages = [{"role": "user", "content": [{"text": "Weather?"}]}]

        _run(
            _collect_stream(model, messages, tool_specs=tool_specs)
        )

        call_kwargs = mock_router.converse_stream.call_args
        tool_config = call_kwargs.kwargs["tool_config"]
        assert len(tool_config["tools"]) == 1
        assert tool_config["tools"][0]["toolSpec"]["name"] == "get_weather"

    def test_stream_passes_tool_choice(self, model, mock_router):
        """tool_choice kwarg should be included in toolConfig."""
        tool_specs = [{"name": "calc", "description": "Calculate", "inputSchema": {"json": {}}}]
        messages = [{"role": "user", "content": [{"text": "Calc"}]}]

        _run(
            _collect_stream(
                model, messages,
                tool_specs=tool_specs,
                tool_choice={"auto": {}},
            )
        )

        call_kwargs = mock_router.converse_stream.call_args
        assert call_kwargs.kwargs["tool_config"]["toolChoice"] == {"auto": {}}

    def test_stream_no_system_prompt(self, model, mock_router):
        """When no system prompt is given, system should be None."""
        messages = [{"role": "user", "content": [{"text": "Hi"}]}]

        _run(
            _collect_stream(model, messages)
        )

        call_kwargs = mock_router.converse_stream.call_args
        assert call_kwargs.kwargs["system"] is None

    def test_stream_no_tools(self, model, mock_router):
        """When no tools are given, tool_config should be None."""
        messages = [{"role": "user", "content": [{"text": "Hi"}]}]

        _run(
            _collect_stream(model, messages)
        )

        call_kwargs = mock_router.converse_stream.call_args
        assert call_kwargs.kwargs["tool_config"] is None


class TestNonStreaming:
    """Non-streaming mode tests."""

    def test_non_streaming_converts_to_events(self, mock_router):
        """Non-streaming response should be converted to stream events."""
        from bedrock_smart_router.strands_model import SmartRouterModel
        model = SmartRouterModel(router=mock_router, streaming=False)
        messages = [{"role": "user", "content": [{"text": "Hello"}]}]

        events = _run(
            _collect_stream(model, messages)
        )

        event_types = [list(e.keys())[0] for e in events]
        assert "messageStart" in event_types
        assert "contentBlockDelta" in event_types
        assert "contentBlockStop" in event_types
        assert "messageStop" in event_types
        assert "metadata" in event_types

    def test_non_streaming_text_content(self, mock_router):
        """Text content should appear in contentBlockDelta."""
        from bedrock_smart_router.strands_model import SmartRouterModel
        model = SmartRouterModel(router=mock_router, streaming=False)
        messages = [{"role": "user", "content": [{"text": "Hello"}]}]

        events = _run(
            _collect_stream(model, messages)
        )

        text_deltas = [
            e["contentBlockDelta"]["delta"]["text"]
            for e in events if "contentBlockDelta" in e and "text" in e["contentBlockDelta"]["delta"]
        ]
        assert text_deltas == ["Hello!"]

    def test_non_streaming_tool_use(self, mock_router):
        """Tool use blocks should be converted correctly."""
        from bedrock_smart_router.strands_model import SmartRouterModel

        mock_router.converse.return_value = {
            **_make_tool_use_response(),
            "routing_decision": _make_routing_decision(stop_reason="tool_use"),
        }
        model = SmartRouterModel(router=mock_router, streaming=False)
        messages = [{"role": "user", "content": [{"text": "Weather?"}]}]

        events = _run(
            _collect_stream(model, messages)
        )

        # Should have contentBlockStart with toolUse
        starts = [e for e in events if "contentBlockStart" in e]
        assert len(starts) == 1
        tool_start = starts[0]["contentBlockStart"]["start"]["toolUse"]
        assert tool_start["name"] == "get_weather"
        assert tool_start["toolUseId"] == "tool_123"

        # Should have contentBlockDelta with toolUse input
        tool_deltas = [
            e for e in events
            if "contentBlockDelta" in e and "toolUse" in e["contentBlockDelta"]["delta"]
        ]
        assert len(tool_deltas) == 1
        tool_input = json.loads(tool_deltas[0]["contentBlockDelta"]["delta"]["toolUse"]["input"])
        assert tool_input == {"city": "Seattle"}

        # Stop reason should be tool_use
        stops = [e for e in events if "messageStop" in e]
        assert stops[0]["messageStop"]["stopReason"] == "tool_use"

    def test_non_streaming_captures_decision(self, mock_router):
        """Routing decision should be captured in non-streaming mode."""
        from bedrock_smart_router.strands_model import SmartRouterModel
        model = SmartRouterModel(router=mock_router, streaming=False)
        messages = [{"role": "user", "content": [{"text": "Hello"}]}]

        _run(
            _collect_stream(model, messages)
        )

        assert model.last_routing_decision is not None
        assert model.last_routing_decision.actual_cost == 0.002


class TestRoutingConfig:
    """Routing configuration forwarding tests."""

    def test_preset_forwarded(self, mock_router):
        from bedrock_smart_router.strands_model import SmartRouterModel
        model = SmartRouterModel(router=mock_router, routing_preset="economy")
        messages = [{"role": "user", "content": [{"text": "Hi"}]}]

        _run(
            _collect_stream(model, messages)
        )

        call_kwargs = mock_router.converse_stream.call_args
        routing: RoutingConfig = call_kwargs.kwargs["routing"]
        assert routing.preset == "economy"

    def test_preferred_model_forwarded(self, mock_router):
        from bedrock_smart_router.strands_model import SmartRouterModel
        model = SmartRouterModel(
            router=mock_router,
            preferred_model="anthropic.claude-sonnet-4-20250514-v1:0",
        )
        messages = [{"role": "user", "content": [{"text": "Hi"}]}]

        _run(
            _collect_stream(model, messages)
        )

        call_kwargs = mock_router.converse_stream.call_args
        routing: RoutingConfig = call_kwargs.kwargs["routing"]
        assert routing.preferred_model == "anthropic.claude-sonnet-4-20250514-v1:0"

    def test_metadata_and_tags_forwarded(self, mock_router):
        from bedrock_smart_router.strands_model import SmartRouterModel
        model = SmartRouterModel(
            router=mock_router,
            tags=["paid-tier"],
            metadata={"tenant": "acme"},
        )
        messages = [{"role": "user", "content": [{"text": "Hi"}]}]

        _run(
            _collect_stream(model, messages)
        )

        call_kwargs = mock_router.converse_stream.call_args
        routing: RoutingConfig = call_kwargs.kwargs["routing"]
        assert routing.tags == ["paid-tier"]
        assert routing.metadata == {"tenant": "acme"}

    def test_update_config_changes_routing(self, model, mock_router):
        """update_config should change routing on subsequent calls."""
        messages = [{"role": "user", "content": [{"text": "Hi"}]}]

        # First call — default
        _run(
            _collect_stream(model, messages)
        )
        call1 = mock_router.converse_stream.call_args
        assert call1.kwargs["routing"].preset is None

        # Update to economy
        model.update_config(routing_preset="economy")
        _run(
            _collect_stream(model, messages)
        )
        call2 = mock_router.converse_stream.call_args
        assert call2.kwargs["routing"].preset == "economy"


class TestInferenceConfig:
    """Inference config forwarding tests."""

    def test_max_tokens_forwarded(self, mock_router):
        from bedrock_smart_router.strands_model import SmartRouterModel
        model = SmartRouterModel(router=mock_router, max_tokens=1024)
        messages = [{"role": "user", "content": [{"text": "Hi"}]}]

        _run(
            _collect_stream(model, messages)
        )

        call_kwargs = mock_router.converse_stream.call_args
        assert call_kwargs.kwargs["inference_config"]["maxTokens"] == 1024

    def test_temperature_forwarded(self, mock_router):
        from bedrock_smart_router.strands_model import SmartRouterModel
        model = SmartRouterModel(router=mock_router, temperature=0.7)
        messages = [{"role": "user", "content": [{"text": "Hi"}]}]

        _run(
            _collect_stream(model, messages)
        )

        call_kwargs = mock_router.converse_stream.call_args
        assert call_kwargs.kwargs["inference_config"]["temperature"] == 0.7

    def test_no_inference_config_when_empty(self, model, mock_router):
        messages = [{"role": "user", "content": [{"text": "Hi"}]}]

        _run(
            _collect_stream(model, messages)
        )

        call_kwargs = mock_router.converse_stream.call_args
        assert call_kwargs.kwargs["inference_config"] is None


class TestErrorHandling:
    """Error mapping tests."""

    def test_throttling_raises_strands_exception(self, mock_router):
        from bedrock_smart_router.strands_model import SmartRouterModel
        from botocore.exceptions import ClientError

        error_response = {"Error": {"Code": "ThrottlingException", "Message": "Rate exceeded"}}
        mock_router.converse_stream.side_effect = ClientError(error_response, "ConverseStream")

        model = SmartRouterModel(router=mock_router)
        messages = [{"role": "user", "content": [{"text": "Hi"}]}]

        with pytest.raises(Exception, match="Rate exceeded"):
            _run(
                _collect_stream(model, messages)
            )

    def test_context_overflow_raises_strands_exception(self, mock_router):
        from bedrock_smart_router.strands_model import SmartRouterModel
        from botocore.exceptions import ClientError

        error_response = {
            "Error": {
                "Code": "ValidationException",
                "Message": "Input is too long for requested model",
            }
        }
        mock_router.converse_stream.side_effect = ClientError(error_response, "ConverseStream")

        model = SmartRouterModel(router=mock_router)
        messages = [{"role": "user", "content": [{"text": "Hi"}]}]

        with pytest.raises(Exception, match="too long"):
            _run(
                _collect_stream(model, messages)
            )

    def test_generic_error_propagates(self, mock_router):
        from bedrock_smart_router.strands_model import SmartRouterModel

        mock_router.converse_stream.side_effect = RuntimeError("All models failed")
        model = SmartRouterModel(router=mock_router)
        messages = [{"role": "user", "content": [{"text": "Hi"}]}]

        with pytest.raises(RuntimeError, match="All models failed"):
            _run(
                _collect_stream(model, messages)
            )


class TestConvertResponseToStream:
    """Unit tests for the non-streaming → streaming conversion."""

    def test_text_response(self):
        from bedrock_smart_router.strands_model import SmartRouterModel

        response = _make_converse_response("Test output", "end_turn")
        events = SmartRouterModel._convert_response_to_stream(response)

        assert events[0] == {"messageStart": {"role": "assistant"}}
        assert events[1]["contentBlockDelta"]["delta"]["text"] == "Test output"
        assert events[2] == {"contentBlockStop": {}}
        assert events[3]["messageStop"]["stopReason"] == "end_turn"
        assert events[4]["metadata"]["usage"]["inputTokens"] == 100

    def test_tool_use_response(self):
        from bedrock_smart_router.strands_model import SmartRouterModel

        response = _make_tool_use_response()
        events = SmartRouterModel._convert_response_to_stream(response)

        # messageStart
        assert events[0] == {"messageStart": {"role": "assistant"}}
        # contentBlockStart with toolUse
        assert events[1]["contentBlockStart"]["start"]["toolUse"]["name"] == "get_weather"
        # contentBlockDelta with toolUse input
        tool_input = json.loads(events[2]["contentBlockDelta"]["delta"]["toolUse"]["input"])
        assert tool_input == {"city": "Seattle"}
        # contentBlockStop
        assert events[3] == {"contentBlockStop": {}}
        # messageStop
        assert events[4]["messageStop"]["stopReason"] == "tool_use"

    def test_empty_content(self):
        from bedrock_smart_router.strands_model import SmartRouterModel

        response = {
            "output": {"message": {"role": "assistant", "content": []}},
            "stopReason": "end_turn",
        }
        events = SmartRouterModel._convert_response_to_stream(response)
        assert events[0] == {"messageStart": {"role": "assistant"}}
        assert events[1]["messageStop"]["stopReason"] == "end_turn"


# ── Async helper ─────────────────────────────────────────────────────

async def _collect_stream(model, messages, **kwargs) -> list[dict]:
    """Collect all events from model.stream() into a list."""
    events = []
    async for event in model.stream(messages, **kwargs):
        events.append(event)
    return events
