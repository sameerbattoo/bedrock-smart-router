# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for **kwargs passthrough to Bedrock Converse API.

Verifies that additional Bedrock parameters (additionalModelRequestFields,
additionalModelResponseFieldPaths, guardrailConfig, promptVariables,
outputConfig, performanceConfig) are passed through to the Bedrock
client unchanged.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from bedrock_smart_router.config import RouterConfig
from bedrock_smart_router.router import BedrockRouter


def _msgs(text: str) -> list[dict]:
    return [{"role": "user", "content": [{"text": text}]}]


@pytest.fixture
def mock_router():
    """Router with a mocked Bedrock client to inspect call args."""
    session = MagicMock()
    mock_client = MagicMock()
    mock_client.converse.return_value = {
        "output": {"message": {"content": [{"text": "ok"}]}},
        "usage": {"inputTokens": 5, "outputTokens": 3},
        "stopReason": "end_turn",
        "metrics": {"latencyMs": 100},
    }
    mock_client.converse_stream.return_value = {
        "stream": iter([
            {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "ok"}}},
            {"messageStop": {"stopReason": "end_turn"}},
            {"metadata": {"usage": {"inputTokens": 5, "outputTokens": 3}, "metrics": {"latencyMs": 100}}},
        ]),
    }
    session.client.return_value = mock_client
    cfg = RouterConfig.from_dict({"strategy": "cost-optimized"})
    router = BedrockRouter(cfg, boto_session=session)
    return router, mock_client


def _get_converse_call_kwargs(mock_client) -> dict:
    """Extract the kwargs from the last converse() call."""
    # RetryHandler wraps the call, so we need to get the actual call
    return mock_client.converse.call_args[1]


def _get_stream_call_kwargs(mock_client) -> dict:
    """Extract the kwargs from the last converse_stream() call."""
    return mock_client.converse_stream.call_args[1]


class TestAdditionalModelRequestFields:
    """additionalModelRequestFields — model-specific inference params."""

    def test_top_k_passthrough(self, mock_router):
        router, client = mock_router
        router.converse(
            messages=_msgs("Hello"),
            additionalModelRequestFields={"top_k": 50},
        )
        kwargs = _get_converse_call_kwargs(client)
        assert kwargs["additionalModelRequestFields"] == {"top_k": 50}

    def test_anthropic_extended_thinking(self, mock_router):
        router, client = mock_router
        router.converse(
            messages=_msgs("Think step by step"),
            additionalModelRequestFields={
                "thinking": {"type": "enabled", "budget_tokens": 5000}
            },
        )
        kwargs = _get_converse_call_kwargs(client)
        assert kwargs["additionalModelRequestFields"]["thinking"]["type"] == "enabled"
        assert kwargs["additionalModelRequestFields"]["thinking"]["budget_tokens"] == 5000

    def test_stream_passthrough(self, mock_router):
        router, client = mock_router
        for _ in router.converse_stream(
            messages=_msgs("Hello"),
            additionalModelRequestFields={"top_k": 25},
        ):
            pass
        kwargs = _get_stream_call_kwargs(client)
        assert kwargs["additionalModelRequestFields"] == {"top_k": 25}


class TestAdditionalModelResponseFieldPaths:
    """additionalModelResponseFieldPaths — request extra response fields."""

    def test_stop_sequence_path(self, mock_router):
        router, client = mock_router
        router.converse(
            messages=_msgs("Hello"),
            additionalModelResponseFieldPaths=["/stop_sequence"],
        )
        kwargs = _get_converse_call_kwargs(client)
        assert kwargs["additionalModelResponseFieldPaths"] == ["/stop_sequence"]

    def test_multiple_paths(self, mock_router):
        router, client = mock_router
        router.converse(
            messages=_msgs("Hello"),
            additionalModelResponseFieldPaths=["/stop_sequence", "/model_id"],
        )
        kwargs = _get_converse_call_kwargs(client)
        assert len(kwargs["additionalModelResponseFieldPaths"]) == 2


class TestGuardrailConfig:
    """guardrailConfig — native Bedrock guardrail on the Converse call."""

    def test_guardrail_config_passthrough(self, mock_router):
        router, client = mock_router
        router.converse(
            messages=_msgs("Hello"),
            guardrailConfig={
                "guardrailIdentifier": "gr-abc123",
                "guardrailVersion": "1",
                "trace": "enabled",
            },
        )
        kwargs = _get_converse_call_kwargs(client)
        assert kwargs["guardrailConfig"]["guardrailIdentifier"] == "gr-abc123"
        assert kwargs["guardrailConfig"]["guardrailVersion"] == "1"
        assert kwargs["guardrailConfig"]["trace"] == "enabled"

    def test_stream_guardrail_passthrough(self, mock_router):
        router, client = mock_router
        for _ in router.converse_stream(
            messages=_msgs("Hello"),
            guardrailConfig={
                "guardrailIdentifier": "gr-xyz",
                "guardrailVersion": "DRAFT",
            },
        ):
            pass
        kwargs = _get_stream_call_kwargs(client)
        assert kwargs["guardrailConfig"]["guardrailIdentifier"] == "gr-xyz"


class TestPromptVariables:
    """promptVariables — for Prompt Management integration."""

    def test_prompt_variables_passthrough(self, mock_router):
        router, client = mock_router
        router.converse(
            messages=_msgs("Hello"),
            promptVariables={
                "topic": {"text": "cloud computing"},
                "tone": {"text": "professional"},
            },
        )
        kwargs = _get_converse_call_kwargs(client)
        assert kwargs["promptVariables"]["topic"] == {"text": "cloud computing"}
        assert kwargs["promptVariables"]["tone"] == {"text": "professional"}


class TestOutputConfig:
    """outputConfig — structured output format (JSON schema)."""

    def test_output_config_passthrough(self, mock_router):
        router, client = mock_router
        router.converse(
            messages=_msgs("List 3 colors"),
            outputConfig={
                "textFormat": {
                    "type": "json",
                    "structure": {
                        "json_schema": {
                            "type": "object",
                            "properties": {
                                "colors": {"type": "array", "items": {"type": "string"}}
                            },
                        }
                    },
                }
            },
        )
        kwargs = _get_converse_call_kwargs(client)
        assert kwargs["outputConfig"]["textFormat"]["type"] == "json"


class TestPerformanceConfig:
    """performanceConfig — latency-optimized inference mode."""

    def test_performance_config_passthrough(self, mock_router):
        router, client = mock_router
        router.converse(
            messages=_msgs("Hello"),
            performanceConfig={"latency": "optimized"},
        )
        kwargs = _get_converse_call_kwargs(client)
        assert kwargs["performanceConfig"]["latency"] == "optimized"

    def test_stream_performance_passthrough(self, mock_router):
        router, client = mock_router
        for _ in router.converse_stream(
            messages=_msgs("Hello"),
            performanceConfig={"latency": "optimized"},
        ):
            pass
        kwargs = _get_stream_call_kwargs(client)
        assert kwargs["performanceConfig"]["latency"] == "optimized"


class TestMultipleKwargs:
    """Multiple kwargs can be combined in a single call."""

    def test_combined_kwargs(self, mock_router):
        router, client = mock_router
        router.converse(
            messages=_msgs("Hello"),
            additionalModelRequestFields={"top_k": 50},
            additionalModelResponseFieldPaths=["/stop_sequence"],
            performanceConfig={"latency": "optimized"},
        )
        kwargs = _get_converse_call_kwargs(client)
        assert kwargs["additionalModelRequestFields"] == {"top_k": 50}
        assert kwargs["additionalModelResponseFieldPaths"] == ["/stop_sequence"]
        assert kwargs["performanceConfig"]["latency"] == "optimized"

    def test_kwargs_dont_override_router_params(self, mock_router):
        """Router-managed params (modelId, messages) should not be overridable."""
        router, client = mock_router
        router.converse(
            messages=_msgs("Hello"),
            system=[{"text": "Be helpful"}],
            inferenceConfig={"maxTokens": 100},
        )
        kwargs = _get_converse_call_kwargs(client)
        # These should be set by the router, not overridden
        assert "modelId" in kwargs
        assert kwargs["messages"] == _msgs("Hello")
        assert kwargs["system"] == [{"text": "Be helpful"}]
        assert kwargs["inferenceConfig"] == {"maxTokens": 100}


class TestRequestMetadata:
    """requestMetadata — forwarded from routing.metadata for CloudWatch logs."""

    def test_metadata_forwarded(self, mock_router):
        from bedrock_smart_router import RoutingConfig
        router, client = mock_router
        router.converse(
            messages=_msgs("Hello"),
            routing=RoutingConfig(metadata={"tenant": "acme", "team": "eng"}),
        )
        kwargs = _get_converse_call_kwargs(client)
        assert kwargs["requestMetadata"]["tenant"] == "acme"
        assert kwargs["requestMetadata"]["team"] == "eng"

    def test_stream_metadata_forwarded(self, mock_router):
        from bedrock_smart_router import RoutingConfig
        router, client = mock_router
        for _ in router.converse_stream(
            messages=_msgs("Hello"),
            routing=RoutingConfig(metadata={"tenant": "globex"}),
        ):
            pass
        kwargs = _get_stream_call_kwargs(client)
        assert kwargs["requestMetadata"]["tenant"] == "globex"

    def test_no_metadata_no_field(self, mock_router):
        router, client = mock_router
        router.converse(messages=_msgs("Hello"))
        kwargs = _get_converse_call_kwargs(client)
        # requestMetadata should not be present when no metadata
        assert "requestMetadata" not in kwargs or kwargs["requestMetadata"] is None


class TestResponseFieldCapture:
    """Verify all Bedrock response fields are captured in RoutingDecision."""

    def _make_rich_response(self):
        return {
            "output": {"message": {"content": [{"text": "ok"}]}},
            "usage": {
                "inputTokens": 100,
                "outputTokens": 50,
                "totalTokens": 150,
                "cacheReadInputTokens": 80,
                "cacheWriteInputTokens": 20,
                "cacheDetails": [{"inputTokens": 80, "ttl": "PT1H"}],
            },
            "stopReason": "end_turn",
            "metrics": {"latencyMs": 450},
            "serviceTier": {"type": "optimized"},
            "performanceConfig": {"latency": "optimized"},
            "trace": {
                "guardrail": {
                    "inputAssessment": {"0": {"topicPolicy": {"topics": []}}},
                }
            },
        }

    def test_total_tokens_captured(self, mock_router):
        router, client = mock_router
        client.converse.return_value = self._make_rich_response()
        response = router.converse(messages=_msgs("Hello"))
        d = response["routing_decision"]
        assert d.total_tokens == 150

    def test_cache_details_captured(self, mock_router):
        router, client = mock_router
        client.converse.return_value = self._make_rich_response()
        response = router.converse(messages=_msgs("Hello"))
        d = response["routing_decision"]
        assert len(d.cache_details) == 1
        assert d.cache_details[0]["ttl"] == "PT1H"

    def test_performance_config_captured(self, mock_router):
        router, client = mock_router
        client.converse.return_value = self._make_rich_response()
        response = router.converse(messages=_msgs("Hello"))
        d = response["routing_decision"]
        assert d.performance_config == {"latency": "optimized"}

    def test_guardrail_trace_captured(self, mock_router):
        router, client = mock_router
        client.converse.return_value = self._make_rich_response()
        response = router.converse(messages=_msgs("Hello"))
        d = response["routing_decision"]
        assert "inputAssessment" in d.guardrail_trace

    def test_prompt_cache_hit_rate_with_real_values(self, mock_router):
        router, client = mock_router
        client.converse.return_value = self._make_rich_response()
        response = router.converse(messages=_msgs("Hello"))
        d = response["routing_decision"]
        # total = 100 input + 80 read + 20 write = 200
        # hit rate = 80 / 200 × 100 = 40.0%
        assert d.total_input_tokens == 200
        assert d.prompt_cache_hit_rate == pytest.approx(40.0)

    def test_service_tier_captured(self, mock_router):
        router, client = mock_router
        client.converse.return_value = self._make_rich_response()
        response = router.converse(messages=_msgs("Hello"))
        d = response["routing_decision"]
        assert d.actual_service_tier == "optimized"
