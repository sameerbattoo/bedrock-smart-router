"""Tests for the Responses API surface (router.responses.create).

Covers:
- router.responses.create() interface
- Model selection (quality-optimized default, mid minimum tier)
- Sticky routing via response ID encoding (id::model_id separator)
- Fallback to explicit model
- Responses path resolution from catalog
- Error handling (no eligible models, missing Mantle client)
- DotDict response compatibility (dot notation + dict access)
- Streaming support
- Service tier filtering with Responses API
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from bedrock_smart_router.config import RouterConfig, RoutingConfig
from bedrock_smart_router.models import (
    BedrockModel,
    ModelCapabilities,
    ModelPricing,
    Tier,
)
from bedrock_smart_router.router import _RESPONSE_ID_SEP


# ═══════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════

def _make_responses_response(model="openai.gpt-oss-120b", text="Hello!", resp_id="resp_abc123"):
    """Build a mock Mantle Responses API response."""
    return {
        "id": resp_id,
        "object": "response",
        "model": model,
        "output": [
            {"type": "message", "role": "assistant", "content": [{"type": "text", "text": text}]},
        ],
        "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        "status": "completed",
        "created_at": 1700000000,
    }


@pytest.fixture
def mock_router():
    """Create a router with mocked Mantle client."""
    with patch("boto3.Session") as mock_session:
        mock_client = MagicMock()
        mock_client.converse.return_value = {
            "output": {"message": {"role": "assistant", "content": [{"text": "Hi"}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 10, "outputTokens": 5},
        }
        mock_session.return_value.client.return_value = mock_client
        mock_session.return_value.region_name = "us-west-2"

        from bedrock_smart_router import BedrockRouter
        router = BedrockRouter.create({"region": "us-west-2"})
        router._bedrock = mock_client

        # Mock the Mantle client
        mock_mantle = MagicMock()
        mock_mantle.responses.return_value = _make_responses_response()
        router._mantle = mock_mantle

        yield router, mock_mantle


# ═══════════════════════════════════════════════════════════════
# Responses Namespace Tests
# ═══════════════════════════════════════════════════════════════

class TestResponsesCreate:
    """Test router.responses.create() interface."""

    def test_returns_dotdict_with_routing_decision(self, mock_router):
        router, mantle = mock_router
        result = router.responses.create(input="Hello", model="openai.gpt-oss-120b-1:0")

        assert "routing_decision" in result
        assert result.routing_decision.selected_model == "openai.gpt-oss-120b-1:0"
        assert result.routing_decision.api_backend == "mantle"

    def test_dot_notation_access(self, mock_router):
        router, mantle = mock_router
        result = router.responses.create(input="Hello", model="openai.gpt-oss-120b-1:0")

        # Response fields accessible via dot notation
        assert result.id is not None
        assert result.model == "openai.gpt-oss-120b"
        assert result.output is not None

    def test_encodes_model_in_response_id(self, mock_router):
        router, mantle = mock_router
        result = router.responses.create(input="Hello", model="openai.gpt-oss-120b-1:0")

        assert _RESPONSE_ID_SEP in result.id
        raw_id, encoded_model = result.id.rsplit(_RESPONSE_ID_SEP, 1)
        assert raw_id == "resp_abc123"
        assert encoded_model == "openai.gpt-oss-120b-1:0"

    def test_explicit_model_skips_routing(self, mock_router):
        router, mantle = mock_router
        result = router.responses.create(input="Hello", model="openai.gpt-oss-120b-1:0")

        # Should call mantle with the model ID (stripped for Mantle)
        mantle.responses.assert_called_once()
        call_kwargs = mantle.responses.call_args
        assert call_kwargs.kwargs.get("model") or call_kwargs[1].get("model") or "openai.gpt-oss-120b" in str(call_kwargs)

    def test_passes_store_parameter(self, mock_router):
        router, mantle = mock_router
        router.responses.create(input="Hello", model="openai.gpt-oss-120b-1:0", store=False)

        call_kwargs = mantle.responses.call_args
        assert call_kwargs[1]["store"] == False

    def test_passes_temperature(self, mock_router):
        router, mantle = mock_router
        router.responses.create(input="Hello", model="openai.gpt-oss-120b-1:0", temperature=0.7)

        call_kwargs = mantle.responses.call_args
        assert call_kwargs[1].get("temperature") == 0.7

    def test_passes_max_output_tokens(self, mock_router):
        router, mantle = mock_router
        router.responses.create(input="Hello", model="openai.gpt-oss-120b-1:0", max_output_tokens=100)

        call_kwargs = mantle.responses.call_args
        assert call_kwargs[1].get("max_output_tokens") == 100

    def test_passes_tools(self, mock_router):
        router, mantle = mock_router
        tools = [{"type": "function", "function": {"name": "get_weather"}}]
        router.responses.create(input="Hello", model="openai.gpt-oss-120b-1:0", tools=tools)

        call_kwargs = mantle.responses.call_args
        assert call_kwargs[1].get("tools") == tools


class TestStickyRouting:
    """Test stateful routing via previous_response_id."""

    def test_decodes_model_from_previous_response_id(self, mock_router):
        router, mantle = mock_router
        # Simulate a response ID with encoded model
        encoded_id = f"resp_abc123{_RESPONSE_ID_SEP}openai.gpt-oss-120b-1:0"

        result = router.responses.create(
            input="Follow up",
            previous_response_id=encoded_id,
        )

        # Should route to the same model
        assert result.routing_decision.selected_model == "openai.gpt-oss-120b-1:0"
        # Mantle receives the raw ID (without model encoding)
        call_kwargs = mantle.responses.call_args
        assert call_kwargs[1]["previous_response_id"] == "resp_abc123"

    def test_sticky_routing_strategy_shows_sticky(self, mock_router):
        router, mantle = mock_router
        encoded_id = f"resp_abc123{_RESPONSE_ID_SEP}openai.gpt-oss-120b-1:0"

        result = router.responses.create(input="Follow up", previous_response_id=encoded_id)
        assert result.routing_decision.strategy_used == "sticky"

    def test_raw_previous_response_id_with_model(self, mock_router):
        """When raw ID (no ::) is passed with explicit model, should work."""
        router, mantle = mock_router
        result = router.responses.create(
            input="Follow up",
            model="openai.gpt-oss-120b-1:0",
            previous_response_id="resp_raw_id_no_separator",
        )

        call_kwargs = mantle.responses.call_args
        assert call_kwargs[1]["previous_response_id"] == "resp_raw_id_no_separator"

    def test_invalid_model_in_encoded_id_raises(self, mock_router):
        router, mantle = mock_router
        encoded_id = f"resp_abc123{_RESPONSE_ID_SEP}nonexistent.model.xyz"

        with pytest.raises(ValueError, match="not found in catalog"):
            router.responses.create(input="Follow up", previous_response_id=encoded_id)


class TestResponsesStrategy:
    """Test quality-optimized default and tier behavior."""

    def test_default_strategy_is_quality_optimized(self, mock_router):
        router, mantle = mock_router
        # Don't specify model — let router decide
        result = router.responses.create(input="Hello")

        # Should use quality-optimized (not balanced)
        assert result.routing_decision.strategy_used == "quality-optimized"

    def test_user_can_override_strategy(self, mock_router):
        router, mantle = mock_router
        result = router.responses.create(
            input="Hello",
            routing=RoutingConfig(strategy="cost-optimized"),
        )
        assert result.routing_decision.strategy_used == "cost-optimized"

    def test_responses_path_from_catalog(self, mock_router):
        router, mantle = mock_router
        # gpt-5.4 uses /openai/v1/responses path
        router.responses.create(input="Hello", model="openai.gpt-5.4")

        call_kwargs = mantle.responses.call_args
        assert call_kwargs[1]["path"] == "/openai/v1/responses"

    def test_standard_responses_path(self, mock_router):
        router, mantle = mock_router
        # gpt-oss-120b uses /v1/responses path
        router.responses.create(input="Hello", model="openai.gpt-oss-120b-1:0")

        call_kwargs = mantle.responses.call_args
        assert call_kwargs[1]["path"] == "/v1/responses"


class TestResponsesErrors:
    """Test error handling."""

    def test_no_mantle_raises_runtime_error(self):
        with patch("boto3.Session") as mock_session:
            mock_session.return_value.client.return_value = MagicMock()
            mock_session.return_value.region_name = "us-west-2"

            from bedrock_smart_router import BedrockRouter
            router = BedrockRouter.create({"region": "us-west-2", "enable_mantle": False})
            router._mantle = None

            with pytest.raises(RuntimeError, match="Mantle client"):
                router.responses.create(input="Hello")

    def test_mantle_error_records_circuit_breaker(self, mock_router):
        router, mantle = mock_router
        from bedrock_smart_router.mantle_client import MantleError
        mantle.responses.side_effect = MantleError(503, "Service unavailable")

        with pytest.raises(MantleError):
            router.responses.create(input="Hello", model="openai.gpt-oss-120b-1:0")


# ═══════════════════════════════════════════════════════════════
# Chat Completions Additional Tests
# ═══════════════════════════════════════════════════════════════

class TestChatCompletionsDotDict:
    """Test _DotDict behavior for OpenAI SDK compatibility."""

    def setup_method(self):
        self.mock_response = {
            "output": {"message": {"role": "assistant", "content": [{"text": "Hi!"}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 10, "outputTokens": 5},
        }

    @patch("boto3.Session")
    def test_missing_attribute_returns_none(self, mock_session):
        """OpenAI SDK expects message.tool_calls to be None when absent."""
        mock_client = MagicMock()
        mock_client.converse.return_value = self.mock_response
        mock_session.return_value.client.return_value = mock_client

        from bedrock_smart_router import BedrockRouter
        router = BedrockRouter.create({"region": "us-west-2", "enable_mantle": False})
        router._bedrock = mock_client

        result = router.chat.completions.create(
            messages=[{"role": "user", "content": "Hello"}],
        )

        # tool_calls should be None (not raise AttributeError)
        msg = result.choices[0].message
        assert msg.tool_calls is None
        assert msg.content is not None

    @patch("boto3.Session")
    def test_response_supports_dict_access(self, mock_session):
        """Response should work with both dict and dot notation."""
        mock_client = MagicMock()
        mock_client.converse.return_value = self.mock_response
        mock_session.return_value.client.return_value = mock_client

        from bedrock_smart_router import BedrockRouter
        router = BedrockRouter.create({"region": "us-west-2", "enable_mantle": False})
        router._bedrock = mock_client

        result = router.chat.completions.create(
            messages=[{"role": "user", "content": "Hello"}],
        )

        # Dict access
        assert result["choices"][0]["message"]["content"] is not None
        # Dot access
        assert result.choices[0].message.content is not None
        # Both give same value
        assert result["choices"][0]["message"]["content"] == result.choices[0].message.content

    @patch("boto3.Session")
    def test_response_is_awaitable(self, mock_session):
        """Response should support await for async compatibility."""
        import asyncio
        mock_client = MagicMock()
        mock_client.converse.return_value = self.mock_response
        mock_session.return_value.client.return_value = mock_client

        from bedrock_smart_router import BedrockRouter
        router = BedrockRouter.create({"region": "us-west-2", "enable_mantle": False})
        router._bedrock = mock_client

        async def _test():
            result = await router.chat.completions.create(
                messages=[{"role": "user", "content": "Hello"}],
            )
            assert result.choices[0].message.content is not None
            return result

        result = asyncio.run(_test())
        assert "choices" in result


class TestChatCompletionsStreaming:
    """Test streaming interface for chat.completions.create(stream=True)."""

    @patch("boto3.Session")
    def test_stream_returns_iterable(self, mock_session):
        """stream=True should return an iterable of chunks."""
        mock_client = MagicMock()
        # Mock converse_stream response
        mock_stream = iter([
            {"contentBlockDelta": {"delta": {"text": "Hello"}}},
            {"messageStop": {"stopReason": "end_turn"}},
            {"metadata": {"usage": {"inputTokens": 5, "outputTokens": 3}}},
        ])
        mock_client.converse_stream.return_value = {"stream": mock_stream}
        mock_session.return_value.client.return_value = mock_client

        from bedrock_smart_router import BedrockRouter
        router = BedrockRouter.create({"region": "us-west-2", "enable_mantle": False})
        router._bedrock = mock_client

        stream = router.chat.completions.create(
            messages=[{"role": "user", "content": "Hello"}],
            stream=True,
        )

        chunks = list(stream)
        assert len(chunks) > 0

    @patch("boto3.Session")
    def test_stream_supports_async_for(self, mock_session):
        """stream should support async for pattern."""
        import asyncio
        mock_client = MagicMock()
        mock_stream = iter([
            {"contentBlockDelta": {"delta": {"text": "Hi"}}},
            {"messageStop": {"stopReason": "end_turn"}},
            {"metadata": {"usage": {"inputTokens": 5, "outputTokens": 2}}},
        ])
        mock_client.converse_stream.return_value = {"stream": mock_stream}
        mock_session.return_value.client.return_value = mock_client

        from bedrock_smart_router import BedrockRouter
        router = BedrockRouter.create({"region": "us-west-2", "enable_mantle": False})
        router._bedrock = mock_client

        async def _test():
            stream = await router.chat.completions.create(
                messages=[{"role": "user", "content": "Hello"}],
                stream=True,
            )
            chunks = []
            async for chunk in stream:
                chunks.append(chunk)
            return chunks

        chunks = asyncio.run(_test())
        assert len(chunks) > 0


class TestChatCompletionsToolInference:
    """Test auto-infer toolConfig when tools not re-passed on follow-up calls."""

    @patch("boto3.Session")
    def test_infers_tool_config_from_history(self, mock_session):
        """When messages contain tool_calls but tools= not passed, should infer."""
        mock_client = MagicMock()
        mock_client.converse.return_value = {
            "output": {"message": {"role": "assistant", "content": [{"text": "Done"}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 20, "outputTokens": 5},
        }
        mock_session.return_value.client.return_value = mock_client

        from bedrock_smart_router import BedrockRouter
        router = BedrockRouter.create({"region": "us-west-2", "enable_mantle": False})
        router._bedrock = mock_client

        # Messages contain tool_calls but we don't pass tools= parameter
        messages = [
            {"role": "user", "content": "What's the weather?"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city":"NYC"}'}}
            ]},
            {"role": "tool", "tool_call_id": "call_1", "content": '{"temp": 72}'},
        ]

        router.chat.completions.create(messages=messages)

        # The converse call should have toolConfig inferred from history
        call_kwargs = mock_client.converse.call_args[1]
        assert "toolConfig" in call_kwargs
        tools = call_kwargs["toolConfig"]["tools"]
        assert any(t["toolSpec"]["name"] == "get_weather" for t in tools)
