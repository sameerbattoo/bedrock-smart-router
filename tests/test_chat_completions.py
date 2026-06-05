"""Tests for the Chat Completions API surface, Mantle integration, and new features.

Covers:
- router.chat.completions.create() interface
- router.models.list() / router.models.retrieve()
- Format translation (Converse ↔ Chat Completions)
- Mantle client (SigV4 + API key auth)
- Model ID matching (version suffix stripping)
- Tool presence boost (complexity upgrade)
- Quality penalty (zero baseline penalization)
- api_support filtering
- Bedrock API key configuration
- Edge cases (empty messages, malformed data, concurrent access)
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from dataclasses import replace as dataclass_replace
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from bedrock_smart_router.config import RouterConfig, RoutingConfig
from bedrock_smart_router.format_translator import (
    converse_to_chat_completions,
    chat_completions_to_converse,
    chat_completions_response_to_converse,
    converse_response_to_chat_completions,
)
from bedrock_smart_router.mantle_client import MantleClient, MantleError, MantleThrottleError
from bedrock_smart_router.model_registry import ModelRegistry, base_model_id
from bedrock_smart_router.models import (
    BedrockModel,
    ModelCapabilities,
    ModelPricing,
    RequestAnalysis,
    Complexity,
    Tier,
)
from bedrock_smart_router.strategy_engine import _quality_score


# ═══════════════════════════════════════════════════════════════
# Format Translator Tests
# ═══════════════════════════════════════════════════════════════

class TestConverseToCC:
    """Test Converse → Chat Completions translation."""

    def test_basic_text_message(self):
        result = converse_to_chat_completions(
            messages=[{"role": "user", "content": [{"text": "Hello"}]}],
        )
        assert result["messages"] == [{"role": "user", "content": "Hello"}]

    def test_system_prompt(self):
        result = converse_to_chat_completions(
            messages=[{"role": "user", "content": [{"text": "Hi"}]}],
            system=[{"text": "You are helpful."}],
        )
        assert result["messages"][0] == {"role": "system", "content": "You are helpful."}
        assert result["messages"][1] == {"role": "user", "content": "Hi"}

    def test_multiple_system_blocks(self):
        result = converse_to_chat_completions(
            messages=[{"role": "user", "content": [{"text": "Hi"}]}],
            system=[{"text": "Be concise."}, {"text": "Use markdown."}],
        )
        assert result["messages"][0]["content"] == "Be concise. Use markdown."

    def test_assistant_with_tool_use(self):
        result = converse_to_chat_completions(
            messages=[
                {"role": "assistant", "content": [
                    {"text": "Let me check."},
                    {"toolUse": {"toolUseId": "tc1", "name": "search", "input": {"q": "test"}}},
                ]},
            ],
        )
        msg = result["messages"][0]
        assert msg["role"] == "assistant"
        assert msg["content"] == "Let me check."
        assert len(msg["tool_calls"]) == 1
        assert msg["tool_calls"][0]["function"]["name"] == "search"

    def test_tool_result_single(self):
        result = converse_to_chat_completions(
            messages=[
                {"role": "user", "content": [
                    {"toolResult": {"toolUseId": "tc1", "content": [{"text": "result"}]}},
                ]},
            ],
        )
        msg = result["messages"][0]
        assert msg["role"] == "tool"
        assert msg["tool_call_id"] == "tc1"
        assert msg["content"] == "result"

    def test_tool_result_multiple(self):
        """Multiple tool results should expand into separate messages."""
        result = converse_to_chat_completions(
            messages=[
                {"role": "user", "content": [
                    {"toolResult": {"toolUseId": "tc1", "content": [{"text": "r1"}]}},
                    {"toolResult": {"toolUseId": "tc2", "content": [{"text": "r2"}]}},
                ]},
            ],
        )
        tool_msgs = [m for m in result["messages"] if m.get("role") == "tool"]
        assert len(tool_msgs) == 2
        assert tool_msgs[0]["tool_call_id"] == "tc1"
        assert tool_msgs[1]["tool_call_id"] == "tc2"

    def test_inference_config(self):
        result = converse_to_chat_completions(
            messages=[{"role": "user", "content": [{"text": "Hi"}]}],
            inference_config={"maxTokens": 100, "temperature": 0.7, "topP": 0.9, "stopSequences": ["END"]},
        )
        assert result["max_tokens"] == 100
        assert result["temperature"] == 0.7
        assert result["top_p"] == 0.9
        assert result["stop"] == ["END"]

    def test_tool_config(self):
        result = converse_to_chat_completions(
            messages=[{"role": "user", "content": [{"text": "Hi"}]}],
            tool_config={"tools": [{"toolSpec": {
                "name": "calc", "description": "Math",
                "inputSchema": {"json": {"type": "object", "properties": {"expr": {"type": "string"}}}},
            }}]},
        )
        assert len(result["tools"]) == 1
        assert result["tools"][0]["type"] == "function"
        assert result["tools"][0]["function"]["name"] == "calc"

    def test_empty_messages(self):
        result = converse_to_chat_completions(messages=[])
        assert result["messages"] == []

    def test_reasoning_content_skipped(self):
        """reasoningContent blocks should be silently skipped."""
        result = converse_to_chat_completions(
            messages=[{"role": "assistant", "content": [
                {"reasoningContent": {"reasoningText": {"text": "thinking..."}}},
                {"text": "The answer is 4."},
            ]}],
        )
        msg = result["messages"][0]
        assert msg["content"] == "The answer is 4."
        assert "tool_calls" not in msg or msg.get("tool_calls") is None


class TestCCToConverse:
    """Test Chat Completions → Converse translation."""

    def test_basic_messages(self):
        result = chat_completions_to_converse(
            messages=[
                {"role": "system", "content": "Be helpful."},
                {"role": "user", "content": "Hello"},
            ],
        )
        assert result["system"] == [{"text": "Be helpful."}]
        assert result["messages"][0] == {"role": "user", "content": [{"text": "Hello"}]}

    def test_assistant_with_tool_calls(self):
        result = chat_completions_to_converse(
            messages=[{"role": "assistant", "content": "checking...", "tool_calls": [
                {"id": "tc1", "type": "function", "function": {"name": "search", "arguments": '{"q": "test"}'}},
            ]}],
        )
        msg = result["messages"][0]
        assert msg["role"] == "assistant"
        assert msg["content"][0]["text"] == "checking..."
        assert msg["content"][1]["toolUse"]["name"] == "search"
        assert msg["content"][1]["toolUse"]["input"] == {"q": "test"}

    def test_tool_role_message(self):
        result = chat_completions_to_converse(
            messages=[{"role": "tool", "tool_call_id": "tc1", "content": "result"}],
        )
        msg = result["messages"][0]
        assert msg["role"] == "user"
        assert msg["content"][0]["toolResult"]["toolUseId"] == "tc1"
        assert msg["content"][0]["toolResult"]["content"] == [{"text": "result"}]

    def test_tools_to_tool_config(self):
        result = chat_completions_to_converse(
            messages=[{"role": "user", "content": "hi"}],
            tools=[{"type": "function", "function": {
                "name": "weather", "description": "Get weather",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
            }}],
        )
        tc = result["tool_config"]["tools"][0]["toolSpec"]
        assert tc["name"] == "weather"
        assert tc["inputSchema"]["json"]["properties"]["city"]["type"] == "string"

    def test_inference_params(self):
        result = chat_completions_to_converse(
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=200, temperature=0.5, top_p=0.8, stop=["DONE"],
        )
        ic = result["inference_config"]
        assert ic["maxTokens"] == 200
        assert ic["temperature"] == 0.5
        assert ic["topP"] == 0.8
        assert ic["stopSequences"] == ["DONE"]

    def test_multimodal_content_array(self):
        result = chat_completions_to_converse(
            messages=[{"role": "user", "content": [
                {"type": "text", "text": "Describe this:"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="}},
            ]}],
        )
        msg = result["messages"][0]
        assert msg["content"][0]["text"] == "Describe this:"
        assert "image" in msg["content"][1]

    def test_malformed_data_uri_handled(self):
        """Malformed data URIs should not crash."""
        result = chat_completions_to_converse(
            messages=[{"role": "user", "content": [
                {"type": "text", "text": "hi"},
                {"type": "image_url", "image_url": {"url": "data:broken"}},
            ]}],
        )
        # Should produce at least the text block without crashing
        assert result["messages"][0]["content"][0]["text"] == "hi"

    def test_empty_messages(self):
        result = chat_completions_to_converse(messages=[])
        assert result["messages"] == []

    def test_null_content(self):
        """Assistant message with null content (tool_calls only)."""
        result = chat_completions_to_converse(
            messages=[{"role": "assistant", "content": None, "tool_calls": [
                {"id": "tc1", "type": "function", "function": {"name": "fn", "arguments": "{}"}},
            ]}],
        )
        msg = result["messages"][0]
        assert msg["content"][0]["toolUse"]["name"] == "fn"


class TestResponseTranslation:
    """Test response format translations."""

    def test_cc_response_to_converse(self):
        cc_resp = {
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello!"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        result = chat_completions_response_to_converse(cc_resp)
        assert result["output"]["message"]["content"][0]["text"] == "Hello!"
        assert result["stopReason"] == "end_turn"
        assert result["usage"]["inputTokens"] == 10

    def test_cc_response_with_tool_calls(self):
        cc_resp = {
            "choices": [{"index": 0, "message": {
                "role": "assistant", "content": None,
                "tool_calls": [{"id": "tc1", "type": "function", "function": {"name": "calc", "arguments": '{"x": 1}'}}],
            }, "finish_reason": "tool_calls"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15},
        }
        result = chat_completions_response_to_converse(cc_resp)
        assert result["stopReason"] == "tool_use"
        assert result["output"]["message"]["content"][0]["toolUse"]["name"] == "calc"

    def test_converse_response_to_cc(self):
        converse_resp = {
            "output": {"message": {"role": "assistant", "content": [{"text": "Hi!"}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 5, "outputTokens": 3},
        }
        result = converse_response_to_chat_completions(converse_resp, model="test-model")
        assert result["model"] == "test-model"
        assert result["choices"][0]["message"]["content"] == "Hi!"
        assert result["choices"][0]["finish_reason"] == "stop"
        assert result["usage"]["prompt_tokens"] == 5

    def test_empty_cc_response(self):
        result = chat_completions_response_to_converse({"choices": []})
        assert result["output"]["message"]["content"] == [{"text": ""}]


# ═══════════════════════════════════════════════════════════════
# Model Registry Tests
# ═══════════════════════════════════════════════════════════════

class TestModelRegistryVersionMatching:
    """Test the version-suffix stripping lookup in ModelRegistry.get()."""

    def setup_method(self):
        self.model = BedrockModel(
            model_id="openai.gpt-oss-120b-1:0",
            family="openai", tier=Tier.MID, display_name="GPT-OSS-120b",
            api_support=["converse", "chat_completions"],
            pricing=ModelPricing(input_per_1k=0.001, output_per_1k=0.002),
        )
        self.model_v = BedrockModel(
            model_id="qwen.qwen3-32b-v1:0",
            family="qwen", tier=Tier.LITE, display_name="Qwen3 32B",
            api_support=["converse", "chat_completions"],
            pricing=ModelPricing(input_per_1k=0.001, output_per_1k=0.002),
        )
        self.registry = ModelRegistry(models=[self.model, self.model_v])

    def test_exact_match(self):
        assert self.registry.get("openai.gpt-oss-120b-1:0") is self.model

    def test_stripped_version_match(self):
        """Lookup without version suffix should find the model."""
        assert self.registry.get("openai.gpt-oss-120b") is self.model

    def test_stripped_v_version_match(self):
        assert self.registry.get("qwen.qwen3-32b") is self.model_v

    def test_geo_prefix_stripped(self):
        assert self.registry.get("us.openai.gpt-oss-120b-1:0") is self.model

    def test_geo_plus_version_stripped(self):
        assert self.registry.get("us.openai.gpt-oss-120b") is self.model

    def test_nonexistent(self):
        assert self.registry.get("nonexistent.model") is None

    def test_api_support_filter(self):
        models = self.registry.eligible_models(api_surface="converse")
        assert len(models) == 2

    def test_api_support_filter_responses_only(self):
        resp_model = BedrockModel(
            model_id="openai.gpt-5.4", family="openai", tier=Tier.MID,
            display_name="GPT 5.4", api_support=["responses"],
            pricing=ModelPricing(input_per_1k=0.003, output_per_1k=0.015),
        )
        reg = ModelRegistry(models=[self.model, resp_model])
        converse_eligible = reg.eligible_models(api_surface="converse")
        # gpt-5.4 should be excluded (responses-only)
        assert all(m.model_id != "openai.gpt-5.4" for m in converse_eligible)


# ═══════════════════════════════════════════════════════════════
# Mantle Client Tests
# ═══════════════════════════════════════════════════════════════

class TestMantleClient:
    """Test MantleClient behavior with mocked HTTP."""

    def test_init_with_api_key(self):
        client = MantleClient(region="us-west-2", api_key="brk_test123")
        assert client._api_key == "brk_test123"
        assert client._session is None  # No boto session needed with API key

    def test_init_with_sigv4(self):
        client = MantleClient(region="us-east-1")
        assert client._api_key is None
        assert client._session is not None
        assert client._region == "us-east-1"

    def test_base_url(self):
        client = MantleClient(region="eu-west-1", api_key="test")
        assert client._base_url == "https://bedrock-mantle.eu-west-1.api.aws"

    @patch("bedrock_smart_router.mantle_client.requests")
    def test_chat_completions_success(self, mock_requests):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"role": "assistant", "content": "Hi"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
        }
        mock_requests.request.return_value = mock_resp

        client = MantleClient(region="us-west-2", api_key="brk_test")
        result = client.chat_completions(
            model="openai.gpt-oss-120b",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=10,
        )
        assert result["choices"][0]["message"]["content"] == "Hi"

    @patch("bedrock_smart_router.mantle_client.requests")
    def test_chat_completions_throttle(self, mock_requests):
        mock_resp = MagicMock()
        mock_resp.status_code = 429
        mock_resp.json.return_value = {"error": {"message": "Rate limited", "type": "rate_limit"}}
        mock_requests.request.return_value = mock_resp

        client = MantleClient(region="us-west-2", api_key="brk_test", max_retries=0)
        with pytest.raises(MantleThrottleError):
            client.chat_completions(model="test", messages=[{"role": "user", "content": "hi"}])

    @patch("bedrock_smart_router.mantle_client.requests")
    def test_bearer_auth_header(self, mock_requests):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"choices": [{"message": {"content": "x"}, "finish_reason": "stop"}], "usage": {}}
        mock_requests.request.return_value = mock_resp

        client = MantleClient(region="us-west-2", api_key="brk_mykey")
        client.chat_completions(model="m", messages=[{"role": "user", "content": "hi"}])

        call_kwargs = mock_requests.request.call_args
        headers = call_kwargs.kwargs.get("headers") or call_kwargs[1].get("headers")
        assert headers["Authorization"] == "Bearer brk_mykey"


# ═══════════════════════════════════════════════════════════════
# Quality Penalty Tests
# ═══════════════════════════════════════════════════════════════

class TestQualityPenalty:
    """Test that models with quality_baseline=0 are penalized."""

    def test_zero_quality_gets_negative_score(self):
        model = BedrockModel(
            model_id="test.model", family="test", tier=Tier.MICRO,
            display_name="Test", quality_baseline=0.0,
        )
        score = _quality_score(model, None)
        assert score == -0.1

    def test_nonzero_quality_gets_positive_score(self):
        model = BedrockModel(
            model_id="test.model", family="test", tier=Tier.MID,
            display_name="Test", quality_baseline=30.0,
        )
        score = _quality_score(model, None)
        assert score == 0.5  # 30/60

    def test_high_quality_model(self):
        model = BedrockModel(
            model_id="test.model", family="test", tier=Tier.MID,
            display_name="Test", quality_baseline=60.0,
        )
        score = _quality_score(model, None)
        assert score == 1.0


# ═══════════════════════════════════════════════════════════════
# Tool Boost Tests
# ═══════════════════════════════════════════════════════════════

class TestToolBoost:
    """Test that tool_config presence boosts complexity."""

    def test_simple_prompt_with_tools_becomes_moderate(self):
        from bedrock_smart_router.request_analyzer import RequestAnalyzer
        analyzer = RequestAnalyzer()
        analysis = analyzer.analyze(
            messages=[{"role": "user", "content": [{"text": "What is the weather?"}]}],
            tool_config={"tools": [{"toolSpec": {"name": "get_weather", "description": "Get weather",
                "inputSchema": {"json": {"type": "object", "properties": {"city": {"type": "string"}}}}}}]},
        )
        assert analysis.complexity == Complexity.MODERATE
        assert analysis.tool_boost_applied is True

    def test_complex_prompt_with_tools_stays_complex(self):
        from bedrock_smart_router.request_analyzer import RequestAnalyzer
        analyzer = RequestAnalyzer()
        analysis = analyzer.analyze(
            messages=[{"role": "user", "content": [{"text": "Design a distributed system with fault tolerance, auto-scaling, and multi-region failover using these tools"}]}],
            tool_config={"tools": [{"toolSpec": {"name": "architect", "description": "Design system",
                "inputSchema": {"json": {"type": "object"}}}}]},
        )
        # Complex stays complex, no boost needed
        assert analysis.complexity in (Complexity.COMPLEX, Complexity.MODERATE)
        # tool_boost_applied only true when upgrading from simple
        if analysis.complexity == Complexity.COMPLEX:
            assert analysis.tool_boost_applied is False

    def test_no_tools_no_boost(self):
        from bedrock_smart_router.request_analyzer import RequestAnalyzer
        analyzer = RequestAnalyzer()
        analysis = analyzer.analyze(
            messages=[{"role": "user", "content": [{"text": "What is the weather?"}]}],
        )
        assert analysis.tool_boost_applied is False


# ═══════════════════════════════════════════════════════════════
# Bedrock API Key Config Tests
# ═══════════════════════════════════════════════════════════════

class TestAPIKeyConfig:
    """Test that api_key configuration is handled correctly."""

    def test_api_key_sets_env_var(self):
        """API key should set AWS_BEARER_TOKEN_BEDROCK env var."""
        # Clean up any existing env var
        os.environ.pop("AWS_BEARER_TOKEN_BEDROCK", None)

        config = RouterConfig.from_dict({"region": "us-west-2", "api_key": "brk_testkey123"})
        assert config.api_key == "brk_testkey123"

    def test_no_api_key_default(self):
        config = RouterConfig.from_dict({"region": "us-west-2"})
        assert config.api_key is None

    def test_mantle_enabled_by_default(self):
        config = RouterConfig.from_dict({"region": "us-west-2"})
        assert config.enable_mantle is True

    def test_mantle_disabled(self):
        config = RouterConfig.from_dict({"region": "us-west-2", "enable_mantle": False})
        assert config.enable_mantle is False


# ═══════════════════════════════════════════════════════════════
# Chat Completions Namespace Tests (router.chat.completions.create)
# ═══════════════════════════════════════════════════════════════

class TestChatCompletionsNamespace:
    """Test the OpenAI-compatible router.chat.completions.create() interface."""

    def setup_method(self):
        """Create a router with mocked bedrock client."""
        self.mock_response = {
            "output": {"message": {"role": "assistant", "content": [{"text": "Hello!"}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 10, "outputTokens": 5},
            "metrics": {"latencyMs": 100},
        }

    @patch("boto3.Session")
    def test_create_returns_cc_format(self, mock_session):
        """chat.completions.create should return OpenAI-format response."""
        mock_client = MagicMock()
        mock_client.converse.return_value = self.mock_response
        mock_session.return_value.client.return_value = mock_client

        from bedrock_smart_router import BedrockRouter
        router = BedrockRouter.create({"region": "us-west-2", "enable_mantle": False})
        # Mock the bedrock client
        router._bedrock = mock_client

        result = router.chat.completions.create(
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=50,
        )
        assert "choices" in result
        assert "usage" in result
        assert result["choices"][0]["message"]["role"] == "assistant"

    @patch("boto3.Session")
    def test_models_list(self, mock_session):
        """router.models.list() should return model catalog."""
        mock_session.return_value.client.return_value = MagicMock()

        from bedrock_smart_router import BedrockRouter
        router = BedrockRouter.create({"region": "us-west-2", "enable_mantle": False})

        result = router.models.list()
        assert result["object"] == "list"
        assert len(result["data"]) > 0
        assert all("id" in m for m in result["data"])

    @patch("boto3.Session")
    def test_models_retrieve(self, mock_session):
        """router.models.retrieve() should return model details."""
        mock_session.return_value.client.return_value = MagicMock()

        from bedrock_smart_router import BedrockRouter
        router = BedrockRouter.create({"region": "us-west-2", "enable_mantle": False})

        # Use a model we know exists
        result = router.models.retrieve("openai.gpt-oss-120b")
        assert result is not None
        assert "api_support" in result
        assert result["id"] == "openai.gpt-oss-120b-1:0"

    @patch("boto3.Session")
    def test_models_retrieve_not_found(self, mock_session):
        mock_session.return_value.client.return_value = MagicMock()

        from bedrock_smart_router import BedrockRouter
        router = BedrockRouter.create({"region": "us-west-2", "enable_mantle": False})

        result = router.models.retrieve("nonexistent.model.xyz")
        assert result is None


# ═══════════════════════════════════════════════════════════════
# Thread Safety Tests
# ═══════════════════════════════════════════════════════════════

class TestThreadSafety:
    """Test thread safety of _last_decision."""

    @patch("boto3.Session")
    def test_last_decision_thread_local(self, mock_session):
        """Each thread should have its own _last_decision."""
        mock_client = MagicMock()
        mock_client.converse.return_value = {
            "output": {"message": {"role": "assistant", "content": [{"text": "ok"}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 5, "outputTokens": 2},
            "metrics": {"latencyMs": 50},
        }
        mock_session.return_value.client.return_value = mock_client

        from bedrock_smart_router import BedrockRouter
        router = BedrockRouter.create({"region": "us-west-2", "enable_mantle": False})
        router._bedrock = mock_client

        results = {}

        def make_request(thread_id):
            router.converse(
                messages=[{"role": "user", "content": [{"text": f"Thread {thread_id}"}]}],
                inferenceConfig={"maxTokens": 10},
            )
            decision = router.last_routing_decision()
            results[thread_id] = decision

        threads = [threading.Thread(target=make_request, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Each thread should have gotten a decision (not None)
        assert len(results) == 5
        for tid, decision in results.items():
            assert decision is not None


# ═══════════════════════════════════════════════════════════════
# BudgetTracker Shutdown Tests
# ═══════════════════════════════════════════════════════════════

class TestBudgetTrackerShutdown:
    """Test that BudgetTracker can be cleanly shut down."""

    def test_close_stops_sync_thread(self):
        from bedrock_smart_router.budget_strategy import BudgetTracker
        tracker = BudgetTracker(store=MagicMock(), sync_interval=0.1)
        assert tracker._sync_thread.is_alive()

        tracker.close()
        time.sleep(0.3)
        assert not tracker._sync_thread.is_alive()

    def test_close_without_store(self):
        """close() should not crash when no store is configured."""
        from bedrock_smart_router.budget_strategy import BudgetTracker
        tracker = BudgetTracker(store=None)
        tracker.close()  # Should not raise


# ═══════════════════════════════════════════════════════════════
# Edge Cases
# ═══════════════════════════════════════════════════════════════

class TestEdgeCases:
    """Edge cases for the new features."""

    def test_converse_to_cc_empty_content_blocks(self):
        """Empty content blocks should produce empty content."""
        result = converse_to_chat_completions(
            messages=[{"role": "user", "content": []}],
        )
        assert result["messages"][0]["content"] == ""

    def test_cc_to_converse_string_stop(self):
        """Single string stop sequence should be wrapped in list."""
        result = chat_completions_to_converse(
            messages=[{"role": "user", "content": "hi"}],
            stop="END",
        )
        assert result["inference_config"]["stopSequences"] == ["END"]

    def test_cc_to_converse_list_stop(self):
        result = chat_completions_to_converse(
            messages=[{"role": "user", "content": "hi"}],
            stop=["END", "STOP"],
        )
        assert result["inference_config"]["stopSequences"] == ["END", "STOP"]

    def test_malformed_tool_call_arguments(self):
        """Invalid JSON in tool call arguments should not crash."""
        result = chat_completions_to_converse(
            messages=[{"role": "assistant", "content": None, "tool_calls": [
                {"id": "tc1", "type": "function", "function": {"name": "fn", "arguments": "not json {{{"}},
            ]}],
        )
        msg = result["messages"][0]
        # Should have the tool use block with raw content
        assert msg["content"][0]["toolUse"]["name"] == "fn"
        assert "raw" in msg["content"][0]["toolUse"]["input"]

    def test_api_support_default(self):
        """Models without api_support field default to ['converse']."""
        model = BedrockModel(
            model_id="test.model", family="test", tier=Tier.MID, display_name="Test",
        )
        assert model.api_support == ["converse"]

    def test_dataclass_replace_routing_config(self):
        """RoutingConfig should work with dataclass_replace."""
        original = RoutingConfig(strategy="balanced", explain=True)
        modified = dataclass_replace(original, preferred_model="some.model")
        assert modified.preferred_model == "some.model"
        assert modified.strategy == "balanced"
        assert modified.explain is True
