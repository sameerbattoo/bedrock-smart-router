"""Tests for the IntentExtractor — auto-extraction of intent and variables."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from bedrock_smart_router.intent_extractor import (
    ExtractionResult,
    IntentExtractor,
    IntentExtractorConfig,
)


def _mock_converse_response(intent: str, variables: dict) -> dict:
    """Build a mock Bedrock converse response with JSON output."""
    return {
        "output": {
            "message": {
                "role": "assistant",
                "content": [
                    {"text": json.dumps({"intent": intent, "variables": variables})}
                ],
            }
        },
        "usage": {"inputTokens": 50, "outputTokens": 30},
    }


@pytest.fixture
def mock_client():
    client = MagicMock()
    client.converse.return_value = _mock_converse_response(
        "Count users by geography for a year with sales above a threshold",
        {"year": "2026", "sales_threshold": "200"},
    )
    return client


@pytest.fixture
def extractor(mock_client):
    ext = IntentExtractor(
        config=IntentExtractorConfig(model_id="amazon.nova-micro-v1:0"),
    )
    ext._client = mock_client
    return ext


class TestSingleTurnExtraction:
    def test_extract_returns_intent_and_variables(self, extractor):
        result = extractor.extract(
            "Count users by geo for 2026 with sales > $200"
        )
        assert result.intent == "Count users by geography for a year with sales above a threshold"
        assert result.variables == {"year": "2026", "sales_threshold": "200"}
        assert result.source == "single-turn"

    def test_extract_no_variables(self, extractor, mock_client):
        mock_client.converse.return_value = _mock_converse_response(
            "What is Amazon S3", {},
        )
        result = extractor.extract("What is Amazon S3?")
        assert result.intent == "What is Amazon S3"
        assert result.variables == {}

    def test_extract_caches_results(self, extractor, mock_client):
        extractor.extract("same query")
        extractor.extract("same query")
        assert mock_client.converse.call_count == 1  # Only one LLM call

    def test_extract_different_queries_not_cached(self, extractor, mock_client):
        extractor.extract("query A")
        extractor.extract("query B")
        assert mock_client.converse.call_count == 2


class TestMultiTurnExtraction:
    def test_extract_from_messages(self, extractor):
        messages = [
            {"role": "user", "content": [{"text": "show me users by geo"}]},
            {"role": "assistant", "content": [{"text": "Here are users..."}]},
            {"role": "user", "content": [{"text": "now for 2026 with sales > $200"}]},
        ]
        result = extractor.extract_from_messages(messages)
        assert result.intent == "Count users by geography for a year with sales above a threshold"
        assert result.variables == {"year": "2026", "sales_threshold": "200"}
        assert result.source == "multi-turn"

    def test_extract_from_messages_caches(self, extractor, mock_client):
        messages = [
            {"role": "user", "content": [{"text": "show me users by geo"}]},
            {"role": "user", "content": [{"text": "for 2026"}]},
        ]
        extractor.extract_from_messages(messages)
        extractor.extract_from_messages(messages)
        assert mock_client.converse.call_count == 1


class TestResponseParsing:
    def test_parse_json_response(self):
        response = _mock_converse_response("test intent", {"k": "v"})
        result = IntentExtractor._parse_response(response)
        assert result.intent == "test intent"
        assert result.variables == {"k": "v"}

    def test_parse_markdown_code_block(self):
        response = {
            "output": {
                "message": {
                    "content": [
                        {"text": '```json\n{"intent": "test", "variables": {}}\n```'}
                    ]
                }
            }
        }
        result = IntentExtractor._parse_response(response)
        assert result.intent == "test"

    def test_parse_invalid_json_fallback(self):
        response = {
            "output": {
                "message": {
                    "content": [{"text": "This is not JSON at all"}]
                }
            }
        }
        result = IntentExtractor._parse_response(response)
        assert result.intent == "This is not JSON at all"
        assert result.variables == {}

    def test_parse_numeric_variables_converted_to_string(self):
        response = _mock_converse_response("intent", {"year": 2026, "amount": 200})
        result = IntentExtractor._parse_response(response)
        assert result.variables == {"year": "2026", "amount": "200"}


class TestRetryLogic:
    def test_retries_on_throttle(self, extractor, mock_client):
        from botocore.exceptions import ClientError

        error_resp = {"Error": {"Code": "ThrottlingException", "Message": "Rate exceeded"}}
        mock_client.converse.side_effect = [
            ClientError(error_resp, "Converse"),
            _mock_converse_response("intent", {}),
        ]
        # Should succeed on second attempt
        with patch("bedrock_smart_router.intent_extractor.time.sleep"):
            result = extractor.extract("test")
        assert result.intent == "intent"
        assert mock_client.converse.call_count == 2

    def test_no_retry_on_validation_error(self, extractor, mock_client):
        from botocore.exceptions import ClientError

        error_resp = {"Error": {"Code": "ValidationException", "Message": "Bad request"}}
        mock_client.converse.side_effect = ClientError(error_resp, "Converse")
        with pytest.raises(ClientError):
            extractor.extract("test")
        assert mock_client.converse.call_count == 1

    def test_exhausts_retries(self, extractor, mock_client):
        from botocore.exceptions import ClientError

        error_resp = {"Error": {"Code": "ThrottlingException", "Message": "Rate exceeded"}}
        mock_client.converse.side_effect = ClientError(error_resp, "Converse")
        with patch("bedrock_smart_router.intent_extractor.time.sleep"):
            with pytest.raises(ClientError):
                extractor.extract("test")
        # 1 initial + 2 retries = 3 calls
        assert mock_client.converse.call_count == 3


class TestConversationFormatting:
    def test_format_conversation(self):
        messages = [
            {"role": "user", "content": [{"text": "Hello"}]},
            {"role": "assistant", "content": [{"text": "Hi there"}]},
            {"role": "user", "content": [{"text": "Help me"}]},
        ]
        text = IntentExtractor._format_conversation(messages)
        assert "User: Hello" in text
        assert "Assistant: Hi there" in text
        assert "User: Help me" in text

    def test_messages_to_text_deterministic(self):
        messages = [
            {"role": "user", "content": [{"text": "A"}]},
            {"role": "assistant", "content": [{"text": "B"}]},
        ]
        t1 = IntentExtractor._messages_to_text(messages)
        t2 = IntentExtractor._messages_to_text(messages)
        assert t1 == t2


class TestCacheEviction:
    def test_cache_evicts_oldest(self, mock_client):
        ext = IntentExtractor(
            config=IntentExtractorConfig(cache_max_entries=2),
        )
        ext._client = mock_client
        ext.extract("query1")
        ext.extract("query2")
        ext.extract("query3")  # Should evict query1
        assert len(ext._cache) == 2
