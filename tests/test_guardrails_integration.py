# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the guardrails integration."""

from unittest.mock import MagicMock, patch

import pytest

from bedrock_smart_router.guardrails_integration import (
    GuardrailBlockedError,
    GuardrailCheckConfig,
    GuardrailsConfig,
    GuardrailsManager,
)


def _msgs(text: str) -> list[dict]:
    return [{"role": "user", "content": [{"text": text}]}]


class TestGuardrailsManager:
    def test_no_config_passes_through(self):
        mgr = GuardrailsManager()
        result = mgr.check_input(_msgs("Hello"))
        assert not result.blocked
        assert result.action == "NONE"

    def test_pre_route_not_blocked(self):
        cfg = GuardrailsConfig(
            pre_route=GuardrailCheckConfig(guardrail_id="gr-123")
        )
        mgr = GuardrailsManager(config=cfg)
        mock_client = MagicMock()
        mock_client.apply_guardrail.return_value = {
            "action": "NONE",
            "outputs": [],
            "assessments": [],
        }
        mgr._client = mock_client

        result = mgr.check_input(_msgs("Hello"))
        assert not result.blocked
        mock_client.apply_guardrail.assert_called_once()

    def test_pre_route_blocked_reject(self):
        cfg = GuardrailsConfig(
            pre_route=GuardrailCheckConfig(
                guardrail_id="gr-123", action_on_block="reject"
            )
        )
        mgr = GuardrailsManager(config=cfg)
        mock_client = MagicMock()
        mock_client.apply_guardrail.return_value = {
            "action": "GUARDRAIL_INTERVENED",
            "outputs": [{"text": "Content blocked"}],
            "assessments": [{"topicPolicy": {"topics": [{"name": "harmful"}]}}],
        }
        mgr._client = mock_client

        with pytest.raises(GuardrailBlockedError):
            mgr.check_input(_msgs("Bad content"))

    def test_pre_route_blocked_sanitize(self):
        cfg = GuardrailsConfig(
            pre_route=GuardrailCheckConfig(
                guardrail_id="gr-123", action_on_block="sanitize"
            )
        )
        mgr = GuardrailsManager(config=cfg)
        mock_client = MagicMock()
        mock_client.apply_guardrail.return_value = {
            "action": "GUARDRAIL_INTERVENED",
            "outputs": [{"text": "Sanitized content"}],
            "assessments": [],
        }
        mgr._client = mock_client

        result = mgr.check_input(_msgs("PII content"))
        assert result.blocked
        assert result.output_text == "Sanitized content"
        # Should NOT raise — sanitize mode

    def test_post_route_check(self):
        cfg = GuardrailsConfig(
            post_route=GuardrailCheckConfig(guardrail_id="gr-456")
        )
        mgr = GuardrailsManager(config=cfg)
        mock_client = MagicMock()
        mock_client.apply_guardrail.return_value = {
            "action": "NONE",
            "outputs": [],
            "assessments": [],
        }
        mgr._client = mock_client

        result = mgr.check_output("Model response text")
        assert not result.blocked

    def test_api_failure_fails_open(self):
        cfg = GuardrailsConfig(
            pre_route=GuardrailCheckConfig(guardrail_id="gr-123")
        )
        mgr = GuardrailsManager(config=cfg)
        mock_client = MagicMock()
        mock_client.apply_guardrail.side_effect = Exception("API down")
        mgr._client = mock_client

        # Should not raise — fails open
        result = mgr.check_input(_msgs("Hello"))
        assert not result.blocked

    def test_empty_messages_passes(self):
        cfg = GuardrailsConfig(
            pre_route=GuardrailCheckConfig(guardrail_id="gr-123")
        )
        mgr = GuardrailsManager(config=cfg)
        result = mgr.check_input([])
        assert not result.blocked

    def test_has_pre_post_flags(self):
        mgr = GuardrailsManager(GuardrailsConfig(
            pre_route=GuardrailCheckConfig(guardrail_id="gr-1"),
        ))
        assert mgr.has_pre_route
        assert not mgr.has_post_route
