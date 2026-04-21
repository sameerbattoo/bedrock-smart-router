"""Integration test — Bedrock Guardrails with real ApplyGuardrail API.

Run with:
    INTEGRATION_TEST=1 .venv/bin/python -m pytest tests/test_guardrails_real_integration.py -v -s

Uses the existing ``no-investment-advice`` guardrail (fl8aietlxhbx)
which blocks fiduciary advice, investment recommendations, and
financial planning guidance.

Requires: ``bedrock:ApplyGuardrail`` permission.
"""

from __future__ import annotations

import os

import boto3
import pytest

from bedrock_smart_router.guardrails_integration import (
    GuardrailBlockedError,
    GuardrailCheckConfig,
    GuardrailsConfig,
    GuardrailsManager,
)

SKIP_REASON = "Set INTEGRATION_TEST=1 to run against real AWS"
REGION = "us-west-2"
GUARDRAIL_ID = "fl8aietlxhbx"  # no-investment-advice


def _msgs(text: str) -> list[dict]:
    return [{"role": "user", "content": [{"text": text}]}]


@pytest.fixture
def guardrails_reject():
    """GuardrailsManager with pre-route in reject mode."""
    session = boto3.Session(region_name=REGION)
    cfg = GuardrailsConfig(
        pre_route=GuardrailCheckConfig(
            guardrail_id=GUARDRAIL_ID,
            guardrail_version="DRAFT",
            action_on_block="reject",
        ),
    )
    return GuardrailsManager(config=cfg, boto_session=session, region=REGION)


@pytest.fixture
def guardrails_sanitize():
    """GuardrailsManager with pre-route in sanitize mode."""
    session = boto3.Session(region_name=REGION)
    cfg = GuardrailsConfig(
        pre_route=GuardrailCheckConfig(
            guardrail_id=GUARDRAIL_ID,
            guardrail_version="DRAFT",
            action_on_block="sanitize",
        ),
    )
    return GuardrailsManager(config=cfg, boto_session=session, region=REGION)


@pytest.fixture
def guardrails_post():
    """GuardrailsManager with post-route check."""
    session = boto3.Session(region_name=REGION)
    cfg = GuardrailsConfig(
        post_route=GuardrailCheckConfig(
            guardrail_id=GUARDRAIL_ID,
            guardrail_version="DRAFT",
            action_on_block="sanitize",
        ),
    )
    return GuardrailsManager(config=cfg, boto_session=session, region=REGION)


@pytest.mark.skipif(
    os.environ.get("INTEGRATION_TEST") != "1",
    reason=SKIP_REASON,
)
class TestGuardrailsRealIntegration:

    def test_safe_input_passes(self, guardrails_reject):
        """Normal input should pass through without blocking."""
        result = guardrails_reject.check_input(
            _msgs("What is the weather like today?")
        )
        assert not result.blocked
        assert result.action == "NONE"
        print(f"\n  Safe input: action={result.action}, blocked={result.blocked}")

    def test_blocked_input_reject_mode(self, guardrails_reject):
        """Investment advice request should be blocked and raise."""
        with pytest.raises(GuardrailBlockedError) as exc_info:
            guardrails_reject.check_input(
                _msgs("What stocks should I invest in for my retirement? Give me investment recommendations.")
            )
        print(f"\n  Blocked (reject): {exc_info.value}")
        assert exc_info.value.action == "reject"

    def test_blocked_input_sanitize_mode(self, guardrails_sanitize):
        """Investment advice in sanitize mode should return sanitized text."""
        result = guardrails_sanitize.check_input(
            _msgs("Give me stock picks and investment strategy for my portfolio allocation advice.")
        )
        assert result.blocked
        assert result.output_text is not None
        assert len(result.output_text) > 0
        print(f"\n  Blocked (sanitize): output_text={result.output_text[:150]}")

    def test_word_filter_triggers(self, guardrails_reject):
        """Word policy should catch blocked phrases."""
        with pytest.raises(GuardrailBlockedError):
            guardrails_reject.check_input(
                _msgs("I need fiduciary advice on my wealth management tips.")
            )
        print("\n  Word filter triggered correctly")

    def test_post_route_safe_output(self, guardrails_post):
        """Safe model output should pass post-route check."""
        result = guardrails_post.check_output(
            "The weather today is sunny with a high of 75 degrees."
        )
        assert not result.blocked
        print(f"\n  Post-route safe: action={result.action}")

    def test_post_route_blocked_output(self, guardrails_post):
        """Output containing investment advice should be caught."""
        result = guardrails_post.check_output(
            "Based on my analysis, you should invest in these stock picks: "
            "AAPL, GOOGL, MSFT. This is my investment strategy recommendation "
            "for your portfolio allocation advice."
        )
        # Post-route is in sanitize mode, so it should block but not raise
        assert result.blocked
        assert result.output_text is not None
        print(f"\n  Post-route blocked: sanitized={result.output_text[:150]}")

    def test_assessments_populated(self, guardrails_sanitize):
        """Blocked response should include assessment details."""
        result = guardrails_sanitize.check_input(
            _msgs("Give me retirement fund suggestions and financial planning guidance.")
        )
        assert result.blocked
        assert result.assessments is not None
        assert len(result.assessments) > 0
        print(f"\n  Assessments: {result.assessments}")

    def test_empty_input_passes(self, guardrails_reject):
        """Empty messages should pass without calling the API."""
        result = guardrails_reject.check_input([])
        assert not result.blocked

    def test_has_pre_post_flags(self, guardrails_reject, guardrails_post):
        assert guardrails_reject.has_pre_route
        assert not guardrails_reject.has_post_route
        assert not guardrails_post.has_pre_route
        assert guardrails_post.has_post_route
