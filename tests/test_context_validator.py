# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the context window validator."""

from bedrock_smart_router.context_validator import ContextValidator
from bedrock_smart_router.models import BedrockModel, ModelCapabilities, ModelPricing, Tier


def _model(max_input: int) -> BedrockModel:
    return BedrockModel(
        model_id="test-model",
        family="test",
        tier=Tier.MID,
        display_name="Test",
        max_input_tokens=max_input,
        max_output_tokens=4096,
    )


class TestContextValidator:
    def setup_method(self):
        self.validator = ContextValidator(safety_margin=0.05)

    def test_short_message_passes(self):
        msgs = [{"role": "user", "content": [{"text": "Hello"}]}]
        result = self.validator.validate(msgs, _model(128_000))
        assert result.valid
        assert result.headroom_pct > 0.9

    def test_long_message_fails(self):
        # ~50K tokens worth of text
        msgs = [{"role": "user", "content": [{"text": "x" * 200_000}]}]
        result = self.validator.validate(msgs, _model(4_096))
        assert not result.valid

    def test_filter_by_context(self):
        small = _model(4_096)
        large = _model(200_000)
        msgs = [{"role": "user", "content": [{"text": "x" * 40_000}]}]
        filtered = self.validator.filter_by_context([small, large], msgs)
        assert large in filtered
        assert small not in filtered

    def test_system_prompt_counted(self):
        msgs = [{"role": "user", "content": [{"text": "Hi"}]}]
        system = [{"text": "x" * 16_000}]  # ~4K tokens
        result = self.validator.validate(msgs, _model(4_096), system=system)
        assert not result.valid
