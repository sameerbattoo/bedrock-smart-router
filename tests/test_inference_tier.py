# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the latency mode selector."""

from bedrock_smart_router.inference_tier import InferenceTierConfig, InferenceTierSelector
from bedrock_smart_router.models import (
    BedrockModel, Complexity, ModelCapabilities, ModelPricing,
    RequestAnalysis, Tier,
)


def _model(modes: list[str]) -> BedrockModel:
    return BedrockModel(
        model_id="test-model", family="test", tier=Tier.MID,
        display_name="Test", capabilities=ModelCapabilities(),
        max_input_tokens=128_000, max_output_tokens=4096,
        pricing=ModelPricing(input_per_1k=0.003, output_per_1k=0.015),
        supported_latency_modes=modes,
    )


def _analysis(complexity: str = "moderate", streaming: bool = True) -> RequestAnalysis:
    return RequestAnalysis(
        complexity=Complexity(complexity),
        estimated_input_tokens=1000,
        estimated_output_tokens=500,
        requires_streaming=streaming,
    )


class TestInferenceTierSelector:
    def test_default_is_standard(self):
        sel = InferenceTierSelector()
        m = _model(["standard", "optimized"])
        assert sel.select_tier(m, _analysis()) == "standard"

    def test_complex_gets_optimized(self):
        sel = InferenceTierSelector()
        m = _model(["standard", "optimized"])
        assert sel.select_tier(m, _analysis("complex")) == "optimized"

    def test_reasoning_gets_optimized(self):
        sel = InferenceTierSelector()
        m = _model(["standard", "optimized"])
        assert sel.select_tier(m, _analysis("reasoning")) == "optimized"

    def test_optimized_disabled(self):
        sel = InferenceTierSelector(InferenceTierConfig(allow_optimized=False))
        m = _model(["standard", "optimized"])
        assert sel.select_tier(m, _analysis("complex")) == "standard"

    def test_model_without_optimized(self):
        sel = InferenceTierSelector()
        m = _model(["standard"])
        assert sel.select_tier(m, _analysis("complex")) == "standard"

    def test_model_with_empty_modes(self):
        """Models like Anthropic that don't support performanceConfig at all."""
        sel = InferenceTierSelector()
        m = _model([])
        assert sel.select_tier(m, _analysis("complex")) == "standard"

    def test_disabled_returns_default(self):
        sel = InferenceTierSelector(InferenceTierConfig(enabled=False, default_tier="optimized"))
        m = _model(["standard", "optimized"])
        assert sel.select_tier(m, _analysis("complex")) == "optimized"

    def test_moderate_stays_standard(self):
        """Moderate complexity should not trigger optimized latency."""
        sel = InferenceTierSelector()
        m = _model(["standard", "optimized"])
        assert sel.select_tier(m, _analysis("moderate")) == "standard"
