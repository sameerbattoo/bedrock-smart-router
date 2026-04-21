"""Tests for the inference tier selector."""

from bedrock_smart_router.inference_tier import InferenceTierConfig, InferenceTierSelector
from bedrock_smart_router.models import (
    BedrockModel, Complexity, ModelCapabilities, ModelPricing,
    RequestAnalysis, Tier,
)


def _model(tiers: list[str]) -> BedrockModel:
    return BedrockModel(
        model_id="test-model", family="test", tier=Tier.MID,
        display_name="Test", capabilities=ModelCapabilities(),
        max_input_tokens=128_000, max_output_tokens=4096,
        pricing=ModelPricing(input_per_1k=0.003, output_per_1k=0.015),
        supported_inference_tiers=tiers,
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
        m = _model(["standard", "priority", "flex"])
        assert sel.select_tier(m, _analysis()) == "standard"

    def test_complex_gets_priority(self):
        sel = InferenceTierSelector()
        m = _model(["standard", "priority", "flex"])
        assert sel.select_tier(m, _analysis("complex")) == "priority"

    def test_reasoning_gets_priority(self):
        sel = InferenceTierSelector()
        m = _model(["standard", "priority"])
        assert sel.select_tier(m, _analysis("reasoning")) == "priority"

    def test_priority_disabled(self):
        sel = InferenceTierSelector(InferenceTierConfig(allow_priority=False))
        m = _model(["standard", "priority"])
        assert sel.select_tier(m, _analysis("complex")) == "standard"

    def test_model_without_priority(self):
        sel = InferenceTierSelector()
        m = _model(["standard", "flex"])
        assert sel.select_tier(m, _analysis("complex")) == "standard"

    def test_budget_triggers_flex(self):
        sel = InferenceTierSelector()
        m = _model(["standard", "flex"])
        # Non-streaming, simple, tight budget
        a = _analysis("simple", streaming=False)
        tier = sel.select_tier(m, a, max_cost_per_request=0.001)
        # Estimated cost for 1000 in + 500 out = 0.003 + 0.0075 = 0.0105
        # That's way over budget, so flex should be selected
        assert tier == "flex"

    def test_disabled_returns_default(self):
        sel = InferenceTierSelector(InferenceTierConfig(enabled=False, default_tier="priority"))
        m = _model(["standard", "priority", "flex"])
        assert sel.select_tier(m, _analysis("complex")) == "priority"

    def test_only_standard_available(self):
        sel = InferenceTierSelector()
        m = _model(["standard"])
        assert sel.select_tier(m, _analysis("complex")) == "standard"
