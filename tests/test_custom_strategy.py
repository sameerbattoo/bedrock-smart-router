"""Tests for custom strategy plugin interface."""

import pytest

from bedrock_smart_router.custom_strategy import (
    list_strategies,
    register_strategy,
    unregister_strategy,
)
from bedrock_smart_router.models import (
    BedrockModel, Complexity, ModelCapabilities, ModelPricing,
    RequestAnalysis, Tier,
)
from bedrock_smart_router.strategy_engine import (
    RoutingStrategy,
    StrategyResult,
    resolve_strategy,
)


class AlwaysFirstStrategy(RoutingStrategy):
    """Test strategy that always picks the first candidate."""

    name = "always-first"

    def select(self, candidates, analysis):
        best = candidates[0]
        return StrategyResult(
            selected_model=best,
            scores={best.model_id: {"composite": 1.0}},
            fallback_chain=candidates[1:2],
        )


CANDIDATES = [
    BedrockModel(
        model_id="model-a", family="test", tier=Tier.MID,
        display_name="A", capabilities=ModelCapabilities(),
        max_input_tokens=128_000, max_output_tokens=4096,
        pricing=ModelPricing(input_per_1k=0.003, output_per_1k=0.015),
    ),
    BedrockModel(
        model_id="model-b", family="test", tier=Tier.LITE,
        display_name="B", capabilities=ModelCapabilities(),
        max_input_tokens=128_000, max_output_tokens=4096,
        pricing=ModelPricing(input_per_1k=0.001, output_per_1k=0.005),
    ),
]

ANALYSIS = RequestAnalysis(complexity=Complexity.MODERATE)


class TestCustomStrategy:
    def teardown_method(self):
        # Clean up any registered strategies
        unregister_strategy("always-first")

    def test_register_and_resolve(self):
        register_strategy("always-first", AlwaysFirstStrategy)
        s = resolve_strategy("always-first")
        assert isinstance(s, AlwaysFirstStrategy)

    def test_registered_strategy_works(self):
        register_strategy("always-first", AlwaysFirstStrategy)
        s = resolve_strategy("always-first")
        result = s.select(CANDIDATES, ANALYSIS)
        assert result.selected_model.model_id == "model-a"

    def test_unregister(self):
        register_strategy("always-first", AlwaysFirstStrategy)
        assert unregister_strategy("always-first")
        with pytest.raises(ValueError):
            resolve_strategy("always-first")

    def test_unregister_nonexistent(self):
        assert not unregister_strategy("nonexistent")

    def test_register_non_strategy_raises(self):
        with pytest.raises(TypeError):
            register_strategy("bad", dict)  # type: ignore

    def test_list_strategies(self):
        names = list_strategies()
        assert "balanced" in names
        assert "cost-optimized" in names
        assert "quality-optimized" in names
