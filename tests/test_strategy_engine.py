"""Tests for the strategy engine."""

from bedrock_smart_router.models import (
    BedrockModel,
    Complexity,
    ModelCapabilities,
    ModelPricing,
    RequestAnalysis,
    Tier,
)
from bedrock_smart_router.strategy_engine import (
    BalancedStrategy,
    CostOptimizedStrategy,
    LatencyOptimizedStrategy,
    resolve_strategy,
)


def _make_model(model_id: str, tier: Tier, input_price: float, quality_baseline: float = 0.0) -> BedrockModel:
    return BedrockModel(
        model_id=model_id,
        family="test",
        tier=tier,
        display_name=model_id,
        capabilities=ModelCapabilities(tool_use=True),
        max_input_tokens=128_000,
        max_output_tokens=4_096,
        pricing=ModelPricing(input_per_1k=input_price, output_per_1k=input_price * 4),
        quality_baseline=quality_baseline,
    )


CANDIDATES = [
    _make_model("cheap-micro", Tier.MICRO, 0.00004, quality_baseline=10.0),
    _make_model("mid-model", Tier.MID, 0.003, quality_baseline=44.0),
    _make_model("expensive-heavy", Tier.HEAVY, 0.015, quality_baseline=50.0),
]

SIMPLE_ANALYSIS = RequestAnalysis(
    complexity=Complexity.SIMPLE,
    complexity_score=0.1,
    estimated_input_tokens=500,
    estimated_output_tokens=200,
)

COMPLEX_ANALYSIS = RequestAnalysis(
    complexity=Complexity.COMPLEX,
    complexity_score=0.7,
    estimated_input_tokens=2000,
    estimated_output_tokens=1000,
)


class TestCostOptimized:
    def test_selects_cheapest(self):
        strategy = CostOptimizedStrategy()
        result = strategy.select(CANDIDATES, SIMPLE_ANALYSIS)
        assert result.selected_model.model_id == "cheap-micro"

    def test_returns_scores_for_all(self):
        strategy = CostOptimizedStrategy()
        result = strategy.select(CANDIDATES, SIMPLE_ANALYSIS)
        assert len(result.scores) == len(CANDIDATES)

    def test_fallback_chain_populated(self):
        strategy = CostOptimizedStrategy()
        result = strategy.select(CANDIDATES, SIMPLE_ANALYSIS)
        assert len(result.fallback_chain) > 0
        assert result.selected_model not in result.fallback_chain


class TestLatencyOptimized:
    def test_prefers_smaller_models(self):
        strategy = LatencyOptimizedStrategy()
        result = strategy.select(CANDIDATES, SIMPLE_ANALYSIS)
        # Micro tier has lowest latency index
        assert result.selected_model.tier == Tier.MICRO


class TestBalanced:
    def test_default_weights(self):
        strategy = BalancedStrategy()
        result = strategy.select(CANDIDATES, SIMPLE_ANALYSIS)
        # Should pick something — exact choice depends on weight balance
        assert result.selected_model is not None
        assert result.selected_model.model_id in [c.model_id for c in CANDIDATES]

    def test_quality_heavy_weights(self):
        strategy = BalancedStrategy(cost_weight=0.0, latency_weight=0.0, quality_weight=1.0)
        result = strategy.select(CANDIDATES, COMPLEX_ANALYSIS)
        # Quality baseline: heavy(50) > mid(44) > micro(10)
        assert result.selected_model.model_id == "expensive-heavy"

    def test_cost_heavy_weights(self):
        strategy = BalancedStrategy(cost_weight=1.0, latency_weight=0.0, quality_weight=0.0)
        result = strategy.select(CANDIDATES, SIMPLE_ANALYSIS)
        assert result.selected_model.model_id == "cheap-micro"


class TestResolveStrategy:
    def test_resolve_balanced(self):
        s = resolve_strategy("balanced")
        assert isinstance(s, BalancedStrategy)

    def test_resolve_cost(self):
        s = resolve_strategy("cost-optimized")
        assert isinstance(s, CostOptimizedStrategy)

    def test_resolve_latency(self):
        s = resolve_strategy("latency-optimized")
        assert isinstance(s, LatencyOptimizedStrategy)

    def test_resolve_balanced_with_weights(self):
        s = resolve_strategy("balanced", weights={"cost": 0.8, "latency": 0.1, "quality": 0.1})
        assert isinstance(s, BalancedStrategy)
        assert s.cost_weight == 0.8

    def test_resolve_unknown_raises(self):
        import pytest
        with pytest.raises(ValueError, match="Unknown strategy"):
            resolve_strategy("nonexistent")
