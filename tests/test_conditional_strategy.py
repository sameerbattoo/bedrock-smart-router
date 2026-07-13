# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for conditional routing."""

from bedrock_smart_router.conditional_strategy import (
    ConditionRule,
    ConditionalRoutingConfig,
    ConditionalStrategy,
)
from bedrock_smart_router.models import (
    BedrockModel, Complexity, ModelCapabilities, ModelPricing,
    RequestAnalysis, Tier,
)


def _model(mid: str, family: str, tier: Tier, price: float) -> BedrockModel:
    return BedrockModel(
        model_id=mid, family=family, tier=tier, display_name=mid,
        capabilities=ModelCapabilities(tool_use=True),
        max_input_tokens=128_000, max_output_tokens=4096,
        pricing=ModelPricing(input_per_1k=price, output_per_1k=price * 4),
    )


CANDIDATES = [
    _model("anthropic-sonnet", "anthropic", Tier.MID, 0.003),
    _model("amazon-nova", "amazon", Tier.MICRO, 0.00004),
    _model("meta-llama", "meta", Tier.MID, 0.0007),
]

ANALYSIS = RequestAnalysis(
    complexity=Complexity.MODERATE,
    estimated_input_tokens=500,
    estimated_output_tokens=200,
)


class TestConditionalStrategy:
    def test_matching_rule_filters_family(self):
        config = ConditionalRoutingConfig(
            rules=[
                ConditionRule(
                    condition={"user_tier": "enterprise"},
                    family="anthropic",
                ),
            ],
            default_strategy="cost-optimized",
        )
        strategy = ConditionalStrategy(config)
        result = strategy.select(
            CANDIDATES, ANALYSIS, metadata={"user_tier": "enterprise"}
        )
        assert result.selected_model.family == "anthropic"

    def test_no_match_uses_default(self):
        config = ConditionalRoutingConfig(
            rules=[
                ConditionRule(
                    condition={"user_tier": "enterprise"},
                    family="anthropic",
                ),
            ],
            default_strategy="cost-optimized",
        )
        strategy = ConditionalStrategy(config)
        result = strategy.select(
            CANDIDATES, ANALYSIS, metadata={"user_tier": "free"}
        )
        # Default is cost-optimized, should pick cheapest
        assert result.selected_model.model_id == "amazon-nova"

    def test_strategy_override(self):
        config = ConditionalRoutingConfig(
            rules=[
                ConditionRule(
                    condition={"department": "research"},
                    strategy="latency-optimized",
                ),
            ],
        )
        strategy = ConditionalStrategy(config)
        result = strategy.select(
            CANDIDATES, ANALYSIS, metadata={"department": "research"}
        )
        # Latency-optimized prefers smaller/faster models
        assert result.selected_model is not None

    def test_explicit_model_list(self):
        config = ConditionalRoutingConfig(
            rules=[
                ConditionRule(
                    condition={"region": "eu"},
                    models=["meta-llama"],
                ),
            ],
        )
        strategy = ConditionalStrategy(config)
        result = strategy.select(
            CANDIDATES, ANALYSIS, metadata={"region": "eu"}
        )
        assert result.selected_model.model_id == "meta-llama"

    def test_first_matching_rule_wins(self):
        config = ConditionalRoutingConfig(
            rules=[
                ConditionRule(
                    condition={"tier": "premium"},
                    family="anthropic",
                ),
                ConditionRule(
                    condition={"tier": "premium"},
                    family="meta",
                ),
            ],
        )
        strategy = ConditionalStrategy(config)
        result = strategy.select(
            CANDIDATES, ANALYSIS, metadata={"tier": "premium"}
        )
        # First rule wins — anthropic
        assert result.selected_model.family == "anthropic"

    def test_empty_metadata(self):
        config = ConditionalRoutingConfig(
            rules=[
                ConditionRule(condition={"x": "y"}, family="anthropic"),
            ],
            default_strategy="cost-optimized",
        )
        strategy = ConditionalStrategy(config)
        result = strategy.select(CANDIDATES, ANALYSIS, metadata=None)
        # No match, uses default
        assert result.selected_model.model_id == "amazon-nova"
