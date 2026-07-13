# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for tag-based routing."""

from bedrock_smart_router.models import (
    BedrockModel, Complexity, ModelCapabilities, ModelPricing,
    RequestAnalysis, Tier,
)
from bedrock_smart_router.tag_strategy import TagRoutingStrategy


def _model(mid: str, tier: Tier, price: float) -> BedrockModel:
    return BedrockModel(
        model_id=mid, family=mid.split("-")[0], tier=tier, display_name=mid,
        capabilities=ModelCapabilities(tool_use=True),
        max_input_tokens=128_000, max_output_tokens=4096,
        pricing=ModelPricing(input_per_1k=price, output_per_1k=price * 4),
    )


CANDIDATES = [
    _model("anthropic-sonnet", Tier.MID, 0.003),
    _model("anthropic-haiku", Tier.LITE, 0.001),
    _model("amazon-nova", Tier.MICRO, 0.00004),
]

ANALYSIS = RequestAnalysis(
    complexity=Complexity.MODERATE,
    estimated_input_tokens=500,
    estimated_output_tokens=200,
)


class TestTagRouting:
    def test_no_tags_uses_all(self):
        strategy = TagRoutingStrategy(tag_rules={"paid": ["anthropic-*"]})
        result = strategy.select(CANDIDATES, ANALYSIS, tags=None)
        assert result.selected_model is not None

    def test_tag_filters_candidates(self):
        strategy = TagRoutingStrategy(
            tag_rules={"paid": ["anthropic-*"]}
        )
        result = strategy.select(CANDIDATES, ANALYSIS, tags=["paid"])
        assert "anthropic" in result.selected_model.model_id

    def test_free_tier_tag(self):
        strategy = TagRoutingStrategy(
            tag_rules={"free": ["amazon-*"]}
        )
        result = strategy.select(CANDIDATES, ANALYSIS, tags=["free"])
        assert result.selected_model.model_id == "amazon-nova"

    def test_unknown_tag_uses_all(self):
        strategy = TagRoutingStrategy(tag_rules={"paid": ["anthropic-*"]})
        result = strategy.select(CANDIDATES, ANALYSIS, tags=["unknown-tag"])
        # No rules match, so all candidates are used
        assert result.selected_model is not None

    def test_multiple_tags_union(self):
        strategy = TagRoutingStrategy(
            tag_rules={
                "tier-a": ["anthropic-sonnet"],
                "tier-b": ["amazon-nova"],
            }
        )
        result = strategy.select(CANDIDATES, ANALYSIS, tags=["tier-a", "tier-b"])
        # Both anthropic-sonnet and amazon-nova are allowed
        assert result.selected_model.model_id in ("anthropic-sonnet", "amazon-nova")
