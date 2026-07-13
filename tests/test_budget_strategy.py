# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the budget-constrained strategy."""

import pytest

from bedrock_smart_router.budget_strategy import (
    BudgetConstrainedStrategy,
    BudgetExceededError,
    BudgetRule,
    BudgetTracker,
)
from bedrock_smart_router.models import (
    BedrockModel,
    Complexity,
    ModelCapabilities,
    ModelPricing,
    RequestAnalysis,
    Tier,
)


def _model(mid: str, tier: Tier, price: float) -> BedrockModel:
    return BedrockModel(
        model_id=mid, family="test", tier=tier, display_name=mid,
        capabilities=ModelCapabilities(tool_use=True),
        max_input_tokens=128_000, max_output_tokens=4096,
        pricing=ModelPricing(input_per_1k=price, output_per_1k=price * 4),
    )


CANDIDATES = [
    _model("cheap", Tier.MICRO, 0.00004),
    _model("mid", Tier.MID, 0.003),
    _model("expensive", Tier.HEAVY, 0.015),
]

ANALYSIS = RequestAnalysis(
    complexity=Complexity.MODERATE,
    estimated_input_tokens=1000,
    estimated_output_tokens=500,
)


class TestBudgetConstrainedStrategy:
    def test_filters_by_max_cost(self):
        rule = BudgetRule(max_cost_per_request=0.001)
        strategy = BudgetConstrainedStrategy(default_rule=rule)
        result = strategy.select(CANDIDATES, ANALYSIS)
        # Only cheap model fits under $0.001
        assert result.selected_model.model_id == "cheap"

    def test_reject_mode(self):
        rule = BudgetRule(max_cost_per_request=0.0000001, on_exceeded="reject")
        strategy = BudgetConstrainedStrategy(default_rule=rule)
        with pytest.raises(BudgetExceededError):
            strategy.select(CANDIDATES, ANALYSIS)

    def test_downgrade_mode(self):
        rule = BudgetRule(max_cost_per_request=0.0000001, on_exceeded="downgrade")
        strategy = BudgetConstrainedStrategy(default_rule=rule)
        # Should not raise — downgrades to cheapest
        result = strategy.select(CANDIDATES, ANALYSIS)
        assert result.selected_model.model_id == "cheap"

    def test_no_budget_passes_all(self):
        rule = BudgetRule()  # No limits
        strategy = BudgetConstrainedStrategy(default_rule=rule)
        result = strategy.select(CANDIDATES, ANALYSIS)
        assert result.selected_model is not None


class TestBudgetTracker:
    def test_spend_tracking(self):
        tracker = BudgetTracker()
        tracker.record_spend("user-1", 0.01)
        tracker.record_spend("user-1", 0.02)
        assert tracker.get_spend("user-1", 3600) == 0.03

    def test_budget_check_hourly(self):
        tracker = BudgetTracker()
        tracker.record_spend("user-1", 5.0)
        rule = BudgetRule(max_hourly_spend=4.0)
        reason = tracker.check_budget("user-1", rule)
        assert reason is not None
        assert "hourly" in reason

    def test_budget_check_ok(self):
        tracker = BudgetTracker()
        tracker.record_spend("user-1", 1.0)
        rule = BudgetRule(max_hourly_spend=10.0)
        assert tracker.check_budget("user-1", rule) is None
