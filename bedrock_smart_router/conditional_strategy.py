"""Conditional routing — route based on request metadata.

Evaluates a list of condition rules in order and applies the first
match.  Conditions can override the strategy, restrict models, or
set specific model families.

Configuration example::

    conditional_routing:
      - condition: {"metadata.user_tier": "enterprise"}
        strategy: "quality-optimized"
      - condition: {"metadata.region": "eu"}
        family: "anthropic"
      - default:
          strategy: "cost-optimized"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from bedrock_smart_router.models import BedrockModel, RequestAnalysis
from bedrock_smart_router.strategy_engine import (
    RoutingStrategy,
    StrategyResult,
    resolve_strategy,
)

logger = logging.getLogger(__name__)


@dataclass
class ConditionRule:
    """A single conditional routing rule."""

    condition: dict[str, str]  # key-value pairs to match against metadata
    strategy: str | None = None
    family: str | None = None
    models: list[str] | None = None  # Explicit model IDs
    weights: dict[str, float] | None = None


@dataclass
class ConditionalRoutingConfig:
    """Ordered list of condition rules with a default fallback."""

    rules: list[ConditionRule] = field(default_factory=list)
    default_strategy: str = "balanced"
    default_weights: dict[str, float] | None = None


def _match_condition(
    condition: dict[str, str], metadata: dict[str, Any]
) -> bool:
    """Check if all key-value pairs in *condition* match *metadata*.

    Supports dotted keys like ``metadata.user_tier`` by flattening
    the metadata dict one level.
    """
    for key, expected in condition.items():
        # Strip leading "metadata." prefix if present
        clean_key = key.removeprefix("metadata.")
        actual = metadata.get(clean_key)
        if str(actual) != str(expected):
            return False
    return True


class ConditionalStrategy(RoutingStrategy):
    """Evaluate condition rules and delegate to the matching strategy."""

    name = "conditional"

    def __init__(self, config: ConditionalRoutingConfig) -> None:
        self.config = config

    def select(
        self,
        candidates: list[BedrockModel],
        analysis: RequestAnalysis,
        metadata: dict[str, Any] | None = None,
    ) -> StrategyResult:
        metadata = metadata or {}

        for rule in self.config.rules:
            if _match_condition(rule.condition, metadata):
                logger.debug("Condition matched: %s", rule.condition)
                filtered = candidates

                # Filter by family
                if rule.family:
                    filtered = [m for m in filtered if m.family == rule.family]

                # Filter by explicit model list
                if rule.models:
                    filtered = [
                        m for m in filtered if m.model_id in rule.models
                    ]

                if not filtered:
                    logger.warning(
                        "Condition %s matched but left no candidates, "
                        "using all",
                        rule.condition,
                    )
                    filtered = candidates

                strategy_name = rule.strategy or self.config.default_strategy
                strategy = resolve_strategy(
                    strategy_name, weights=rule.weights
                )
                return strategy.select(filtered, analysis)

        # No rule matched — use default
        strategy = resolve_strategy(
            self.config.default_strategy,
            weights=self.config.default_weights,
        )
        return strategy.select(candidates, analysis)
