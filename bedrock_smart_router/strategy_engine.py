"""Routing strategy engine.

Provides cost-optimized, latency-optimized, quality-optimized, and
balanced strategies.  Each strategy scores candidate models and returns
the best match for the given request analysis.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from bedrock_smart_router.models import (
    BedrockModel,
    Complexity,
    RequestAnalysis,
    Tier,
)
from bedrock_smart_router.model_registry import (
    COMPLEXITY_MIN_TIER,
    TIER_QUALITY_HEURISTIC,
    ModelRegistry,
)

logger = logging.getLogger(__name__)

# Tier ordering for numeric comparisons
_TIER_ORDER: dict[Tier, int] = {
    Tier.MICRO: 0,
    Tier.LITE: 1,
    Tier.MID: 2,
    Tier.HEAVY: 3,
    Tier.REASONING: 4,
}

# Approximate relative cost index per tier (normalised 0-1, lower = cheaper)
_TIER_COST_INDEX: dict[Tier, float] = {
    Tier.MICRO: 0.05,
    Tier.LITE: 0.15,
    Tier.MID: 0.45,
    Tier.HEAVY: 0.80,
    Tier.REASONING: 1.00,
}

# Approximate relative latency index per tier (normalised 0-1, lower = faster)
_TIER_LATENCY_INDEX: dict[Tier, float] = {
    Tier.MICRO: 0.10,
    Tier.LITE: 0.25,
    Tier.MID: 0.50,
    Tier.HEAVY: 0.75,
    Tier.REASONING: 0.90,
}


@dataclass
class StrategyResult:
    """Outcome of a strategy evaluation."""

    selected_model: BedrockModel
    scores: dict[str, dict[str, float]]  # model_id -> {cost, latency, quality, composite}
    fallback_chain: list[BedrockModel]


# ── Base class ──────────────────────────────────────────────────────

class RoutingStrategy(ABC):
    """Abstract base for routing strategies."""

    name: str = "base"

    @abstractmethod
    def select(
        self,
        candidates: list[BedrockModel],
        analysis: RequestAnalysis,
    ) -> StrategyResult:
        ...

    def _build_fallback_chain(
        self,
        selected: BedrockModel,
        candidates: list[BedrockModel],
        scores: dict[str, dict[str, float]],
    ) -> list[BedrockModel]:
        """Build a fallback chain from remaining candidates sorted by score."""
        others = [c for c in candidates if c.model_id != selected.model_id]
        others.sort(
            key=lambda m: scores.get(m.model_id, {}).get("composite", 0),
            reverse=True,
        )
        return others[:5]  # Top 5 fallbacks


# ── Cost-Optimized Strategy ─────────────────────────────────────────

class CostOptimizedStrategy(RoutingStrategy):
    """Route to the cheapest model that meets the complexity requirement."""

    name = "cost-optimized"

    def select(
        self,
        candidates: list[BedrockModel],
        analysis: RequestAnalysis,
    ) -> StrategyResult:
        scores: dict[str, dict[str, float]] = {}
        est_tokens = analysis.estimated_input_tokens + analysis.estimated_output_tokens

        for m in candidates:
            cost_raw = m.pricing.estimate_cost(
                analysis.estimated_input_tokens,
                analysis.estimated_output_tokens,
            )
            # Normalise: lower cost = higher score
            max_cost = max(
                c.pricing.estimate_cost(
                    analysis.estimated_input_tokens,
                    analysis.estimated_output_tokens,
                )
                for c in candidates
            ) or 0.001
            cost_score = 1.0 - min(1.0, cost_raw / max_cost)
            quality_score = TIER_QUALITY_HEURISTIC.get(m.tier, 0.5)
            latency_score = 1.0 - _TIER_LATENCY_INDEX.get(m.tier, 0.5)

            scores[m.model_id] = {
                "cost": round(cost_score, 4),
                "latency": round(latency_score, 4),
                "quality": round(quality_score, 4),
                "composite": round(cost_score, 4),  # Cost is the only factor
            }

        best = max(candidates, key=lambda m: scores[m.model_id]["composite"])
        return StrategyResult(
            selected_model=best,
            scores=scores,
            fallback_chain=self._build_fallback_chain(best, candidates, scores),
        )


# ── Latency-Optimized Strategy ─────────────────────────────────────

class LatencyOptimizedStrategy(RoutingStrategy):
    """Route to the model with lowest expected latency."""

    name = "latency-optimized"

    def select(
        self,
        candidates: list[BedrockModel],
        analysis: RequestAnalysis,
    ) -> StrategyResult:
        scores: dict[str, dict[str, float]] = {}

        for m in candidates:
            latency_score = 1.0 - _TIER_LATENCY_INDEX.get(m.tier, 0.5)
            # Bonus for CRIS availability (cross-region = less queue time)
            if m.is_cris_available:
                latency_score = min(1.0, latency_score + 0.05)
            # Bonus for prompt caching support (faster TTFT)
            if m.capabilities.prompt_caching and analysis.is_multi_turn:
                latency_score = min(1.0, latency_score + 0.05)

            cost_raw = m.pricing.estimate_cost(
                analysis.estimated_input_tokens,
                analysis.estimated_output_tokens,
            )
            max_cost = max(
                c.pricing.estimate_cost(
                    analysis.estimated_input_tokens,
                    analysis.estimated_output_tokens,
                )
                for c in candidates
            ) or 0.001
            cost_score = 1.0 - min(1.0, cost_raw / max_cost)
            quality_score = TIER_QUALITY_HEURISTIC.get(m.tier, 0.5)

            scores[m.model_id] = {
                "cost": round(cost_score, 4),
                "latency": round(latency_score, 4),
                "quality": round(quality_score, 4),
                "composite": round(latency_score, 4),  # Latency is the only factor
            }

        best = max(candidates, key=lambda m: scores[m.model_id]["composite"])
        return StrategyResult(
            selected_model=best,
            scores=scores,
            fallback_chain=self._build_fallback_chain(best, candidates, scores),
        )


# ── Balanced Strategy ───────────────────────────────────────────────

class BalancedStrategy(RoutingStrategy):
    """Composite score: w_cost * cost + w_latency * latency + w_quality * quality."""

    name = "balanced"

    def __init__(
        self,
        cost_weight: float = 0.4,
        latency_weight: float = 0.3,
        quality_weight: float = 0.3,
    ) -> None:
        self.cost_weight = cost_weight
        self.latency_weight = latency_weight
        self.quality_weight = quality_weight

    def select(
        self,
        candidates: list[BedrockModel],
        analysis: RequestAnalysis,
    ) -> StrategyResult:
        scores: dict[str, dict[str, float]] = {}

        max_cost = max(
            (
                c.pricing.estimate_cost(
                    analysis.estimated_input_tokens,
                    analysis.estimated_output_tokens,
                )
                for c in candidates
            ),
            default=0.001,
        ) or 0.001

        for m in candidates:
            cost_raw = m.pricing.estimate_cost(
                analysis.estimated_input_tokens,
                analysis.estimated_output_tokens,
            )
            cost_score = 1.0 - min(1.0, cost_raw / max_cost)
            latency_score = 1.0 - _TIER_LATENCY_INDEX.get(m.tier, 0.5)
            quality_score = TIER_QUALITY_HEURISTIC.get(m.tier, 0.5)

            if m.is_cris_available:
                latency_score = min(1.0, latency_score + 0.03)
            if m.capabilities.prompt_caching and analysis.is_multi_turn:
                cost_score = min(1.0, cost_score + 0.05)

            composite = (
                self.cost_weight * cost_score
                + self.latency_weight * latency_score
                + self.quality_weight * quality_score
            )

            scores[m.model_id] = {
                "cost": round(cost_score, 4),
                "latency": round(latency_score, 4),
                "quality": round(quality_score, 4),
                "composite": round(composite, 4),
            }

        best = max(candidates, key=lambda m: scores[m.model_id]["composite"])
        return StrategyResult(
            selected_model=best,
            scores=scores,
            fallback_chain=self._build_fallback_chain(best, candidates, scores),
        )


# ── Strategy resolver ───────────────────────────────────────────────

BUILTIN_STRATEGIES: dict[str, type[RoutingStrategy]] = {
    "cost-optimized": CostOptimizedStrategy,
    "latency-optimized": LatencyOptimizedStrategy,
    "balanced": BalancedStrategy,
}


def resolve_strategy(
    name: str,
    *,
    weights: dict[str, float] | None = None,
    metrics_store: Any | None = None,
) -> RoutingStrategy:
    """Instantiate a strategy by name.

    Args:
        name: One of ``"cost-optimized"``, ``"latency-optimized"``,
            ``"quality-optimized"``, ``"balanced"``, or a custom class.
        weights: For the balanced strategy, override the default
            ``{cost, latency, quality}`` weights.
        metrics_store: For the quality-optimized strategy, the
            ``MetricsStore`` to read historical data from.
    """
    if name == "balanced" and weights:
        return BalancedStrategy(
            cost_weight=weights.get("cost", 0.4),
            latency_weight=weights.get("latency", 0.3),
            quality_weight=weights.get("quality", 0.3),
        )

    if name == "quality-optimized":
        from bedrock_smart_router.quality_strategy import QualityOptimizedStrategy
        return QualityOptimizedStrategy(metrics_store=metrics_store)

    cls = BUILTIN_STRATEGIES.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown strategy '{name}'. "
            f"Available: {list(BUILTIN_STRATEGIES.keys()) + ['quality-optimized']}"
        )
    return cls()
