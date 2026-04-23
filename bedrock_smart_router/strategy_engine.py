"""Routing strategy engine.

Provides cost-optimized, latency-optimized, quality-optimized, and
balanced strategies.  Each strategy scores candidate models and returns
the best match for the given request analysis.

When a ``MetricsStore`` is provided, strategies blend real historical
data (latency, quality, error rate) with tier-based heuristics.  With
no metrics store, pure heuristics are used — sensible defaults that
improve as data accumulates.
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

# Minimum samples before we trust historical latency over the heuristic
_MIN_LATENCY_SAMPLES = 5
# Rough upper bound for latency normalisation (ms)
_MAX_LATENCY_MS = 5000.0


@dataclass
class StrategyResult:
    """Outcome of a strategy evaluation."""

    selected_model: BedrockModel
    scores: dict[str, dict[str, float]]  # model_id -> {cost, latency, quality, composite}
    fallback_chain: list[BedrockModel]


# ── Shared scoring helpers ──────────────────────────────────────────

def _latency_score(
    model: BedrockModel,
    metrics: Any | None,
    analysis: RequestAnalysis,
) -> float:
    """Compute a latency score (0–1, higher = faster).

    Uses real P50 latency from the metrics store when available (5+
    samples), otherwise falls back to the tier heuristic.
    """
    heuristic = 1.0 - _TIER_LATENCY_INDEX.get(model.tier, 0.5)

    if metrics is not None and metrics.sample_count >= _MIN_LATENCY_SAMPLES:
        # Real data: normalise against a rough max
        real = max(0.0, 1.0 - metrics.avg_latency_ms / _MAX_LATENCY_MS)
        return real

    score = heuristic
    # Bonus for CRIS availability (cross-region = less queue time)
    if model.is_cris_available:
        score = min(1.0, score + 0.05)
    # Bonus for prompt caching support (faster TTFT on multi-turn)
    if model.capabilities.prompt_caching and analysis.is_multi_turn:
        score = min(1.0, score + 0.05)
    return score


def _quality_score(
    model: BedrockModel,
    metrics: Any | None,
) -> float:
    """Compute a quality score (0–1, higher = better).

    Uses historical avg_quality_score from the metrics store when
    available (5+ samples), otherwise falls back to the tier heuristic.
    Penalises models with high error rates.
    """
    heuristic = TIER_QUALITY_HEURISTIC.get(model.tier, 0.5)

    if metrics is None or metrics.sample_count == 0:
        return heuristic

    # Penalise for errors even without quality scores
    if metrics.avg_quality_score is None:
        error_penalty = metrics.error_rate * 0.2
        return max(0.0, heuristic - error_penalty)

    historical = metrics.avg_quality_score

    if metrics.sample_count >= 20:
        score = historical
    elif metrics.sample_count >= 5:
        trust = (metrics.sample_count - 5) / 15.0
        score = heuristic * (1 - trust) + historical * trust
    else:
        score = heuristic

    # Penalise high error rates
    if metrics.error_rate > 0:
        score *= (1.0 - metrics.error_rate * 0.5)
    return score


def _cost_score(
    model: BedrockModel,
    analysis: RequestAnalysis,
    max_cost: float,
) -> float:
    """Compute a cost score (0–1, higher = cheaper)."""
    cost_raw = model.pricing.estimate_cost(
        analysis.estimated_input_tokens,
        analysis.estimated_output_tokens,
    )
    return 1.0 - min(1.0, cost_raw / max_cost)


def _max_cost_for_candidates(
    candidates: list[BedrockModel],
    analysis: RequestAnalysis,
) -> float:
    """Compute the max estimated cost across all candidates (for normalisation)."""
    return max(
        (
            c.pricing.estimate_cost(
                analysis.estimated_input_tokens,
                analysis.estimated_output_tokens,
            )
            for c in candidates
        ),
        default=0.001,
    ) or 0.001


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

    def __init__(self, metrics_store: Any | None = None) -> None:
        self._metrics = metrics_store

    def select(
        self,
        candidates: list[BedrockModel],
        analysis: RequestAnalysis,
    ) -> StrategyResult:
        all_metrics = {}
        if self._metrics is not None:
            all_metrics = self._metrics.get_all_metrics(window_seconds=3600)

        max_cost = _max_cost_for_candidates(candidates, analysis)
        scores: dict[str, dict[str, float]] = {}

        for m in candidates:
            mm = all_metrics.get(m.model_id)
            cs = _cost_score(m, analysis, max_cost)
            ls = _latency_score(m, mm, analysis)
            qs = _quality_score(m, mm)

            scores[m.model_id] = {
                "cost": round(cs, 4),
                "latency": round(ls, 4),
                "quality": round(qs, 4),
                "composite": round(cs, 4),  # Cost is the only factor
            }

        best = max(candidates, key=lambda m: scores[m.model_id]["composite"])
        return StrategyResult(
            selected_model=best,
            scores=scores,
            fallback_chain=self._build_fallback_chain(best, candidates, scores),
        )


# ── Latency-Optimized Strategy ─────────────────────────────────────

class LatencyOptimizedStrategy(RoutingStrategy):
    """Route to the model with lowest expected latency.

    Uses real P50 latency from the metrics store when available,
    otherwise falls back to tier-based heuristics.
    """

    name = "latency-optimized"

    def __init__(self, metrics_store: Any | None = None) -> None:
        self._metrics = metrics_store

    def select(
        self,
        candidates: list[BedrockModel],
        analysis: RequestAnalysis,
    ) -> StrategyResult:
        all_metrics = {}
        if self._metrics is not None:
            all_metrics = self._metrics.get_all_metrics(window_seconds=3600)

        max_cost = _max_cost_for_candidates(candidates, analysis)
        scores: dict[str, dict[str, float]] = {}

        for m in candidates:
            mm = all_metrics.get(m.model_id)
            cs = _cost_score(m, analysis, max_cost)
            ls = _latency_score(m, mm, analysis)
            qs = _quality_score(m, mm)

            scores[m.model_id] = {
                "cost": round(cs, 4),
                "latency": round(ls, 4),
                "quality": round(qs, 4),
                "composite": round(ls, 4),  # Latency is the only factor
            }

        best = max(candidates, key=lambda m: scores[m.model_id]["composite"])
        return StrategyResult(
            selected_model=best,
            scores=scores,
            fallback_chain=self._build_fallback_chain(best, candidates, scores),
        )


# ── Balanced Strategy ───────────────────────────────────────────────

class BalancedStrategy(RoutingStrategy):
    """Composite score: w_cost * cost + w_latency * latency + w_quality * quality.

    Uses real historical data from the metrics store when available,
    blending with tier heuristics for models with insufficient data.
    """

    name = "balanced"

    def __init__(
        self,
        cost_weight: float = 0.4,
        latency_weight: float = 0.3,
        quality_weight: float = 0.3,
        metrics_store: Any | None = None,
    ) -> None:
        self.cost_weight = cost_weight
        self.latency_weight = latency_weight
        self.quality_weight = quality_weight
        self._metrics = metrics_store

    def select(
        self,
        candidates: list[BedrockModel],
        analysis: RequestAnalysis,
    ) -> StrategyResult:
        all_metrics = {}
        if self._metrics is not None:
            all_metrics = self._metrics.get_all_metrics(window_seconds=3600)

        max_cost = _max_cost_for_candidates(candidates, analysis)
        scores: dict[str, dict[str, float]] = {}

        for m in candidates:
            mm = all_metrics.get(m.model_id)
            cs = _cost_score(m, analysis, max_cost)
            ls = _latency_score(m, mm, analysis)
            qs = _quality_score(m, mm)

            if m.capabilities.prompt_caching and analysis.is_multi_turn:
                cs = min(1.0, cs + 0.05)

            composite = (
                self.cost_weight * cs
                + self.latency_weight * ls
                + self.quality_weight * qs
            )

            scores[m.model_id] = {
                "cost": round(cs, 4),
                "latency": round(ls, 4),
                "quality": round(qs, 4),
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
            ``"quality-optimized"``, ``"balanced"``, or a custom name.
        weights: For the balanced strategy, override the default
            ``{cost, latency, quality}`` weights.
        metrics_store: The ``MetricsStore`` for historical data.
            Passed to all strategies so they can use real latency,
            quality, and error rate data when available.
    """
    if name == "balanced":
        return BalancedStrategy(
            cost_weight=(weights or {}).get("cost", 0.4),
            latency_weight=(weights or {}).get("latency", 0.3),
            quality_weight=(weights or {}).get("quality", 0.3),
            metrics_store=metrics_store,
        )

    if name == "quality-optimized":
        from bedrock_smart_router.quality_strategy import QualityOptimizedStrategy
        return QualityOptimizedStrategy(metrics_store=metrics_store)

    if name == "cost-optimized":
        return CostOptimizedStrategy(metrics_store=metrics_store)

    if name == "latency-optimized":
        return LatencyOptimizedStrategy(metrics_store=metrics_store)

    cls = BUILTIN_STRATEGIES.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown strategy '{name}'. "
            f"Available: {list(BUILTIN_STRATEGIES.keys()) + ['quality-optimized']}"
        )
    # Custom strategies may not accept metrics_store — try with, fall back without
    try:
        return cls(metrics_store=metrics_store)
    except TypeError:
        return cls()
