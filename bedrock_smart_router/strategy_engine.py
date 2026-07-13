# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

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


@dataclass
class StrategyExplanation:
    """Structured explanation for the routing decision JSON.

    Returned by ``RoutingStrategy.explain()`` so the router can include
    strategy-specific reasoning in the decision output without needing
    to know the strategy's internals.
    """

    reason: str
    weights: dict[str, float] | None = None
    custom_dimensions: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class StrategyContext:
    """Shared context passed to custom strategy scoring and filtering methods.

    Provides access to the full candidate list, request analysis, and
    any metadata passed via the router config (e.g., approved model lists,
    compliance tiers, tenant info).
    """

    candidates: list[BedrockModel]
    analysis: RequestAnalysis
    metadata: dict[str, Any] = field(default_factory=dict)


# ── Shared scoring helpers ──────────────────────────────────────────

# Minimum score floor for cost and latency dimensions.
#
# WHY: With min-max normalization, the most expensive (or slowest) model
# in a small candidate pool always scores 0.0.  In the balanced strategy
# (0.4×cost + 0.3×latency + 0.3×quality), multiplying by 0 on 70% of
# the weight means quality becomes irrelevant — a high-quality expensive
# model can NEVER win, even when it's only marginally more expensive.
#
# Example: REASONING tier with 2 models (Kimi $0.60 vs Opus $4.50).
# Without floor: Opus cost=0.0, latency=0.0 → composite dominated by
# quality alone at 30% weight → always loses.
# With floor: Opus cost=0.133, latency=0.10 → quality (0.86) can
# compensate, giving a fair composite comparison.
#
# The 0.10 value ensures even the worst model on a dimension still
# contributes 10% credit, preventing the "multiply by zero" collapse.
_SCORE_FLOOR = 0.10

def _latency_score(
    model: BedrockModel,
    metrics: Any | None,
    analysis: RequestAnalysis,
) -> float:
    """Compute a latency score (0–1, higher = faster).

    Uses real P50 latency from the metrics store when available (5+
    samples) with ratio-based normalization against the fastest observed
    model. Otherwise falls back to the tier heuristic.
    Floor of _SCORE_FLOOR ensures no model ever scores zero.
    """
    heuristic = 1.0 - _TIER_LATENCY_INDEX.get(model.tier, 0.5)

    if metrics is not None and metrics.sample_count >= _MIN_LATENCY_SAMPLES:
        # Real data available — store raw latency for ratio normalization
        # (actual ratio computation happens in select() with all candidates)
        # For now return the raw-normalized value; the select() methods
        # that have access to all candidates will override if needed.
        real = max(_SCORE_FLOOR, 1.0 - metrics.avg_latency_ms / _MAX_LATENCY_MS)
        return real

    score = heuristic
    # Bonus for CRIS availability (cross-region = less queue time)
    if model.is_cris_available:
        score = min(1.0, score + 0.05)
    # Bonus for prompt caching support (faster TTFT on multi-turn)
    if model.capabilities.prompt_caching and analysis.is_multi_turn:
        score = min(1.0, score + 0.05)
    return max(_SCORE_FLOOR, score)


def _latency_score_ratio(
    model: BedrockModel,
    metrics: Any | None,
    analysis: RequestAnalysis,
    fastest_latency_ms: float | None,
) -> float:
    """Compute latency score using ratio normalization against fastest candidate.

    WHY RATIO: When real metrics are available, models in the same tier
    can have vastly different latencies (e.g. Opus 4.7 at 4.5s vs Kimi K2
    at 22s).  The old fixed-max normalization (1 - latency/5000ms) would
    give both a score near 0 since both exceed 5000ms.  Ratio-based
    scoring (fastest/model) gives proper differentiation:
    - Opus: 4500/4500 = 1.0 (fastest)
    - Kimi: 4500/22000 = 0.20 (5x slower → 5x lower score)

    Falls back to tier heuristic when no real data exists for any
    candidate in the pool.
    """
    if metrics is not None and metrics.sample_count >= _MIN_LATENCY_SAMPLES:
        if fastest_latency_ms and fastest_latency_ms > 0:
            return max(_SCORE_FLOOR, fastest_latency_ms / metrics.avg_latency_ms)
        return max(_SCORE_FLOOR, 1.0 - metrics.avg_latency_ms / _MAX_LATENCY_MS)

    # No real data — use tier heuristic
    score = 1.0 - _TIER_LATENCY_INDEX.get(model.tier, 0.5)
    if model.is_cris_available:
        score = min(1.0, score + 0.05)
    if model.capabilities.prompt_caching and analysis.is_multi_turn:
        score = min(1.0, score + 0.05)
    return max(_SCORE_FLOOR, score)


def _quality_score(
    model: BedrockModel,
    metrics: Any | None,
) -> float:
    """Compute a quality score (0–1, higher = better).

    Uses the model's quality_baseline from the catalog (Artificial
    Analysis Intelligence Index), normalized to 0-1 scale.
    Penalises models with high error rates from historical metrics.
    Models with quality_baseline=0 (no benchmark data) are penalized
    to prevent unknown-quality models from winning over proven ones.
    """
    # Normalize from AA Intelligence Index (0-60) to 0-1 scale
    # Max observed score is ~60 (GPT-5.5 xhigh)
    score = model.quality_baseline / 60.0

    # Penalize unknown quality: models with no benchmark data get a
    # strong negative score (-1.0) so they only win as a last resort.
    # With balanced weights (0.3 quality), this translates to -0.3 on composite,
    # ensuring any model with proven quality (even minimal) ranks above.
    if model.quality_baseline <= 0:
        score = -1.0

    # Penalise for high error rates if we have metrics
    if metrics is not None and metrics.sample_count > 0 and metrics.error_rate > 0:
        score *= (1.0 - metrics.error_rate * 0.5)

    return score


def _cost_score(
    model: BedrockModel,
    analysis: RequestAnalysis,
    min_cost: float,
) -> float:
    """Compute a cost score (0–1, higher = cheaper).

    Uses ratio-based normalization: ``min_cost / model_cost``.
    The cheapest model scores 1.0; more expensive models score
    proportionally lower but never below the floor (0.10).

    This avoids the "multiply by zero" problem of min-max normalization
    where the most expensive model always scores 0 regardless of how
    close it is to the cheapest.
    """
    cost_raw = model.pricing.estimate_cost(
        analysis.estimated_input_tokens,
        analysis.estimated_output_tokens,
    )
    if cost_raw <= 0:
        return 1.0
    return max(_SCORE_FLOOR, min_cost / cost_raw)


def _min_cost_for_candidates(
    candidates: list[BedrockModel],
    analysis: RequestAnalysis,
) -> float:
    """Compute the minimum estimated cost across all candidates (for ratio normalization)."""
    return min(
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
    """Abstract base for routing strategies.

    To implement a custom strategy, you have two paths:

    **Path 1 (Minimal)** — Override ``weights`` and ``score_model()``:
        The base class handles filtering, ranking, fallback chains,
        and explanation assembly. You just define how to score each
        model on your custom dimensions.

    **Path 2 (Full control)** — Override ``select()``:
        You own the entire selection pipeline. Use this when your
        logic can't be expressed as "score each model independently."

    Built-in scoring dimensions available to all strategies:
        - ``"quality"``: from ``model.quality_baseline`` (0–60 → 0–1)
        - ``"cost"``: inverse cost (cheaper = higher score)
        - ``"latency"``: tier-based heuristic or real metrics

    Required interface (abstract — must implement):
        - ``weights``: property returning ``{dimension: weight}`` dict
        - ``score_model()``: return scores for your custom dimensions

    Optional overrides:
        - ``filter_candidates()``: hard-gate filtering before scoring
        - ``explain_metadata()``: extra context for the decision JSON
        - ``select()``: full pipeline override (Path 2)
    """

    name: str = "base"

    # ── REQUIRED: Define your scoring weights ────────────────────────

    @property
    @abstractmethod
    def weights(self) -> dict[str, float]:
        """Scoring weights. Keys are dimension names, values are weights.

        Must include at least one dimension. Can mix built-in dimensions
        (``quality``, ``cost``, ``latency``) with custom ones.
        Weights should sum to approximately 1.0.

        Example::

            @property
            def weights(self):
                return {"compliance": 0.5, "quality": 0.3, "cost": 0.2}
        """
        ...

    # ── REQUIRED: Score your custom dimensions ───────────────────────

    @abstractmethod
    def score_model(
        self,
        model: BedrockModel,
        analysis: RequestAnalysis,
        context: StrategyContext,
    ) -> dict[str, float]:
        """Return scores for your CUSTOM dimensions only (0.0 to 1.0).

        You do NOT need to score ``quality``, ``cost``, or ``latency`` —
        the base class computes those automatically.

        Only return scores for dimensions that are YOUR additions.

        Args:
            model: The candidate model being scored.
            analysis: The analyzed request (complexity, tokens, flags).
            context: Shared context (config metadata, all candidates).

        Returns:
            Dict of ``{dimension_name: score}`` for your custom dimensions.

        Example::

            def score_model(self, model, analysis, context):
                tier = context.metadata.get("compliance_tier", "general")
                score = COMPLIANCE_SCORES.get(tier, {}).get(model.family, 0.5)
                return {"compliance": score}
        """
        ...

    # ── OPTIONAL: Hard-filter candidates before scoring ──────────────

    def filter_candidates(
        self,
        candidates: list[BedrockModel],
        analysis: RequestAnalysis,
        context: StrategyContext,
    ) -> tuple[list[BedrockModel], dict[str, Any]]:
        """Optional: Remove candidates that fail hard requirements.

        Override this to enforce approval lists, region locks, etc.
        Default: no filtering (all candidates pass).

        Returns:
            Tuple of ``(filtered_candidates, filter_metadata)``.
            ``filter_metadata`` is included in the explanation JSON.
        """
        return candidates, {}

    # ── OPTIONAL: Custom explanation metadata ────────────────────────

    def explain_metadata(
        self,
        result: StrategyResult,
        analysis: RequestAnalysis,
    ) -> dict[str, Any]:
        """Optional: Add strategy-specific metadata to the explanation JSON.

        Default: empty dict. Override to add audit context, rule traces, etc.
        """
        return {}

    # ── Selection pipeline (override only for Path 2) ────────────────

    def select(
        self,
        candidates: list[BedrockModel],
        analysis: RequestAnalysis,
    ) -> StrategyResult:
        """Full selection pipeline using weights + score_model().

        Override this ONLY if you need full control (Path 2).
        For most custom strategies, implementing ``weights`` and
        ``score_model()`` is sufficient.
        """
        context = StrategyContext(
            candidates=candidates,
            analysis=analysis,
            metadata=getattr(self, "_metadata", {}),
        )

        # Step 1: Filter
        eligible, self._filter_meta = self.filter_candidates(
            candidates, analysis, context
        )
        if not eligible:
            eligible = candidates  # Safety fallback

        # Step 2: Score all dimensions for each model
        min_cost = _min_cost_for_candidates(eligible, analysis)
        scores: dict[str, dict[str, float]] = {}

        for model in eligible:
            # Built-in dimensions (computed by base class)
            model_scores: dict[str, float] = {
                "quality": _quality_score(model, None),
                "cost": _cost_score(model, analysis, min_cost),
                "latency": _latency_score(model, None, analysis),
            }
            # Custom dimensions (computed by implementor)
            custom_scores = self.score_model(model, analysis, context)
            model_scores.update(custom_scores)

            # Composite = weighted sum (only dimensions in self.weights)
            composite = sum(
                self.weights.get(dim, 0.0) * model_scores.get(dim, 0.0)
                for dim in self.weights
            )
            model_scores["composite"] = round(composite, 4)

            scores[model.model_id] = {
                k: round(v, 4) for k, v in model_scores.items()
            }

        # Step 3: Rank and select
        ranked = sorted(
            eligible,
            key=lambda m: scores[m.model_id]["composite"],
            reverse=True,
        )
        selected = ranked[0]
        fallback_chain = self._build_fallback_chain(selected, ranked, scores)

        return StrategyResult(
            selected_model=selected,
            scores=scores,
            fallback_chain=fallback_chain,
        )

    # ── Explanation assembly ─────────────────────────────────────────

    def explain(
        self,
        result: StrategyResult,
        analysis: RequestAnalysis,
    ) -> StrategyExplanation:
        """Assemble explanation from scores + metadata.

        The base class builds a reasonable explanation automatically.
        Custom strategies can override ``explain_metadata()`` to add
        strategy-specific context without replacing the whole method.
        """
        selected = result.selected_model
        selected_scores = result.scores.get(selected.model_id, {})

        # Auto-generate reason from top scores
        score_parts = [
            f"{k}: {v:.2f}"
            for k, v in selected_scores.items()
            if k != "composite"
        ]
        reason = (
            f"Selected {selected.display_name} "
            f"(composite: {selected_scores.get('composite', 0):.4f}, "
            f"{', '.join(score_parts)})."
        )

        filter_meta = getattr(self, "_filter_meta", {})
        if filter_meta:
            reason += f" Filters: {filter_meta}"

        return StrategyExplanation(
            reason=reason,
            weights=self.weights,
            custom_dimensions=self.explain_metadata(result, analysis),
            metadata=filter_meta or None,
        )

    # ── Shared helpers ───────────────────────────────────────────────

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


# ── Shared scoring loop for built-in strategies ─────────────────────

def _score_and_select(
    candidates: list[BedrockModel],
    analysis: RequestAnalysis,
    metrics_store: Any | None,
    composite_fn: Any,
    strategy: RoutingStrategy,
) -> StrategyResult:
    """Shared scoring loop used by all built-in strategies.

    Computes cost/latency/quality scores for each candidate and applies
    the strategy-specific composite function to determine the best model.

    Args:
        candidates: Eligible models to score.
        analysis: The analyzed request.
        metrics_store: Optional metrics backend for real latency data.
        composite_fn: Callable(cost, latency, quality, model, analysis) → float
        strategy: The strategy instance (for fallback chain building).
    """
    all_metrics: dict = {}
    if metrics_store is not None:
        all_metrics = metrics_store.get_all_metrics(window_seconds=3600)

    min_cost = _min_cost_for_candidates(candidates, analysis)

    # Find fastest latency among candidates with real data
    fastest_latency_ms = None
    for m in candidates:
        mm = all_metrics.get(m.model_id)
        if mm and mm.sample_count >= _MIN_LATENCY_SAMPLES:
            if fastest_latency_ms is None or mm.avg_latency_ms < fastest_latency_ms:
                fastest_latency_ms = mm.avg_latency_ms

    scores: dict[str, dict[str, float]] = {}
    for m in candidates:
        mm = all_metrics.get(m.model_id)
        cs = _cost_score(m, analysis, min_cost)
        ls = _latency_score_ratio(m, mm, analysis, fastest_latency_ms)
        qs = _quality_score(m, mm)

        composite = composite_fn(cs, ls, qs, m, analysis)

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
        fallback_chain=strategy._build_fallback_chain(best, candidates, scores),
    )


# ── Cost-Optimized Strategy ─────────────────────────────────────────

class CostOptimizedStrategy(RoutingStrategy):
    """Route to the cheapest model that meets the complexity requirement."""

    name = "cost-optimized"

    @property
    def weights(self) -> dict[str, float]:
        return {"cost": 1.0}

    def score_model(self, model, analysis, context):
        return {}  # No custom dimensions — uses select() override

    def __init__(self, metrics_store: Any | None = None) -> None:
        self._metrics = metrics_store

    def select(
        self,
        candidates: list[BedrockModel],
        analysis: RequestAnalysis,
    ) -> StrategyResult:
        return _score_and_select(
            candidates, analysis, self._metrics,
            composite_fn=lambda cs, ls, qs, m, analysis: cs,
            strategy=self,
        )


# ── Latency-Optimized Strategy ─────────────────────────────────────

class LatencyOptimizedStrategy(RoutingStrategy):
    """Route to the model with lowest expected latency.

    Uses real P50 latency from the metrics store when available,
    otherwise falls back to tier-based heuristics.
    """

    name = "latency-optimized"

    @property
    def weights(self) -> dict[str, float]:
        return {"latency": 1.0}

    def score_model(self, model, analysis, context):
        return {}  # No custom dimensions — uses select() override

    def __init__(self, metrics_store: Any | None = None) -> None:
        self._metrics = metrics_store

    def select(
        self,
        candidates: list[BedrockModel],
        analysis: RequestAnalysis,
    ) -> StrategyResult:
        return _score_and_select(
            candidates, analysis, self._metrics,
            composite_fn=lambda cs, ls, qs, m, analysis: ls,
            strategy=self,
        )


# ── Balanced Strategy ───────────────────────────────────────────────

class BalancedStrategy(RoutingStrategy):
    """Composite score: w_cost * cost + w_latency * latency + w_quality * quality.

    Uses real historical data from the metrics store when available,
    blending with tier heuristics for models with insufficient data.
    """

    name = "balanced"

    @property
    def weights(self) -> dict[str, float]:
        return {
            "cost": self.cost_weight,
            "latency": self.latency_weight,
            "quality": self.quality_weight,
        }

    def score_model(self, model, analysis, context):
        return {}  # No custom dimensions — uses select() override

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
        def _balanced_composite(cs, ls, qs, m, analysis):
            # Slight boost for prompt-caching models in multi-turn conversations
            if m.capabilities.prompt_caching and analysis.is_multi_turn:
                cs = min(1.0, cs + 0.05)
            return self.cost_weight * cs + self.latency_weight * ls + self.quality_weight * qs

        return _score_and_select(
            candidates, analysis, self._metrics,
            composite_fn=_balanced_composite,
            strategy=self,
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
