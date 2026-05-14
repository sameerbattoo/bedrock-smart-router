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

    Uses the model's quality_baseline from the catalog (Artificial
    Analysis Intelligence Index), normalized to 0-1 scale.
    Penalises models with high error rates from historical metrics.
    """
    # Normalize from AA Intelligence Index (0-60) to 0-1 scale
    # Max observed score is ~60 (GPT-5.5 xhigh)
    score = model.quality_baseline / 60.0

    # Penalise for high error rates if we have metrics
    if metrics is not None and metrics.sample_count > 0 and metrics.error_rate > 0:
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
        max_cost = _max_cost_for_candidates(eligible, analysis)
        scores: dict[str, dict[str, float]] = {}

        for model in eligible:
            # Built-in dimensions (computed by base class)
            model_scores: dict[str, float] = {
                "quality": _quality_score(model, None),
                "cost": _cost_score(model, analysis, max_cost),
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
