"""Quality-optimized routing strategy.

Routes to the model with the highest historical quality scores from
your own evaluation / judge data stored in the metrics store.  Falls
back to tier-based heuristics when insufficient historical data exists
for a model.

This is the strategy that makes the router *learn* — the more you use
it, the better it gets at picking the right model for your workload.
"""

from __future__ import annotations

import logging
from typing import Any

from bedrock_smart_router.metrics_store import MetricsStore, ModelMetrics
from bedrock_smart_router.model_registry import TIER_QUALITY_HEURISTIC
from bedrock_smart_router.models import BedrockModel, RequestAnalysis, Tier
from bedrock_smart_router.strategy_engine import (
    RoutingStrategy,
    StrategyResult,
    _TIER_COST_INDEX,
    _TIER_LATENCY_INDEX,
)

logger = logging.getLogger(__name__)

# Minimum number of samples before we trust historical data over the
# tier heuristic.  Below this threshold we blend historical and
# heuristic scores.
MIN_SAMPLES_FULL_TRUST = 20
MIN_SAMPLES_PARTIAL = 5


def _quality_score_for_model(
    model: BedrockModel,
    metrics: ModelMetrics | None,
) -> float:
    """Compute a quality score blending historical data and tier heuristic.

    - 0 samples          → pure tier heuristic
    - 5–19 samples       → weighted blend (more samples = more trust in history)
    - 20+ samples        → pure historical score
    """
    heuristic = TIER_QUALITY_HEURISTIC.get(model.tier, 0.5)

    if metrics is None or metrics.sample_count == 0:
        return heuristic

    if metrics.avg_quality_score is None:
        # Records exist but none have quality scores — use heuristic
        # but give a small boost for low error rate
        error_penalty = metrics.error_rate * 0.2
        return max(0.0, heuristic - error_penalty)

    historical = metrics.avg_quality_score

    if metrics.sample_count >= MIN_SAMPLES_FULL_TRUST:
        return historical

    # Blend: linearly interpolate between heuristic and historical
    # based on how many samples we have
    trust = (metrics.sample_count - MIN_SAMPLES_PARTIAL) / (
        MIN_SAMPLES_FULL_TRUST - MIN_SAMPLES_PARTIAL
    )
    trust = max(0.0, min(1.0, trust))
    return heuristic * (1 - trust) + historical * trust


class QualityOptimizedStrategy(RoutingStrategy):
    """Route to the model with highest historical quality scores.

    Requires a ``MetricsStore`` to read historical data from.  When
    no store is provided or a model has insufficient data, falls back
    to tier-based heuristics.

    The quality score also factors in error rate — a model with high
    quality but frequent failures gets penalised.
    """

    name = "quality-optimized"

    def __init__(self, metrics_store: MetricsStore | None = None) -> None:
        self._metrics = metrics_store

    def select(
        self,
        candidates: list[BedrockModel],
        analysis: RequestAnalysis,
    ) -> StrategyResult:
        # Fetch historical metrics for all candidates in one call
        all_metrics: dict[str, ModelMetrics] = {}
        if self._metrics is not None:
            all_metrics = self._metrics.get_all_metrics(window_seconds=86400)  # 24h

        scores: dict[str, dict[str, float]] = {}

        # Compute max cost for normalisation
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
            model_metrics = all_metrics.get(m.model_id)

            # Quality: blend historical + heuristic
            quality = _quality_score_for_model(m, model_metrics)

            # Penalise models with high error rates
            if model_metrics and model_metrics.error_rate > 0:
                quality *= (1.0 - model_metrics.error_rate * 0.5)

            # Cost and latency for reference (not used in composite)
            cost_raw = m.pricing.estimate_cost(
                analysis.estimated_input_tokens,
                analysis.estimated_output_tokens,
            )
            cost_score = 1.0 - min(1.0, cost_raw / max_cost)
            latency_score = 1.0 - _TIER_LATENCY_INDEX.get(m.tier, 0.5)

            # Use historical latency if available
            if model_metrics and model_metrics.sample_count >= MIN_SAMPLES_PARTIAL:
                # Normalise: lower latency = higher score
                # Assume 5000ms as a rough max for normalisation
                latency_score = max(
                    0.0, 1.0 - model_metrics.avg_latency_ms / 5000
                )

            scores[m.model_id] = {
                "cost": round(cost_score, 4),
                "latency": round(latency_score, 4),
                "quality": round(quality, 4),
                "composite": round(quality, 4),  # Quality is the primary factor
                "sample_count": float(
                    model_metrics.sample_count if model_metrics else 0
                ),
            }

        best = max(candidates, key=lambda m: scores[m.model_id]["composite"])
        return StrategyResult(
            selected_model=best,
            scores=scores,
            fallback_chain=self._build_fallback_chain(best, candidates, scores),
        )
