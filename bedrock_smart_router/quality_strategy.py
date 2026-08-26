# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

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

from bedrock_smart_router.metrics_store import MetricsStore, ModelMetrics
from bedrock_smart_router.models import BedrockModel, RequestAnalysis
from bedrock_smart_router.strategy_engine import (
    RoutingStrategy,
    StrategyResult,
    _cost_score,
    _latency_score_ratio,
    _min_cost_for_candidates,
    _quality_score,
)

logger = logging.getLogger(__name__)


class QualityOptimizedStrategy(RoutingStrategy):
    """Route to the model with highest historical quality scores.

    Requires a ``MetricsStore`` to read historical data from.  When
    no store is provided or a model has insufficient data, falls back
    to tier-based heuristics.

    The quality score also factors in error rate — a model with high
    quality but frequent failures gets penalised.
    """

    name = "quality-optimized"

    @property
    def weights(self) -> dict[str, float]:
        return {"quality": 1.0}

    def score_model(self, model, analysis, context):
        return {}  # No custom dimensions — uses select() override

    def __init__(self, metrics_store: MetricsStore | None = None) -> None:
        self._metrics = metrics_store

    def select(
        self,
        candidates: list[BedrockModel],
        analysis: RequestAnalysis,
    ) -> StrategyResult:
        all_metrics: dict[str, ModelMetrics] = {}
        if self._metrics is not None:
            all_metrics = self._metrics.get_all_metrics(window_seconds=86400)  # 24h

        min_cost = _min_cost_for_candidates(candidates, analysis)
        fastest_latency_ms = None
        for m in candidates:
            mm = all_metrics.get(m.model_id)
            if mm and mm.sample_count >= 5:
                if fastest_latency_ms is None or mm.avg_latency_ms < fastest_latency_ms:
                    fastest_latency_ms = mm.avg_latency_ms

        scores: dict[str, dict[str, float]] = {}

        for m in candidates:
            mm = all_metrics.get(m.model_id)
            cs = _cost_score(m, analysis, min_cost)
            ls = _latency_score_ratio(m, mm, analysis, fastest_latency_ms)
            qs = _quality_score(m, mm)

            scores[m.model_id] = {
                "cost": round(cs, 4),
                "latency": round(ls, 4),
                "quality": round(qs, 4),
                "composite": round(qs, 4),  # Quality is the primary factor
                "sample_count": float(mm.sample_count if mm else 0),
            }

        best = max(candidates, key=lambda m: scores[m.model_id]["composite"])
        return StrategyResult(
            selected_model=best,
            scores=scores,
            fallback_chain=self._build_fallback_chain(best, candidates, scores),
        )
