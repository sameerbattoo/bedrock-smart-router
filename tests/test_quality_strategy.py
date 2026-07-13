# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the quality-optimized routing strategy."""

import time

import pytest

from bedrock_smart_router.metrics_store import (
    InMemoryMetricsStore,
    RequestRecord,
)
from bedrock_smart_router.models import (
    BedrockModel,
    Complexity,
    ModelCapabilities,
    ModelPricing,
    RequestAnalysis,
    Tier,
)
from bedrock_smart_router.quality_strategy import (
    QualityOptimizedStrategy,
)
from bedrock_smart_router.strategy_engine import resolve_strategy, _quality_score


def _model(mid: str, tier: Tier, price: float, quality_baseline: float = 0.0) -> BedrockModel:
    return BedrockModel(
        model_id=mid, family="test", tier=tier, display_name=mid,
        capabilities=ModelCapabilities(tool_use=True),
        max_input_tokens=128_000, max_output_tokens=4096,
        pricing=ModelPricing(input_per_1k=price, output_per_1k=price * 4),
        quality_baseline=quality_baseline,
    )


CANDIDATES = [
    _model("cheap-micro", Tier.MICRO, 0.00004, quality_baseline=10.0),
    _model("mid-model", Tier.MID, 0.003, quality_baseline=44.0),
    _model("expensive-heavy", Tier.HEAVY, 0.015, quality_baseline=50.0),
]

ANALYSIS = RequestAnalysis(
    complexity=Complexity.MODERATE,
    estimated_input_tokens=1000,
    estimated_output_tokens=500,
)


class TestQualityScoreBlending:
    """Test the quality_baseline scoring logic."""

    def test_no_metrics_uses_baseline(self):
        model = _model("m", Tier.MID, 0.003, quality_baseline=44.0)
        score = _quality_score(model, None)
        assert score == pytest.approx(44.0 / 60.0)

    def test_zero_baseline_returns_zero(self):
        model = _model("m", Tier.MID, 0.003, quality_baseline=0.0)
        score = _quality_score(model, None)
        assert score == -1.0  # Zero quality_baseline gets strong penalty

    def test_high_error_rate_penalises(self):
        from bedrock_smart_router.metrics_store import ModelMetrics
        model = _model("m", Tier.MID, 0.003, quality_baseline=44.0)
        metrics = ModelMetrics(
            model_id="m", window_seconds=3600,
            sample_count=10, error_rate=0.5,
        )
        score = _quality_score(model, metrics)
        expected = (44.0 / 60.0) * (1 - 0.5 * 0.5)
        assert score == pytest.approx(expected)

    def test_no_errors_no_penalty(self):
        from bedrock_smart_router.metrics_store import ModelMetrics
        model = _model("m", Tier.MID, 0.003, quality_baseline=44.0)
        metrics = ModelMetrics(
            model_id="m", window_seconds=3600,
            sample_count=25, error_rate=0.0,
        )
        score = _quality_score(model, metrics)
        assert score == pytest.approx(44.0 / 60.0)  # No penalty


class TestQualityOptimizedStrategy:
    """Test the full strategy with a metrics store."""

    def _store_with_data(self) -> InMemoryMetricsStore:
        store = InMemoryMetricsStore()
        now = time.monotonic()
        # cheap-micro: some requests, no errors
        for _ in range(25):
            store.record(RequestRecord(
                model_id="cheap-micro", timestamp=now,
                latency_ms=50, success=True,
            ))
        # mid-model: some requests, no errors
        for _ in range(25):
            store.record(RequestRecord(
                model_id="mid-model", timestamp=now,
                latency_ms=200, success=True,
            ))
        # expensive-heavy: some requests with errors (20% error rate)
        for i in range(25):
            store.record(RequestRecord(
                model_id="expensive-heavy", timestamp=now,
                latency_ms=500,
                success=(i < 20),  # 20% error rate
            ))
        return store

    def test_picks_highest_quality(self):
        store = self._store_with_data()
        strategy = QualityOptimizedStrategy(metrics_store=store)
        result = strategy.select(CANDIDATES, ANALYSIS)
        # expensive-heavy has baseline 50.0 but 20% errors → 50 * 0.9 = 45
        # mid-model has baseline 44.0 with 0% errors → 44.0
        # expensive-heavy (45) > mid-model (44) so expensive-heavy wins
        assert result.selected_model.model_id == "expensive-heavy"

    def test_high_errors_can_flip_ranking(self):
        """With very high error rate, a lower-baseline model can win."""
        store = InMemoryMetricsStore()
        now = time.monotonic()
        # mid-model: no errors → quality = 44/60 = 0.733
        for _ in range(25):
            store.record(RequestRecord(
                model_id="mid-model", timestamp=now,
                latency_ms=200, success=True,
            ))
        # expensive-heavy: 80% error rate → (50/60) * (1 - 0.8*0.5) = 0.833 * 0.6 = 0.5
        for i in range(25):
            store.record(RequestRecord(
                model_id="expensive-heavy", timestamp=now,
                latency_ms=500, success=(i < 5),  # 80% error rate
            ))
        strategy = QualityOptimizedStrategy(metrics_store=store)
        result = strategy.select(CANDIDATES, ANALYSIS)
        # mid-model (0.733) > expensive-heavy (0.5)
        assert result.selected_model.model_id == "mid-model"

    def test_scores_include_sample_count(self):
        store = self._store_with_data()
        strategy = QualityOptimizedStrategy(metrics_store=store)
        result = strategy.select(CANDIDATES, ANALYSIS)
        for model_id, score_dict in result.scores.items():
            assert "sample_count" in score_dict

    def test_no_metrics_store_uses_baselines(self):
        strategy = QualityOptimizedStrategy(metrics_store=None)
        result = strategy.select(CANDIDATES, ANALYSIS)
        # Without data, quality_baseline: heavy(50) > mid(44) > micro(10)
        assert result.selected_model.model_id == "expensive-heavy"

    def test_empty_store_uses_baselines(self):
        store = InMemoryMetricsStore()
        strategy = QualityOptimizedStrategy(metrics_store=store)
        result = strategy.select(CANDIDATES, ANALYSIS)
        assert result.selected_model.model_id == "expensive-heavy"

    def test_fallback_chain_populated(self):
        store = self._store_with_data()
        strategy = QualityOptimizedStrategy(metrics_store=store)
        result = strategy.select(CANDIDATES, ANALYSIS)
        assert len(result.fallback_chain) > 0
        assert result.selected_model not in result.fallback_chain


class TestResolveQualityStrategy:
    def test_resolve_quality_optimized(self):
        s = resolve_strategy("quality-optimized")
        assert isinstance(s, QualityOptimizedStrategy)

    def test_resolve_quality_with_store(self):
        store = InMemoryMetricsStore()
        s = resolve_strategy("quality-optimized", metrics_store=store)
        assert isinstance(s, QualityOptimizedStrategy)
        assert s._metrics is store
