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
    _quality_score_for_model,
)
from bedrock_smart_router.strategy_engine import resolve_strategy


def _model(mid: str, tier: Tier, price: float) -> BedrockModel:
    return BedrockModel(
        model_id=mid, family="test", tier=tier, display_name=mid,
        capabilities=ModelCapabilities(tool_use=True),
        max_input_tokens=128_000, max_output_tokens=4096,
        pricing=ModelPricing(input_per_1k=price, output_per_1k=price * 4),
    )


CANDIDATES = [
    _model("cheap-micro", Tier.MICRO, 0.00004),
    _model("mid-model", Tier.MID, 0.003),
    _model("expensive-heavy", Tier.HEAVY, 0.015),
]

ANALYSIS = RequestAnalysis(
    complexity=Complexity.MODERATE,
    estimated_input_tokens=1000,
    estimated_output_tokens=500,
)


class TestQualityScoreBlending:
    """Test the heuristic/historical blending logic."""

    def test_no_metrics_uses_heuristic(self):
        model = _model("m", Tier.MID, 0.003)
        score = _quality_score_for_model(model, None)
        assert score == 0.82  # TIER_QUALITY_HEURISTIC[MID]

    def test_no_quality_scores_in_metrics(self):
        from bedrock_smart_router.metrics_store import ModelMetrics
        model = _model("m", Tier.MID, 0.003)
        metrics = ModelMetrics(
            model_id="m", window_seconds=3600,
            sample_count=10, error_rate=0.0,
        )
        score = _quality_score_for_model(model, metrics)
        assert score == 0.82  # Heuristic, no quality data

    def test_high_error_rate_penalises(self):
        from bedrock_smart_router.metrics_store import ModelMetrics
        model = _model("m", Tier.MID, 0.003)
        metrics = ModelMetrics(
            model_id="m", window_seconds=3600,
            sample_count=10, error_rate=0.5,
        )
        score = _quality_score_for_model(model, metrics)
        assert score < 0.82  # Penalised

    def test_full_trust_with_enough_samples(self):
        from bedrock_smart_router.metrics_store import ModelMetrics
        model = _model("m", Tier.MICRO, 0.00004)
        metrics = ModelMetrics(
            model_id="m", window_seconds=3600,
            sample_count=25, avg_quality_score=0.95,
        )
        score = _quality_score_for_model(model, metrics)
        # With 25 samples (>= 20), should fully trust historical
        assert score == 0.95

    def test_partial_trust_blends(self):
        from bedrock_smart_router.metrics_store import ModelMetrics
        model = _model("m", Tier.MICRO, 0.00004)  # heuristic = 0.55
        metrics = ModelMetrics(
            model_id="m", window_seconds=3600,
            sample_count=12, avg_quality_score=0.95,
        )
        score = _quality_score_for_model(model, metrics)
        # 12 samples: partial trust, should be between 0.55 and 0.95
        assert 0.55 < score < 0.95


class TestQualityOptimizedStrategy:
    """Test the full strategy with a metrics store."""

    def _store_with_data(self) -> InMemoryMetricsStore:
        store = InMemoryMetricsStore()
        now = time.monotonic()
        # cheap-micro: low quality scores
        for _ in range(25):
            store.record(RequestRecord(
                model_id="cheap-micro", timestamp=now,
                latency_ms=50, quality_score=0.5, success=True,
            ))
        # mid-model: high quality scores
        for _ in range(25):
            store.record(RequestRecord(
                model_id="mid-model", timestamp=now,
                latency_ms=200, quality_score=0.92, success=True,
            ))
        # expensive-heavy: highest quality but some errors
        for i in range(25):
            store.record(RequestRecord(
                model_id="expensive-heavy", timestamp=now,
                latency_ms=500, quality_score=0.95,
                success=(i < 20),  # 20% error rate
            ))
        return store

    def test_picks_highest_quality(self):
        store = self._store_with_data()
        strategy = QualityOptimizedStrategy(metrics_store=store)
        result = strategy.select(CANDIDATES, ANALYSIS)
        # mid-model has 0.92 quality with 0% errors
        # expensive-heavy has 0.95 quality but 20% errors (penalised)
        assert result.selected_model.model_id == "mid-model"

    def test_scores_include_sample_count(self):
        store = self._store_with_data()
        strategy = QualityOptimizedStrategy(metrics_store=store)
        result = strategy.select(CANDIDATES, ANALYSIS)
        for model_id, score_dict in result.scores.items():
            assert "sample_count" in score_dict
            assert score_dict["sample_count"] == 25.0

    def test_no_metrics_store_uses_heuristics(self):
        strategy = QualityOptimizedStrategy(metrics_store=None)
        result = strategy.select(CANDIDATES, ANALYSIS)
        # Without data, tier heuristic: heavy(0.90) > mid(0.82) > micro(0.55)
        assert result.selected_model.model_id == "expensive-heavy"

    def test_empty_store_uses_heuristics(self):
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
