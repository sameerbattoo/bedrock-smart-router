"""Tests for the metrics store."""

import time

import pytest

from bedrock_smart_router.metrics_store import InMemoryMetricsStore, RequestRecord


class TestInMemoryMetricsStore:
    def setup_method(self):
        self.store = InMemoryMetricsStore()

    def test_empty_metrics(self):
        m = self.store.get_metrics("model-a")
        assert m.sample_count == 0
        assert m.avg_latency_ms == 0.0

    def test_record_and_retrieve(self):
        now = time.monotonic()
        self.store.record(RequestRecord(
            model_id="model-a", timestamp=now,
            latency_ms=100, cost=0.001, success=True,
        ))
        self.store.record(RequestRecord(
            model_id="model-a", timestamp=now,
            latency_ms=200, cost=0.002, success=True,
        ))
        m = self.store.get_metrics("model-a")
        assert m.sample_count == 2
        assert m.avg_latency_ms == 150.0
        assert m.error_rate == 0.0

    def test_error_rate(self):
        now = time.monotonic()
        for i in range(10):
            self.store.record(RequestRecord(
                model_id="model-a", timestamp=now,
                latency_ms=100, success=(i < 7),
            ))
        m = self.store.get_metrics("model-a")
        assert m.error_rate == 0.3

    def test_throttle_rate(self):
        now = time.monotonic()
        for i in range(5):
            self.store.record(RequestRecord(
                model_id="model-a", timestamp=now,
                latency_ms=100, success=False, is_throttle=(i < 2),
            ))
        m = self.store.get_metrics("model-a")
        assert m.throttle_rate == 0.4

    def test_quality_score(self):
        now = time.monotonic()
        self.store.record(RequestRecord(
            model_id="model-a", timestamp=now,
            latency_ms=100, quality_score=0.8, success=True,
        ))
        self.store.record(RequestRecord(
            model_id="model-a", timestamp=now,
            latency_ms=100, quality_score=0.9, success=True,
        ))
        m = self.store.get_metrics("model-a")
        assert m.avg_quality_score == pytest.approx(0.85)

    def test_window_filtering(self):
        old = time.monotonic() - 7200  # 2 hours ago
        now = time.monotonic()
        self.store.record(RequestRecord(
            model_id="model-a", timestamp=old, latency_ms=999, success=True,
        ))
        self.store.record(RequestRecord(
            model_id="model-a", timestamp=now, latency_ms=100, success=True,
        ))
        m = self.store.get_metrics("model-a", window_seconds=3600)
        assert m.sample_count == 1
        assert m.avg_latency_ms == 100.0

    def test_get_all_metrics(self):
        now = time.monotonic()
        self.store.record(RequestRecord(
            model_id="model-a", timestamp=now, latency_ms=100, success=True,
        ))
        self.store.record(RequestRecord(
            model_id="model-b", timestamp=now, latency_ms=200, success=True,
        ))
        all_m = self.store.get_all_metrics()
        assert "model-a" in all_m
        assert "model-b" in all_m

    def test_percentiles(self):
        now = time.monotonic()
        for lat in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
            self.store.record(RequestRecord(
                model_id="model-a", timestamp=now,
                latency_ms=lat, success=True,
            ))
        m = self.store.get_metrics("model-a")
        assert m.p50_latency_ms >= 50.0
        assert m.p95_latency_ms >= 90.0
