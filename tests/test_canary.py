# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for canary deployments."""

from bedrock_smart_router.canary import CanaryConfig, CanaryManager, CanaryThresholds


def _config(**overrides) -> CanaryConfig:
    defaults = dict(
        enabled=True,
        baseline_model="model-baseline",
        canary_model="model-canary",
        canary_percentage=50.0,  # High % for test determinism
        auto_rollback=CanaryThresholds(max_error_rate=0.20, max_latency_p95_ms=5000),
        auto_promote=CanaryThresholds(min_requests=10, max_error_rate=0.05, max_latency_p95_ms=3000),
    )
    defaults.update(overrides)
    return CanaryConfig(**defaults)


class TestCanaryManager:
    def test_inactive_when_disabled(self):
        mgr = CanaryManager(CanaryConfig(enabled=False))
        assert not mgr.is_active
        model, is_canary = mgr.select_model()
        assert not is_canary

    def test_selects_canary_or_baseline(self):
        mgr = CanaryManager(_config())
        models = set()
        for _ in range(100):
            model, _ = mgr.select_model()
            models.add(model)
        assert "model-baseline" in models
        assert "model-canary" in models

    def test_auto_rollback_on_high_errors(self):
        mgr = CanaryManager(_config())
        # Simulate 10 canary requests, all failing
        for _ in range(15):
            mgr.record_result(is_canary=True, latency_ms=100, success=False)
        assert mgr.is_rolled_back
        assert not mgr.is_active

    def test_auto_rollback_on_high_latency(self):
        cfg = _config(
            auto_rollback=CanaryThresholds(max_error_rate=0.5, max_latency_p95_ms=200),
        )
        mgr = CanaryManager(cfg)
        for _ in range(15):
            mgr.record_result(is_canary=True, latency_ms=500, success=True)
        assert mgr.is_rolled_back

    def test_auto_promote_on_good_metrics(self):
        mgr = CanaryManager(_config())
        for _ in range(15):
            mgr.record_result(is_canary=True, latency_ms=100, success=True)
        assert mgr.is_promoted
        assert not mgr.is_active  # Promoted = no longer active

    def test_no_promote_below_min_requests(self):
        cfg = _config(
            auto_promote=CanaryThresholds(min_requests=100, max_error_rate=0.05),
        )
        mgr = CanaryManager(cfg)
        for _ in range(5):
            mgr.record_result(is_canary=True, latency_ms=100, success=True)
        assert not mgr.is_promoted

    def test_baseline_records_dont_trigger_rollback(self):
        mgr = CanaryManager(_config())
        for _ in range(20):
            mgr.record_result(is_canary=False, latency_ms=100, success=False)
        assert not mgr.is_rolled_back
        assert mgr.is_active

    def test_stats(self):
        mgr = CanaryManager(_config())
        mgr.record_result(is_canary=True, latency_ms=100, success=True)
        mgr.record_result(is_canary=False, latency_ms=200, success=True)
        stats = mgr.stats
        assert stats["canary_requests"] == 1
        assert stats["baseline_requests"] == 1
        assert stats["canary_error_rate"] == 0.0
