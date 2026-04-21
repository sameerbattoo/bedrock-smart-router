"""Tests for the observability module."""

from bedrock_smart_router.models import RoutingDecision
from bedrock_smart_router.observability import CostTracker, ObservabilityManager


def _decision(model: str = "model-a", cost: float = 0.01) -> RoutingDecision:
    return RoutingDecision(
        selected_model=model,
        strategy_used="balanced",
        complexity_detected="moderate",
        complexity_score=0.5,
        candidates_evaluated=3,
        estimated_cost=cost,
        actual_cost=cost,
    )


class TestCostTracker:
    def test_empty(self):
        ct = CostTracker()
        assert ct.total_cost == 0.0
        assert ct.avg_cost_per_request == 0.0

    def test_record_cost(self):
        ct = CostTracker()
        ct.record(_decision(cost=0.01))
        ct.record(_decision(cost=0.02))
        assert ct.total_cost == 0.03
        assert ct.total_requests == 2
        assert ct.avg_cost_per_request == 0.015

    def test_cache_hit_tracking(self):
        ct = CostTracker()
        ct.record(_decision(cost=0.01), cache_hit=True)
        assert ct.total_cache_hits == 1
        assert ct.total_cost == 0.0  # Cache hits are free
        assert ct.cost_saved_by_cache == 0.01

    def test_routing_savings(self):
        ct = CostTracker()
        ct.record(_decision(cost=0.01), most_expensive_cost=0.05)
        assert ct.cost_saved_by_routing == 0.04

    def test_cost_by_model(self):
        ct = CostTracker()
        ct.record(_decision("model-a", 0.01))
        ct.record(_decision("model-b", 0.02))
        ct.record(_decision("model-a", 0.03))
        assert ct.cost_by_model["model-a"] == 0.04
        assert ct.cost_by_model["model-b"] == 0.02


class TestObservabilityManager:
    def test_emit_calls_callbacks(self):
        events = []
        mgr = ObservabilityManager(
            callbacks=[lambda e: events.append(e)],
            log_decisions=False,
        )
        mgr.emit(_decision())
        assert len(events) == 1
        assert events[0].decision.selected_model == "model-a"

    def test_emit_tracks_cost(self):
        mgr = ObservabilityManager(log_decisions=False)
        mgr.emit(_decision(cost=0.01))
        mgr.emit(_decision(cost=0.02))
        assert mgr.cost_tracker.total_cost == 0.03

    def test_callback_failure_doesnt_crash(self):
        def bad_callback(e):
            raise ValueError("boom")

        mgr = ObservabilityManager(
            callbacks=[bad_callback], log_decisions=False
        )
        # Should not raise
        mgr.emit(_decision())

    def test_request_id_increments(self):
        mgr = ObservabilityManager(log_decisions=False)
        e1 = mgr.emit(_decision())
        e2 = mgr.emit(_decision())
        assert e1.request_id != e2.request_id
