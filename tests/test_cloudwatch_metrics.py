# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the CloudWatch metrics publisher."""

from unittest.mock import MagicMock, call

from bedrock_smart_router.cloudwatch_metrics import CloudWatchMetricsPublisher
from bedrock_smart_router.models import RoutingDecision


def _decision(
    model: str = "model-a",
    cost: float = 0.01,
    latency: float = 150.0,
    fallback: bool = False,
    cb_skipped: list | None = None,
) -> RoutingDecision:
    return RoutingDecision(
        selected_model=model,
        strategy_used="balanced",
        complexity_detected="moderate",
        complexity_score=0.5,
        candidates_evaluated=3,
        estimated_cost=cost,
        actual_cost=cost,
        latency_ms=latency,
        fallback_used=fallback,
        circuit_breaker_skipped=cb_skipped or [],
    )


class TestCloudWatchMetricsPublisher:
    def setup_method(self):
        self.mock_session = MagicMock()
        self.mock_client = MagicMock()
        self.mock_session.client.return_value = self.mock_client
        self.publisher = CloudWatchMetricsPublisher(
            namespace="TestNamespace",
            boto_session=self.mock_session,
            region="us-west-2",
        )

    def test_record_adds_to_batch(self):
        self.publisher.record(_decision())
        assert self.publisher.stats["pending_batch_size"] > 0

    def test_flush_calls_put_metric_data(self):
        self.publisher.record(_decision())
        count = self.publisher.flush()
        assert count > 0
        self.mock_client.put_metric_data.assert_called_once()
        call_kwargs = self.mock_client.put_metric_data.call_args
        assert call_kwargs[1]["Namespace"] == "TestNamespace"
        assert len(call_kwargs[1]["MetricData"]) > 0

    def test_flush_empty_batch_returns_zero(self):
        count = self.publisher.flush()
        assert count == 0
        self.mock_client.put_metric_data.assert_not_called()

    def test_metrics_include_routing_decisions(self):
        self.publisher.record(_decision())
        self.publisher.flush()
        data = self.mock_client.put_metric_data.call_args[1]["MetricData"]
        names = {d["MetricName"] for d in data}
        assert "RoutingDecisions" in names

    def test_metrics_include_latency(self):
        self.publisher.record(_decision(latency=200.0))
        self.publisher.flush()
        data = self.mock_client.put_metric_data.call_args[1]["MetricData"]
        latency_points = [d for d in data if d["MetricName"] == "Latency"]
        assert len(latency_points) == 1
        assert latency_points[0]["Value"] == 200.0
        assert latency_points[0]["Unit"] == "Milliseconds"

    def test_metrics_include_cost(self):
        self.publisher.record(_decision(cost=0.005))
        self.publisher.flush()
        data = self.mock_client.put_metric_data.call_args[1]["MetricData"]
        cost_points = [d for d in data if d["MetricName"] == "Cost"]
        assert len(cost_points) == 1
        assert cost_points[0]["Value"] == 0.005

    def test_cache_hit_metric(self):
        self.publisher.record(_decision(), cache_hit=True)
        self.publisher.flush()
        data = self.mock_client.put_metric_data.call_args[1]["MetricData"]
        cache_points = [d for d in data if d["MetricName"] == "CacheHits"]
        assert len(cache_points) == 1

    def test_fallback_metric(self):
        self.publisher.record(_decision(fallback=True))
        self.publisher.flush()
        data = self.mock_client.put_metric_data.call_args[1]["MetricData"]
        fb_points = [d for d in data if d["MetricName"] == "FallbacksUsed"]
        assert len(fb_points) == 1

    def test_circuit_breaker_skip_metric(self):
        self.publisher.record(_decision(cb_skipped=["model-x", "model-y"]))
        self.publisher.flush()
        data = self.mock_client.put_metric_data.call_args[1]["MetricData"]
        cb_points = [d for d in data if d["MetricName"] == "CircuitBreakerSkips"]
        assert len(cb_points) == 1
        assert cb_points[0]["Value"] == 2

    def test_cost_savings_metric(self):
        self.publisher.record(_decision(cost=0.01), most_expensive_cost=0.05)
        self.publisher.flush()
        data = self.mock_client.put_metric_data.call_args[1]["MetricData"]
        savings = [d for d in data if d["MetricName"] == "CostSavings"]
        assert len(savings) == 1
        assert savings[0]["Value"] == 0.04

    def test_dimensions_present(self):
        self.publisher.record(_decision(model="anthropic.claude-sonnet-4-6"))
        self.publisher.flush()
        data = self.mock_client.put_metric_data.call_args[1]["MetricData"]
        dims = data[0]["Dimensions"]
        dim_names = {d["Name"] for d in dims}
        assert dim_names == {"Model", "Strategy", "Complexity"}
        model_dim = next(d for d in dims if d["Name"] == "Model")
        assert model_dim["Value"] == "anthropic.claude-sonnet-4-6"

    def test_api_failure_doesnt_crash(self):
        self.mock_client.put_metric_data.side_effect = Exception("Access denied")
        self.publisher.record(_decision())
        count = self.publisher.flush()
        assert count == 0
        assert self.publisher.stats["total_errors"] == 1

    def test_stats(self):
        self.publisher.record(_decision())
        self.publisher.flush()
        stats = self.publisher.stats
        assert stats["namespace"] == "TestNamespace"
        assert stats["total_published"] > 0
        assert stats["pending_batch_size"] == 0
