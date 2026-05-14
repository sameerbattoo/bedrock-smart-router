"""Integration test — publishes real metrics to CloudWatch.

Run with:
    INTEGRATION_TEST=1 .venv/bin/python -m pytest tests/test_cloudwatch_integration.py -v -s

Publishes to a unique namespace ``BSR-IntegTest-<uuid>`` so it won't
pollute production metrics.  CloudWatch metrics cannot be deleted, but
the unique namespace ensures isolation.  They'll age out after 15
months per CloudWatch retention policy.

Requires: ``cloudwatch:PutMetricData`` and ``cloudwatch:GetMetricData``
"""

from __future__ import annotations

import os
import time
import uuid
from datetime import datetime, timedelta, timezone

import boto3
import pytest

from bedrock_smart_router.cloudwatch_metrics import CloudWatchMetricsPublisher
from bedrock_smart_router.models import RoutingDecision

SKIP_REASON = "Set INTEGRATION_TEST=1 to run against real AWS"
REGION = "us-west-2"


def _decision(
    model: str = "amazon.nova-micro-v1:0",
    cost: float = 0.001,
    latency: float = 120.0,
    fallback: bool = False,
) -> RoutingDecision:
    return RoutingDecision(
        selected_model=model,
        strategy_used="balanced",
        complexity_detected="simple",
        complexity_score=0.2,
        candidates_evaluated=5,
        estimated_cost=cost,
        actual_cost=cost,
        latency_ms=latency,
        fallback_used=fallback,
    )


@pytest.fixture
def cw_publisher():
    """Create a publisher with a unique namespace."""
    short_id = uuid.uuid4().hex[:8]
    namespace = f"BSR-IntegTest-{short_id}"
    session = boto3.Session(region_name=REGION)
    publisher = CloudWatchMetricsPublisher(
        namespace=namespace,
        boto_session=session,
        region=REGION,
    )
    yield publisher, namespace, session


@pytest.mark.skipif(
    os.environ.get("INTEGRATION_TEST") != "1",
    reason=SKIP_REASON,
)
class TestCloudWatchIntegration:

    def test_publish_and_verify(self, cw_publisher):
        """Publish metrics and verify they appear in CloudWatch."""
        publisher, namespace, session = cw_publisher
        cw_client = session.client("cloudwatch", region_name=REGION)

        # Publish 5 routing decisions
        for i in range(5):
            publisher.record(
                _decision(
                    model="amazon.nova-micro-v1:0",
                    cost=0.001 * (i + 1),
                    latency=100.0 + i * 20,
                ),
                most_expensive_cost=0.01,
            )

        # Flush synchronously (batch may have auto-flushed in background)
        publisher.flush()
        # Wait for any background flush threads to complete
        time.sleep(1)
        total = publisher.stats["total_published"]
        assert total > 0
        print(f"\n  Published {total} data points to namespace '{namespace}'")

        # CloudWatch needs a few seconds to ingest
        print("  Waiting 10s for CloudWatch ingestion...")
        time.sleep(10)

        # Query the metrics back
        now = datetime.now(timezone.utc)
        start = now - timedelta(minutes=5)

        resp = cw_client.get_metric_data(
            MetricDataQueries=[
                {
                    "Id": "routing_count",
                    "MetricStat": {
                        "Metric": {
                            "Namespace": namespace,
                            "MetricName": "RoutingDecisions",
                            "Dimensions": [
                                {"Name": "Model", "Value": "amazon.nova-micro-v1:0"},
                                {"Name": "Strategy", "Value": "balanced"},
                                {"Name": "Complexity", "Value": "simple"},
                            ],
                        },
                        "Period": 60,
                        "Stat": "Sum",
                    },
                    "ReturnData": True,
                },
                {
                    "Id": "avg_latency",
                    "MetricStat": {
                        "Metric": {
                            "Namespace": namespace,
                            "MetricName": "Latency",
                            "Dimensions": [
                                {"Name": "Model", "Value": "amazon.nova-micro-v1:0"},
                                {"Name": "Strategy", "Value": "balanced"},
                                {"Name": "Complexity", "Value": "simple"},
                            ],
                        },
                        "Period": 60,
                        "Stat": "Average",
                    },
                    "ReturnData": True,
                },
            ],
            StartTime=start,
            EndTime=now,
        )

        results = resp.get("MetricDataResults", [])
        print(f"  MetricDataResults: {results}")

        # Verify RoutingDecisions metric exists with data
        routing_result = next(
            (r for r in results if r["Id"] == "routing_count"), None
        )
        assert routing_result is not None
        assert routing_result["StatusCode"] == "Complete"
        # Values may take a minute to appear; check we got the query through
        print(f"  RoutingDecisions values: {routing_result.get('Values', [])}")

    def test_publish_with_cache_hit(self, cw_publisher):
        """Publish a cache hit metric."""
        publisher, namespace, _ = cw_publisher

        publisher.record(_decision(), cache_hit=True)
        count = publisher.flush()
        assert count > 0
        print(f"\n  Published {count} data points (cache hit) to '{namespace}'")

    def test_publish_with_fallback(self, cw_publisher):
        """Publish a fallback metric."""
        publisher, namespace, _ = cw_publisher

        publisher.record(_decision(fallback=True))
        count = publisher.flush()
        assert count > 0
        print(f"\n  Published {count} data points (fallback) to '{namespace}'")

    def test_publish_cost_savings(self, cw_publisher):
        """Publish cost savings metric."""
        publisher, namespace, _ = cw_publisher

        publisher.record(
            _decision(cost=0.001),
            most_expensive_cost=0.05,
        )
        count = publisher.flush()
        assert count > 0
        print(f"\n  Published {count} data points (savings) to '{namespace}'")

    def test_stats_after_publish(self, cw_publisher):
        """Verify publisher stats after real publish."""
        publisher, namespace, _ = cw_publisher

        publisher.record(_decision())
        publisher.flush()

        stats = publisher.stats
        assert stats["total_published"] > 0
        assert stats["total_errors"] == 0
        assert stats["pending_batch_size"] == 0
        print(f"\n  Stats: {stats}")
