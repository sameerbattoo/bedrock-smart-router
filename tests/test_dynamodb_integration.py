"""Integration test — runs against a REAL DynamoDB table in your AWS account.

Run with:
    INTEGRATION_TEST=1 .venv/bin/python -m pytest tests/test_dynamodb_integration.py -v -s

The test creates a table, writes data, reads it back, and deletes the
table on teardown.  Requires valid AWS credentials with DynamoDB
permissions (see docs/iam-permissions.md).
"""

from __future__ import annotations

import os
import time
import uuid

import boto3
import pytest

from bedrock_smart_router.dynamodb_metrics_store import DynamoDBMetricsStore
from bedrock_smart_router.metrics_store import RequestRecord

SKIP_REASON = "Set INTEGRATION_TEST=1 to run against real AWS"
REGION = "us-west-2"


def _table_name() -> str:
    """Unique table name per test run to avoid collisions."""
    short_id = uuid.uuid4().hex[:8]
    return f"bsr-integ-test-{short_id}"


@pytest.fixture
def live_store():
    """Create a DynamoDBMetricsStore against real AWS, clean up after."""
    table = _table_name()
    session = boto3.Session(region_name=REGION)
    store = DynamoDBMetricsStore(
        table_name=table,
        ttl_hours=1,  # Short TTL for test data
        boto_session=session,
        region=REGION,
        auto_create_table=True,
    )

    yield store

    # Teardown — delete the table
    try:
        client = session.client("dynamodb", region_name=REGION)
        client.delete_table(TableName=table)
        print(f"\n  Deleted table {table}")
    except Exception as exc:
        print(f"\n  Warning: could not delete table {table}: {exc}")


@pytest.mark.skipif(
    os.environ.get("INTEGRATION_TEST") != "1",
    reason=SKIP_REASON,
)
class TestDynamoDBIntegration:

    def test_table_created(self, live_store):
        """Table should exist after first record."""
        live_store.record(RequestRecord(
            model_id="integ-test-model",
            timestamp=time.monotonic(),
            latency_ms=42,
            success=True,
        ))
        client = live_store._dynamodb.meta.client
        resp = client.describe_table(TableName=live_store._table_name)
        assert resp["Table"]["TableStatus"] == "ACTIVE"
        key_names = {k["AttributeName"] for k in resp["Table"]["KeySchema"]}
        assert key_names == {"model_id", "timestamp"}
        print(f"\n  Table {live_store._table_name} is ACTIVE")

    def test_write_and_read_single_model(self, live_store):
        """Write 5 records, read them back, verify aggregation."""
        model = "us.anthropic.claude-sonnet-4-6"
        for i in range(5):
            live_store.record(RequestRecord(
                model_id=model,
                timestamp=time.monotonic(),
                latency_ms=100 + i * 50,
                ttft_ms=40 + i * 10,
                input_tokens=500,
                output_tokens=200,
                cost=0.001 + i * 0.0005,
                quality_score=0.8 + i * 0.02,
                success=True,
            ))

        metrics = live_store.get_metrics(model, window_seconds=300)
        assert metrics.sample_count == 5
        assert metrics.avg_latency_ms > 0
        assert metrics.avg_ttft_ms > 0
        assert metrics.avg_quality_score is not None
        assert metrics.avg_quality_score > 0.8
        assert metrics.error_rate == 0.0
        print(f"\n  Metrics: {metrics}")

    def test_write_and_read_multiple_models(self, live_store):
        """Write records for 3 models, verify get_all_metrics."""
        models = [
            "us.amazon.nova-micro-v1:0",
            "us.anthropic.claude-sonnet-4-6",
            "us.amazon.nova-pro-v1:0",
        ]
        for model in models:
            for _ in range(3):
                live_store.record(RequestRecord(
                    model_id=model,
                    timestamp=time.monotonic(),
                    latency_ms=150,
                    cost=0.002,
                    success=True,
                ))

        all_metrics = live_store.get_all_metrics(window_seconds=300)
        assert len(all_metrics) >= 3
        for model in models:
            assert model in all_metrics
            assert all_metrics[model].sample_count == 3
        print(f"\n  All metrics keys: {list(all_metrics.keys())}")

    def test_error_and_throttle_tracking(self, live_store):
        """Verify error and throttle flags persist correctly."""
        model = "us.amazon.nova-lite-v1:0"
        # 7 success, 2 errors, 1 throttle
        for i in range(10):
            live_store.record(RequestRecord(
                model_id=model,
                timestamp=time.monotonic(),
                latency_ms=100,
                success=(i < 7),
                is_throttle=(i == 9),
            ))

        metrics = live_store.get_metrics(model, window_seconds=300)
        assert metrics.sample_count == 10
        assert metrics.error_rate == pytest.approx(0.3)
        assert metrics.throttle_rate == pytest.approx(0.1)
        print(f"\n  Error rate: {metrics.error_rate}, Throttle rate: {metrics.throttle_rate}")

    def test_quality_score_none_handling(self, live_store):
        """Records without quality_score should not break aggregation."""
        model = "us.meta.llama3-3-70b-instruct-v1:0"
        live_store.record(RequestRecord(
            model_id=model,
            timestamp=time.monotonic(),
            latency_ms=200,
            success=True,
            # No quality_score
        ))
        live_store.record(RequestRecord(
            model_id=model,
            timestamp=time.monotonic(),
            latency_ms=300,
            quality_score=0.9,
            success=True,
        ))

        metrics = live_store.get_metrics(model, window_seconds=300)
        assert metrics.sample_count == 2
        # Only 1 record has a quality score
        assert metrics.avg_quality_score == pytest.approx(0.9)
        print(f"\n  Quality score (partial): {metrics.avg_quality_score}")
