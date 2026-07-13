# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the DynamoDB metrics store using moto mock."""

import time

import boto3
import pytest
from moto import mock_aws

from bedrock_smart_router.dynamodb_metrics_store import DynamoDBMetricsStore
from bedrock_smart_router.metrics_store import RequestRecord


@pytest.fixture
def ddb_store():
    """Create a DynamoDBMetricsStore backed by a moto-mocked table."""
    with mock_aws():
        session = boto3.Session(
            region_name="us-east-1",
            aws_access_key_id="testing",
            aws_secret_access_key="testing",
        )
        store = DynamoDBMetricsStore(
            table_name="TestMetrics",
            ttl_hours=24,
            boto_session=session,
            region="us-east-1",
            auto_create_table=True,
        )
        yield store


class TestDynamoDBMetricsStore:
    def test_auto_creates_table(self, ddb_store):
        """Table should be created automatically on first use."""
        # Recording triggers lazy table init
        ddb_store.record(RequestRecord(
            model_id="model-a",
            timestamp=time.monotonic(),
            latency_ms=100,
            success=True,
        ))
        # Verify table exists
        client = ddb_store._dynamodb.meta.client
        tables = client.list_tables()["TableNames"]
        assert "TestMetrics" in tables

    def test_record_and_get_metrics(self, ddb_store):
        ddb_store.record(RequestRecord(
            model_id="model-a", timestamp=time.monotonic(),
            latency_ms=100, cost=0.001, success=True,
        ))
        ddb_store.record(RequestRecord(
            model_id="model-a", timestamp=time.monotonic(),
            latency_ms=200, cost=0.002, success=True,
        ))
        m = ddb_store.get_metrics("model-a", window_seconds=3600)
        assert m.sample_count == 2
        assert m.avg_latency_ms == 150.0
        assert m.error_rate == 0.0

    def test_error_and_throttle_rates(self, ddb_store):
        for i in range(10):
            ddb_store.record(RequestRecord(
                model_id="model-a", timestamp=time.monotonic(),
                latency_ms=100,
                success=(i < 7),
                is_throttle=(i >= 8),
            ))
        m = ddb_store.get_metrics("model-a", window_seconds=3600)
        assert m.sample_count == 10
        assert m.error_rate == 0.3
        assert m.throttle_rate == 0.2

    def test_quality_scores(self, ddb_store):
        ddb_store.record(RequestRecord(
            model_id="model-a", timestamp=time.monotonic(),
            latency_ms=100, quality_score=0.8, success=True,
        ))
        ddb_store.record(RequestRecord(
            model_id="model-a", timestamp=time.monotonic(),
            latency_ms=100, quality_score=0.9, success=True,
        ))
        m = ddb_store.get_metrics("model-a", window_seconds=3600)
        assert m.avg_quality_score == pytest.approx(0.85)

    def test_no_quality_score_returns_none(self, ddb_store):
        ddb_store.record(RequestRecord(
            model_id="model-a", timestamp=time.monotonic(),
            latency_ms=100, success=True,
        ))
        m = ddb_store.get_metrics("model-a", window_seconds=3600)
        assert m.avg_quality_score is None

    def test_empty_model_returns_zero_metrics(self, ddb_store):
        m = ddb_store.get_metrics("nonexistent", window_seconds=3600)
        assert m.sample_count == 0
        assert m.avg_latency_ms == 0.0

    def test_get_all_metrics(self, ddb_store):
        ddb_store.record(RequestRecord(
            model_id="model-a", timestamp=time.monotonic(),
            latency_ms=100, success=True,
        ))
        ddb_store.record(RequestRecord(
            model_id="model-b", timestamp=time.monotonic(),
            latency_ms=200, success=True,
        ))
        all_m = ddb_store.get_all_metrics(window_seconds=3600)
        assert "model-a" in all_m
        assert "model-b" in all_m
        assert all_m["model-a"].avg_latency_ms == 100.0
        assert all_m["model-b"].avg_latency_ms == 200.0

    def test_multiple_models_isolated(self, ddb_store):
        ddb_store.record(RequestRecord(
            model_id="model-a", timestamp=time.monotonic(),
            latency_ms=100, cost=0.01, success=True,
        ))
        ddb_store.record(RequestRecord(
            model_id="model-b", timestamp=time.monotonic(),
            latency_ms=500, cost=0.05, success=False,
        ))
        ma = ddb_store.get_metrics("model-a", window_seconds=3600)
        mb = ddb_store.get_metrics("model-b", window_seconds=3600)
        assert ma.sample_count == 1
        assert ma.error_rate == 0.0
        assert mb.sample_count == 1
        assert mb.error_rate == 1.0

    def test_ttft_tracking(self, ddb_store):
        ddb_store.record(RequestRecord(
            model_id="model-a", timestamp=time.monotonic(),
            latency_ms=500, ttft_ms=120, success=True,
        ))
        ddb_store.record(RequestRecord(
            model_id="model-a", timestamp=time.monotonic(),
            latency_ms=600, ttft_ms=180, success=True,
        ))
        m = ddb_store.get_metrics("model-a", window_seconds=3600)
        assert m.avg_ttft_ms == 150.0
