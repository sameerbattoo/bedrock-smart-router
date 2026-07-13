# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""CloudWatch custom metrics publisher.

Publishes routing decision metrics to CloudWatch using PutMetricData.
Metrics are batched and flushed periodically or when the batch is full
to minimize API calls.

Published metrics (namespace: ``BedrockSmartRouter`` by default):

  - ``RoutingDecisions`` (Count) — one per request
  - ``Latency`` (Milliseconds) — end-to-end latency
  - ``Cost`` (None/USD) — actual cost per request
  - ``CacheHits`` (Count) — response cache hits
  - ``FallbacksUsed`` (Count) — requests that fell back to another model
  - ``CircuitBreakerSkips`` (Count) — models skipped due to open breakers
  - ``CostSavings`` (None/USD) — savings vs most expensive model

Dimensions on every metric:
  - Model, Strategy, Complexity

IAM permission required: ``cloudwatch:PutMetricData``
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

from bedrock_smart_router.models import RoutingDecision

logger = logging.getLogger(__name__)

# CloudWatch PutMetricData accepts max 1000 metric data points per call
_MAX_BATCH_SIZE = 20  # We publish fewer dimensions, so keep batches small
_FLUSH_INTERVAL_SECONDS = 60.0


class CloudWatchMetricsPublisher:
    """Batched CloudWatch metrics publisher.

    Collects metric data points and flushes them to CloudWatch
    periodically or when the batch reaches ``_MAX_BATCH_SIZE``.
    Flushing happens in a background thread to avoid blocking the
    request path.
    """

    def __init__(
        self,
        namespace: str = "BedrockSmartRouter",
        boto_session: Any | None = None,
        region: str = "us-west-2",
    ) -> None:
        self._namespace = namespace
        self._region = region
        self._session = boto_session
        self._client: Any | None = None
        self._batch: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._last_flush = time.monotonic()
        self._total_published = 0
        self._total_errors = 0

    def _get_client(self) -> Any:
        if self._client is None:
            if self._session is None:
                import boto3
                self._session = boto3.Session(region_name=self._region)
            self._client = self._session.client(
                "cloudwatch", region_name=self._region
            )
        return self._client

    def record(
        self,
        decision: RoutingDecision,
        cache_hit: bool = False,
        duration_ms: float = 0.0,
        most_expensive_cost: float = 0.0,
    ) -> None:
        """Add metric data points for a routing decision to the batch."""
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc)
        dims = [
            {"Name": "Model", "Value": decision.selected_model},
            {"Name": "Strategy", "Value": decision.strategy_used},
            {"Name": "Complexity", "Value": decision.complexity_detected},
        ]

        points: list[dict[str, Any]] = []

        # Request count
        points.append({
            "MetricName": "RoutingDecisions",
            "Dimensions": dims,
            "Timestamp": now,
            "Value": 1,
            "Unit": "Count",
        })

        # Latency
        if decision.latency_ms and decision.latency_ms > 0:
            points.append({
                "MetricName": "Latency",
                "Dimensions": dims,
                "Timestamp": now,
                "Value": decision.latency_ms,
                "Unit": "Milliseconds",
            })

        # Cost
        actual_cost = decision.actual_cost or 0.0
        if actual_cost > 0:
            points.append({
                "MetricName": "Cost",
                "Dimensions": dims,
                "Timestamp": now,
                "Value": actual_cost,
                "Unit": "None",
            })

        # Cache hit
        if cache_hit:
            points.append({
                "MetricName": "CacheHits",
                "Dimensions": dims,
                "Timestamp": now,
                "Value": 1,
                "Unit": "Count",
            })

        # Fallback used
        if decision.fallback_used:
            points.append({
                "MetricName": "FallbacksUsed",
                "Dimensions": dims,
                "Timestamp": now,
                "Value": 1,
                "Unit": "Count",
            })

        # Circuit breaker skips
        if decision.circuit_breaker_skipped:
            points.append({
                "MetricName": "CircuitBreakerSkips",
                "Dimensions": dims,
                "Timestamp": now,
                "Value": len(decision.circuit_breaker_skipped),
                "Unit": "Count",
            })

        # Cost savings
        if most_expensive_cost > actual_cost:
            points.append({
                "MetricName": "CostSavings",
                "Dimensions": dims,
                "Timestamp": now,
                "Value": most_expensive_cost - actual_cost,
                "Unit": "None",
            })

        with self._lock:
            self._batch.extend(points)
            if len(self._batch) >= _MAX_BATCH_SIZE:
                self._flush_async()
            elif time.monotonic() - self._last_flush >= _FLUSH_INTERVAL_SECONDS:
                self._flush_async()

    def flush(self) -> int:
        """Flush the current batch to CloudWatch synchronously.

        Returns the number of data points published.
        """
        with self._lock:
            batch = list(self._batch)
            self._batch.clear()
            self._last_flush = time.monotonic()

        if not batch:
            return 0

        return self._put_metrics(batch)

    def _flush_async(self) -> None:
        """Flush in a background thread to avoid blocking."""
        batch = list(self._batch)
        self._batch.clear()
        self._last_flush = time.monotonic()

        if batch:
            thread = threading.Thread(
                target=self._put_metrics, args=(batch,), daemon=True
            )
            thread.start()

    def _put_metrics(self, batch: list[dict[str, Any]]) -> int:
        """Call CloudWatch PutMetricData."""
        try:
            client = self._get_client()
            # PutMetricData accepts max 1000 data points per call
            for i in range(0, len(batch), 1000):
                chunk = batch[i : i + 1000]
                client.put_metric_data(
                    Namespace=self._namespace,
                    MetricData=chunk,
                )
            self._total_published += len(batch)
            logger.debug(
                "Published %d metrics to CloudWatch %s",
                len(batch), self._namespace,
            )
            return len(batch)
        except Exception as exc:
            self._total_errors += 1
            logger.warning("Failed to publish CloudWatch metrics: %s", exc)
            return 0

    @property
    def stats(self) -> dict[str, Any]:
        with self._lock:
            pending = len(self._batch)
        return {
            "namespace": self._namespace,
            "total_published": self._total_published,
            "total_errors": self._total_errors,
            "pending_batch_size": pending,
        }
