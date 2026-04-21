"""Historical metrics store — pluggable backend for routing decisions.

Tracks per-model performance (latency, cost, error rate, quality) over
sliding time windows.  The strategy engine uses this data to make
informed routing decisions instead of relying solely on tier heuristics.

Backends:
  - InMemoryMetricsStore (default): sliding window, Lambda-friendly
  - Custom: implement the MetricsStore protocol
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class RequestRecord:
    """A single recorded request outcome."""

    model_id: str
    timestamp: float  # monotonic seconds
    latency_ms: float = 0.0
    ttft_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0
    quality_score: float | None = None  # From judge, if available
    success: bool = True
    is_throttle: bool = False
    # Routing context (enriches DynamoDB items for analytics)
    strategy: str = ""
    complexity: str = ""
    tenant_id: str = ""
    inference_tier: str = ""
    cris_profile: str = ""
    fallback_used: bool = False
    cache_hit: bool = False  # Response cache hit (our cache, not Bedrock's)
    # Bedrock prompt cache metrics
    prompt_cache_read_tokens: int = 0
    prompt_cache_write_tokens: int = 0


@dataclass
class ModelMetrics:
    """Aggregated metrics for a model over a time window."""

    model_id: str
    window_seconds: float
    avg_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    avg_ttft_ms: float = 0.0
    avg_cost_per_request: float = 0.0
    error_rate: float = 0.0
    throttle_rate: float = 0.0
    avg_quality_score: float | None = None
    sample_count: int = 0


def _percentile(sorted_vals: list[float], pct: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = int(len(sorted_vals) * pct / 100)
    return sorted_vals[min(idx, len(sorted_vals) - 1)]


class MetricsStore(ABC):
    """Abstract interface for metrics backends."""

    @abstractmethod
    def record(self, record: RequestRecord) -> None:
        """Record a completed request."""
        ...

    @abstractmethod
    def get_metrics(
        self, model_id: str, window_seconds: float = 3600.0
    ) -> ModelMetrics:
        """Get aggregated metrics for a model over the given window."""
        ...

    @abstractmethod
    def get_all_metrics(
        self, window_seconds: float = 3600.0
    ) -> dict[str, ModelMetrics]:
        """Get metrics for all tracked models."""
        ...


class InMemoryMetricsStore(MetricsStore):
    """Sliding-window in-memory metrics store.

    Good for Lambda (resets on cold start, warms up quickly) and
    single-instance deployments.
    """

    def __init__(self, max_records_per_model: int = 1000) -> None:
        self._max = max_records_per_model
        self._records: dict[str, deque[RequestRecord]] = defaultdict(
            lambda: deque(maxlen=self._max)
        )

    def record(self, rec: RequestRecord) -> None:
        self._records[rec.model_id].append(rec)

    def get_metrics(
        self, model_id: str, window_seconds: float = 3600.0
    ) -> ModelMetrics:
        cutoff = time.monotonic() - window_seconds
        records = [
            r for r in self._records.get(model_id, []) if r.timestamp >= cutoff
        ]
        return self._aggregate(model_id, records, window_seconds)

    def get_all_metrics(
        self, window_seconds: float = 3600.0
    ) -> dict[str, ModelMetrics]:
        cutoff = time.monotonic() - window_seconds
        result: dict[str, ModelMetrics] = {}
        for model_id, recs in self._records.items():
            filtered = [r for r in recs if r.timestamp >= cutoff]
            result[model_id] = self._aggregate(model_id, filtered, window_seconds)
        return result

    @staticmethod
    def _aggregate(
        model_id: str,
        records: list[RequestRecord],
        window: float,
    ) -> ModelMetrics:
        if not records:
            return ModelMetrics(model_id=model_id, window_seconds=window)

        n = len(records)
        latencies = sorted(r.latency_ms for r in records if r.success)
        ttfts = [r.ttft_ms for r in records if r.success and r.ttft_ms > 0]
        costs = [r.cost for r in records]
        quality_scores = [
            r.quality_score for r in records if r.quality_score is not None
        ]
        errors = sum(1 for r in records if not r.success)
        throttles = sum(1 for r in records if r.is_throttle)

        return ModelMetrics(
            model_id=model_id,
            window_seconds=window,
            avg_latency_ms=sum(latencies) / max(1, len(latencies)),
            p50_latency_ms=_percentile(latencies, 50),
            p95_latency_ms=_percentile(latencies, 95),
            avg_ttft_ms=sum(ttfts) / max(1, len(ttfts)) if ttfts else 0.0,
            avg_cost_per_request=sum(costs) / n if costs else 0.0,
            error_rate=errors / n,
            throttle_rate=throttles / n,
            avg_quality_score=(
                sum(quality_scores) / len(quality_scores)
                if quality_scores
                else None
            ),
            sample_count=n,
        )
