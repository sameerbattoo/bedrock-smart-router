# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canary deployments — gradually roll out a new model with auto-rollback.

Sends a configurable percentage of traffic to a canary model while
monitoring error rate and latency.  If the canary exceeds rollback
thresholds, it is automatically disabled.
"""

from __future__ import annotations

import logging
import random
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CanaryThresholds:
    """Thresholds for auto-promote and auto-rollback."""

    min_requests: int = 100
    max_error_rate: float = 0.05
    max_latency_p95_ms: float = 5000.0
    min_quality_baseline: float = 0.0  # Minimum AA Intelligence Index score (0 = don't check)


@dataclass
class CanaryConfig:
    """Canary deployment configuration."""

    enabled: bool = False
    baseline_model: str = ""
    canary_model: str = ""
    canary_percentage: float = 5.0  # 0–100
    auto_rollback: CanaryThresholds = field(
        default_factory=lambda: CanaryThresholds(max_error_rate=0.10)
    )
    auto_promote: CanaryThresholds = field(
        default_factory=lambda: CanaryThresholds(
            min_requests=100, max_error_rate=0.02
        )
    )


@dataclass
class _CanaryRecord:
    timestamp: float
    latency_ms: float
    success: bool


class CanaryManager:
    """Manages canary traffic splitting and health monitoring."""

    def __init__(self, config: CanaryConfig | None = None) -> None:
        self.config = config or CanaryConfig()
        self._canary_records: deque[_CanaryRecord] = deque(maxlen=5000)
        self._baseline_records: deque[_CanaryRecord] = deque(maxlen=5000)
        self._rolled_back: bool = False
        self._promoted: bool = False

    @property
    def is_active(self) -> bool:
        return (
            self.config.enabled
            and bool(self.config.baseline_model)
            and bool(self.config.canary_model)
            and not self._rolled_back
            and not self._promoted
        )

    def select_model(self) -> tuple[str, bool]:
        """Select baseline or canary model.

        Returns:
            Tuple of (model_id, is_canary).
        """
        if not self.is_active:
            return self.config.baseline_model, False

        if random.random() * 100 < self.config.canary_percentage:
            return self.config.canary_model, True
        return self.config.baseline_model, False

    def record_result(
        self,
        is_canary: bool,
        latency_ms: float,
        success: bool,
    ) -> None:
        """Record a request result and check rollback thresholds."""
        rec = _CanaryRecord(
            timestamp=time.monotonic(),
            latency_ms=latency_ms,
            success=success,
        )
        if is_canary:
            self._canary_records.append(rec)
            self._check_rollback()
            self._check_promote()
        else:
            self._baseline_records.append(rec)

    def _check_rollback(self) -> None:
        """Auto-rollback if canary exceeds error thresholds."""
        records = list(self._canary_records)
        if len(records) < 10:  # Need minimum data
            return

        error_rate = sum(1 for r in records if not r.success) / len(records)
        if error_rate > self.config.auto_rollback.max_error_rate:
            logger.warning(
                "Canary %s rolled back: error rate %.1f%% > %.1f%%",
                self.config.canary_model,
                error_rate * 100,
                self.config.auto_rollback.max_error_rate * 100,
            )
            self._rolled_back = True
            return

        latencies = sorted(r.latency_ms for r in records if r.success)
        if latencies:
            p95_idx = int(len(latencies) * 0.95)
            p95 = latencies[min(p95_idx, len(latencies) - 1)]
            if p95 > self.config.auto_rollback.max_latency_p95_ms:
                logger.warning(
                    "Canary %s rolled back: P95 latency %.0fms > %.0fms",
                    self.config.canary_model, p95,
                    self.config.auto_rollback.max_latency_p95_ms,
                )
                self._rolled_back = True

    def _check_promote(self) -> None:
        """Auto-promote if canary meets all promotion thresholds."""
        records = list(self._canary_records)
        thresholds = self.config.auto_promote

        if len(records) < thresholds.min_requests:
            return

        error_rate = sum(1 for r in records if not r.success) / len(records)
        if error_rate > thresholds.max_error_rate:
            return

        latencies = sorted(r.latency_ms for r in records if r.success)
        if latencies:
            p95_idx = int(len(latencies) * 0.95)
            p95 = latencies[min(p95_idx, len(latencies) - 1)]
            if p95 > thresholds.max_latency_p95_ms:
                return

        logger.info(
            "Canary %s promoted: %d requests, %.1f%% error rate, P95=%.0fms",
            self.config.canary_model, len(records),
            error_rate * 100, latencies[-1] if latencies else 0,
        )
        self._promoted = True

    @property
    def is_rolled_back(self) -> bool:
        return self._rolled_back

    @property
    def is_promoted(self) -> bool:
        return self._promoted

    @property
    def stats(self) -> dict[str, Any]:
        canary = list(self._canary_records)
        baseline = list(self._baseline_records)
        # Calculate P95 latency for canary
        canary_latencies = sorted(r.latency_ms for r in canary if r.success)
        canary_p95 = 0.0
        if canary_latencies:
            p95_idx = int(len(canary_latencies) * 0.95)
            canary_p95 = canary_latencies[min(p95_idx, len(canary_latencies) - 1)]
        return {
            "active": self.is_active,
            "rolled_back": self._rolled_back,
            "promoted": self._promoted,
            "canary_model": self.config.canary_model,
            "baseline_model": self.config.baseline_model,
            "canary_requests": len(canary),
            "baseline_requests": len(baseline),
            "canary_error_rate": (
                sum(1 for r in canary if not r.success) / len(canary)
                if canary else 0.0
            ),
            "canary_p95_latency_ms": round(canary_p95, 0),
            "max_latency_threshold_ms": self.config.auto_rollback.max_latency_p95_ms,
        }
