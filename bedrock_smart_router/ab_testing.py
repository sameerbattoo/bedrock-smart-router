# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""A/B testing — split traffic between models for production comparison.

When active, the A/B splitter overrides the strategy engine and assigns
each request to a variant based on weight.  Sticky mode hashes a user
identifier so the same user always sees the same variant.

Results are tagged with the variant name for analysis via the
observability layer.
"""

from __future__ import annotations

import hashlib
import logging
import random
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ABVariant:
    """A single A/B test variant."""

    name: str
    model: str
    weight: float  # 0.0–1.0, weights across variants should sum to 1.0


@dataclass
class ABTestConfig:
    """A/B test configuration."""

    name: str = ""
    variants: list[ABVariant] = field(default_factory=list)
    sticky: bool = True  # Same user_id always gets same variant
    enabled: bool = True


@dataclass
class ABTestResult:
    """Outcome of an A/B split decision."""

    variant_name: str
    model_id: str
    test_name: str


class ABTestManager:
    """Manages A/B test traffic splitting."""

    def __init__(self, config: ABTestConfig | None = None) -> None:
        self.config = config or ABTestConfig(enabled=False)
        self._request_counts: dict[str, int] = {}

    @property
    def is_active(self) -> bool:
        return self.config.enabled and len(self.config.variants) >= 2

    def assign(self, user_id: str | None = None) -> ABTestResult | None:
        """Assign a request to a variant.

        Args:
            user_id: Optional user identifier for sticky assignment.
                When provided and ``sticky=True``, the same user always
                gets the same variant.

        Returns:
            The assigned variant, or *None* if A/B testing is inactive.
        """
        if not self.is_active:
            return None

        variants = self.config.variants

        if self.config.sticky and user_id:
            variant = self._sticky_assign(user_id, variants)
        else:
            variant = self._weighted_random(variants)

        self._request_counts[variant.name] = (
            self._request_counts.get(variant.name, 0) + 1
        )

        return ABTestResult(
            variant_name=variant.name,
            model_id=variant.model,
            test_name=self.config.name,
        )

    def _sticky_assign(
        self, user_id: str, variants: list[ABVariant]
    ) -> ABVariant:
        """Deterministic assignment based on user_id hash."""
        h = int(hashlib.sha256(user_id.encode()).hexdigest(), 16)
        bucket = (h % 10000) / 10000.0  # 0.0–1.0

        cumulative = 0.0
        for v in variants:
            cumulative += v.weight
            if bucket < cumulative:
                return v
        return variants[-1]

    @staticmethod
    def _weighted_random(variants: list[ABVariant]) -> ABVariant:
        """Random assignment weighted by variant weights."""
        r = random.random()
        cumulative = 0.0
        for v in variants:
            cumulative += v.weight
            if r < cumulative:
                return v
        return variants[-1]

    @property
    def stats(self) -> dict[str, Any]:
        total = sum(self._request_counts.values())
        return {
            "test_name": self.config.name,
            "active": self.is_active,
            "total_requests": total,
            "variant_counts": dict(self._request_counts),
            "variant_percentages": {
                k: round(v / total * 100, 1) if total > 0 else 0
                for k, v in self._request_counts.items()
            },
        }
