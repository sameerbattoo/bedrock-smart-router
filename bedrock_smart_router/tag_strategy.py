# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tag-based routing strategy.

Routes requests to specific models based on request tags.  Useful for
free/paid tiers, team-based access control, and data residency rules.

Configuration example::

    tag_routing:
      paid-tier: ["us.anthropic.claude-sonnet-4-6", "us.amazon.nova-pro-v1:0"]
      free-tier: ["us.amazon.nova-micro-v1:0", "us.amazon.nova-lite-v1:0"]
      team-alpha: ["us.anthropic.*"]
"""

from __future__ import annotations

import fnmatch
import logging

from bedrock_smart_router.models import BedrockModel, RequestAnalysis
from bedrock_smart_router.strategy_engine import RoutingStrategy, StrategyResult

logger = logging.getLogger(__name__)


class TagRoutingStrategy(RoutingStrategy):
    """Filter candidates by tag rules, then delegate to an inner strategy."""

    name = "tag-based"

    @property
    def weights(self) -> dict[str, float]:
        return self.inner.weights

    def score_model(self, model, analysis, context):
        return {}  # Delegates to inner strategy via select() override

    def __init__(
        self,
        tag_rules: dict[str, list[str]],
        inner: RoutingStrategy | None = None,
    ) -> None:
        """
        Args:
            tag_rules: Mapping of tag name to list of allowed model ID
                patterns (glob syntax).
            inner: Strategy to use after filtering.  Defaults to the
                balanced strategy.
        """
        from bedrock_smart_router.strategy_engine import BalancedStrategy

        self.tag_rules = tag_rules
        self.inner = inner or BalancedStrategy()

    def select(
        self,
        candidates: list[BedrockModel],
        analysis: RequestAnalysis,
        tags: list[str] | None = None,
    ) -> StrategyResult:
        if not tags:
            return self.inner.select(candidates, analysis)

        # Intersect allowed models across all tags
        allowed_patterns: set[str] = set()
        for tag in tags:
            patterns = self.tag_rules.get(tag)
            if patterns is not None:
                allowed_patterns.update(patterns)

        if not allowed_patterns:
            logger.debug("No tag rules matched for tags=%s, using all candidates", tags)
            return self.inner.select(candidates, analysis)

        filtered = [
            m
            for m in candidates
            if any(fnmatch.fnmatch(m.model_id, pat) for pat in allowed_patterns)
        ]

        if not filtered:
            logger.warning(
                "Tag filter left no candidates (tags=%s, patterns=%s), "
                "falling back to all candidates",
                tags,
                allowed_patterns,
            )
            filtered = candidates

        return self.inner.select(filtered, analysis)
