"""Latency mode selector.

Bedrock offers two latency modes via performanceConfig:
  - **standard**: default, no performanceConfig needed
  - **optimized**: lower latency via performanceConfig={"latency": "optimized"}

The selector picks the right mode based on request complexity
and model support.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from bedrock_smart_router.models import BedrockModel, RequestAnalysis

logger = logging.getLogger(__name__)


@dataclass
class InferenceTierConfig:
    """Latency mode selection configuration."""

    enabled: bool = True
    default_tier: str = "standard"
    allow_optimized: bool = True
    # Use optimized latency for complex/reasoning tasks
    optimized_for_complex: bool = True


class InferenceTierSelector:
    """Selects the optimal latency mode for a request + model pair."""

    def __init__(self, config: InferenceTierConfig | None = None) -> None:
        self.config = config or InferenceTierConfig()

    def select_tier(
        self,
        model: BedrockModel,
        analysis: RequestAnalysis,
        *,
        max_cost_per_request: float | None = None,
    ) -> str:
        """Return the best latency mode for this request.

        Returns one of ``"standard"`` or ``"optimized"``.
        """
        if not self.config.enabled:
            return self.config.default_tier

        supported = set(model.supported_latency_modes)

        # Optimized: complex/reasoning tasks where latency matters
        if (
            self.config.allow_optimized
            and self.config.optimized_for_complex
            and "optimized" in supported
            and analysis.complexity.value in ("complex", "reasoning")
        ):
            logger.debug("Selecting optimized latency for complex request")
            return "optimized"

        # Standard: default
        return "standard"
