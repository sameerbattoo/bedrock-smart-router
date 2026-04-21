"""Inference tier selector.

Bedrock offers three on-demand tiers:
  - **Standard**: everyday workloads, regular pricing
  - **Priority**: up to 25% better OTPS latency, premium pricing
  - **Flex**: discounted pricing for latency-tolerant workloads

The tier selector picks the right tier based on request urgency,
budget constraints, and model support.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from bedrock_smart_router.models import BedrockModel, RequestAnalysis

logger = logging.getLogger(__name__)


@dataclass
class InferenceTierConfig:
    """Inference tier selection configuration."""

    enabled: bool = True
    default_tier: str = "standard"
    allow_priority: bool = True
    allow_flex: bool = True
    # Use priority tier when estimated latency matters (e.g. real-time chat)
    priority_for_complex: bool = True
    # Use flex tier for batch-eligible requests
    flex_for_batch: bool = True


class InferenceTierSelector:
    """Selects the optimal inference tier for a request + model pair."""

    def __init__(self, config: InferenceTierConfig | None = None) -> None:
        self.config = config or InferenceTierConfig()

    def select_tier(
        self,
        model: BedrockModel,
        analysis: RequestAnalysis,
        *,
        max_cost_per_request: float | None = None,
    ) -> str:
        """Return the best inference tier for this request.

        Returns one of ``"standard"``, ``"priority"``, or ``"flex"``.
        """
        if not self.config.enabled:
            return self.config.default_tier

        supported = set(model.supported_inference_tiers)

        # Flex: batch-eligible, latency-tolerant workloads
        if (
            self.config.allow_flex
            and self.config.flex_for_batch
            and "flex" in supported
            and not analysis.requires_streaming
            and analysis.complexity.value in ("simple", "moderate")
        ):
            # If there's a tight budget, prefer flex for savings
            if max_cost_per_request is not None:
                estimated = model.pricing.estimate_cost(
                    analysis.estimated_input_tokens,
                    analysis.estimated_output_tokens,
                )
                if estimated > max_cost_per_request * 0.8:
                    logger.debug("Selecting flex tier for budget savings")
                    return "flex"

        # Priority: complex/reasoning tasks where latency matters
        if (
            self.config.allow_priority
            and self.config.priority_for_complex
            and "priority" in supported
            and analysis.complexity.value in ("complex", "reasoning")
        ):
            logger.debug("Selecting priority tier for complex request")
            return "priority"

        # Standard: default
        if "standard" in supported:
            return "standard"

        # Fallback to whatever is available
        return model.supported_inference_tiers[0] if model.supported_inference_tiers else "standard"
