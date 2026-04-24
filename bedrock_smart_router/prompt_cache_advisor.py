"""Prompt cache advisor.

Estimates the cost benefit of Bedrock's native prompt caching for a
given request.  When the benefit is significant, the strategy engine
can boost the score of cache-capable models.

Bedrock prompt caching caches the prefix (system prompt + early
conversation turns) server-side, so subsequent requests with the same
prefix pay the cheaper ``cache_read`` rate instead of the full
``input`` rate.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from bedrock_smart_router.models import BedrockModel

logger = logging.getLogger(__name__)

# Rough chars-per-token for estimation
_CHARS_PER_TOKEN = 4


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // _CHARS_PER_TOKEN)


@dataclass
class CacheBenefit:
    """Estimated benefit of prompt caching for a request + model."""

    model_id: str
    cacheable_tokens: int
    savings_per_request: float  # USD saved per request vs no caching
    cache_eligible: bool
    reason: str


class PromptCacheAdvisor:
    """Estimates prompt caching benefit for routing decisions."""

    def __init__(self, min_savings_threshold: float = 0.0005) -> None:
        """
        Args:
            min_savings_threshold: Minimum USD savings per request to
                consider caching worthwhile (default $0.0005).
        """
        self.threshold = min_savings_threshold

    def estimate(
        self,
        model: BedrockModel,
        messages: list[dict[str, Any]],
        system: list[dict[str, Any]] | None = None,
    ) -> CacheBenefit:
        """Estimate the caching benefit for a model + request pair."""
        if not model.capabilities.prompt_caching:
            return CacheBenefit(
                model_id=model.model_id,
                cacheable_tokens=0,
                savings_per_request=0.0,
                cache_eligible=False,
                reason="Model does not support prompt caching",
            )

        if model.pricing.cache_read_per_1k <= 0:
            return CacheBenefit(
                model_id=model.model_id,
                cacheable_tokens=0,
                savings_per_request=0.0,
                cache_eligible=False,
                reason="No cache pricing available",
            )

        # Estimate cacheable tokens: system prompt + all but the last user message
        cacheable_chars = 0
        if system:
            for block in system:
                if isinstance(block, dict) and "text" in block:
                    cacheable_chars += len(block["text"])

        # All messages except the last user message are cacheable prefix
        if len(messages) > 1:
            for msg in messages[:-1]:
                content = msg.get("content", [])
                if isinstance(content, str):
                    cacheable_chars += len(content)
                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and "text" in block:
                            cacheable_chars += len(block["text"])

        cacheable_tokens = _estimate_tokens("x" * cacheable_chars) if cacheable_chars > 0 else 0

        if cacheable_tokens < 100:
            return CacheBenefit(
                model_id=model.model_id,
                cacheable_tokens=cacheable_tokens,
                savings_per_request=0.0,
                cache_eligible=False,
                reason=f"Only {cacheable_tokens} cacheable tokens (minimum ~100 for benefit)",
            )

        # Savings = cacheable_tokens * (input_price - cache_read_price) / 1000
        savings = (
            cacheable_tokens
            * (model.pricing.input_per_1k - model.pricing.cache_read_per_1k)
            / 1000
        )

        return CacheBenefit(
            model_id=model.model_id,
            cacheable_tokens=cacheable_tokens,
            savings_per_request=round(savings, 6),
            cache_eligible=savings >= self.threshold,
            reason=(
                f"${savings:.6f}/req savings on {cacheable_tokens} cached tokens"
                if savings >= self.threshold
                else f"Savings ${savings:.6f} below threshold ${self.threshold}"
            ),
        )

    def rank_models_by_cache_benefit(
        self,
        models: list[BedrockModel],
        messages: list[dict[str, Any]],
        system: list[dict[str, Any]] | None = None,
    ) -> list[tuple[BedrockModel, CacheBenefit]]:
        """Rank models by their caching benefit, highest first."""
        results = [
            (m, self.estimate(m, messages, system)) for m in models
        ]
        results.sort(key=lambda x: x[1].savings_per_request, reverse=True)
        return results
