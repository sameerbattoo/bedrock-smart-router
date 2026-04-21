"""Multi-level fallback handler.

Fallback chain:
  1. Primary model (selected by strategy)
  2. Same-family downgrade (e.g., Sonnet -> Haiku)
  3. Cross-family equivalent tier (e.g., Sonnet -> Nova Pro)
  4. CRIS profile retry
  5. Default safe model
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from bedrock_smart_router.models import BedrockModel, Tier
from bedrock_smart_router.model_registry import ModelRegistry

logger = logging.getLogger(__name__)

_TIER_ORDER = list(Tier)


@dataclass
class FallbackConfig:
    """Fallback behaviour configuration."""

    enabled: bool = True
    max_depth: int = 5
    default_safe_model: str = "us.amazon.nova-lite-v1:0"
    context_window_fallback: bool = True
    content_policy_fallback: bool = True


class FallbackHandler:
    """Builds and walks a fallback chain when the primary model fails."""

    def __init__(
        self,
        registry: ModelRegistry,
        config: FallbackConfig | None = None,
    ) -> None:
        self.registry = registry
        self.config = config or FallbackConfig()

    def build_chain(
        self,
        primary: BedrockModel,
        strategy_fallbacks: list[BedrockModel] | None = None,
    ) -> list[BedrockModel]:
        """Build an ordered fallback chain for *primary*.

        The chain is de-duplicated and capped at ``config.max_depth``.
        """
        if not self.config.enabled:
            return []

        chain: list[BedrockModel] = []
        seen: set[str] = {primary.model_id}

        def _add(model: BedrockModel | None) -> None:
            if model and model.model_id not in seen:
                chain.append(model)
                seen.add(model.model_id)

        # Level 1 — strategy-provided fallbacks (already scored)
        for fb in (strategy_fallbacks or []):
            _add(fb)

        # Level 2 — same-family downgrade
        same_family = self.registry.list_models(family=primary.family)
        primary_idx = _TIER_ORDER.index(primary.tier)
        for t in reversed(_TIER_ORDER[:primary_idx]):
            for m in same_family:
                if m.tier == t:
                    _add(m)
                    break

        # Level 3 — cross-family equivalent tier
        for t in [primary.tier] + list(reversed(_TIER_ORDER[:primary_idx])):
            for m in self.registry.list_models(tier=t):
                if m.family != primary.family:
                    _add(m)

        # Level 4 — default safe model
        safe = self.registry.get(self.config.default_safe_model)
        _add(safe)

        return chain[: self.config.max_depth]

    def find_context_window_fallback(
        self,
        estimated_tokens: int,
        exclude: set[str] | None = None,
    ) -> BedrockModel | None:
        """Find a model with a large enough context window."""
        if not self.config.context_window_fallback:
            return None
        exclude = exclude or set()
        candidates = [
            m
            for m in self.registry.all_models
            if m.max_input_tokens >= estimated_tokens and m.model_id not in exclude
        ]
        if not candidates:
            return None
        # Prefer cheapest model that fits
        candidates.sort(key=lambda m: m.pricing.input_per_1k)
        return candidates[0]
