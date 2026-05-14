"""Multi-level fallback handler.

Fallback chain:
  1. Primary model (selected by strategy)
  2. Same-family downgrade (e.g., Sonnet -> Haiku)
  3. Cross-family equivalent tier (e.g., Sonnet -> Nova Pro)
  4. CRIS profile retry
  5. Default safe model

Geography deduplication: if the primary is ``global.anthropic.claude-sonnet-4-6``,
the chain will not include ``us.anthropic.claude-sonnet-4-6`` (same underlying
model, different CRIS profile — not a useful fallback).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from bedrock_smart_router.models import BedrockModel, Tier
from bedrock_smart_router.model_registry import ModelRegistry, base_model_id

logger = logging.getLogger(__name__)

_TIER_ORDER = list(Tier)


@dataclass
class FallbackConfig:
    """Fallback behaviour configuration."""

    enabled: bool = True
    max_depth: int = 5
    default_safe_model: str = "amazon.nova-micro-v1:0"
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

        The chain is de-duplicated by base model identity — geography
        variants (``us.*`` vs ``global.*``) of the same model are
        treated as duplicates since they hit the same underlying model.
        """
        if not self.config.enabled:
            return []

        chain: list[BedrockModel] = []
        seen_ids: set[str] = {primary.model_id}
        seen_base: set[str] = {primary.base_model_id}

        def _add(model: BedrockModel | None) -> None:
            if model is None:
                return
            bm = model.base_model_id
            if model.model_id not in seen_ids and bm not in seen_base:
                chain.append(model)
                seen_ids.add(model.model_id)
                seen_base.add(bm)

        # Level 1 — strategy-provided fallbacks (already scored)
        for fb in (strategy_fallbacks or []):
            _add(fb)

        # Level 2 — same-family downgrade
        same_family = self.registry.list_models(family=primary.family)
        primary_idx = _TIER_ORDER.index(primary.tier)
        # Sort by quality_baseline (highest first) within each tier
        for t in reversed(_TIER_ORDER[:primary_idx]):
            tier_models = sorted(
                [m for m in same_family if m.tier == t],
                key=lambda m: m.quality_baseline,
                reverse=True,
            )
            for m in tier_models:
                _add(m)

        # Level 3 — cross-family equivalent tier
        for t in [primary.tier] + list(reversed(_TIER_ORDER[:primary_idx])):
            cross_family = sorted(
                [m for m in self.registry.list_models(tier=t) if m.family != primary.family],
                key=lambda m: m.quality_baseline,
                reverse=True,
            )
            for m in cross_family:
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
        # Deduplicate by base model — prefer cheapest variant
        seen_base: set[str] = set()
        candidates: list[BedrockModel] = []
        for m in self.registry.all_models:
            if m.max_input_tokens < estimated_tokens or m.model_id in exclude:
                continue
            bm = m.base_model_id
            if bm in seen_base:
                continue
            seen_base.add(bm)
            candidates.append(m)
        if not candidates:
            return None
        candidates.sort(key=lambda m: m.pricing.input_per_1k)
        return candidates[0]
