"""Distilled model support for the model registry.

Bedrock Model Distillation produces smaller, faster models from a
teacher model with minimal quality loss (up to 500% faster, 75%
cheaper, <2% accuracy loss).

This module helps the registry track distilled variants so the
strategy engine can consider them as routing candidates — a distilled
Nova Pro might be a better choice than the base Nova Lite for a
workload where the distilled model was specifically trained.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from bedrock_smart_router.model_registry import ModelRegistry
from bedrock_smart_router.models import (
    BedrockModel,
    ModelCapabilities,
    ModelPricing,
    Tier,
)

logger = logging.getLogger(__name__)


@dataclass
class DistilledModelInfo:
    """Metadata about a distilled model variant."""

    model_id: str
    teacher_model_id: str
    display_name: str
    tier: Tier
    quality_delta: float  # e.g. -0.02 means 2% quality loss vs teacher
    speed_multiplier: float  # e.g. 5.0 means 5x faster than teacher
    cost_multiplier: float  # e.g. 0.25 means 75% cheaper than teacher


class DistilledModelManager:
    """Registers distilled model variants into the model registry.

    Users can register their custom distilled models so the router
    considers them alongside base models during routing decisions.
    """

    def __init__(self, registry: ModelRegistry) -> None:
        self._registry = registry

    def register_distilled(
        self,
        model_id: str,
        teacher_model_id: str,
        *,
        display_name: str | None = None,
        quality_delta: float = -0.02,
        speed_multiplier: float = 3.0,
        cost_multiplier: float = 0.5,
        capabilities: ModelCapabilities | None = None,
        max_input_tokens: int | None = None,
        max_output_tokens: int | None = None,
    ) -> BedrockModel:
        """Register a distilled model derived from a teacher.

        Pricing and capabilities are derived from the teacher model
        unless explicitly overridden.
        """
        teacher = self._registry.get(teacher_model_id)
        if teacher is None:
            raise ValueError(
                f"Teacher model '{teacher_model_id}' not found in registry"
            )

        # Derive pricing from teacher
        teacher_pricing = teacher.pricing
        derived_pricing = ModelPricing(
            input_per_1k=round(teacher_pricing.input_per_1k * cost_multiplier, 8),
            output_per_1k=round(teacher_pricing.output_per_1k * cost_multiplier, 8),
            cache_read_per_1k=round(teacher_pricing.cache_read_per_1k * cost_multiplier, 8),
            cache_write_per_1k=round(teacher_pricing.cache_write_per_1k * cost_multiplier, 8),
        )

        # Derive tier: one step down from teacher
        tier_order = list(Tier)
        teacher_idx = tier_order.index(teacher.tier)
        derived_tier = tier_order[max(0, teacher_idx - 1)]

        distilled = BedrockModel(
            model_id=model_id,
            family=teacher.family,
            tier=derived_tier,
            display_name=display_name or f"{teacher.display_name} (distilled)",
            capabilities=capabilities or teacher.capabilities,
            max_input_tokens=max_input_tokens or teacher.max_input_tokens,
            max_output_tokens=max_output_tokens or teacher.max_output_tokens,
            pricing=derived_pricing,
            supports_prompt_caching=teacher.supports_prompt_caching,
            supports_extended_thinking=False,  # Distilled models typically don't
            cris_profiles=[],  # Custom models don't have CRIS
            supported_inference_tiers=["standard"],
            guardrail_compatible=teacher.guardrail_compatible,
            distilled_from=teacher_model_id,
            distilled_quality_delta=quality_delta,
        )

        self._registry.register(distilled)
        logger.info(
            "Registered distilled model %s (teacher=%s, quality_delta=%.2f, "
            "cost_multiplier=%.2f)",
            model_id, teacher_model_id, quality_delta, cost_multiplier,
        )
        return distilled

    def list_distilled(self) -> list[BedrockModel]:
        """Return all distilled models in the registry."""
        return [
            m for m in self._registry.all_models
            if m.distilled_from is not None
        ]
