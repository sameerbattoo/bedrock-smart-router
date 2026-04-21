"""Model registry — catalog of Bedrock models with capabilities and pricing."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Sequence

from bedrock_smart_router.models import (
    BedrockModel,
    ModelCapabilities,
    ModelPricing,
    Tier,
)

logger = logging.getLogger(__name__)

# Path to the bundled JSON catalog shipped with the package
_DEFAULT_CATALOG_PATH = Path(__file__).parent / "data" / "models.json"


def _model_from_dict(d: dict[str, Any]) -> BedrockModel:
    """Deserialise a single model entry from the JSON catalog."""
    caps = d.get("capabilities", {})
    pricing = d.get("pricing", {})
    return BedrockModel(
        model_id=d["model_id"],
        family=d["family"],
        tier=Tier(d["tier"]),
        display_name=d["display_name"],
        capabilities=ModelCapabilities(
            tool_use=caps.get("tool_use", False),
            vision=caps.get("vision", False),
            streaming=caps.get("streaming", True),
            document_support=caps.get("document_support", False),
            extended_thinking=caps.get("extended_thinking", False),
        ),
        max_input_tokens=d.get("max_input_tokens", 4096),
        max_output_tokens=d.get("max_output_tokens", 4096),
        pricing=ModelPricing(
            input_per_1k=pricing.get("input_per_1k", 0.0),
            output_per_1k=pricing.get("output_per_1k", 0.0),
            cache_read_per_1k=pricing.get("cache_read_per_1k", 0.0),
            cache_write_per_1k=pricing.get("cache_write_per_1k", 0.0),
        ),
        supports_prompt_caching=d.get("supports_prompt_caching", False),
        supports_extended_thinking=d.get("supports_extended_thinking", False),
        cris_profiles=d.get("cris_profiles", []),
        supported_inference_tiers=d.get("supported_inference_tiers", ["standard"]),
        guardrail_compatible=d.get("guardrail_compatible", True),
    )


def load_catalog(path: Path | str | None = None) -> list[BedrockModel]:
    """Load a model catalog from a JSON file.

    Args:
        path: Path to a JSON catalog file.  When *None* the bundled
            ``data/models.json`` shipped with the package is used.

    Returns:
        A list of :class:`BedrockModel` instances.
    """
    catalog_path = Path(path) if path else _DEFAULT_CATALOG_PATH
    try:
        with open(catalog_path, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        logger.warning("Catalog file not found at %s, using empty catalog", catalog_path)
        return []
    except json.JSONDecodeError as exc:
        logger.error("Invalid JSON in catalog %s: %s", catalog_path, exc)
        return []

    models: list[BedrockModel] = []
    for entry in data.get("models", []):
        try:
            models.append(_model_from_dict(entry))
        except (KeyError, ValueError) as exc:
            logger.warning("Skipping invalid model entry: %s", exc)
    logger.info("Loaded %d models from %s", len(models), catalog_path)
    return models


# ── Tier-based quality heuristics (used when no historical data) ────
TIER_QUALITY_HEURISTIC: dict[Tier, float] = {
    Tier.MICRO: 0.55,
    Tier.LITE: 0.70,
    Tier.MID: 0.82,
    Tier.HEAVY: 0.90,
    Tier.REASONING: 0.93,
}

# ── Minimum tier required per complexity level ──────────────────────
COMPLEXITY_MIN_TIER: dict[str, Tier] = {
    "simple": Tier.MICRO,
    "moderate": Tier.LITE,
    "complex": Tier.MID,
    "reasoning": Tier.REASONING,
}


class ModelRegistry:
    """In-memory catalog of Bedrock models.

    Initialised from the bundled JSON catalog (``data/models.json``) by
    default.  Users can supply their own catalog path or a pre-built
    list of models.
    """

    def __init__(
        self,
        models: Sequence[BedrockModel] | None = None,
        catalog_path: Path | str | None = None,
    ) -> None:
        self._models: dict[str, BedrockModel] = {}
        if models is not None:
            source = models
        else:
            source = load_catalog(catalog_path)
        for m in source:
            self._models[m.model_id] = m

    def load_overlay(self, path: Path | str) -> int:
        """Merge an additional JSON catalog on top of the current one.

        Models in *path* override existing entries with the same
        ``model_id``.  Returns the number of models loaded.
        """
        overlay = load_catalog(path)
        for m in overlay:
            self._models[m.model_id] = m
        logger.info("Overlay loaded %d models from %s", len(overlay), path)
        return len(overlay)

    # ── Queries ─────────────────────────────────────────────────

    def get(self, model_id: str) -> BedrockModel | None:
        return self._models.get(model_id)

    def list_models(
        self,
        *,
        family: str | None = None,
        tier: Tier | str | None = None,
        min_tier: Tier | str | None = None,
        capability: str | None = None,
        min_context: int | None = None,
    ) -> list[BedrockModel]:
        """Return models matching the given filters."""
        tier_order = list(Tier)
        if isinstance(tier, str):
            tier = Tier(tier)
        if isinstance(min_tier, str):
            min_tier = Tier(min_tier)

        results: list[BedrockModel] = []
        for m in self._models.values():
            if family and m.family != family:
                continue
            if tier and m.tier != tier:
                continue
            if min_tier and tier_order.index(m.tier) < tier_order.index(min_tier):
                continue
            if capability and not getattr(m.capabilities, capability, False):
                continue
            if min_context and m.max_input_tokens < min_context:
                continue
            results.append(m)
        return results

    def eligible_models(
        self,
        *,
        min_tier: Tier | str | None = None,
        requires_vision: bool = False,
        requires_tool_use: bool = False,
        min_context: int | None = None,
        exclude_patterns: list[str] | None = None,
        family: str | None = None,
    ) -> list[BedrockModel]:
        """Return models that meet the given requirements."""
        import fnmatch

        if isinstance(min_tier, str):
            min_tier = Tier(min_tier)

        tier_order = list(Tier)
        results: list[BedrockModel] = []
        for m in self._models.values():
            if min_tier and tier_order.index(m.tier) < tier_order.index(min_tier):
                continue
            if requires_vision and not m.capabilities.vision:
                continue
            if requires_tool_use and not m.capabilities.tool_use:
                continue
            if min_context and m.max_input_tokens < min_context:
                continue
            if family and m.family != family:
                continue
            if exclude_patterns:
                skip = False
                for pat in exclude_patterns:
                    if fnmatch.fnmatch(m.model_id, pat):
                        skip = True
                        break
                if skip:
                    continue
            results.append(m)
        return results

    # ── Mutations ───────────────────────────────────────────────

    def register(self, model: BedrockModel) -> None:
        """Add or replace a model in the registry."""
        self._models[model.model_id] = model
        logger.info("Registered model %s", model.model_id)

    def update_pricing(self, model_id: str, pricing: ModelPricing) -> None:
        """Update pricing for an existing model."""
        m = self._models.get(model_id)
        if m is None:
            logger.warning("Cannot update pricing: model %s not found", model_id)
            return
        # BedrockModel is not frozen, so we can replace the pricing field
        m.pricing = pricing  # type: ignore[misc]

    @property
    def all_models(self) -> list[BedrockModel]:
        return list(self._models.values())

    def __len__(self) -> int:
        return len(self._models)
