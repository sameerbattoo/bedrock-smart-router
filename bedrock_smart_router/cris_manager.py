"""CRIS (Cross-Region Inference) profile manager.

Selects the best cross-region inference profile for a model based on
availability and geography preference.  CRIS profiles route requests
across AWS regions for higher throughput and availability at no extra
cost.

Profile naming conventions:
  - ``us.anthropic.claude-sonnet-4-6``   → US regions only
  - ``eu.anthropic.claude-sonnet-4-6``   → EU regions only
  - ``global.anthropic.claude-sonnet-4-6`` → any commercial region
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from bedrock_smart_router.models import BedrockModel

logger = logging.getLogger(__name__)


@dataclass
class CRISConfig:
    """CRIS configuration."""

    enabled: bool = True
    preferred_geography: str | None = None  # "us" | "eu" | "global" | None
    allow_global: bool = True


class CRISManager:
    """Selects the optimal CRIS inference profile for a model."""

    def __init__(self, config: CRISConfig | None = None) -> None:
        self.config = config or CRISConfig()

    def select_profile(self, model: BedrockModel, region: str = "") -> str:
        """Return the best inference profile ID for the model.

        Uses the model's ``regions`` data to find available CRIS profiles
        for the given region. Falls back to geography preference or global.

        Priority:
        1. Exact region match with preferred geography prefix
        2. Exact region match with global prefix
        3. Any CRIS profile available in the region
        4. The raw model_id (direct access, no CRIS)
        """
        if not self.config.enabled or not model.regions:
            return model.model_id

        # Find the region entry for the user's region
        region_entry = None
        for r in model.regions:
            if r.get("name") == region:
                region_entry = r
                break

        # If no entry for this region, model may not be available here
        # Fall back to checking if any profile matches preferred geography
        if not region_entry:
            # Try to find any region with a matching geography prefix
            pref = self.config.preferred_geography
            if pref:
                for r in model.regions:
                    if pref in r.get("cris_profiles", []):
                        return f"{pref}.{model.model_id}"
            # Try global
            if self.config.allow_global:
                for r in model.regions:
                    if "global" in r.get("cris_profiles", []):
                        return f"global.{model.model_id}"
            # Direct access (no CRIS)
            return model.model_id

        # We have a region entry — pick the best profile
        cris_prefixes = region_entry.get("cris_profiles", [])
        if not cris_prefixes:
            # Direct access only in this region
            return model.model_id

        pref = self.config.preferred_geography

        # 1. Preferred geography
        if pref and pref in cris_prefixes:
            return f"{pref}.{model.model_id}"

        # 2. Global
        if self.config.allow_global and "global" in cris_prefixes:
            return f"global.{model.model_id}"

        # 3. Any available prefix (skip global if disabled)
        for prefix in cris_prefixes:
            if prefix == "global" and not self.config.allow_global:
                continue
            return f"{prefix}.{model.model_id}"

        return model.model_id

    def is_available_in_region(self, model: BedrockModel, region: str) -> bool:
        """Check if a model is available (via CRIS or direct) in a given region."""
        for r in model.regions:
            if r.get("name") == region:
                return True
        return False

    def get_geography(self, profile_id: str) -> str | None:
        """Extract the geography prefix from a profile ID."""
        for prefix in ("us.", "eu.", "ap.", "global."):
            if profile_id.startswith(prefix):
                return prefix.rstrip(".")
        return None
