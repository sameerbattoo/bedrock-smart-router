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
    blocked_prefixes: list[str] | None = None  # Prefixes to never use (e.g., ["global", "eu"])
    allowed_prefixes: list[str] | None = None  # If set, ONLY these prefixes are permitted (allowlist)


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

        # If model_id already has a geography prefix, return as-is
        for prefix in ("us.", "eu.", "ap.", "global.", "apac.", "au.", "ca.", "jp."):
            if model.model_id.startswith(prefix):
                return model.model_id

        blocked = set(self.config.blocked_prefixes or [])
        allowed = set(self.config.allowed_prefixes) if self.config.allowed_prefixes else None

        def _is_prefix_allowed(prefix: str) -> bool:
            """Check if a prefix passes both allowlist and blocklist."""
            if prefix in blocked:
                return False
            if allowed is not None and prefix not in allowed:
                return False
            if prefix == "global" and not self.config.allow_global:
                return False
            return True

        # Find the region entry for the user's region
        region_entry = None
        for r in model.regions:
            if r.get("name") == region:
                region_entry = r
                break

        # If no entry for this region, model may not be available here
        if not region_entry:
            pref = self.config.preferred_geography
            if pref and _is_prefix_allowed(pref):
                for r in model.regions:
                    if pref in r.get("cris_profiles", []):
                        return f"{pref}.{model.model_id}"
            if _is_prefix_allowed("global"):
                for r in model.regions:
                    if "global" in r.get("cris_profiles", []):
                        return f"global.{model.model_id}"
            return model.model_id

        # We have a region entry — pick the best profile
        cris_prefixes = region_entry.get("cris_profiles", [])
        if not cris_prefixes:
            return model.model_id

        # Filter to allowed prefixes
        available_prefixes = [p for p in cris_prefixes if _is_prefix_allowed(p)]
        if not available_prefixes:
            # All CRIS prefixes are blocked/not allowed — use direct invocation
            return model.model_id

        pref = self.config.preferred_geography

        # 1. Preferred geography
        if pref and pref in available_prefixes:
            return f"{pref}.{model.model_id}"

        # 2. Global (if allowed)
        if "global" in available_prefixes:
            return f"global.{model.model_id}"

        # 3. Any available prefix
        return f"{available_prefixes[0]}.{model.model_id}"

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
