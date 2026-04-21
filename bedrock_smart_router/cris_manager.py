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

    def select_profile(self, model: BedrockModel) -> str:
        """Return the best inference profile ID for the model.

        Priority:
        1. Geography-preferred profile (e.g. ``us.`` if preferred_geography="us")
        2. Global profile (``global.``) if allowed
        3. Any available CRIS profile
        4. The raw model_id (no CRIS)
        """
        if not self.config.enabled or not model.cris_profiles:
            return model.model_id

        profiles = model.cris_profiles
        pref = self.config.preferred_geography

        # 1. Exact geography match
        if pref:
            for p in profiles:
                if p.startswith(f"{pref}."):
                    return p

        # 2. Global profile
        if self.config.allow_global:
            for p in profiles:
                if p.startswith("global."):
                    return p

        # 3. Any CRIS profile (prefer us. as a sensible default)
        for prefix in ("us.", "eu.", "ap."):
            for p in profiles:
                if p.startswith(prefix):
                    return p

        # 4. First available
        return profiles[0] if profiles else model.model_id

    def get_geography(self, profile_id: str) -> str | None:
        """Extract the geography prefix from a profile ID."""
        for prefix in ("us.", "eu.", "ap.", "global."):
            if profile_id.startswith(prefix):
                return prefix.rstrip(".")
        return None
