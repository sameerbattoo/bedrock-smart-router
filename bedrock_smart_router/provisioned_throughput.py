"""Provisioned throughput detector.

Queries Bedrock for active provisioned model throughput and prefers
provisioned capacity when available (already paid for), falling back
to on-demand when provisioned is not available or saturated.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ProvisionedCapacity:
    """A provisioned throughput entry."""

    provisioned_model_arn: str
    model_id: str
    model_units: int
    status: str  # "InService" | "Creating" | etc.
    commitment_expiration: str | None = None


@dataclass
class ProvisionedThroughputConfig:
    """Configuration for provisioned throughput routing."""

    enabled: bool = True
    prefer_provisioned: bool = True  # Route to provisioned when available
    refresh_interval_seconds: float = 300.0  # Re-check every 5 minutes


class ProvisionedThroughputManager:
    """Detects and manages provisioned throughput for routing.

    Queries ``bedrock:ListProvisionedModelThroughputs`` to discover
    active provisioned capacity, then the router can prefer those
    models (since the capacity is already paid for).
    """

    def __init__(
        self,
        config: ProvisionedThroughputConfig | None = None,
        boto_session: Any | None = None,
        region: str = "us-west-2",
    ) -> None:
        self.config = config or ProvisionedThroughputConfig()
        self._region = region
        self._session = boto_session
        self._cache: dict[str, ProvisionedCapacity] = {}
        self._last_refresh: float = 0.0

    def _get_client(self) -> Any:
        if self._session is None:
            import boto3
            self._session = boto3.Session(region_name=self._region)
        return self._session.client("bedrock", region_name=self._region)

    def refresh(self) -> int:
        """Query Bedrock for provisioned throughputs and cache them.

        Returns the number of active provisioned models found.
        """
        if not self.config.enabled:
            return 0

        try:
            client = self._get_client()
            resp = client.list_provisioned_model_throughputs(
                statusEquals="InService"
            )
        except Exception as exc:
            logger.warning("Failed to list provisioned throughputs: %s", exc)
            return 0

        self._cache.clear()
        for item in resp.get("provisionedModelSummaries", []):
            model_arn = item.get("modelArn", "")
            # Extract model ID from ARN
            model_id = model_arn.split("/")[-1] if "/" in model_arn else model_arn
            entry = ProvisionedCapacity(
                provisioned_model_arn=item.get("provisionedModelArn", ""),
                model_id=model_id,
                model_units=item.get("modelUnits", 0),
                status=item.get("status", "Unknown"),
                commitment_expiration=item.get("commitmentExpirationTime"),
            )
            self._cache[model_id] = entry

        self._last_refresh = time.monotonic()
        logger.info("Found %d provisioned throughputs", len(self._cache))
        return len(self._cache)

    def _maybe_refresh(self) -> None:
        """Refresh if the cache is stale."""
        elapsed = time.monotonic() - self._last_refresh
        if elapsed >= self.config.refresh_interval_seconds:
            self.refresh()

    def get_provisioned(self, model_id: str) -> ProvisionedCapacity | None:
        """Return provisioned capacity for a model, or None."""
        self._maybe_refresh()
        return self._cache.get(model_id)

    def has_provisioned(self, model_id: str) -> bool:
        """Check if a model has active provisioned throughput."""
        self._maybe_refresh()
        return model_id in self._cache

    def get_provisioned_model_id(self, model_id: str) -> str:
        """Return the provisioned ARN if available, else the original model_id."""
        entry = self.get_provisioned(model_id)
        if entry and self.config.prefer_provisioned:
            return entry.provisioned_model_arn
        return model_id

    @property
    def all_provisioned(self) -> dict[str, ProvisionedCapacity]:
        self._maybe_refresh()
        return dict(self._cache)
