# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Application Inference Profile (AIP) manager for multi-tenant cost tracking.

Bedrock AIPs are logical wrappers around a model that allow per-tenant
cost allocation via custom tags.  This manager automatically creates
and caches AIPs so the router can invoke Bedrock using the AIP ARN
instead of the raw model ID, enabling Cost Explorer attribution.

IAM permissions required::

    bedrock:CreateInferenceProfile
    bedrock:GetInferenceProfile
    bedrock:ListInferenceProfiles
    bedrock:TagResource
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class AIPConfig:
    """Application Inference Profile configuration."""

    enabled: bool = False
    auto_create: bool = True
    tag_keys: list[str] = field(default_factory=lambda: ["tenant", "team"])
    profile_name_prefix: str = "bsr"  # Prefix for auto-created profile names


@dataclass
class AIPEntry:
    """A cached AIP entry."""

    profile_arn: str
    profile_name: str
    model_id: str
    tags: dict[str, str]


class AIPManager:
    """Manages Application Inference Profiles for multi-tenant routing.

    When a request arrives with tenant metadata, the manager:
    1. Looks up an existing AIP for that tenant+model combination
    2. Creates one if ``auto_create=True`` and none exists
    3. Returns the AIP ARN for the router to use in the Bedrock call
    """

    def __init__(
        self,
        config: AIPConfig | None = None,
        boto_session: Any | None = None,
        region: str = "us-west-2",
    ) -> None:
        self.config = config or AIPConfig()
        self._region = region
        self._session = boto_session
        self._client: Any | None = None
        self._account_id: str | None = None
        # Cache: (model_id, tenant_key) -> AIPEntry
        self._cache: dict[tuple[str, str], AIPEntry] = {}

    def _get_client(self) -> Any:
        if self._client is None:
            if self._session is None:
                import boto3
                self._session = boto3.Session(region_name=self._region)
            self._client = self._session.client("bedrock", region_name=self._region)
        return self._client

    def get_model_id_for_tenant(
        self,
        model_id: str,
        tenant_tags: dict[str, str],
    ) -> str:
        """Return the AIP ARN for a tenant+model, or the raw model_id.

        If AIPs are disabled or no tenant tags are provided, returns
        the original model_id unchanged.
        
        AIPs can only be created for models with system-defined inference
        profiles (CRIS profiles like us.*, global.*, eu.*). Direct-access
        models (no geography prefix) don't support AIPs.
        """
        if not self.config.enabled or not tenant_tags:
            return model_id

        # AIPs require a system-defined inference profile as source.
        # Only models invoked via CRIS profiles (us.*, global.*, eu.*, etc.) support this.
        _CRIS_PREFIXES = ("us.", "global.", "eu.", "ap.", "apac.", "au.", "ca.", "jp.")
        if not any(model_id.startswith(p) for p in _CRIS_PREFIXES):
            return model_id

        # Build a cache key from the tenant tags
        tag_key = "|".join(f"{k}={v}" for k, v in sorted(tenant_tags.items()))
        cache_key = (model_id, tag_key)

        if cache_key in self._cache:
            return self._cache[cache_key].profile_arn

        if not self.config.auto_create:
            return model_id

        # Create a new AIP
        try:
            entry = self._create_profile(model_id, tenant_tags)
            self._cache[cache_key] = entry
            return entry.profile_arn
        except Exception as exc:
            logger.warning(
                "Failed to create AIP for %s (tags=%s): %s. "
                "Falling back to raw model_id.",
                model_id, tenant_tags, exc,
            )
            return model_id

    def _create_profile(
        self,
        model_id: str,
        tenant_tags: dict[str, str],
    ) -> AIPEntry:
        """Create an Application Inference Profile via the Bedrock API.
        
        First checks if a profile with the same name already exists
        to avoid creating duplicates across backend restarts.
        """
        client = self._get_client()

        # Build a descriptive name (include full model path to avoid conflicts)
        tag_suffix = "-".join(
            f"{v}" for k, v in sorted(tenant_tags.items())
        )[:50]
        # Use full model_id in name (replace dots with hyphens) to differentiate CRIS profiles
        model_slug = model_id.replace(".", "-").replace(":", "-")[:40]
        profile_name = f"{self.config.profile_name_prefix}-{tag_suffix}-{model_slug}"
        # Sanitise: AIP names allow alphanumeric, hyphens, underscores
        profile_name = "".join(
            c if c.isalnum() or c in "-_" else "-" for c in profile_name
        )[:64]

        # Check if profile already exists (avoid duplicates across restarts)
        existing_arn = self._find_existing_profile(profile_name)
        if existing_arn:
            logger.info("Reusing existing AIP %s -> %s", profile_name, existing_arn)
            return AIPEntry(
                profile_arn=existing_arn,
                profile_name=profile_name,
                model_id=model_id,
                tags=tenant_tags,
            )

        # Build tags list
        tags = [{"key": k, "value": v} for k, v in tenant_tags.items()]

        # The API requires a full ARN for modelSource.copyFrom
        model_arn = self._to_model_arn(model_id)

        resp = client.create_inference_profile(
            inferenceProfileName=profile_name,
            modelSource={"copyFrom": model_arn},
            tags=tags,
        )

        arn = resp.get("inferenceProfileArn", "")
        logger.info("Created AIP %s -> %s (tags=%s)", profile_name, arn, tenant_tags)

        return AIPEntry(
            profile_arn=arn,
            profile_name=profile_name,
            model_id=model_id,
            tags=tenant_tags,
        )

    def _find_existing_profile(self, profile_name: str) -> str | None:
        """Check if an AIP with this name already exists and is usable. Returns ARN or None."""
        try:
            client = self._get_client()
            kwargs: dict[str, Any] = {"maxResults": 100, "typeEquals": "APPLICATION"}
            while True:
                resp = client.list_inference_profiles(**kwargs)
                for p in resp.get("inferenceProfileSummaries", []):
                    if p.get("inferenceProfileName") == profile_name:
                        status = p.get("status", "")
                        if status == "ACTIVE":
                            return p.get("inferenceProfileArn", "")
                        # Skip non-active profiles
                nt = resp.get("nextToken")
                if not nt:
                    break
                kwargs["nextToken"] = nt
        except Exception as exc:
            logger.debug("Could not list profiles to check for existing: %s", exc)
        return None

    def _to_model_arn(self, model_id: str) -> str:
        """Convert a short model ID to a full Bedrock inference profile ARN.

        ``us.amazon.nova-micro-v1:0``
        → ``arn:aws:bedrock:us-west-2:<account>:inference-profile/us.amazon.nova-micro-v1:0``

        If the model_id is already an ARN, return it unchanged.
        """
        if model_id.startswith("arn:"):
            return model_id

        if self._account_id is None:
            try:
                if self._session is None:
                    import boto3
                    self._session = boto3.Session(region_name=self._region)
                sts = self._session.client("sts", region_name=self._region)
                self._account_id = sts.get_caller_identity()["Account"]
            except Exception as exc:
                logger.warning("Could not get account ID from STS: %s", exc)
                self._account_id = ""

        return (
            f"arn:aws:bedrock:{self._region}:{self._account_id}"
            f":inference-profile/{model_id}"
        )

    def invalidate_cache(self) -> None:
        """Clear the AIP cache, forcing re-lookup on next request."""
        self._cache.clear()

    @property
    def cached_profiles(self) -> dict[tuple[str, str], AIPEntry]:
        return dict(self._cache)
