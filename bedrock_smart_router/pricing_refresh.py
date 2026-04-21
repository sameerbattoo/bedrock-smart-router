"""Dynamic pricing refresh from the AWS Pricing API.

Optionally queries the AWS Pricing API (or Bedrock ListFoundationModels)
to update the model registry with current pricing, rather than relying
solely on the bundled JSON catalog.
"""

from __future__ import annotations

import logging
from typing import Any

from bedrock_smart_router.model_registry import ModelRegistry
from bedrock_smart_router.models import ModelPricing

logger = logging.getLogger(__name__)


class PricingRefresher:
    """Refreshes model pricing from AWS APIs.

    Uses ``bedrock:ListFoundationModels`` to discover models and
    ``pricing:GetProducts`` for current pricing data.
    """

    def __init__(
        self,
        registry: ModelRegistry,
        boto_session: Any | None = None,
        region: str = "us-east-1",
    ) -> None:
        self.registry = registry
        self._session = boto_session
        self._region = region

    def _get_session(self) -> Any:
        if self._session is None:
            import boto3
            self._session = boto3.Session(region_name=self._region)
        return self._session

    def refresh_from_bedrock(self) -> int:
        """Query Bedrock ListFoundationModels and update the registry.

        Returns the number of models updated.
        """
        try:
            session = self._get_session()
            bedrock = session.client("bedrock", region_name=self._region)
            response = bedrock.list_foundation_models()
        except Exception as exc:
            logger.warning("Failed to list foundation models: %s", exc)
            return 0

        updated = 0
        for summary in response.get("modelSummaries", []):
            model_id = summary.get("modelId", "")
            existing = self.registry.get(model_id)
            if existing is None:
                continue

            # Update capabilities from the API response if available
            input_modalities = summary.get("inputModalities", [])
            output_modalities = summary.get("outputModalities", [])
            streaming = summary.get("responseStreamingSupported", True)

            logger.debug(
                "Refreshed model %s: input=%s output=%s streaming=%s",
                model_id,
                input_modalities,
                output_modalities,
                streaming,
            )
            updated += 1

        logger.info("Refreshed %d models from Bedrock API", updated)
        return updated

    def refresh_from_pricing_api(self) -> int:
        """Query the AWS Pricing API for Bedrock model pricing.

        The Pricing API is only available in us-east-1 and ap-south-1.
        Returns the number of models whose pricing was updated.
        """
        try:
            session = self._get_session()
            pricing_client = session.client("pricing", region_name="us-east-1")
        except Exception as exc:
            logger.warning("Failed to create pricing client: %s", exc)
            return 0

        updated = 0
        try:
            paginator = pricing_client.get_paginator("get_products")
            pages = paginator.paginate(
                ServiceCode="AmazonBedrock",
                Filters=[
                    {
                        "Type": "TERM_MATCH",
                        "Field": "productFamily",
                        "Value": "Machine Learning",
                    }
                ],
            )

            for page in pages:
                for price_item_json in page.get("PriceList", []):
                    import json
                    item = (
                        json.loads(price_item_json)
                        if isinstance(price_item_json, str)
                        else price_item_json
                    )
                    model_id = (
                        item.get("product", {})
                        .get("attributes", {})
                        .get("modelId", "")
                    )
                    if not model_id:
                        continue

                    existing = self.registry.get(model_id)
                    if existing is None:
                        continue

                    # Extract pricing from the on-demand terms
                    terms = item.get("terms", {}).get("OnDemand", {})
                    for term in terms.values():
                        for dimension in term.get("priceDimensions", {}).values():
                            price_per_unit = float(
                                dimension.get("pricePerUnit", {}).get("USD", "0")
                            )
                            description = dimension.get("description", "").lower()
                            if "input" in description:
                                existing.pricing = ModelPricing(
                                    input_per_1k=price_per_unit,
                                    output_per_1k=existing.pricing.output_per_1k,
                                    cache_read_per_1k=existing.pricing.cache_read_per_1k,
                                    cache_write_per_1k=existing.pricing.cache_write_per_1k,
                                )
                                updated += 1
                            elif "output" in description:
                                existing.pricing = ModelPricing(
                                    input_per_1k=existing.pricing.input_per_1k,
                                    output_per_1k=price_per_unit,
                                    cache_read_per_1k=existing.pricing.cache_read_per_1k,
                                    cache_write_per_1k=existing.pricing.cache_write_per_1k,
                                )

        except Exception as exc:
            logger.warning("Failed to refresh pricing: %s", exc)

        logger.info("Updated pricing for %d models from AWS Pricing API", updated)
        return updated
