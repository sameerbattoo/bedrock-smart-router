"""Tests for the pricing refresher using mocked boto3 clients."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from bedrock_smart_router.model_registry import ModelRegistry
from bedrock_smart_router.pricing_refresh import PricingRefresher


class TestRefreshFromBedrock:
    def setup_method(self):
        self.registry = ModelRegistry()
        self.session = MagicMock()
        self.refresher = PricingRefresher(
            registry=self.registry,
            boto_session=self.session,
            region="us-west-2",
        )

    def test_updates_known_models(self):
        mock_client = MagicMock()
        mock_client.list_foundation_models.return_value = {
            "modelSummaries": [
                {
                    "modelId": "amazon.nova-micro-v1:0",
                    "inputModalities": ["TEXT"],
                    "outputModalities": ["TEXT"],
                    "responseStreamingSupported": True,
                },
                {
                    "modelId": "amazon.nova-pro-v1:0",
                    "inputModalities": ["TEXT", "IMAGE"],
                    "outputModalities": ["TEXT"],
                    "responseStreamingSupported": True,
                },
            ]
        }
        self.session.client.return_value = mock_client

        count = self.refresher.refresh_from_bedrock()
        assert count == 2
        mock_client.list_foundation_models.assert_called_once()

    def test_skips_unknown_models(self):
        mock_client = MagicMock()
        mock_client.list_foundation_models.return_value = {
            "modelSummaries": [
                {
                    "modelId": "some-unknown-model-v1",
                    "inputModalities": ["TEXT"],
                    "outputModalities": ["TEXT"],
                },
            ]
        }
        self.session.client.return_value = mock_client

        count = self.refresher.refresh_from_bedrock()
        assert count == 0

    def test_handles_api_failure(self):
        mock_client = MagicMock()
        mock_client.list_foundation_models.side_effect = Exception("Access denied")
        self.session.client.return_value = mock_client

        count = self.refresher.refresh_from_bedrock()
        assert count == 0

    def test_empty_response(self):
        mock_client = MagicMock()
        mock_client.list_foundation_models.return_value = {"modelSummaries": []}
        self.session.client.return_value = mock_client

        count = self.refresher.refresh_from_bedrock()
        assert count == 0


def _pricing_item(model_id: str, input_price: str, output_price: str) -> dict:
    """Build a realistic AWS Pricing API response item."""
    return {
        "product": {
            "attributes": {
                "modelId": model_id,
            }
        },
        "terms": {
            "OnDemand": {
                "term1": {
                    "priceDimensions": {
                        "dim_input": {
                            "description": "Input token price per 1K tokens",
                            "pricePerUnit": {"USD": input_price},
                        },
                        "dim_output": {
                            "description": "Output token price per 1K tokens",
                            "pricePerUnit": {"USD": output_price},
                        },
                    }
                }
            }
        },
    }


class TestRefreshFromPricingAPI:
    def setup_method(self):
        self.registry = ModelRegistry()
        self.session = MagicMock()
        self.refresher = PricingRefresher(
            registry=self.registry,
            boto_session=self.session,
            region="us-east-1",
        )

    def _mock_paginator(self, items: list[dict]) -> None:
        """Set up a mock paginator that returns the given items."""
        mock_client = MagicMock()
        mock_paginator = MagicMock()
        mock_paginator.paginate.return_value = [
            {"PriceList": [json.dumps(item) for item in items]}
        ]
        mock_client.get_paginator.return_value = mock_paginator
        self.session.client.return_value = mock_client

    def test_updates_input_pricing(self):
        original = self.registry.get("amazon.nova-micro-v1:0")
        original_input = original.pricing.input_per_1k

        self._mock_paginator([
            _pricing_item("amazon.nova-micro-v1:0", "0.00005", "0.0002"),
        ])

        count = self.refresher.refresh_from_pricing_api()
        assert count >= 1

        updated = self.registry.get("amazon.nova-micro-v1:0")
        assert updated.pricing.input_per_1k == 0.00005
        assert updated.pricing.output_per_1k == 0.0002

    def test_preserves_cache_pricing(self):
        """Cache read/write pricing should not be overwritten."""
        model = self.registry.get("anthropic.claude-sonnet-4-6")
        original_cache_read = model.pricing.cache_read_per_1k

        self._mock_paginator([
            _pricing_item("anthropic.claude-sonnet-4-6", "0.004", "0.02"),
        ])

        self.refresher.refresh_from_pricing_api()

        updated = self.registry.get("anthropic.claude-sonnet-4-6")
        assert updated.pricing.input_per_1k == 0.004
        assert updated.pricing.cache_read_per_1k == original_cache_read

    def test_skips_unknown_models(self):
        self._mock_paginator([
            _pricing_item("unknown-model-xyz", "0.01", "0.05"),
        ])

        count = self.refresher.refresh_from_pricing_api()
        assert count == 0

    def test_handles_api_failure(self):
        mock_client = MagicMock()
        mock_client.get_paginator.side_effect = Exception("Service unavailable")
        self.session.client.return_value = mock_client

        count = self.refresher.refresh_from_pricing_api()
        assert count == 0

    def test_handles_malformed_item(self):
        """Items without proper structure should be skipped."""
        self._mock_paginator([
            {"product": {"attributes": {}}},  # No modelId
            _pricing_item("amazon.nova-micro-v1:0", "0.00005", "0.0002"),
        ])

        count = self.refresher.refresh_from_pricing_api()
        assert count >= 1  # Only the valid item counted

    def test_handles_dict_items(self):
        """PriceList items can be dicts (not just JSON strings)."""
        mock_client = MagicMock()
        mock_paginator = MagicMock()
        # Return items as dicts, not JSON strings
        mock_paginator.paginate.return_value = [
            {"PriceList": [
                _pricing_item("amazon.nova-micro-v1:0", "0.00005", "0.0002"),
            ]}
        ]
        mock_client.get_paginator.return_value = mock_paginator
        self.session.client.return_value = mock_client

        count = self.refresher.refresh_from_pricing_api()
        assert count >= 1

    def test_multiple_models_updated(self):
        self._mock_paginator([
            _pricing_item("amazon.nova-micro-v1:0", "0.00005", "0.0002"),
            _pricing_item("amazon.nova-pro-v1:0", "0.001", "0.004"),
        ])

        count = self.refresher.refresh_from_pricing_api()
        assert count >= 2

        micro = self.registry.get("amazon.nova-micro-v1:0")
        pro = self.registry.get("amazon.nova-pro-v1:0")
        assert micro.pricing.input_per_1k == 0.00005
        assert pro.pricing.input_per_1k == 0.001
