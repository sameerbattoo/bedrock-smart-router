"""Tests for the async router."""

import asyncio

from bedrock_smart_router.async_router import AsyncBedrockRouter
from bedrock_smart_router.config import RouterConfig


class TestAsyncRouterCreate:
    def test_create_defaults(self):
        """AsyncBedrockRouter.create() should resolve config without crashing."""
        # We can't actually call create() without mocking boto3,
        # but we can verify the config path works
        cfg = RouterConfig()
        assert cfg.strategy == "balanced"

    def test_create_from_dict(self):
        cfg = RouterConfig.from_dict({
            "region": "eu-west-1",
            "strategy": "cost-optimized",
        })
        assert cfg.region == "eu-west-1"

    def test_delegates_properties(self):
        """Verify the async router delegates accessors to sync router."""
        # This tests the class structure, not actual Bedrock calls
        assert hasattr(AsyncBedrockRouter, "config")
        assert hasattr(AsyncBedrockRouter, "registry")
        assert hasattr(AsyncBedrockRouter, "metrics")
        assert hasattr(AsyncBedrockRouter, "cache")
        assert hasattr(AsyncBedrockRouter, "observability")
        assert hasattr(AsyncBedrockRouter, "last_routing_decision")

    def test_converse_is_coroutine(self):
        """The converse method should be a coroutine function."""
        assert asyncio.iscoroutinefunction(AsyncBedrockRouter.converse)
