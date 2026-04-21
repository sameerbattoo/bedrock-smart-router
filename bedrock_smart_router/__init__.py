"""Bedrock Smart Router — Intelligent model routing for Amazon Bedrock."""

from bedrock_smart_router.router import BedrockRouter
from bedrock_smart_router.config import RouterConfig, RoutingConfig, MetricsConfig
from bedrock_smart_router.models import BedrockModel, ModelCapabilities, ModelPricing
from bedrock_smart_router.observability import RoutingEvent

__all__ = [
    "BedrockRouter",
    "RouterConfig",
    "RoutingConfig",
    "MetricsConfig",
    "BedrockModel",
    "ModelCapabilities",
    "ModelPricing",
    "RoutingEvent",
]

__version__ = "0.1.0"
