"""Bedrock Smart Router — Intelligent model routing for Amazon Bedrock."""

from bedrock_smart_router.router import BedrockRouter
from bedrock_smart_router.config import RouterConfig, RoutingConfig, MetricsConfig, ROUTING_PRESETS
from bedrock_smart_router.models import BedrockModel, ModelCapabilities, ModelPricing
from bedrock_smart_router.observability import RoutingEvent
from bedrock_smart_router.exceptions import NoModelsMatchError

__all__ = [
    "BedrockRouter",
    "RouterConfig",
    "RoutingConfig",
    "MetricsConfig",
    "ROUTING_PRESETS",
    "BedrockModel",
    "ModelCapabilities",
    "ModelPricing",
    "RoutingEvent",
    "NoModelsMatchError",
]

__version__ = "0.1.0"
