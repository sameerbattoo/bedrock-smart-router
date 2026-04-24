"""Bedrock Smart Router — Intelligent model routing for Amazon Bedrock."""

from bedrock_smart_router.router import BedrockRouter
from bedrock_smart_router.config import RouterConfig, RoutingConfig, MetricsConfig, ROUTING_PRESETS
from bedrock_smart_router.models import BedrockModel, ModelCapabilities, ModelPricing, TIER_PRICING_MULTIPLIER
from bedrock_smart_router.model_registry import base_model_id
from bedrock_smart_router.observability import RoutingEvent
from bedrock_smart_router.exceptions import NoModelsMatchError

# Strands integration (optional — only available when strands-agents is installed)
try:
    from bedrock_smart_router.strands_model import SmartRouterModel as SmartRouterModel
    _HAS_STRANDS = True
except ImportError:
    _HAS_STRANDS = False

__all__ = [
    "BedrockRouter",
    "RouterConfig",
    "RoutingConfig",
    "MetricsConfig",
    "ROUTING_PRESETS",
    "BedrockModel",
    "ModelCapabilities",
    "ModelPricing",
    "TIER_PRICING_MULTIPLIER",
    "RoutingEvent",
    "NoModelsMatchError",
    *((["SmartRouterModel"] if _HAS_STRANDS else [])),
]

__version__ = "0.1.0"
