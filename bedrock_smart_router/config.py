"""Configuration loader for the Bedrock Smart Router.

Everything can be driven from a single dict (or YAML file)::

    router = BedrockRouter.from_config({
        "region": "us-west-2",
        "strategy": "balanced",
        "weights": {"cost": 0.5, "latency": 0.2, "quality": 0.3},
        "cache": {"enabled": true, "ttl_seconds": 1800},
        "metrics": {"backend": "dynamodb", "table_name": "MyMetrics"},
        "observability": {"log_decisions": true},
        "fallback": {"max_depth": 5},
        "circuit_breaker": {"failure_threshold": 5},
    })
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from bedrock_smart_router.cache_layer import CacheConfig
from bedrock_smart_router.circuit_breaker import CircuitBreakerConfig
from bedrock_smart_router.cris_manager import CRISConfig
from bedrock_smart_router.fallback_handler import FallbackConfig
from bedrock_smart_router.guardrails_integration import GuardrailCheckConfig, GuardrailsConfig
from bedrock_smart_router.inference_tier import InferenceTierConfig
from bedrock_smart_router.aip_manager import AIPConfig
from bedrock_smart_router.retry_handler import RetryConfig


@dataclass
class RoutingConfig:
    """Per-request routing overrides.

    Passed as the ``routing`` parameter to ``BedrockRouter.converse()``.
    """

    strategy: str | None = None
    weights: dict[str, float] | None = None
    preferred_family: str | None = None
    required_capabilities: list[str] | None = None
    min_context_window: int | None = None
    exclude_models: list[str] | None = None
    max_cost_per_request: float | None = None
    tags: list[str] | None = None
    metadata: dict[str, Any] | None = None
    fallback_enabled: bool | None = None


@dataclass
class MetricsConfig:
    """Metrics store configuration."""

    backend: str = "memory"  # "memory" | "dynamodb"
    # DynamoDB-specific
    table_name: str = "BedrockSmartRouterMetrics"
    ttl_hours: int = 168  # 7 days
    auto_create_table: bool = True
    # In-memory-specific
    max_records_per_model: int = 1000


@dataclass
class ObservabilityConfig:
    """Observability configuration."""

    log_decisions: bool = True


@dataclass
class RouterConfig:
    """Global router configuration.

    Can be constructed from a dict (e.g. parsed YAML) via
    ``RouterConfig.from_dict()``.  All sub-configs (cache, metrics,
    observability, fallback, circuit breaker, retry) are nested.
    """

    region: str = "us-west-2"
    strategy: str = "balanced"
    weights: dict[str, float] = field(
        default_factory=lambda: {"cost": 0.4, "latency": 0.3, "quality": 0.3}
    )

    # Sub-configs
    cache: CacheConfig = field(default_factory=CacheConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    observability: ObservabilityConfig = field(default_factory=ObservabilityConfig)
    fallback: FallbackConfig = field(default_factory=FallbackConfig)
    circuit_breaker: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)
    retry: RetryConfig = field(default_factory=RetryConfig)

    # Phase 3: Bedrock-native features
    cris: CRISConfig = field(default_factory=CRISConfig)
    inference_tier: InferenceTierConfig = field(default_factory=InferenceTierConfig)
    guardrails: GuardrailsConfig = field(default_factory=GuardrailsConfig)
    aip: AIPConfig = field(default_factory=AIPConfig)
    prompt_cache_boost: bool = True  # Boost score of cache-capable models

    excluded_models: list[str] = field(default_factory=list)
    catalog_path: str | None = None  # Custom model catalog JSON path

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RouterConfig:
        """Build a ``RouterConfig`` from a plain dict (e.g. YAML).

        Example::

            config = RouterConfig.from_dict({
                "region": "us-west-2",
                "strategy": "balanced",
                "weights": {"cost": 0.5, "latency": 0.2, "quality": 0.3},
                "cache": {"enabled": True, "ttl_seconds": 1800},
                "metrics": {"backend": "dynamodb", "table_name": "MyMetrics"},
                "observability": {"log_decisions": True},
                "fallback": {"max_depth": 5},
                "circuit_breaker": {"failure_threshold": 10},
                "retry": {"max_retries": 5},
                "excluded_models": ["us.meta.*"],
            })
        """
        return cls(
            region=data.get("region", "us-west-2"),
            strategy=data.get("strategy", "balanced"),
            weights=data.get("weights", {"cost": 0.4, "latency": 0.3, "quality": 0.3}),
            cache=_build_sub(CacheConfig, data.get("cache", {})),
            metrics=_build_sub(MetricsConfig, data.get("metrics", {})),
            observability=_build_sub(ObservabilityConfig, data.get("observability", {})),
            fallback=_build_sub(FallbackConfig, data.get("fallback", {})),
            circuit_breaker=_build_sub(CircuitBreakerConfig, data.get("circuit_breaker", {})),
            retry=_build_sub(
                RetryConfig,
                {k: v for k, v in data.get("retry", {}).items()
                 if k in ("max_retries", "backoff_base_seconds",
                          "backoff_max_seconds", "backoff_multiplier")},
            ),
            cris=_build_sub(CRISConfig, data.get("cris", {})),
            inference_tier=_build_sub(InferenceTierConfig, data.get("inference_tier", {})),
            guardrails=_build_guardrails(data.get("guardrails", {})),
            aip=_build_sub(AIPConfig, data.get("aip", {})),
            prompt_cache_boost=data.get("prompt_cache_boost", True),
            excluded_models=data.get("excluded_models", []),
            catalog_path=data.get("catalog_path"),
        )


def _build_sub(cls: type, data: dict[str, Any]) -> Any:
    """Instantiate a dataclass from a dict, ignoring unknown keys."""
    known = {f for f in cls.__dataclass_fields__}
    return cls(**{k: v for k, v in data.items() if k in known})


def _build_guardrails(data: dict[str, Any]) -> GuardrailsConfig:
    """Build GuardrailsConfig from a dict with nested pre_route/post_route."""
    pre = data.get("pre_route")
    post = data.get("post_route")
    return GuardrailsConfig(
        pre_route=_build_sub(GuardrailCheckConfig, pre) if isinstance(pre, dict) else None,
        post_route=_build_sub(GuardrailCheckConfig, post) if isinstance(post, dict) else None,
    )
