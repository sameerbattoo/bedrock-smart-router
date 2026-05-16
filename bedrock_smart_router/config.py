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

from bedrock_smart_router.ab_testing import ABTestConfig, ABVariant
from bedrock_smart_router.cache_layer import CacheConfig
from bedrock_smart_router.canary import CanaryConfig, CanaryThresholds
from bedrock_smart_router.circuit_breaker import CircuitBreakerConfig
from bedrock_smart_router.cris_manager import CRISConfig
from bedrock_smart_router.fallback_handler import FallbackConfig
from bedrock_smart_router.guardrails_integration import GuardrailCheckConfig, GuardrailsConfig
from bedrock_smart_router.inference_tier import InferenceTierConfig
from bedrock_smart_router.aip_manager import AIPConfig
from bedrock_smart_router.retry_handler import RetryConfig
from bedrock_smart_router.shadow_mode import ShadowConfig


@dataclass
class RoutingConfig:
    """Per-request routing overrides.

    Passed as the ``routing`` parameter to ``BedrockRouter.converse()``.

    Use ``preset`` for a named shortcut::

        routing=RoutingConfig(preset="economy")   # cheapest model
        routing=RoutingConfig(preset="speed")      # lowest latency
        routing=RoutingConfig(preset="quality")    # best quality
        routing=RoutingConfig(preset="balanced")   # default weighted

    Any explicit field overrides the preset value.
    """

    preset: str | None = None  # "economy" | "speed" | "balanced" | "quality"
    strategy: str | None = None
    weights: dict[str, float] | None = None
    preferred_model: str | None = None  # Exact model ID to use (bypasses strategy selection)
    preferred_family: str | None = None
    required_capabilities: list[str] | None = None
    min_context_window: int | None = None
    exclude_models: list[str] | None = None
    max_cost_per_request: float | None = None
    tags: list[str] | None = None
    metadata: dict[str, Any] | None = None
    fallback_enabled: bool | None = None
    explain: bool = False  # Include detailed explanation in RoutingDecision


# ── Named presets ───────────────────────────────────────────────────

ROUTING_PRESETS: dict[str, dict[str, Any]] = {
    "economy": {
        "strategy": "cost-optimized",
        "max_cost_per_request": 0.002,
    },
    "speed": {
        "strategy": "latency-optimized",
    },
    "balanced": {
        "strategy": "balanced",
        "weights": {"cost": 0.4, "latency": 0.3, "quality": 0.3},
    },
    "quality": {
        "strategy": "quality-optimized",
    },
}


def resolve_preset(config: RoutingConfig) -> RoutingConfig:
    """Apply preset defaults, then layer explicit overrides on top.

    Returns a new ``RoutingConfig`` with preset values filled in for
    any field the caller didn't explicitly set.
    """
    if not config.preset:
        return config

    preset_values = ROUTING_PRESETS.get(config.preset)
    if preset_values is None:
        raise ValueError(
            f"Unknown preset '{config.preset}'. "
            f"Available: {list(ROUTING_PRESETS.keys())}"
        )

    # Start from preset, override with any explicit values
    merged = RoutingConfig(
        preset=config.preset,
        strategy=config.strategy or preset_values.get("strategy"),
        weights=config.weights or preset_values.get("weights"),
        preferred_family=config.preferred_family or preset_values.get("preferred_family"),
        max_cost_per_request=(
            config.max_cost_per_request
            if config.max_cost_per_request is not None
            else preset_values.get("max_cost_per_request")
        ),
        # Pass through everything else unchanged
        required_capabilities=config.required_capabilities,
        min_context_window=config.min_context_window,
        exclude_models=config.exclude_models,
        tags=config.tags,
        metadata=config.metadata,
        fallback_enabled=config.fallback_enabled,
        preferred_model=config.preferred_model,
        explain=config.explain,
    )
    return merged


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
    cloudwatch_enabled: bool = False
    cloudwatch_namespace: str = "BedrockSmartRouter"
    otel_enabled: bool = False
    otel_service_name: str = "bedrock-smart-router"


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
    prompt_cache_boost: bool = True

    # Phase 4: Advanced deployment
    ab_test: ABTestConfig = field(default_factory=lambda: ABTestConfig(enabled=False))
    canary: CanaryConfig = field(default_factory=CanaryConfig)
    shadow: ShadowConfig = field(default_factory=ShadowConfig)

    excluded_models: list[str] = field(default_factory=list)
    catalog_path: str | None = None
    boto_config: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

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
                "excluded_models": ["meta.*"],
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
            ab_test=_build_ab_test(data.get("ab_test", {})),
            canary=_build_canary(data.get("canary", {})),
            shadow=_build_sub(ShadowConfig, data.get("shadow", {})),
            excluded_models=data.get("excluded_models", []),
            catalog_path=data.get("catalog_path"),
            boto_config=data.get("boto_config"),
            metadata=data.get("metadata", {}),
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


def _build_ab_test(data: dict[str, Any]) -> ABTestConfig:
    """Build ABTestConfig from a dict with nested variants."""
    if not data or not data.get("enabled"):
        return ABTestConfig(enabled=False)
    variants_raw = data.get("variants", {})
    variants = []
    for name, v in variants_raw.items():
        variants.append(ABVariant(
            name=name,
            model=v.get("model", ""),
            weight=v.get("weight", 0.5),
        ))
    return ABTestConfig(
        name=data.get("name", ""),
        variants=variants,
        sticky=data.get("sticky", True),
        enabled=True,
    )


def _build_canary(data: dict[str, Any]) -> CanaryConfig:
    """Build CanaryConfig from a dict with nested thresholds."""
    if not data or not data.get("enabled"):
        return CanaryConfig()
    rollback = data.get("auto_rollback", {})
    promote = data.get("auto_promote", {})
    return CanaryConfig(
        enabled=True,
        baseline_model=data.get("baseline", ""),
        canary_model=data.get("canary_model", ""),
        canary_percentage=data.get("canary_percentage", 5.0),
        auto_rollback=_build_sub(CanaryThresholds, rollback) if rollback else CanaryThresholds(),
        auto_promote=_build_sub(CanaryThresholds, promote) if promote else CanaryThresholds(),
    )
