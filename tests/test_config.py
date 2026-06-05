"""Tests for configuration loading."""

from bedrock_smart_router.config import MetricsConfig, RouterConfig, RoutingConfig


class TestRouterConfig:
    def test_defaults(self):
        cfg = RouterConfig()
        assert cfg.region == ""  # Empty = auto-detect from boto3 session
        assert cfg.strategy == "balanced"
        assert cfg.fallback.enabled is True
        assert cfg.circuit_breaker.failure_threshold == 5
        assert cfg.cache.enabled is True
        assert cfg.metrics.backend == "memory"
        assert cfg.observability.log_decisions is True

    def test_from_dict_minimal(self):
        cfg = RouterConfig.from_dict({"region": "eu-west-1"})
        assert cfg.region == "eu-west-1"
        assert cfg.strategy == "balanced"

    def test_from_dict_full(self):
        cfg = RouterConfig.from_dict({
            "region": "us-east-1",
            "strategy": "cost-optimized",
            "weights": {"cost": 0.8, "latency": 0.1, "quality": 0.1},
            "cache": {"enabled": True, "ttl_seconds": 1800, "max_entries": 5000},
            "metrics": {
                "backend": "dynamodb",
                "table_name": "MyMetrics",
                "ttl_hours": 48,
            },
            "observability": {"log_decisions": False},
            "fallback": {"enabled": True, "max_depth": 3},
            "circuit_breaker": {"failure_threshold": 10, "cooldown_seconds": 60},
            "retry": {"max_retries": 5},
            "excluded_models": ["meta.*"],
        })
        assert cfg.strategy == "cost-optimized"
        assert cfg.weights["cost"] == 0.8
        assert cfg.cache.ttl_seconds == 1800
        assert cfg.cache.max_entries == 5000
        assert cfg.metrics.backend == "dynamodb"
        assert cfg.metrics.table_name == "MyMetrics"
        assert cfg.metrics.ttl_hours == 48
        assert cfg.observability.log_decisions is False
        assert cfg.fallback.max_depth == 3
        assert cfg.circuit_breaker.failure_threshold == 10
        assert cfg.retry.max_retries == 5
        assert "meta.*" in cfg.excluded_models

    def test_from_dict_ignores_unknown_keys(self):
        cfg = RouterConfig.from_dict({
            "region": "us-west-2",
            "cache": {"enabled": True, "unknown_field": 42},
        })
        assert cfg.cache.enabled is True


class TestRoutingConfig:
    def test_defaults(self):
        rc = RoutingConfig()
        assert rc.strategy is None
        assert rc.tags is None

    def test_override(self):
        rc = RoutingConfig(
            strategy="cost-optimized",
            preferred_family="anthropic",
            max_cost_per_request=0.01,
        )
        assert rc.strategy == "cost-optimized"
        assert rc.preferred_family == "anthropic"


class TestMetricsConfig:
    def test_defaults(self):
        mc = MetricsConfig()
        assert mc.backend == "memory"
        assert mc.table_name == "BedrockSmartRouterMetrics"
        assert mc.ttl_hours == 168

    def test_dynamodb(self):
        mc = MetricsConfig(backend="dynamodb", table_name="Custom")
        assert mc.backend == "dynamodb"
        assert mc.table_name == "Custom"


class TestRouterCreate:
    def test_create_defaults(self):
        """BedrockRouter.create() with no args should not crash."""
        # We can't actually call create() without mocking boto3,
        # but we can verify config resolution
        cfg = RouterConfig.from_dict({})
        assert cfg.region == ""  # Empty = auto-detect from boto3 session
        assert cfg.strategy == "balanced"
        assert cfg.metrics.backend == "memory"

    def test_create_from_dict(self):
        cfg = RouterConfig.from_dict({
            "region": "eu-west-1",
            "strategy": "latency-optimized",
            "metrics": {"backend": "dynamodb", "table_name": "Test"},
        })
        assert cfg.region == "eu-west-1"
        assert cfg.strategy == "latency-optimized"
        assert cfg.metrics.backend == "dynamodb"
        assert cfg.metrics.table_name == "Test"
