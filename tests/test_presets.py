"""Tests for named routing presets."""

import pytest

from bedrock_smart_router.config import (
    ROUTING_PRESETS,
    RoutingConfig,
    resolve_preset,
)


class TestPresets:
    def test_economy_preset(self):
        rc = resolve_preset(RoutingConfig(preset="economy"))
        assert rc.strategy == "cost-optimized"
        assert rc.max_cost_per_request == 0.002

    def test_speed_preset(self):
        rc = resolve_preset(RoutingConfig(preset="speed"))
        assert rc.strategy == "latency-optimized"

    def test_balanced_preset(self):
        rc = resolve_preset(RoutingConfig(preset="balanced"))
        assert rc.strategy == "balanced"
        assert rc.weights == {"cost": 0.4, "latency": 0.3, "quality": 0.3}

    def test_quality_preset(self):
        rc = resolve_preset(RoutingConfig(preset="quality"))
        assert rc.strategy == "quality-optimized"

    def test_no_preset_passes_through(self):
        rc = RoutingConfig(strategy="cost-optimized")
        resolved = resolve_preset(rc)
        assert resolved.strategy == "cost-optimized"
        assert resolved.preset is None

    def test_explicit_overrides_preset(self):
        """Explicit values should override preset defaults."""
        rc = resolve_preset(RoutingConfig(
            preset="economy",
            strategy="balanced",  # Override the economy preset's strategy
            max_cost_per_request=0.05,  # Override the economy preset's cost limit
        ))
        assert rc.strategy == "balanced"
        assert rc.max_cost_per_request == 0.05

    def test_partial_override(self):
        """Override one field, keep the rest from preset."""
        rc = resolve_preset(RoutingConfig(
            preset="economy",
            preferred_family="anthropic",  # Not in preset, added by user
        ))
        assert rc.strategy == "cost-optimized"  # From preset
        assert rc.max_cost_per_request == 0.002  # From preset
        assert rc.preferred_family == "anthropic"  # User override

    def test_unknown_preset_raises(self):
        with pytest.raises(ValueError, match="Unknown preset"):
            resolve_preset(RoutingConfig(preset="turbo"))

    def test_all_presets_defined(self):
        assert set(ROUTING_PRESETS.keys()) == {"economy", "speed", "balanced", "quality"}

    def test_preset_preserves_tags_and_metadata(self):
        rc = resolve_preset(RoutingConfig(
            preset="economy",
            tags=["paid-tier"],
            metadata={"user_id": "u123"},
        ))
        assert rc.tags == ["paid-tier"]
        assert rc.metadata == {"user_id": "u123"}
