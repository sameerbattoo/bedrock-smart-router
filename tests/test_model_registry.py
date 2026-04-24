"""Tests for the model registry."""

from bedrock_smart_router.model_registry import ModelRegistry
from bedrock_smart_router.models import Tier


class TestModelRegistry:
    def test_default_catalog_not_empty(self):
        reg = ModelRegistry()
        assert len(reg) > 0

    def test_get_known_model(self):
        reg = ModelRegistry()
        m = reg.get("us.amazon.nova-micro-v1:0")
        assert m is not None
        assert m.tier == Tier.MICRO
        assert m.family == "amazon"

    def test_get_unknown_returns_none(self):
        reg = ModelRegistry()
        assert reg.get("nonexistent-model") is None

    def test_list_by_family(self):
        reg = ModelRegistry()
        anthropic = reg.list_models(family="anthropic")
        assert len(anthropic) >= 7
        assert all(m.family == "anthropic" for m in anthropic)

    def test_list_by_tier(self):
        reg = ModelRegistry()
        micros = reg.list_models(tier="micro")
        assert len(micros) >= 1
        assert all(m.tier == Tier.MICRO for m in micros)

    def test_list_by_min_tier(self):
        reg = ModelRegistry()
        heavy_plus = reg.list_models(min_tier="heavy")
        assert all(m.tier in (Tier.HEAVY, Tier.REASONING) for m in heavy_plus)

    def test_eligible_models_vision(self):
        reg = ModelRegistry()
        vision = reg.eligible_models(requires_vision=True)
        assert all(m.capabilities.vision for m in vision)

    def test_eligible_models_tool_use(self):
        reg = ModelRegistry()
        tools = reg.eligible_models(requires_tool_use=True)
        assert all(m.capabilities.tool_use for m in tools)

    def test_eligible_models_streaming_tool_use(self):
        """Models with streaming_tool_use=False should be excluded when required."""
        reg = ModelRegistry()
        # Without streaming filter — Scout should be included (it has tool_use=True)
        all_tool = reg.eligible_models(requires_tool_use=True)
        scout_ids = [m.model_id for m in all_tool if "scout" in m.model_id]
        assert len(scout_ids) > 0, "Scout should be in tool_use candidates"

        # With streaming filter — Scout should be excluded
        streaming_tool = reg.eligible_models(
            requires_tool_use=True, requires_streaming_tool_use=True,
        )
        scout_ids = [m.model_id for m in streaming_tool if "scout" in m.model_id]
        assert len(scout_ids) == 0, "Scout should NOT be in streaming tool_use candidates"

    def test_streaming_tool_use_loaded_from_catalog(self):
        """streaming_tool_use capability should be loaded from models.json."""
        reg = ModelRegistry()
        scout = reg.get("us.meta.llama4-scout-17b-instruct-v1:0")
        assert scout is not None
        assert scout.capabilities.tool_use is True
        assert scout.capabilities.streaming_tool_use is False

    def test_streaming_tool_use_defaults_true(self):
        """Models without explicit streaming_tool_use should default to True."""
        reg = ModelRegistry()
        nova = reg.get("us.amazon.nova-2-lite-v1:0")
        assert nova is not None
        assert nova.capabilities.streaming_tool_use is True

    def test_eligible_models_exclude_pattern(self):
        reg = ModelRegistry()
        no_meta = reg.eligible_models(exclude_patterns=["us.meta.*"])
        assert all("meta" not in m.model_id for m in no_meta)

    def test_register_custom_model(self):
        reg = ModelRegistry()
        from bedrock_smart_router.models import BedrockModel, ModelCapabilities
        custom = BedrockModel(
            model_id="custom-model-v1",
            family="custom",
            tier=Tier.MID,
            display_name="My Custom Model",
            capabilities=ModelCapabilities(tool_use=True),
            max_input_tokens=32_000,
            max_output_tokens=4_096,
        )
        reg.register(custom)
        assert reg.get("custom-model-v1") is not None


class TestJsonCatalog:
    def test_loads_from_bundled_json(self):
        """Default registry loads from data/models.json."""
        reg = ModelRegistry()
        assert len(reg) == 25  # regional + global CRIS profiles (legacy models excluded)

    def test_loads_from_custom_path(self, tmp_path):
        """Registry can load from a user-provided JSON file."""
        import json
        custom = {
            "models": [{
                "model_id": "custom-model",
                "family": "custom",
                "tier": "mid",
                "display_name": "Custom Model",
                "max_input_tokens": 8192,
                "max_output_tokens": 2048,
            }]
        }
        p = tmp_path / "custom.json"
        p.write_text(json.dumps(custom))
        reg = ModelRegistry(catalog_path=p)
        assert len(reg) == 1
        assert reg.get("custom-model") is not None

    def test_overlay_merges(self, tmp_path):
        """Overlay adds/overrides models on top of the default catalog."""
        import json
        overlay = {
            "models": [{
                "model_id": "us.amazon.nova-micro-v1:0",
                "family": "amazon",
                "tier": "micro",
                "display_name": "Nova Micro PATCHED",
                "max_input_tokens": 256000,
                "max_output_tokens": 5000,
            }]
        }
        p = tmp_path / "overlay.json"
        p.write_text(json.dumps(overlay))
        reg = ModelRegistry()
        original_count = len(reg)
        reg.load_overlay(p)
        # Count unchanged (override, not add)
        assert len(reg) == original_count
        # But the display name is patched
        m = reg.get("us.amazon.nova-micro-v1:0")
        assert m.display_name == "Nova Micro PATCHED"
        assert m.max_input_tokens == 256000

    def test_missing_file_returns_empty(self, tmp_path):
        """Missing catalog file produces an empty registry, not a crash."""
        reg = ModelRegistry(catalog_path=tmp_path / "nonexistent.json")
        assert len(reg) == 0
