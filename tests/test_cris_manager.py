"""Tests for the CRIS profile manager."""

from bedrock_smart_router.cris_manager import CRISConfig, CRISManager
from bedrock_smart_router.models import BedrockModel, ModelCapabilities, Tier


def _model(profiles: list[str]) -> BedrockModel:
    return BedrockModel(
        model_id="us.anthropic.claude-sonnet-4-6",
        family="anthropic", tier=Tier.MID, display_name="Sonnet",
        capabilities=ModelCapabilities(tool_use=True),
        max_input_tokens=200_000, max_output_tokens=16_384,
        cris_profiles=profiles,
    )


class TestCRISManager:
    def test_no_profiles_returns_model_id(self):
        mgr = CRISManager()
        m = _model([])
        assert mgr.select_profile(m) == m.model_id

    def test_disabled_returns_model_id(self):
        mgr = CRISManager(CRISConfig(enabled=False))
        m = _model(["us.anthropic.claude-sonnet-4-6", "global.anthropic.claude-sonnet-4-6"])
        assert mgr.select_profile(m) == m.model_id

    def test_prefers_geography(self):
        mgr = CRISManager(CRISConfig(preferred_geography="eu"))
        m = _model(["us.anthropic.claude-sonnet-4-6", "eu.anthropic.claude-sonnet-4-6"])
        assert mgr.select_profile(m).startswith("eu.")

    def test_falls_back_to_global(self):
        mgr = CRISManager(CRISConfig(preferred_geography="ap"))
        m = _model(["us.anthropic.claude-sonnet-4-6", "global.anthropic.claude-sonnet-4-6"])
        assert mgr.select_profile(m).startswith("global.")

    def test_global_disabled_skips_global(self):
        mgr = CRISManager(CRISConfig(preferred_geography="ap", allow_global=False))
        m = _model(["us.anthropic.claude-sonnet-4-6", "global.anthropic.claude-sonnet-4-6"])
        # No ap. profile, global disabled, falls back to us.
        assert mgr.select_profile(m).startswith("us.")

    def test_default_prefers_global(self):
        mgr = CRISManager()
        m = _model(["us.anthropic.claude-sonnet-4-6", "global.anthropic.claude-sonnet-4-6"])
        # No geography preference, should pick global
        assert mgr.select_profile(m).startswith("global.")

    def test_get_geography(self):
        mgr = CRISManager()
        assert mgr.get_geography("us.anthropic.claude-sonnet-4-6") == "us"
        assert mgr.get_geography("global.anthropic.claude-sonnet-4-6") == "global"
        assert mgr.get_geography("mistral.mistral-large") is None
