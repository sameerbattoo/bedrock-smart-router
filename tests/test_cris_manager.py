# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the CRIS profile manager."""

from bedrock_smart_router.cris_manager import CRISConfig, CRISManager
from bedrock_smart_router.models import BedrockModel, ModelCapabilities, Tier


def _model(regions: list[dict]) -> BedrockModel:
    return BedrockModel(
        model_id="anthropic.claude-sonnet-4-6",
        family="anthropic", tier=Tier.MID, display_name="Sonnet",
        capabilities=ModelCapabilities(tool_use=True),
        max_input_tokens=200_000, max_output_tokens=16_384,
        regions=regions,
    )


class TestCRISManager:
    def test_no_regions_returns_model_id(self):
        mgr = CRISManager()
        m = _model([])
        assert mgr.select_profile(m, "us-east-1") == m.model_id

    def test_disabled_returns_model_id(self):
        mgr = CRISManager(CRISConfig(enabled=False))
        m = _model([{"name": "us-east-1", "cris_profiles": ["us", "global"]}])
        assert mgr.select_profile(m, "us-east-1") == m.model_id

    def test_prefers_geography(self):
        mgr = CRISManager(CRISConfig(preferred_geography="eu"))
        m = _model([
            {"name": "us-east-1", "cris_profiles": ["us", "global"]},
            {"name": "eu-west-1", "cris_profiles": ["eu", "global"]},
        ])
        assert mgr.select_profile(m, "eu-west-1").startswith("eu.")

    def test_falls_back_to_global(self):
        mgr = CRISManager(CRISConfig(preferred_geography="apac"))
        m = _model([
            {"name": "us-east-1", "cris_profiles": ["us", "global"]},
            {"name": "ap-south-1", "cris_profiles": ["global"]},
        ])
        assert mgr.select_profile(m, "ap-south-1").startswith("global.")

    def test_global_disabled_picks_available_prefix(self):
        mgr = CRISManager(CRISConfig(preferred_geography="apac", allow_global=False))
        m = _model([
            {"name": "us-east-1", "cris_profiles": ["us", "global"]},
            {"name": "ap-south-1", "cris_profiles": ["global", "apac"]},
        ])
        # apac preferred, global disabled → picks apac
        result = mgr.select_profile(m, "ap-south-1")
        assert result.startswith("apac.")

    def test_default_prefers_global(self):
        mgr = CRISManager()
        m = _model([
            {"name": "us-east-1", "cris_profiles": ["us", "global"]},
        ])
        # No geography preference, should pick global
        assert mgr.select_profile(m, "us-east-1").startswith("global.")

    def test_direct_only_model(self):
        mgr = CRISManager()
        m = _model([
            {"name": "us-east-1", "direct": True},
            {"name": "ap-south-1", "direct": True},
        ])
        # No CRIS profiles, returns raw model_id
        assert mgr.select_profile(m, "us-east-1") == "anthropic.claude-sonnet-4-6"

    def test_region_not_available(self):
        mgr = CRISManager()
        m = _model([
            {"name": "us-east-1", "cris_profiles": ["us", "global"]},
        ])
        # Model not available in ap-south-1, but has global elsewhere
        result = mgr.select_profile(m, "ap-south-1")
        assert result.startswith("global.")

    def test_is_available_in_region(self):
        mgr = CRISManager()
        m = _model([
            {"name": "us-east-1", "cris_profiles": ["us", "global"]},
            {"name": "eu-west-1", "cris_profiles": ["eu"]},
        ])
        assert mgr.is_available_in_region(m, "us-east-1") is True
        assert mgr.is_available_in_region(m, "eu-west-1") is True
        assert mgr.is_available_in_region(m, "ap-south-1") is False

    def test_get_geography(self):
        mgr = CRISManager()
        assert mgr.get_geography("us.anthropic.claude-sonnet-4-6") == "us"
        assert mgr.get_geography("global.anthropic.claude-sonnet-4-6") == "global"
        assert mgr.get_geography("mistral.mistral-large") is None
