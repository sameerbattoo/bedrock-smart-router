# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for distilled model support."""

import pytest

from bedrock_smart_router.distilled_models import DistilledModelManager
from bedrock_smart_router.model_registry import ModelRegistry
from bedrock_smart_router.models import Tier


class TestDistilledModelManager:
    def setup_method(self):
        self.registry = ModelRegistry()
        self.mgr = DistilledModelManager(self.registry)

    def test_register_distilled(self):
        original_count = len(self.registry)
        distilled = self.mgr.register_distilled(
            model_id="my-distilled-nova-pro",
            teacher_model_id="amazon.nova-pro-v1:0",
            quality_delta=-0.02,
            cost_multiplier=0.25,
            speed_multiplier=5.0,
        )
        assert len(self.registry) == original_count + 1
        assert distilled.distilled_from == "amazon.nova-pro-v1:0"
        assert distilled.distilled_quality_delta == -0.02

    def test_pricing_derived_from_teacher(self):
        teacher = self.registry.get("amazon.nova-pro-v1:0")
        distilled = self.mgr.register_distilled(
            model_id="my-distilled",
            teacher_model_id="amazon.nova-pro-v1:0",
            cost_multiplier=0.25,
        )
        assert distilled.pricing.input_per_1k == pytest.approx(
            teacher.pricing.input_per_1k * 0.25
        )
        assert distilled.pricing.output_per_1k == pytest.approx(
            teacher.pricing.output_per_1k * 0.25
        )

    def test_tier_one_step_down(self):
        distilled = self.mgr.register_distilled(
            model_id="my-distilled",
            teacher_model_id="amazon.nova-pro-v1:0",  # MID tier
        )
        # One step down from MID = LITE
        assert distilled.tier == Tier.LITE

    def test_capabilities_inherited(self):
        teacher = self.registry.get("amazon.nova-pro-v1:0")
        distilled = self.mgr.register_distilled(
            model_id="my-distilled",
            teacher_model_id="amazon.nova-pro-v1:0",
        )
        assert distilled.capabilities.tool_use == teacher.capabilities.tool_use
        assert distilled.capabilities.vision == teacher.capabilities.vision

    def test_unknown_teacher_raises(self):
        with pytest.raises(ValueError, match="not found"):
            self.mgr.register_distilled(
                model_id="bad",
                teacher_model_id="nonexistent-model",
            )

    def test_list_distilled(self):
        self.mgr.register_distilled(
            model_id="distilled-a",
            teacher_model_id="amazon.nova-pro-v1:0",
        )
        self.mgr.register_distilled(
            model_id="distilled-b",
            teacher_model_id="anthropic.claude-sonnet-4-6",
        )
        distilled = self.mgr.list_distilled()
        assert len(distilled) == 2
        ids = {m.model_id for m in distilled}
        assert ids == {"distilled-a", "distilled-b"}

    def test_distilled_in_registry_queries(self):
        self.mgr.register_distilled(
            model_id="distilled-nova",
            teacher_model_id="amazon.nova-pro-v1:0",
        )
        m = self.registry.get("distilled-nova")
        assert m is not None
        assert m.family == "amazon"

    def test_custom_display_name(self):
        distilled = self.mgr.register_distilled(
            model_id="my-model",
            teacher_model_id="amazon.nova-pro-v1:0",
            display_name="My Custom Distilled Model",
        )
        assert distilled.display_name == "My Custom Distilled Model"
