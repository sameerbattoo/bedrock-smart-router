# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the fallback handler."""

from bedrock_smart_router.fallback_handler import FallbackConfig, FallbackHandler
from bedrock_smart_router.model_registry import ModelRegistry


class TestFallbackHandler:
    def setup_method(self):
        self.registry = ModelRegistry()
        self.handler = FallbackHandler(self.registry)

    def test_chain_not_empty(self):
        primary = self.registry.get("anthropic.claude-sonnet-4-5-20250929-v1:0")
        assert primary is not None
        chain = self.handler.build_chain(primary)
        assert len(chain) > 0

    def test_chain_excludes_primary(self):
        primary = self.registry.get("anthropic.claude-sonnet-4-5-20250929-v1:0")
        chain = self.handler.build_chain(primary)
        assert all(m.model_id != primary.model_id for m in chain)

    def test_chain_respects_max_depth(self):
        config = FallbackConfig(max_depth=2)
        handler = FallbackHandler(self.registry, config)
        primary = self.registry.get("anthropic.claude-sonnet-4-5-20250929-v1:0")
        chain = handler.build_chain(primary)
        assert len(chain) <= 2

    def test_disabled_returns_empty(self):
        config = FallbackConfig(enabled=False)
        handler = FallbackHandler(self.registry, config)
        primary = self.registry.get("anthropic.claude-sonnet-4-5-20250929-v1:0")
        chain = handler.build_chain(primary)
        assert chain == []

    def test_context_window_fallback(self):
        fb = self.handler.find_context_window_fallback(150_000)
        assert fb is not None
        assert fb.max_input_tokens >= 150_000

    def test_same_family_downgrade_first(self):
        primary = self.registry.get("anthropic.claude-sonnet-4-5-20250929-v1:0")
        chain = self.handler.build_chain(primary)
        # Haiku should appear before non-Anthropic models
        anthropic_in_chain = [m for m in chain if m.family == "anthropic"]
        if anthropic_in_chain:
            first_anthropic_idx = chain.index(anthropic_in_chain[0])
            non_anthropic = [m for m in chain if m.family != "anthropic"]
            if non_anthropic:
                first_other_idx = chain.index(non_anthropic[0])
                assert first_anthropic_idx < first_other_idx
