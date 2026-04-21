"""Tests for the prompt cache advisor."""

from bedrock_smart_router.models import (
    BedrockModel, ModelCapabilities, ModelPricing, Tier,
)
from bedrock_smart_router.prompt_cache_advisor import PromptCacheAdvisor


def _model(caching: bool, input_price: float = 0.003, cache_read: float = 0.0003) -> BedrockModel:
    return BedrockModel(
        model_id="test-model", family="anthropic", tier=Tier.MID,
        display_name="Test", capabilities=ModelCapabilities(),
        max_input_tokens=200_000, max_output_tokens=16_384,
        pricing=ModelPricing(
            input_per_1k=input_price, output_per_1k=0.015,
            cache_read_per_1k=cache_read, cache_write_per_1k=0.00375,
        ),
        supports_prompt_caching=caching,
    )


def _msgs(text: str) -> list[dict]:
    return [{"role": "user", "content": [{"text": text}]}]


def _multi_turn(turns: int) -> list[dict]:
    msgs = []
    for i in range(turns):
        role = "user" if i % 2 == 0 else "assistant"
        msgs.append({"role": role, "content": [{"text": f"Message {i} " * 100}]})
    return msgs


class TestPromptCacheAdvisor:
    def setup_method(self):
        self.advisor = PromptCacheAdvisor()

    def test_no_caching_support(self):
        m = _model(caching=False)
        result = self.advisor.estimate(m, _msgs("Hello"))
        assert not result.cache_eligible
        assert result.savings_per_request == 0.0

    def test_short_message_not_eligible(self):
        m = _model(caching=True)
        result = self.advisor.estimate(m, _msgs("Hello"))
        assert not result.cache_eligible
        assert result.cacheable_tokens < 100

    def test_long_system_prompt_eligible(self):
        m = _model(caching=True)
        system = [{"text": "You are a helpful assistant. " * 200}]  # ~1200 tokens
        result = self.advisor.estimate(m, _msgs("Hi"), system=system)
        assert result.cache_eligible
        assert result.cacheable_tokens > 100
        assert result.savings_per_request > 0

    def test_multi_turn_conversation_eligible(self):
        m = _model(caching=True)
        msgs = _multi_turn(8)  # 8 turns, 7 are cacheable prefix
        result = self.advisor.estimate(m, msgs)
        assert result.cache_eligible
        assert result.cacheable_tokens > 100

    def test_savings_calculation(self):
        m = _model(caching=True, input_price=0.003, cache_read=0.0003)
        system = [{"text": "x" * 4000}]  # ~1000 tokens
        result = self.advisor.estimate(m, _msgs("Hi"), system=system)
        # Savings = 1000 * (0.003 - 0.0003) / 1000 = 0.0027
        assert result.savings_per_request > 0.002

    def test_rank_models(self):
        cache_model = _model(caching=True, input_price=0.003, cache_read=0.0003)
        cache_model.model_id = "cache-model"  # type: ignore
        no_cache = _model(caching=False)
        no_cache.model_id = "no-cache"  # type: ignore

        system = [{"text": "x" * 4000}]
        ranked = self.advisor.rank_models_by_cache_benefit(
            [no_cache, cache_model], _msgs("Hi"), system=system,
        )
        assert ranked[0][0].model_id == "cache-model"
        assert ranked[0][1].savings_per_request > 0
        assert ranked[1][1].savings_per_request == 0
