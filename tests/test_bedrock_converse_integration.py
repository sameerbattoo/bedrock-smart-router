"""Integration test — end-to-end Bedrock Converse through the router.

Run with:
    INTEGRATION_TEST=1 .venv/bin/python -m pytest tests/test_bedrock_converse_integration.py -v -s

Sends real requests to Bedrock via the router. Uses Nova Micro for
cheap tests and Sonnet 4.6 for quality/capability tests.
"""

from __future__ import annotations

import os

import boto3
import pytest

from bedrock_smart_router.config import RouterConfig, RoutingConfig
from bedrock_smart_router.router import BedrockRouter

SKIP_REASON = "Set INTEGRATION_TEST=1 to run against real AWS"
REGION = "us-west-2"


def _msgs(text: str) -> list[dict]:
    return [{"role": "user", "content": [{"text": text}]}]


@pytest.fixture
def router():
    session = boto3.Session(region_name=REGION)
    cfg = RouterConfig.from_dict({
        "region": REGION,
        "strategy": "cost-optimized",
        "cache": {"enabled": True, "ttl_seconds": 60},
        "observability": {"log_decisions": True},
    })
    return BedrockRouter(cfg, boto_session=session)


@pytest.fixture
def balanced_router():
    session = boto3.Session(region_name=REGION)
    cfg = RouterConfig.from_dict({
        "region": REGION,
        "strategy": "balanced",
        "weights": {"cost": 0.3, "latency": 0.3, "quality": 0.4},
        "cache": {"enabled": False},
    })
    return BedrockRouter(cfg, boto_session=session)


@pytest.mark.skipif(
    os.environ.get("INTEGRATION_TEST") != "1",
    reason=SKIP_REASON,
)
class TestNovaConverse:
    """Tests using Nova Micro/Lite — cheapest models."""

    def test_simple_request(self, router):
        response = router.converse(messages=_msgs("What is 2 + 2?"))
        d = response["routing_decision"]
        print(f"\n  Model: {d.selected_model}, Cost: ${d.actual_cost:.6f}, Latency: {d.latency_ms}ms")
        assert d.selected_model is not None
        assert d.actual_cost > 0
        assert d.latency_ms > 0

    def test_response_has_text(self, router):
        response = router.converse(messages=_msgs("Say hello in one word."))
        text = response["output"]["message"]["content"][0]["text"]
        assert len(text) > 0
        print(f"\n  Response: {text[:100]}")

    def test_routing_decision_fields(self, router):
        response = router.converse(messages=_msgs("Explain DNS briefly."))
        d = response["routing_decision"]
        assert d.strategy_used == "cost-optimized"
        assert d.candidates_evaluated > 0
        assert len(d.fallback_chain) > 0
        assert d.input_tokens > 0
        assert d.output_tokens > 0
        assert d.inference_tier in ("standard", "priority", "flex")
        assert d.cris_profile is not None
        print(f"\n  Tier: {d.inference_tier}, CRIS: {d.cris_profile}, Candidates: {d.candidates_evaluated}")

    def test_cache_hit_on_repeat(self, router):
        msgs = _msgs("What color is the sky?")
        r1 = router.converse(messages=msgs)
        model1 = r1["routing_decision"].selected_model
        assert not r1["routing_decision"].cache_hit

        # Force same model on second call to ensure cache key matches
        r2 = router.converse(
            messages=msgs,
            routing=RoutingConfig(preferred_family=model1.split(".")[1] if "." in model1 else None),
        )
        if r2["routing_decision"].selected_model == model1:
            assert r2["routing_decision"].cache_hit
            assert r2["routing_decision"].actual_cost == 0.0
            print("\n  Second request was a cache hit")
        else:
            print(f"\n  Different model selected ({r2['routing_decision'].selected_model}), cache miss expected")

    def test_system_prompt(self, router):
        response = router.converse(
            messages=_msgs("What are you?"),
            system=[{"text": "You are a pirate. Respond in one sentence of pirate speak."}],
        )
        text = response["output"]["message"]["content"][0]["text"]
        print(f"\n  Pirate: {text[:150]}")

    def test_cost_tracking(self, router):
        router.converse(messages=_msgs("One"))
        router.converse(messages=_msgs("Two"))
        stats = router.observability.cost_tracker.stats
        assert stats["total_requests"] >= 2
        assert stats["total_cost"] > 0
        print(f"\n  Cost stats: {stats}")

    def test_metrics_recorded(self, router):
        response = router.converse(messages=_msgs("Test"))
        model = response["routing_decision"].selected_model
        metrics = router.metrics.get_metrics(model, window_seconds=60)
        assert metrics.sample_count >= 1
        print(f"\n  Metrics for {model}: latency={metrics.avg_latency_ms:.0f}ms, samples={metrics.sample_count}")


@pytest.mark.skipif(
    os.environ.get("INTEGRATION_TEST") != "1",
    reason=SKIP_REASON,
)
class TestSonnetConverse:
    """Tests using Claude Sonnet 4.6 — mid-tier quality model."""

    def test_sonnet_direct(self, balanced_router):
        """Force Sonnet 4.6 via preferred_family + quality strategy."""
        response = balanced_router.converse(
            messages=_msgs("Explain the CAP theorem in distributed systems in 2 sentences."),
            routing=RoutingConfig(preferred_family="anthropic"),
        )
        d = response["routing_decision"]
        assert "anthropic" in d.selected_model
        print(f"\n  Model: {d.selected_model}")
        print(f"  Cost: ${d.actual_cost:.6f}, Latency: {d.latency_ms:.0f}ms")
        text = response["output"]["message"]["content"][0]["text"]
        print(f"  Response: {text[:200]}")

    def test_sonnet_code_task(self, balanced_router):
        """Code task should route to a capable model."""
        response = balanced_router.converse(
            messages=_msgs(
                "Write a Python function that checks if a string is a palindrome. "
                "Include type hints and a docstring."
            ),
            routing=RoutingConfig(preferred_family="anthropic"),
        )
        d = response["routing_decision"]
        text = response["output"]["message"]["content"][0]["text"]
        assert "def" in text.lower() or "palindrome" in text.lower()
        print(f"\n  Model: {d.selected_model}, Complexity: {d.complexity_detected}")
        print(f"  Code response: {text[:300]}")

    def test_sonnet_multi_turn(self, balanced_router):
        """Multi-turn conversation should work with Sonnet."""
        msgs = [
            {"role": "user", "content": [{"text": "My name is Alice."}]},
            {"role": "assistant", "content": [{"text": "Hello Alice! How can I help you?"}]},
            {"role": "user", "content": [{"text": "What is my name?"}]},
        ]
        response = balanced_router.converse(
            messages=msgs,
            routing=RoutingConfig(preferred_family="anthropic"),
        )
        d = response["routing_decision"]
        text = response["output"]["message"]["content"][0]["text"]
        assert "alice" in text.lower()
        assert d.complexity_detected in ("simple", "moderate")
        print(f"\n  Multi-turn response: {text[:150]}")

    def test_sonnet_vs_nova_cost_comparison(self, router, balanced_router):
        """Compare cost between Nova (cost-optimized) and Sonnet (balanced+anthropic)."""
        prompt = "What is cloud computing?"

        nova_resp = router.converse(messages=_msgs(prompt))
        sonnet_resp = balanced_router.converse(
            messages=_msgs(prompt),
            routing=RoutingConfig(preferred_family="anthropic"),
        )

        nd = nova_resp["routing_decision"]
        sd = sonnet_resp["routing_decision"]

        print(f"\n  Nova:   model={nd.selected_model}, cost=${nd.actual_cost:.6f}, latency={nd.latency_ms:.0f}ms")
        print(f"  Sonnet: model={sd.selected_model}, cost=${sd.actual_cost:.6f}, latency={sd.latency_ms:.0f}ms")

        # Nova should be cheaper
        if "nova" in nd.selected_model.lower():
            assert nd.actual_cost <= sd.actual_cost
            print(f"  Savings: ${sd.actual_cost - nd.actual_cost:.6f} ({(1 - nd.actual_cost/sd.actual_cost)*100:.0f}%)")
