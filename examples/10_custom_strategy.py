"""Custom Strategy Plugin — extend the router with your own logic.

Demonstrates:
  - Implementing a custom RoutingStrategy
  - Registering it for use in config
  - Using it in the router
"""

from bedrock_smart_router import BedrockRouter
from bedrock_smart_router.custom_strategy import register_strategy
from bedrock_smart_router.strategy_engine import RoutingStrategy, StrategyResult
from bedrock_smart_router.models import BedrockModel, RequestAnalysis


# ── Example 1: Prefer Anthropic for code, Nova for everything else ───

class CodeAwareStrategy(RoutingStrategy):
    name = "code-aware"

    def select(self, candidates, analysis):
        if analysis.is_code_task:
            # Prefer Anthropic for code tasks
            preferred = [c for c in candidates if c.family == "anthropic"]
            pool = preferred or candidates
        else:
            # Prefer cheapest for non-code
            pool = sorted(candidates, key=lambda m: m.pricing.input_per_1k)

        best = pool[0]
        scores = {m.model_id: {"composite": 1.0 if m == best else 0.5} for m in candidates}
        return StrategyResult(
            selected_model=best,
            scores=scores,
            fallback_chain=[c for c in candidates if c != best][:3],
        )

register_strategy("code-aware", CodeAwareStrategy)

router = BedrockRouter.create({"strategy": "code-aware"})

# Code task → Anthropic
r1 = router.converse(messages=[{"role": "user", "content": [
    {"text": "Write a Python function to sort a list"}
]}])
print(f"Code task → {r1['routing_decision'].selected_model}")

# Non-code → cheapest
r2 = router.converse(messages=[{"role": "user", "content": [
    {"text": "What is the capital of France?"}
]}])
print(f"Simple Q&A → {r2['routing_decision'].selected_model}")


# ── Example 2: Region-locked strategy ────────────────────────────────
# Only allow models with EU CRIS profiles.

class EUOnlyStrategy(RoutingStrategy):
    name = "eu-only"

    def select(self, candidates, analysis):
        eu_models = [
            c for c in candidates
            if any(p.startswith("eu.") for p in c.cris_profiles)
        ]
        pool = eu_models or candidates  # Fallback if no EU models
        best = min(pool, key=lambda m: m.pricing.input_per_1k)
        return StrategyResult(
            selected_model=best,
            scores={best.model_id: {"composite": 1.0}},
            fallback_chain=pool[1:3],
        )

register_strategy("eu-only", EUOnlyStrategy)
# Now usable: BedrockRouter.create({"strategy": "eu-only"})


# ── Example 3: Time-of-day strategy ─────────────────────────────────
# Use cheap models during off-peak, quality models during business hours.

import datetime

class TimeOfDayStrategy(RoutingStrategy):
    name = "time-of-day"

    def select(self, candidates, analysis):
        hour = datetime.datetime.now().hour
        is_business_hours = 9 <= hour <= 17

        if is_business_hours:
            # Quality during business hours
            pool = sorted(candidates, key=lambda m: m.tier.value, reverse=True)
        else:
            # Cheap during off-peak
            pool = sorted(candidates, key=lambda m: m.pricing.input_per_1k)

        best = pool[0]
        return StrategyResult(
            selected_model=best,
            scores={best.model_id: {"composite": 1.0}},
            fallback_chain=pool[1:3],
        )

register_strategy("time-of-day", TimeOfDayStrategy)
print(f"\nRegistered strategies: time-of-day, eu-only, code-aware")
