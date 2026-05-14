"""Custom Strategy Plugin — extend the router with your own logic.

Demonstrates:
  - Path 1 (Minimal): Override weights + score_model() — base class does the rest
  - Path 2 (Full control): Override select() for complex logic
  - Registering custom strategies for use in config
  - Using filter_candidates() for hard-gate filtering
  - Using explain_metadata() for decision JSON context

Interface contract:
  ┌─────────────────────────┬──────────┬─────────────────────────────────────┐
  │ Method                  │ Required │ What it does                        │
  ├─────────────────────────┼──────────┼─────────────────────────────────────┤
  │ weights (property)      │ Yes      │ Declares dimensions + their weights │
  │ score_model()           │ Yes      │ Scores YOUR custom dimensions (0-1) │
  │ filter_candidates()     │ No       │ Hard-gate filtering before scoring  │
  │ explain_metadata()      │ No       │ Extra context for decision JSON     │
  │ select()                │ No       │ Full pipeline override (Path 2)     │
  └─────────────────────────┴──────────┴─────────────────────────────────────┘
"""

from bedrock_smart_router import BedrockRouter
from bedrock_smart_router.config import RoutingConfig
from bedrock_smart_router.custom_strategy import register_strategy
from bedrock_smart_router.strategy_engine import (
    RoutingStrategy,
    StrategyResult,
    StrategyContext,
)
from bedrock_smart_router.models import BedrockModel, RequestAnalysis


# ── Example 1: Compliance Strategy (Path 1 — Minimal) ───────────────
# Only ~20 lines of logic. Base class handles scoring, ranking,
# fallback chains, and explanation assembly.

COMPLIANCE_SCORES = {
    "hipaa": {"anthropic": 0.95, "amazon": 0.98, "meta": 0.70, "mistral": 0.75},
    "pci":   {"anthropic": 0.90, "amazon": 0.95, "meta": 0.60, "mistral": 0.65},
    "general": {"anthropic": 1.0, "amazon": 1.0, "meta": 1.0, "mistral": 1.0},
}


class ComplianceStrategy(RoutingStrategy):
    name = "compliance"

    @property
    def weights(self) -> dict[str, float]:
        return {"compliance": 0.50, "quality": 0.30, "cost": 0.20}

    def score_model(self, model: BedrockModel, analysis: RequestAnalysis,
                    context: StrategyContext) -> dict[str, float]:
        tier = context.metadata.get("compliance_tier", "general")
        score = COMPLIANCE_SCORES.get(tier, {}).get(model.family, 0.5)
        return {"compliance": score}

    def filter_candidates(self, candidates, analysis, context: StrategyContext):
        approved = set(context.metadata.get("approved_models", []))
        if not approved:
            return candidates, {}
        filtered = [m for m in candidates
                    if m.model_id in approved or m.base_model_id in approved]
        return filtered, {"rejected": len(candidates) - len(filtered),
                          "reason": "not in approved list"}

    def explain_metadata(self, result, analysis):
        return {"compliance_tier": "hipaa", "policy": "HIPAA-2024-v3"}


register_strategy("compliance", ComplianceStrategy)

router = BedrockRouter.create({
    "strategy": "compliance",
    "metadata": {
        "approved_models": [
            "anthropic.claude-sonnet-4-6",
            "anthropic.claude-haiku-4-5-20251001-v1:0",
            "amazon.nova-pro-v1:0",
        ],
        "compliance_tier": "hipaa",
    },
})

r1 = router.converse(messages=[{"role": "user", "content": [
    {"text": "Summarize this patient record"}
]}])
print(f"Compliance strategy → {r1['routing_decision'].selected_model}")


# ── Example 2: Code-Aware Strategy (Path 2 — Full control) ──────────
# Override select() when logic can't be expressed as per-model scoring.

class CodeAwareStrategy(RoutingStrategy):
    name = "code-aware"

    @property
    def weights(self) -> dict[str, float]:
        return {"quality": 0.7, "cost": 0.3}

    def score_model(self, model, analysis, context):
        return {}  # Not used — select() override handles everything

    def select(self, candidates, analysis):
        if analysis.is_code_task:
            preferred = [c for c in candidates if c.family == "anthropic"]
            pool = preferred or candidates
        else:
            pool = sorted(candidates, key=lambda m: m.pricing.input_per_1k)

        best = pool[0]
        scores = {m.model_id: {"composite": 1.0 if m == best else 0.5} for m in candidates}
        return StrategyResult(
            selected_model=best,
            scores=scores,
            fallback_chain=[c for c in candidates if c != best][:3],
        )


register_strategy("code-aware", CodeAwareStrategy)

router2 = BedrockRouter.create({"strategy": "code-aware"})

r2 = router2.converse(messages=[{"role": "user", "content": [
    {"text": "Write a Python function to sort a list"}
]}])
print(f"Code task → {r2['routing_decision'].selected_model}")

r3 = router2.converse(messages=[{"role": "user", "content": [
    {"text": "What is the capital of France?"}
]}])
print(f"Simple Q&A → {r3['routing_decision'].selected_model}")


# ── Example 3: EU Data Residency (Path 1 — filter only) ─────────────
# Uses filter_candidates() to enforce region lock, then lets the base
# class handle scoring with default quality+cost weights.

class EUOnlyStrategy(RoutingStrategy):
    name = "eu-only"

    @property
    def weights(self) -> dict[str, float]:
        return {"quality": 0.5, "cost": 0.5}

    def score_model(self, model, analysis, context):
        return {}  # No custom dimensions — just filtering

    def filter_candidates(self, candidates, analysis, context):
        eu_models = [
            c for c in candidates
            if any(p.startswith("eu.") for p in c.cris_profiles)
        ]
        if eu_models:
            return eu_models, {"region_filter": "eu-only"}
        return candidates, {"region_filter": "eu-only", "fallback": "no EU models found"}


register_strategy("eu-only", EUOnlyStrategy)


# ── Example 4: Time-of-Day Strategy (Path 1 — custom dimension) ─────
# Scores a "time_fit" dimension: quality models during business hours,
# cheap models off-peak.

import datetime


class TimeOfDayStrategy(RoutingStrategy):
    name = "time-of-day"

    @property
    def weights(self) -> dict[str, float]:
        return {"time_fit": 0.40, "quality": 0.30, "cost": 0.30}

    def score_model(self, model, analysis, context):
        hour = datetime.datetime.now().hour
        is_business_hours = 9 <= hour <= 17

        if is_business_hours:
            # During business hours, prefer higher-tier models
            tier_score = {"micro": 0.1, "lite": 0.3, "mid": 0.6, "heavy": 0.9, "reasoning": 1.0}
            return {"time_fit": tier_score.get(model.tier.value, 0.5)}
        else:
            # Off-peak: prefer cheaper models (invert tier)
            tier_score = {"micro": 1.0, "lite": 0.8, "mid": 0.5, "heavy": 0.2, "reasoning": 0.1}
            return {"time_fit": tier_score.get(model.tier.value, 0.5)}


register_strategy("time-of-day", TimeOfDayStrategy)


# ── Example 5: Multi-Tenant SaaS Strategy (Path 1) ──────────────────
# Routes based on customer plan tier. Enterprise customers get the best
# models; free-tier users get cheap ones. The plan is passed via
# context.metadata at request time.

# Plan → minimum quality floor (0-1 scale)
PLAN_QUALITY_FLOOR = {
    "free": 0.0,       # Any model is fine
    "starter": 0.4,    # At least mid-tier
    "enterprise": 0.7, # Heavy/reasoning only
}

# Plan → cost tolerance (higher = more willing to spend)
PLAN_COST_TOLERANCE = {
    "free": 0.0,       # Cheapest possible
    "starter": 0.5,    # Moderate spend OK
    "enterprise": 1.0, # Cost is not a concern
}


class MultiTenantStrategy(RoutingStrategy):
    name = "multi-tenant"

    @property
    def weights(self) -> dict[str, float]:
        return {"plan_fit": 0.50, "quality": 0.35, "cost": 0.15}

    def score_model(self, model, analysis, context):
        plan = context.metadata.get("plan", "free")
        quality_floor = PLAN_QUALITY_FLOOR.get(plan, 0.0)
        cost_tolerance = PLAN_COST_TOLERANCE.get(plan, 0.0)

        # How well does this model's quality match the plan's expectations?
        model_quality = model.quality_baseline / 60.0
        if model_quality < quality_floor:
            # Below the plan's floor — penalize heavily
            plan_fit = 0.1
        elif cost_tolerance < 0.5 and model_quality > 0.7:
            # Free/starter plan but expensive model — wasteful
            plan_fit = 0.3
        else:
            # Good match
            plan_fit = min(1.0, 0.5 + model_quality * cost_tolerance)

        return {"plan_fit": plan_fit}

    def filter_candidates(self, candidates, analysis, context):
        plan = context.metadata.get("plan", "free")
        if plan == "free":
            # Free tier: only allow micro/lite models
            filtered = [m for m in candidates if m.tier.value in ("micro", "lite")]
            if filtered:
                return filtered, {"plan": plan, "restricted_to": "micro/lite"}
        return candidates, {"plan": plan}

    def explain_metadata(self, result, analysis):
        return {"strategy_type": "multi-tenant-saas"}


register_strategy("multi-tenant", MultiTenantStrategy)

# Usage:
router5 = BedrockRouter.create({"strategy": "multi-tenant"})

# Free-tier user → routed to cheap micro/lite models
r5a = router5.converse(
    messages=[{"role": "user", "content": [{"text": "What time is it in Tokyo?"}]}],
    routing=RoutingConfig(metadata={"plan": "free"}),
)
print(f"\nFree plan → {r5a['routing_decision'].selected_model}")

# Enterprise user → routed to best quality model
r5b = router5.converse(
    messages=[{"role": "user", "content": [{"text": "Analyze this quarterly earnings report and identify risks"}]}],
    routing=RoutingConfig(metadata={"plan": "enterprise"}),
)
print(f"Enterprise plan → {r5b['routing_decision'].selected_model}")

# Starter user → mid-tier models
r5c = router5.converse(
    messages=[{"role": "user", "content": [{"text": "Summarize this article in 3 bullet points"}]}],
    routing=RoutingConfig(metadata={"plan": "starter"}),
)
print(f"Starter plan → {r5c['routing_decision'].selected_model}")


# ── Example 6: Capability-Match Strategy (Path 1) ───────────────────
# Scores models based on how well their capabilities match what the
# request actually needs. A model with exactly the right capabilities
# scores higher than an overpowered (expensive) one.


class CapabilityMatchStrategy(RoutingStrategy):
    name = "capability-match"

    @property
    def weights(self) -> dict[str, float]:
        return {"capability_fit": 0.45, "quality": 0.30, "cost": 0.25}

    def score_model(self, model, analysis, context):
        # Count how many required capabilities the model satisfies
        requirements_met = 0
        requirements_total = 0

        if analysis.requires_vision:
            requirements_total += 1
            if model.capabilities.vision:
                requirements_met += 1

        if analysis.requires_tool_use:
            requirements_total += 1
            if model.capabilities.tool_use:
                requirements_met += 1

        if analysis.requires_long_context:
            requirements_total += 1
            if model.max_input_tokens >= 100_000:
                requirements_met += 1

        if analysis.requires_extended_thinking:
            requirements_total += 1
            if model.capabilities.extended_thinking:
                requirements_met += 1

        if analysis.requires_document_support:
            requirements_total += 1
            if model.capabilities.document_support:
                requirements_met += 1

        if analysis.requires_streaming:
            requirements_total += 1
            if model.capabilities.streaming:
                requirements_met += 1

        if requirements_total == 0:
            # No special requirements — all models are equally fit
            return {"capability_fit": 0.8}

        # Base score: percentage of requirements met
        fit_score = requirements_met / requirements_total

        # Bonus: if model has prompt caching and it's multi-turn, slight boost
        if model.capabilities.prompt_caching and analysis.is_multi_turn:
            fit_score = min(1.0, fit_score + 0.1)

        return {"capability_fit": round(fit_score, 4)}

    def filter_candidates(self, candidates, analysis, context):
        # Hard filter: if vision is required, exclude models without it
        if analysis.requires_vision:
            vision_models = [m for m in candidates if m.capabilities.vision]
            if vision_models:
                return vision_models, {"hard_filter": "vision_required"}

        # Hard filter: if extended thinking is required, exclude models without it
        if analysis.requires_extended_thinking:
            thinking_models = [m for m in candidates if m.capabilities.extended_thinking]
            if thinking_models:
                return thinking_models, {"hard_filter": "extended_thinking_required"}

        return candidates, {}


register_strategy("capability-match", CapabilityMatchStrategy)

# Usage:
router6 = BedrockRouter.create({"strategy": "capability-match"})

# Simple text request → any model works, cheapest wins via cost weight
r6a = router6.converse(
    messages=[{"role": "user", "content": [{"text": "What is 2 + 2?"}]}],
)
print(f"\nSimple text → {r6a['routing_decision'].selected_model}")

# Request requiring tool use → filters to tool-capable models
r6b = router6.converse(
    messages=[{"role": "user", "content": [
        {"text": "Use the calculator tool to compute 15% of $2,340"}
    ]}],
    tool_config={"tools": [{"toolSpec": {
        "name": "calculator",
        "description": "Performs arithmetic",
        "inputSchema": {"json": {"type": "object", "properties": {"expression": {"type": "string"}}}},
    }}]},
)
print(f"Tool use request → {r6b['routing_decision'].selected_model}")

# Vision request → hard-filters to vision-capable models
# (Uncomment to run — requires an actual image)
# import base64
# with open("diagram.png", "rb") as f:
#     image_bytes = f.read()
# r6c = router6.converse(
#     messages=[{"role": "user", "content": [
#         {"image": {"format": "png", "source": {"bytes": base64.b64encode(image_bytes).decode()}}},
#         {"text": "Describe this architecture diagram"},
#     ]}],
# )
# print(f"Vision request → {r6c['routing_decision'].selected_model}")

print(f"\nRegistered strategies: compliance, code-aware, eu-only, "
      f"time-of-day, multi-tenant, capability-match")
