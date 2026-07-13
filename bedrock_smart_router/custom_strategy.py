# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Custom strategy plugin interface.

Users can implement their own routing strategy by subclassing
``RoutingStrategy`` and registering it with the strategy engine.

There are two paths for custom strategies:

**Path 1 (Minimal — recommended):** Override ``weights`` and ``score_model()``.
The base class handles filtering, ranking, fallback chains, cost/quality/latency
scoring, and explanation assembly. You only define your custom scoring dimensions.

**Path 2 (Full control):** Override ``select()`` directly for logic that can't
be expressed as "score each model independently."

Example (Path 1 — Minimal)::

    from bedrock_smart_router.custom_strategy import register_strategy
    from bedrock_smart_router.strategy_engine import RoutingStrategy, StrategyContext
    from bedrock_smart_router.models import BedrockModel, RequestAnalysis

    COMPLIANCE_SCORES = {
        "hipaa": {"anthropic": 0.95, "amazon": 0.98, "meta": 0.70},
        "general": {"anthropic": 1.0, "amazon": 1.0, "meta": 1.0},
    }

    class ComplianceStrategy(RoutingStrategy):
        name = "compliance"

        @property
        def weights(self):
            return {"compliance": 0.50, "quality": 0.30, "cost": 0.20}

        def score_model(self, model, analysis, context):
            tier = context.metadata.get("compliance_tier", "general")
            score = COMPLIANCE_SCORES.get(tier, {}).get(model.family, 0.5)
            return {"compliance": score}

        def filter_candidates(self, candidates, analysis, context):
            approved = set(context.metadata.get("approved_models", []))
            if not approved:
                return candidates, {}
            filtered = [m for m in candidates
                        if m.model_id in approved or m.base_model_id in approved]
            return filtered, {"rejected": len(candidates) - len(filtered)}

    register_strategy("compliance", ComplianceStrategy)

    # Now usable in config:
    router = BedrockRouter.create({
        "strategy": "compliance",
        "metadata": {
            "approved_models": ["anthropic.claude-sonnet-4-20250514"],
            "compliance_tier": "hipaa",
        },
    })

Example (Path 2 — Full control)::

    from bedrock_smart_router.custom_strategy import register_strategy
    from bedrock_smart_router.strategy_engine import RoutingStrategy, StrategyResult

    class PreferAnthropicForCode(RoutingStrategy):
        name = "anthropic-code"

        @property
        def weights(self):
            return {"quality": 1.0}

        def score_model(self, model, analysis, context):
            return {}  # No custom dimensions

        def select(self, candidates, analysis):
            if analysis.is_code_task:
                candidates = [c for c in candidates
                              if c.family == "anthropic"] or candidates
            best = max(candidates, key=lambda m: m.quality_baseline)
            scores = {best.model_id: {"quality": 1.0, "composite": 1.0}}
            return StrategyResult(
                selected_model=best,
                scores=scores,
                fallback_chain=candidates[:3],
            )

    register_strategy("anthropic-code", PreferAnthropicForCode)

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

Built-in dimensions (computed by base class — do NOT score these):
    - "quality": model.quality_baseline / 60.0
    - "cost": inverse cost (cheaper = higher)
    - "latency": tier-based heuristic
"""

from __future__ import annotations

from typing import Any

from bedrock_smart_router.strategy_engine import (
    BUILTIN_STRATEGIES,
    RoutingStrategy,
)


def register_strategy(name: str, cls: type[RoutingStrategy]) -> None:
    """Register a custom strategy class so it can be resolved by name.

    Args:
        name: The strategy name to use in config (e.g. ``"compliance"``).
        cls: A subclass of ``RoutingStrategy``.

    Raises:
        TypeError: If *cls* is not a subclass of ``RoutingStrategy``.
    """
    if not (isinstance(cls, type) and issubclass(cls, RoutingStrategy)):
        raise TypeError(
            f"{cls} is not a subclass of RoutingStrategy"
        )
    BUILTIN_STRATEGIES[name] = cls


def unregister_strategy(name: str) -> bool:
    """Remove a custom strategy registration.

    Returns True if the strategy was found and removed.
    """
    return BUILTIN_STRATEGIES.pop(name, None) is not None


def list_strategies() -> list[str]:
    """Return all registered strategy names."""
    return list(BUILTIN_STRATEGIES.keys()) + ["quality-optimized"]
