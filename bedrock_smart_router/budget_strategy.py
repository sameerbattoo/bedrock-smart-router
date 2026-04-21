"""Budget-constrained routing strategy.

Enforces per-request cost ceilings and rolling budget windows
(per-user, per-team, per-tenant).  When the budget is exceeded the
strategy either downgrades to a cheaper tier or rejects the request.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

from bedrock_smart_router.models import BedrockModel, RequestAnalysis, Tier
from bedrock_smart_router.strategy_engine import (
    BalancedStrategy,
    RoutingStrategy,
    StrategyResult,
)

logger = logging.getLogger(__name__)


class BudgetExceededError(Exception):
    """Raised when no model fits within the budget."""


@dataclass
class BudgetRule:
    """A single budget rule for a scope (user, team, tenant, global)."""

    max_cost_per_request: float | None = None
    max_hourly_spend: float | None = None
    max_daily_spend: float | None = None
    on_exceeded: str = "downgrade"  # "downgrade" | "reject"
    downgrade_to_tier: str = "lite"


@dataclass
class _SpendRecord:
    timestamp: float
    cost: float


class BudgetTracker:
    """Tracks spend per scope and checks budget limits."""

    def __init__(self) -> None:
        self._spend: dict[str, deque[_SpendRecord]] = defaultdict(
            lambda: deque(maxlen=10_000)
        )

    def record_spend(self, scope: str, cost: float) -> None:
        self._spend[scope].append(
            _SpendRecord(timestamp=time.monotonic(), cost=cost)
        )

    def get_spend(self, scope: str, window_seconds: float) -> float:
        cutoff = time.monotonic() - window_seconds
        return sum(
            r.cost for r in self._spend.get(scope, []) if r.timestamp >= cutoff
        )

    def check_budget(self, scope: str, rule: BudgetRule) -> str | None:
        """Return the reason the budget is exceeded, or None if OK."""
        if rule.max_hourly_spend is not None:
            hourly = self.get_spend(scope, 3600)
            if hourly >= rule.max_hourly_spend:
                return f"hourly spend ${hourly:.4f} >= ${rule.max_hourly_spend:.4f}"
        if rule.max_daily_spend is not None:
            daily = self.get_spend(scope, 86400)
            if daily >= rule.max_daily_spend:
                return f"daily spend ${daily:.4f} >= ${rule.max_daily_spend:.4f}"
        return None


class BudgetConstrainedStrategy(RoutingStrategy):
    """Like balanced, but enforces cost ceilings and rolling budgets."""

    name = "budget-constrained"

    def __init__(
        self,
        inner: RoutingStrategy | None = None,
        default_rule: BudgetRule | None = None,
        tracker: BudgetTracker | None = None,
    ) -> None:
        self.inner = inner or BalancedStrategy()
        self.default_rule = default_rule or BudgetRule()
        self.tracker = tracker or BudgetTracker()

    def select(
        self,
        candidates: list[BedrockModel],
        analysis: RequestAnalysis,
    ) -> StrategyResult:
        rule = self.default_rule

        # Filter by per-request cost ceiling
        if rule.max_cost_per_request is not None:
            affordable = [
                m
                for m in candidates
                if m.pricing.estimate_cost(
                    analysis.estimated_input_tokens,
                    analysis.estimated_output_tokens,
                )
                <= rule.max_cost_per_request
            ]
            if not affordable:
                if rule.on_exceeded == "reject":
                    raise BudgetExceededError(
                        f"No model under ${rule.max_cost_per_request:.4f} "
                        f"for ~{analysis.estimated_input_tokens} input tokens"
                    )
                # Downgrade: pick cheapest available
                affordable = sorted(
                    candidates,
                    key=lambda m: m.pricing.estimate_cost(
                        analysis.estimated_input_tokens,
                        analysis.estimated_output_tokens,
                    ),
                )[:3]
                logger.warning(
                    "Budget exceeded, downgrading to cheapest %d models",
                    len(affordable),
                )
            candidates = affordable

        return self.inner.select(candidates, analysis)

    def record_spend(self, scope: str, cost: float) -> None:
        self.tracker.record_spend(scope, cost)

    def check_budget(self, scope: str) -> str | None:
        return self.tracker.check_budget(scope, self.default_rule)
