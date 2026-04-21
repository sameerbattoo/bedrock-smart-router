"""Observability — structured logging, callbacks, and cost tracking.

Every routing decision is emitted as a structured event that can be:
  - Logged via Python logging (default)
  - Sent to custom callbacks (Datadog, Splunk, etc.)
  - Published to CloudWatch custom metrics (optional)
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from bedrock_smart_router.models import RoutingDecision

logger = logging.getLogger(__name__)

# Type alias for callback functions
RoutingCallback = Callable[["RoutingEvent"], None]


@dataclass
class RoutingEvent:
    """A structured routing event for observability."""

    timestamp: str
    request_id: str
    decision: RoutingDecision
    cache_hit: bool = False
    duration_ms: float = 0.0
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


@dataclass
class CostTracker:
    """Tracks cumulative cost with breakdowns."""

    total_cost: float = 0.0
    total_requests: int = 0
    total_cache_hits: int = 0
    cost_by_model: dict[str, float] = field(default_factory=dict)
    cost_by_strategy: dict[str, float] = field(default_factory=dict)
    cost_by_complexity: dict[str, float] = field(default_factory=dict)
    cost_saved_by_cache: float = 0.0
    cost_saved_by_routing: float = 0.0  # vs always using most expensive model

    def record(
        self,
        decision: RoutingDecision,
        cache_hit: bool = False,
        most_expensive_cost: float = 0.0,
    ) -> None:
        self.total_requests += 1
        actual = decision.actual_cost or 0.0

        if cache_hit:
            self.total_cache_hits += 1
            self.cost_saved_by_cache += decision.estimated_cost
            return

        self.total_cost += actual
        model = decision.selected_model
        self.cost_by_model[model] = self.cost_by_model.get(model, 0.0) + actual
        strategy = decision.strategy_used
        self.cost_by_strategy[strategy] = (
            self.cost_by_strategy.get(strategy, 0.0) + actual
        )
        complexity = decision.complexity_detected
        self.cost_by_complexity[complexity] = (
            self.cost_by_complexity.get(complexity, 0.0) + actual
        )
        if most_expensive_cost > actual:
            self.cost_saved_by_routing += most_expensive_cost - actual

    @property
    def avg_cost_per_request(self) -> float:
        non_cache = self.total_requests - self.total_cache_hits
        return self.total_cost / non_cache if non_cache > 0 else 0.0

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "total_cost": round(self.total_cost, 6),
            "total_requests": self.total_requests,
            "total_cache_hits": self.total_cache_hits,
            "avg_cost_per_request": round(self.avg_cost_per_request, 6),
            "cost_saved_by_cache": round(self.cost_saved_by_cache, 6),
            "cost_saved_by_routing": round(self.cost_saved_by_routing, 6),
            "cost_by_model": {
                k: round(v, 6) for k, v in self.cost_by_model.items()
            },
            "cost_by_strategy": {
                k: round(v, 6) for k, v in self.cost_by_strategy.items()
            },
        }


class ObservabilityManager:
    """Central hub for routing event emission and cost tracking."""

    def __init__(
        self,
        callbacks: list[RoutingCallback] | None = None,
        log_decisions: bool = True,
    ) -> None:
        self.callbacks = callbacks or []
        self.log_decisions = log_decisions
        self.cost_tracker = CostTracker()
        self._request_counter = 0

    def emit(
        self,
        decision: RoutingDecision,
        cache_hit: bool = False,
        duration_ms: float = 0.0,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        most_expensive_cost: float = 0.0,
    ) -> RoutingEvent:
        """Emit a routing event — log it, call callbacks, track cost."""
        self._request_counter += 1
        request_id = f"req_{self._request_counter:08d}"

        event = RoutingEvent(
            timestamp=datetime.now(timezone.utc).isoformat(),
            request_id=request_id,
            decision=decision,
            cache_hit=cache_hit,
            duration_ms=round(duration_ms, 1),
            tags=tags or [],
            metadata=metadata or {},
        )

        # Cost tracking
        self.cost_tracker.record(
            decision,
            cache_hit=cache_hit,
            most_expensive_cost=most_expensive_cost,
        )

        # Structured logging
        if self.log_decisions:
            logger.info(
                "routing_decision model=%s strategy=%s complexity=%s "
                "cost=%.6f latency_ms=%.1f cache_hit=%s fallback=%s",
                decision.selected_model,
                decision.strategy_used,
                decision.complexity_detected,
                decision.actual_cost or 0.0,
                decision.latency_ms or 0.0,
                cache_hit,
                decision.fallback_used,
            )

        # Custom callbacks
        for cb in self.callbacks:
            try:
                cb(event)
            except Exception:
                logger.exception("Callback %s failed", cb)

        return event
