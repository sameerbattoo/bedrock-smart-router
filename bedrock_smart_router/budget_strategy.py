"""Budget-constrained routing strategy.

Enforces per-request cost ceilings and rolling budget windows
(per-user, per-team, per-tenant).  When the budget is exceeded the
strategy either downgrades to a cheaper tier or rejects the request.

Persistence:
  The ``BudgetTracker`` is in-memory by default (fast, no I/O on the
  hot path).  When a ``BudgetStore`` is provided, spend records are
  flushed to the store asynchronously and loaded on startup for
  recovery after restarts.

  Supported backends: ``sqlite``, ``dynamodb``, or any custom
  ``BudgetStore`` subclass.
"""

from __future__ import annotations

import logging
import threading
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
    """Tracks spend per scope and checks budget limits.

    In-memory by default.  When a ``BudgetStore`` is provided:
    - Spend records are flushed to the store every ``sync_interval`` seconds
    - On init, recent spend is loaded from the store (recovery after restart)
    - The hot path (record_spend, check_budget) always uses in-memory data

    Args:
        store: Optional persistent backend (SQLite, DynamoDB, or custom).
        sync_interval: Seconds between flushes to the persistent store.
        cleanup_interval: Seconds between cleanup of old records in the store.
    """

    def __init__(
        self,
        store: Any | None = None,
        sync_interval: float = 5.0,
        cleanup_interval: float = 3600.0,
    ) -> None:
        self._spend: dict[str, deque[_SpendRecord]] = defaultdict(
            lambda: deque(maxlen=10_000)
        )
        self._lock = threading.Lock()
        self._store = store
        self._pending: list[Any] = []  # SpendRecords waiting to be flushed
        self._pending_lock = threading.Lock()

        # Load existing spend from store on init
        if self._store is not None:
            self._hydrate_from_store()
            # Start background sync thread
            self._sync_interval = sync_interval
            self._cleanup_interval = cleanup_interval
            self._sync_thread = threading.Thread(
                target=self._sync_loop, daemon=True, name="budget-sync"
            )
            self._sync_thread.start()

    def _hydrate_from_store(self) -> None:
        """Load recent spend from the persistent store into memory."""
        try:
            # Load last 24 hours (covers both hourly and daily windows)
            all_spend = self._store.get_all_spend(86400)
            with self._lock:
                for scope, total in all_spend.items():
                    # Add as a single aggregated record at current time
                    # (we lose per-record granularity but get correct totals)
                    if total > 0:
                        self._spend[scope].append(
                            _SpendRecord(timestamp=time.monotonic(), cost=total)
                        )
            logger.info(
                "BudgetTracker hydrated from store: %d scopes loaded",
                len(all_spend),
            )
        except Exception as e:
            logger.warning("Failed to hydrate BudgetTracker from store: %s", e)

    def _sync_loop(self) -> None:
        """Background thread: flush pending records and cleanup old data."""
        last_cleanup = time.monotonic()
        while True:
            time.sleep(self._sync_interval)
            self._flush_pending()
            # Periodic cleanup (every cleanup_interval)
            if time.monotonic() - last_cleanup >= self._cleanup_interval:
                try:
                    self._store.cleanup(older_than_seconds=86400)
                except Exception as e:
                    logger.debug("Budget store cleanup failed: %s", e)
                last_cleanup = time.monotonic()

    def _flush_pending(self) -> None:
        """Flush pending spend records to the persistent store."""
        with self._pending_lock:
            if not self._pending:
                return
            batch = self._pending[:]
            self._pending.clear()
        try:
            self._store.write_batch(batch)
        except Exception as e:
            logger.warning("Failed to flush budget records to store: %s", e)
            # Re-queue failed records (will retry on next sync)
            with self._pending_lock:
                self._pending = batch + self._pending

    def record_spend(
        self, scope: str, cost: float, model_id: str = "", metadata: dict | None = None
    ) -> None:
        """Record a spend event for a scope.

        Writes to in-memory immediately (fast). If a persistent store
        is configured, queues the record for async flush.
        """
        now_mono = time.monotonic()
        with self._lock:
            self._spend[scope].append(_SpendRecord(timestamp=now_mono, cost=cost))

        # Queue for persistence (non-blocking)
        if self._store is not None:
            from bedrock_smart_router.budget_store import SpendRecord
            with self._pending_lock:
                self._pending.append(SpendRecord(
                    scope=scope,
                    cost=cost,
                    timestamp=time.time(),  # Wall clock for persistence
                    model_id=model_id,
                    metadata=metadata,
                ))

    def get_spend(self, scope: str, window_seconds: float) -> float:
        """Get total spend for a scope within a rolling time window."""
        cutoff = time.monotonic() - window_seconds
        with self._lock:
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

    @property
    def weights(self) -> dict[str, float]:
        return self.inner.weights

    def score_model(self, model, analysis, context):
        return {}  # Delegates to inner strategy via select() override

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
                # Downgrade: pick highest quality_baseline among cheapest
                affordable = sorted(
                    candidates,
                    key=lambda m: (
                        m.pricing.estimate_cost(
                            analysis.estimated_input_tokens,
                            analysis.estimated_output_tokens,
                        ),
                        -m.quality_baseline,  # Tiebreaker: prefer higher quality
                    ),
                )[:3]
                logger.warning(
                    "Budget exceeded, downgrading to cheapest %d models",
                    len(affordable),
                )
            else:
                # Within budget: sort by quality_baseline (best quality first)
                # so the inner strategy has the best options at the top
                affordable.sort(key=lambda m: m.quality_baseline, reverse=True)
            candidates = affordable

        return self.inner.select(candidates, analysis)

    def record_spend(self, scope: str, cost: float) -> None:
        self.tracker.record_spend(scope, cost)

    def check_budget(self, scope: str) -> str | None:
        return self.tracker.check_budget(scope, self.default_rule)
