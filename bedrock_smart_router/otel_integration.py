"""OpenTelemetry integration — traces and metrics for routing decisions.

Provides spans wrapping each phase of the routing pipeline and OTEL
metrics mirroring the CloudWatch metrics.  Lazy-imports the
``opentelemetry`` package so it's only required when enabled.

When disabled (default), all methods return no-op context managers
and do nothing — zero overhead.

Install::

    pip install bedrock-smart-router[otel]

Enable::

    observability:
      otel_enabled: true
      otel_service_name: "my-app"
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Generator

logger = logging.getLogger(__name__)


# ── No-op fallbacks when OTEL is not installed ──────────────────────

class _NoOpSpan:
    """Dummy span that does nothing."""

    def set_attribute(self, key: str, value: Any) -> None:
        pass

    def set_status(self, status: Any) -> None:
        pass

    def record_exception(self, exc: Exception) -> None:
        pass


@contextmanager
def _noop_span(name: str, **kwargs: Any) -> Generator[_NoOpSpan, None, None]:
    yield _NoOpSpan()


class OTelIntegration:
    """OpenTelemetry tracing and metrics for the Bedrock Smart Router.

    When ``enabled=False`` or the ``opentelemetry`` package is not
    installed, all methods are no-ops with zero overhead.
    """

    def __init__(
        self,
        enabled: bool = False,
        service_name: str = "bedrock-smart-router",
    ) -> None:
        self._enabled = enabled
        self._service_name = service_name
        self._tracer: Any = None
        self._meter: Any = None
        # OTEL metric instruments
        self._request_counter: Any = None
        self._latency_histogram: Any = None
        self._ttft_histogram: Any = None
        self._cost_counter: Any = None
        self._cache_hit_counter: Any = None
        self._fallback_counter: Any = None
        self._error_counter: Any = None

        if enabled:
            self._init_otel()

    def _init_otel(self) -> None:
        """Initialize OTEL tracer and meter instruments."""
        try:
            from opentelemetry import trace, metrics

            self._tracer = trace.get_tracer(
                self._service_name,
                schema_url="https://opentelemetry.io/schemas/1.11.0",
            )
            self._meter = metrics.get_meter(
                self._service_name,
                schema_url="https://opentelemetry.io/schemas/1.11.0",
            )

            # Create metric instruments
            self._request_counter = self._meter.create_counter(
                "bedrock_router.requests",
                description="Total routing decisions",
                unit="1",
            )
            self._latency_histogram = self._meter.create_histogram(
                "bedrock_router.latency",
                description="End-to-end latency",
                unit="ms",
            )
            self._ttft_histogram = self._meter.create_histogram(
                "bedrock_router.ttft",
                description="Time to first token (streaming)",
                unit="ms",
            )
            self._cost_counter = self._meter.create_counter(
                "bedrock_router.cost",
                description="Cumulative inference cost",
                unit="USD",
            )
            self._cache_hit_counter = self._meter.create_counter(
                "bedrock_router.cache_hits",
                description="Response cache hits",
                unit="1",
            )
            self._fallback_counter = self._meter.create_counter(
                "bedrock_router.fallbacks",
                description="Fallback events",
                unit="1",
            )
            self._error_counter = self._meter.create_counter(
                "bedrock_router.errors",
                description="Routing errors",
                unit="1",
            )

            logger.info("OpenTelemetry initialized for %s", self._service_name)

        except ImportError:
            logger.warning(
                "otel_enabled=true but opentelemetry not installed. "
                "Install with: pip install bedrock-smart-router[otel]"
            )
            self._enabled = False
        except Exception as exc:
            logger.warning("Failed to initialize OpenTelemetry: %s", exc)
            self._enabled = False

    # ── Span helpers ────────────────────────────────────────────

    @contextmanager
    def span(self, name: str, attributes: dict[str, Any] | None = None):
        """Create a traced span.  No-op when disabled."""
        if not self._enabled or self._tracer is None:
            yield _NoOpSpan()
            return

        with self._tracer.start_as_current_span(name) as otel_span:
            if attributes:
                for k, v in attributes.items():
                    if v is not None:
                        otel_span.set_attribute(k, v)
            yield otel_span

    def set_span_attributes(self, span: Any, attrs: dict[str, Any]) -> None:
        """Set attributes on a span (works with both real and no-op spans)."""
        if not self._enabled:
            return
        for k, v in attrs.items():
            if v is not None:
                span.set_attribute(k, v)

    def record_exception(self, span: Any, exc: Exception) -> None:
        """Record an exception on a span."""
        if not self._enabled:
            return
        try:
            span.set_status({"status_code": "ERROR", "description": str(exc)})
            span.record_exception(exc)
        except Exception:
            pass

    # ── Metric helpers ──────────────────────────────────────────

    def record_request(
        self,
        model: str,
        strategy: str,
        complexity: str,
        latency_ms: float,
        cost: float,
        cache_hit: bool = False,
        fallback_used: bool = False,
        ttft_ms: float | None = None,
    ) -> None:
        """Record metrics for a completed routing decision."""
        if not self._enabled:
            return

        labels = {
            "model": model,
            "strategy": strategy,
            "complexity": complexity,
        }

        try:
            if self._request_counter:
                self._request_counter.add(1, labels)
            if self._latency_histogram and latency_ms > 0:
                self._latency_histogram.record(latency_ms, labels)
            if self._ttft_histogram and ttft_ms is not None and ttft_ms > 0:
                self._ttft_histogram.record(ttft_ms, labels)
            if self._cost_counter and cost > 0:
                self._cost_counter.add(cost, labels)
            if self._cache_hit_counter and cache_hit:
                self._cache_hit_counter.add(1, labels)
            if self._fallback_counter and fallback_used:
                self._fallback_counter.add(1, labels)
        except Exception as exc:
            logger.debug("OTEL metric recording failed: %s", exc)

    def record_error(self, model: str, error_type: str) -> None:
        """Record an error metric."""
        if not self._enabled or not self._error_counter:
            return
        try:
            self._error_counter.add(1, {"model": model, "error_type": error_type})
        except Exception:
            pass

    @property
    def enabled(self) -> bool:
        return self._enabled
