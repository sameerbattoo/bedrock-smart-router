"""Tests for OpenTelemetry integration."""

from unittest.mock import MagicMock, patch

import pytest

from bedrock_smart_router.otel_integration import OTelIntegration, _NoOpSpan


class TestOTelDisabled:
    """When disabled, everything is a no-op."""

    def test_disabled_by_default(self):
        otel = OTelIntegration()
        assert not otel.enabled

    def test_span_returns_noop(self):
        otel = OTelIntegration(enabled=False)
        with otel.span("test") as span:
            assert isinstance(span, _NoOpSpan)
            span.set_attribute("key", "value")  # Should not crash

    def test_record_request_noop(self):
        otel = OTelIntegration(enabled=False)
        # Should not crash
        otel.record_request(
            model="m", strategy="s", complexity="c",
            latency_ms=100, cost=0.01,
        )

    def test_record_error_noop(self):
        otel = OTelIntegration(enabled=False)
        otel.record_error("model-a", "ThrottlingException")

    def test_set_span_attributes_noop(self):
        otel = OTelIntegration(enabled=False)
        otel.set_span_attributes(_NoOpSpan(), {"key": "value"})

    def test_record_exception_noop(self):
        otel = OTelIntegration(enabled=False)
        otel.record_exception(_NoOpSpan(), RuntimeError("test"))


class TestOTelMissingPackage:
    """When enabled but opentelemetry not installed, falls back to disabled."""

    def test_missing_package_disables(self):
        with patch.dict("sys.modules", {"opentelemetry": None, "opentelemetry.trace": None, "opentelemetry.metrics": None}):
            otel = OTelIntegration(enabled=True)
            # Should fall back to disabled without crashing
            assert not otel.enabled


class TestOTelEnabled:
    """When enabled with mocked opentelemetry."""

    def _make_otel(self):
        """Create an OTelIntegration with mocked tracer and meter."""
        otel = OTelIntegration(enabled=False)  # Don't init yet
        otel._enabled = True
        otel._tracer = MagicMock()
        otel._meter = MagicMock()
        otel._request_counter = MagicMock()
        otel._latency_histogram = MagicMock()
        otel._ttft_histogram = MagicMock()
        otel._cost_counter = MagicMock()
        otel._cache_hit_counter = MagicMock()
        otel._fallback_counter = MagicMock()
        otel._error_counter = MagicMock()
        return otel

    def test_record_request_calls_counter(self):
        otel = self._make_otel()
        otel.record_request(
            model="model-a", strategy="balanced", complexity="moderate",
            latency_ms=150, cost=0.005,
        )
        otel._request_counter.add.assert_called_once()
        otel._latency_histogram.record.assert_called_once()
        otel._cost_counter.add.assert_called_once()

    def test_record_request_with_cache_hit(self):
        otel = self._make_otel()
        otel.record_request(
            model="m", strategy="s", complexity="c",
            latency_ms=0, cost=0, cache_hit=True,
        )
        otel._cache_hit_counter.add.assert_called_once()

    def test_record_request_with_fallback(self):
        otel = self._make_otel()
        otel.record_request(
            model="m", strategy="s", complexity="c",
            latency_ms=100, cost=0.01, fallback_used=True,
        )
        otel._fallback_counter.add.assert_called_once()

    def test_record_request_with_ttft(self):
        otel = self._make_otel()
        otel.record_request(
            model="m", strategy="s", complexity="c",
            latency_ms=500, cost=0.01, ttft_ms=45.0,
        )
        otel._ttft_histogram.record.assert_called_once_with(
            45.0, {"model": "m", "strategy": "s", "complexity": "c"}
        )

    def test_record_error(self):
        otel = self._make_otel()
        otel.record_error("model-a", "ThrottlingException")
        otel._error_counter.add.assert_called_once_with(
            1, {"model": "model-a", "error_type": "ThrottlingException"}
        )

    def test_span_creates_tracer_span(self):
        otel = self._make_otel()
        mock_span = MagicMock()
        otel._tracer.start_as_current_span.return_value.__enter__ = MagicMock(return_value=mock_span)
        otel._tracer.start_as_current_span.return_value.__exit__ = MagicMock(return_value=False)

        with otel.span("test_span", attributes={"key": "value"}) as span:
            assert span == mock_span
        otel._tracer.start_as_current_span.assert_called_once_with("test_span")

    def test_labels_include_model_strategy_complexity(self):
        otel = self._make_otel()
        otel.record_request(
            model="anthropic.claude-sonnet-4-6",
            strategy="quality-optimized",
            complexity="reasoning",
            latency_ms=2000, cost=0.05,
        )
        expected_labels = {
            "model": "anthropic.claude-sonnet-4-6",
            "strategy": "quality-optimized",
            "complexity": "reasoning",
        }
        otel._request_counter.add.assert_called_once_with(1, expected_labels)

    def test_zero_cost_not_recorded(self):
        otel = self._make_otel()
        otel.record_request(
            model="m", strategy="s", complexity="c",
            latency_ms=100, cost=0.0,
        )
        otel._cost_counter.add.assert_not_called()

    def test_metric_failure_doesnt_crash(self):
        otel = self._make_otel()
        otel._request_counter.add.side_effect = Exception("OTEL down")
        # Should not raise
        otel.record_request(
            model="m", strategy="s", complexity="c",
            latency_ms=100, cost=0.01,
        )


class TestOTelWithRealImports:
    """Tests using the actual opentelemetry package (not mocked)."""

    def test_init_with_real_otel(self):
        """OTelIntegration should initialize with real opentelemetry."""
        otel = OTelIntegration(enabled=True, service_name="test-router")
        assert otel.enabled
        assert otel._tracer is not None
        assert otel._meter is not None

    def test_real_span(self):
        """Real OTEL span should work end-to-end."""
        otel = OTelIntegration(enabled=True, service_name="test-router")
        with otel.span("test.operation", attributes={"key": "value"}) as span:
            assert span is not None
            # Real span should accept set_attribute
            span.set_attribute("extra_key", 42)

    def test_real_nested_spans(self):
        """Nested spans should work."""
        otel = OTelIntegration(enabled=True, service_name="test-router")
        with otel.span("parent") as parent:
            parent.set_attribute("level", "parent")
            with otel.span("child") as child:
                child.set_attribute("level", "child")

    def test_real_record_request(self):
        """record_request should work with real OTEL instruments."""
        otel = OTelIntegration(enabled=True, service_name="test-router")
        # Should not raise
        otel.record_request(
            model="anthropic.claude-sonnet-4-6",
            strategy="balanced",
            complexity="complex",
            latency_ms=1500,
            cost=0.045,
            cache_hit=False,
            fallback_used=True,
            ttft_ms=120.0,
        )

    def test_real_record_error(self):
        """record_error should work with real OTEL instruments."""
        otel = OTelIntegration(enabled=True, service_name="test-router")
        otel.record_error("model-a", "ThrottlingException")

    def test_real_record_exception_on_span(self):
        """record_exception should annotate a real span."""
        otel = OTelIntegration(enabled=True, service_name="test-router")
        with otel.span("failing_op") as span:
            try:
                raise ValueError("test error")
            except ValueError as exc:
                otel.record_exception(span, exc)

    def test_metric_instruments_created(self):
        """All metric instruments should be created on init."""
        otel = OTelIntegration(enabled=True, service_name="test-router")
        assert otel._request_counter is not None
        assert otel._latency_histogram is not None
        assert otel._ttft_histogram is not None
        assert otel._cost_counter is not None
        assert otel._cache_hit_counter is not None
        assert otel._fallback_counter is not None
        assert otel._error_counter is not None

    def test_multiple_requests_accumulate(self):
        """Multiple record_request calls should not crash."""
        otel = OTelIntegration(enabled=True, service_name="test-router")
        for i in range(10):
            otel.record_request(
                model=f"model-{i % 3}",
                strategy="balanced",
                complexity="moderate",
                latency_ms=100 + i * 10,
                cost=0.001 * (i + 1),
            )

    def test_service_name_propagated(self):
        """Custom service name should be used for tracer and meter."""
        otel = OTelIntegration(enabled=True, service_name="my-custom-app")
        assert otel._service_name == "my-custom-app"
        assert otel.enabled
