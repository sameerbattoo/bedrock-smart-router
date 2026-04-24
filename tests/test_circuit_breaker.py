"""Tests for the circuit breaker."""

import time

from bedrock_smart_router.circuit_breaker import (
    CircuitBreakerConfig,
    CircuitBreakerRegistry,
)
from bedrock_smart_router.models import CircuitState


class TestCircuitBreaker:
    def setup_method(self):
        self.config = CircuitBreakerConfig(
            failure_threshold=3,
            window_seconds=10.0,
            cooldown_seconds=0.2,
            throttle_cooldown_seconds=0.1,
            half_open_max_requests=1,
        )
        self.cb = CircuitBreakerRegistry(self.config)

    def test_starts_closed(self):
        assert self.cb.is_available("model-a")
        assert self.cb.get_state("model-a") == CircuitState.CLOSED

    def test_opens_after_threshold(self):
        for _ in range(3):
            self.cb.record_failure("model-a")
        assert self.cb.get_state("model-a") == CircuitState.OPEN
        assert not self.cb.is_available("model-a")

    def test_stays_closed_below_threshold(self):
        self.cb.record_failure("model-a")
        self.cb.record_failure("model-a")
        assert self.cb.get_state("model-a") == CircuitState.CLOSED
        assert self.cb.is_available("model-a")

    def test_transitions_to_half_open_after_cooldown(self):
        for _ in range(3):
            self.cb.record_failure("model-a")
        assert not self.cb.is_available("model-a")
        time.sleep(0.25)  # Wait for cooldown
        assert self.cb.is_available("model-a")
        assert self.cb.get_state("model-a") == CircuitState.HALF_OPEN

    def test_closes_on_successful_probe(self):
        for _ in range(3):
            self.cb.record_failure("model-a")
        time.sleep(0.25)
        self.cb.is_available("model-a")  # Triggers HALF_OPEN
        self.cb.record_success("model-a")
        assert self.cb.get_state("model-a") == CircuitState.CLOSED

    def test_reopens_on_failed_probe(self):
        for _ in range(3):
            self.cb.record_failure("model-a")
        time.sleep(0.25)
        self.cb.is_available("model-a")  # Triggers HALF_OPEN
        self.cb.record_failure("model-a")
        assert self.cb.get_state("model-a") == CircuitState.OPEN

    def test_manual_reset(self):
        for _ in range(3):
            self.cb.record_failure("model-a")
        assert self.cb.get_state("model-a") == CircuitState.OPEN
        self.cb.reset("model-a")
        assert self.cb.get_state("model-a") == CircuitState.CLOSED

    def test_independent_per_model(self):
        for _ in range(3):
            self.cb.record_failure("model-a")
        assert not self.cb.is_available("model-a")
        assert self.cb.is_available("model-b")
