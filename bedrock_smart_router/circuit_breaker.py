"""Circuit breaker — prevents cascading failures by temporarily
blocking requests to models that are failing.

States:
  CLOSED    -> Normal operation.  Requests flow through.
  OPEN      -> Model is failing.  Requests immediately skip to fallback.
  HALF_OPEN -> After cooldown, allow one probe request to test recovery.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field

from bedrock_smart_router.models import CircuitState

logger = logging.getLogger(__name__)


@dataclass
class CircuitBreakerConfig:
    """Tuning knobs for the circuit breaker."""

    failure_threshold: int = 5
    window_seconds: float = 60.0
    cooldown_seconds: float = 30.0
    throttle_cooldown_seconds: float = 10.0
    half_open_max_requests: int = 1


@dataclass
class _ModelCircuit:
    """Per-model circuit breaker state."""

    state: CircuitState = CircuitState.CLOSED
    failures: deque = field(default_factory=deque)  # timestamps of failures
    last_opened_at: float = 0.0
    half_open_attempts: int = 0
    is_throttle: bool = False  # Was the last failure a throttle (429)?


class CircuitBreakerRegistry:
    """Manages circuit breakers for all models.

    Thread-safe: all state reads and mutations are protected by a lock.
    """

    def __init__(self, config: CircuitBreakerConfig | None = None) -> None:
        self.config = config or CircuitBreakerConfig()
        self._circuits: dict[str, _ModelCircuit] = {}
        self._lock = threading.Lock()

    def _get(self, model_id: str) -> _ModelCircuit:
        if model_id not in self._circuits:
            self._circuits[model_id] = _ModelCircuit()
        return self._circuits[model_id]

    def is_available(self, model_id: str) -> bool:
        """Check whether a model is available (circuit not OPEN)."""
        with self._lock:
            circuit = self._get(model_id)
            now = time.monotonic()

            if circuit.state == CircuitState.CLOSED:
                return True

            if circuit.state == CircuitState.OPEN:
                cooldown = (
                    self.config.throttle_cooldown_seconds
                    if circuit.is_throttle
                    else self.config.cooldown_seconds
                )
                if now - circuit.last_opened_at >= cooldown:
                    circuit.state = CircuitState.HALF_OPEN
                    circuit.half_open_attempts = 0
                    logger.info("Circuit for %s -> HALF_OPEN (probing)", model_id)
                    return True
                return False

            # HALF_OPEN — allow limited probe requests
            if circuit.state == CircuitState.HALF_OPEN:
                return circuit.half_open_attempts < self.config.half_open_max_requests

            return True  # pragma: no cover

    def record_success(self, model_id: str) -> None:
        """Record a successful request — may close a half-open circuit."""
        with self._lock:
            circuit = self._get(model_id)
            if circuit.state == CircuitState.HALF_OPEN:
                circuit.state = CircuitState.CLOSED
                circuit.failures.clear()
                logger.info("Circuit for %s -> CLOSED (probe succeeded)", model_id)

    def record_failure(self, model_id: str, *, is_throttle: bool = False) -> None:
        """Record a failed request — may open the circuit."""
        with self._lock:
            circuit = self._get(model_id)
            now = time.monotonic()

            if circuit.state == CircuitState.HALF_OPEN:
                circuit.state = CircuitState.OPEN
                circuit.last_opened_at = now
                circuit.is_throttle = is_throttle
                circuit.half_open_attempts = 0
                logger.warning("Circuit for %s -> OPEN (probe failed)", model_id)
                return

            # Prune old failures outside the window
            cutoff = now - self.config.window_seconds
            while circuit.failures and circuit.failures[0] < cutoff:
                circuit.failures.popleft()

            circuit.failures.append(now)

            if len(circuit.failures) >= self.config.failure_threshold:
                circuit.state = CircuitState.OPEN
                circuit.last_opened_at = now
                circuit.is_throttle = is_throttle
                logger.warning(
                    "Circuit for %s -> OPEN (%d failures in %.0fs)",
                    model_id,
                    len(circuit.failures),
                    self.config.window_seconds,
                )

    def get_state(self, model_id: str) -> CircuitState:
        with self._lock:
            return self._get(model_id).state

    def get_all_states(self) -> dict[str, CircuitState]:
        with self._lock:
            return {mid: c.state for mid, c in self._circuits.items()}

    def reset(self, model_id: str) -> None:
        """Manually reset a circuit to CLOSED."""
        with self._lock:
            circuit = self._get(model_id)
            circuit.state = CircuitState.CLOSED
            circuit.failures.clear()
            logger.info("Circuit for %s manually reset -> CLOSED", model_id)
