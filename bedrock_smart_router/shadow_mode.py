"""Shadow mode — mirror production traffic to a secondary model.

Sends a sampled copy of requests to a shadow model asynchronously.
The shadow response is logged for offline comparison but never returned
to the caller.  Useful for evaluating a new model before any traffic
shift.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class ShadowConfig:
    """Shadow mode configuration."""

    enabled: bool = False
    shadow_model: str = ""
    sample_rate: float = 0.1  # 0.0–1.0, fraction of traffic to mirror


@dataclass
class ShadowResult:
    """Logged result of a shadow invocation."""

    shadow_model: str
    primary_model: str
    latency_ms: float
    success: bool
    error: str | None = None
    timestamp: float = 0.0


class ShadowManager:
    """Manages shadow traffic mirroring."""

    def __init__(
        self,
        config: ShadowConfig | None = None,
        invoke_fn: Callable[..., dict[str, Any]] | None = None,
    ) -> None:
        self.config = config or ShadowConfig()
        self._invoke_fn = invoke_fn
        self._results: list[ShadowResult] = []
        self._lock = threading.Lock()

    @property
    def is_active(self) -> bool:
        return self.config.enabled and bool(self.config.shadow_model)

    def should_shadow(self) -> bool:
        """Decide whether to shadow this request based on sample rate."""
        if not self.is_active:
            return False
        import random
        return random.random() < self.config.sample_rate

    def mirror(
        self,
        *,
        primary_model: str,
        messages: list[dict[str, Any]],
        system: list[dict[str, Any]] | None = None,
        tool_config: dict[str, Any] | None = None,
        inference_config: dict[str, Any] | None = None,
    ) -> None:
        """Send a shadow request in a background thread.

        The shadow response is logged but never affects the primary
        response or latency.
        """
        if self._invoke_fn is None:
            logger.warning("Shadow mode active but no invoke function set")
            return

        def _run() -> None:
            t0 = time.monotonic()
            try:
                self._invoke_fn(
                    modelId=self.config.shadow_model,
                    messages=messages,
                    **({"system": system} if system else {}),
                    **({"toolConfig": tool_config} if tool_config else {}),
                    **({"inferenceConfig": inference_config} if inference_config else {}),
                )
                elapsed = (time.monotonic() - t0) * 1000
                result = ShadowResult(
                    shadow_model=self.config.shadow_model,
                    primary_model=primary_model,
                    latency_ms=elapsed,
                    success=True,
                    timestamp=time.time(),
                )
            except Exception as exc:
                elapsed = (time.monotonic() - t0) * 1000
                result = ShadowResult(
                    shadow_model=self.config.shadow_model,
                    primary_model=primary_model,
                    latency_ms=elapsed,
                    success=False,
                    error=str(exc),
                    timestamp=time.time(),
                )
                logger.debug("Shadow request failed: %s", exc)

            with self._lock:
                self._results.append(result)
                # Keep last 1000 results
                if len(self._results) > 1000:
                    self._results = self._results[-1000:]

            logger.debug(
                "Shadow %s: %.0fms, success=%s",
                self.config.shadow_model, result.latency_ms, result.success,
            )

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()

    @property
    def results(self) -> list[ShadowResult]:
        with self._lock:
            return list(self._results)

    @property
    def stats(self) -> dict[str, Any]:
        with self._lock:
            results = list(self._results)
        if not results:
            return {"active": self.is_active, "total": 0}
        successes = sum(1 for r in results if r.success)
        latencies = [r.latency_ms for r in results if r.success]
        return {
            "active": self.is_active,
            "shadow_model": self.config.shadow_model,
            "total": len(results),
            "success_rate": round(successes / len(results), 4),
            "avg_latency_ms": round(sum(latencies) / len(latencies), 1) if latencies else 0,
        }
