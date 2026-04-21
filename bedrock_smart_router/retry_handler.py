"""Retry handler with configurable exponential backoff.

Distinguishes between retryable errors (throttles, transient 5xx) and
non-retryable errors (validation, auth) to avoid wasting time on
requests that will never succeed.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Bedrock error codes that are safe to retry
DEFAULT_RETRYABLE = frozenset({
    "ThrottlingException",
    "ServiceUnavailableException",
    "ModelTimeoutException",
    "InternalServerException",
    "ServiceException",
})

DEFAULT_NON_RETRYABLE = frozenset({
    "ValidationException",
    "AccessDeniedException",
    "ResourceNotFoundException",
    "ModelNotReadyException",
})


@dataclass
class RetryConfig:
    """Retry policy configuration."""

    max_retries: int = 3
    backoff_base_seconds: float = 0.5
    backoff_max_seconds: float = 8.0
    backoff_multiplier: float = 2.0
    retryable_errors: frozenset[str] = DEFAULT_RETRYABLE
    non_retryable_errors: frozenset[str] = DEFAULT_NON_RETRYABLE


def _get_error_code(exc: Exception) -> str:
    """Extract the Bedrock/botocore error code from an exception."""
    # botocore ClientError
    if hasattr(exc, "response"):
        return exc.response.get("Error", {}).get("Code", type(exc).__name__)  # type: ignore[union-attr]
    return type(exc).__name__


def _is_throttle(exc: Exception) -> bool:
    return _get_error_code(exc) in ("ThrottlingException",)


class RetryHandler:
    """Executes a callable with retries and exponential backoff."""

    def __init__(self, config: RetryConfig | None = None) -> None:
        self.config = config or RetryConfig()

    def execute(
        self,
        fn: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Call *fn* with retries.  Returns the result or raises the last error."""
        last_exc: Exception | None = None
        for attempt in range(1 + self.config.max_retries):
            try:
                return fn(*args, **kwargs)
            except Exception as exc:
                last_exc = exc
                code = _get_error_code(exc)

                if code in self.config.non_retryable_errors:
                    logger.debug("Non-retryable error %s, raising immediately", code)
                    raise

                if code not in self.config.retryable_errors:
                    logger.debug("Unknown error %s, raising immediately", code)
                    raise

                if attempt >= self.config.max_retries:
                    logger.warning(
                        "Max retries (%d) exhausted for %s",
                        self.config.max_retries,
                        code,
                    )
                    raise

                delay = min(
                    self.config.backoff_base_seconds
                    * (self.config.backoff_multiplier ** attempt),
                    self.config.backoff_max_seconds,
                )
                logger.info(
                    "Retryable error %s (attempt %d/%d), backing off %.2fs",
                    code,
                    attempt + 1,
                    self.config.max_retries,
                    delay,
                )
                time.sleep(delay)

        # Should not reach here, but satisfy the type checker
        raise last_exc  # type: ignore[misc]

    @staticmethod
    def is_throttle(exc: Exception) -> bool:
        """Check if an exception is a throttle (429)."""
        return _is_throttle(exc)

    @staticmethod
    def get_error_code(exc: Exception) -> str:
        return _get_error_code(exc)
