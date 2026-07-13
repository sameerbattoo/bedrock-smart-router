# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pre-call context window validation.

Checks that the estimated token count fits within a candidate model's
context window *before* sending the request to Bedrock, avoiding wasted
API calls and latency from context-too-long errors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from bedrock_smart_router.models import BedrockModel


@dataclass
class ValidationResult:
    """Outcome of a context-window validation check."""

    valid: bool
    estimated_tokens: int
    model_limit: int
    headroom_pct: float = 0.0  # How much room is left (0.0–1.0)


from bedrock_smart_router.utils import estimate_tokens_from_messages as _estimate_tokens_from_messages


class ContextValidator:
    """Validates that a request fits within a model's context window."""

    def __init__(self, safety_margin: float = 0.05) -> None:
        """
        Args:
            safety_margin: Reserve this fraction of the context window as
                headroom (default 5%).  Helps avoid edge-case rejections
                when the token estimate is slightly off.
        """
        self.safety_margin = safety_margin

    def validate(
        self,
        messages: list[dict[str, Any]],
        model: BedrockModel,
        system: list[dict[str, Any]] | None = None,
    ) -> ValidationResult:
        estimated = _estimate_tokens_from_messages(messages, system)
        effective_limit = int(model.max_input_tokens * (1 - self.safety_margin))
        valid = estimated <= effective_limit
        headroom = max(0.0, (effective_limit - estimated) / max(1, effective_limit))
        return ValidationResult(
            valid=valid,
            estimated_tokens=estimated,
            model_limit=model.max_input_tokens,
            headroom_pct=round(headroom, 4),
        )

    def filter_by_context(
        self,
        models: list[BedrockModel],
        messages: list[dict[str, Any]],
        system: list[dict[str, Any]] | None = None,
    ) -> list[BedrockModel]:
        """Return only models whose context window fits the request."""
        estimated = _estimate_tokens_from_messages(messages, system)
        return [
            m for m in models
            if estimated <= int(m.max_input_tokens * (1 - self.safety_margin))
        ]
