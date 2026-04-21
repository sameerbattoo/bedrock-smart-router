"""Custom exceptions for the Bedrock Smart Router."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ModelRejection:
    """Why a specific model was rejected during filtering."""

    model_id: str
    display_name: str
    reasons: list[str]


class NoModelsMatchError(Exception):
    """Raised when no models satisfy the routing constraints.

    Includes structured feedback about which models were checked and
    why each was rejected, so the user knows what to relax.

    Attributes:
        rejections: Per-model rejection reasons.
        constraints: The constraints that were applied.
        suggestions: Actionable suggestions for the user.
    """

    def __init__(
        self,
        message: str,
        rejections: list[ModelRejection] | None = None,
        constraints: dict[str, Any] | None = None,
        suggestions: list[str] | None = None,
    ) -> None:
        self.rejections = rejections or []
        self.constraints = constraints or {}
        self.suggestions = suggestions or []
        detail = self._format_detail()
        super().__init__(f"{message}\n{detail}" if detail else message)

    def _format_detail(self) -> str:
        parts: list[str] = []
        if self.constraints:
            parts.append(f"  Constraints: {self.constraints}")
        if self.rejections:
            parts.append("  Models checked:")
            for r in self.rejections[:10]:  # Cap at 10
                reasons = ", ".join(r.reasons)
                parts.append(f"    - {r.display_name} ({r.model_id}): {reasons}")
        if self.suggestions:
            parts.append("  Suggestions:")
            for s in self.suggestions:
                parts.append(f"    - {s}")
        return "\n".join(parts)

    def to_dict(self) -> dict[str, Any]:
        """Structured representation for API responses."""
        return {
            "error": "no_models_match",
            "message": str(self),
            "constraints": self.constraints,
            "rejections": [
                {
                    "model_id": r.model_id,
                    "display_name": r.display_name,
                    "reasons": r.reasons,
                }
                for r in self.rejections
            ],
            "suggestions": self.suggestions,
        }
