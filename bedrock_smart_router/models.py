"""Data models for the Bedrock Smart Router."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Tier(str, Enum):
    """Model capability tiers."""

    MICRO = "micro"
    LITE = "lite"
    MID = "mid"
    HEAVY = "heavy"
    REASONING = "reasoning"


class Complexity(str, Enum):
    """Request complexity levels."""

    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    REASONING = "reasoning"


class CircuitState(str, Enum):
    """Circuit breaker states."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half-open"


class HealthStatus(str, Enum):
    """Model health status."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    DOWN = "down"


@dataclass(frozen=True)
class ModelCapabilities:
    """What a model can do."""

    tool_use: bool = False
    vision: bool = False
    streaming: bool = True
    document_support: bool = False
    extended_thinking: bool = False


@dataclass(frozen=True)
class ModelPricing:
    """Per-1K-token pricing for a model."""

    input_per_1k: float = 0.0
    output_per_1k: float = 0.0
    cache_read_per_1k: float = 0.0
    cache_write_per_1k: float = 0.0

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate the cost for a request in dollars."""
        return (
            (input_tokens / 1000) * self.input_per_1k
            + (output_tokens / 1000) * self.output_per_1k
        )


@dataclass
class BedrockModel:
    """A Bedrock model with its capabilities, pricing, and metadata."""

    model_id: str
    family: str
    tier: Tier
    display_name: str
    capabilities: ModelCapabilities = field(default_factory=ModelCapabilities)
    max_input_tokens: int = 4096
    max_output_tokens: int = 4096
    pricing: ModelPricing = field(default_factory=ModelPricing)
    supports_prompt_caching: bool = False
    supports_extended_thinking: bool = False
    cris_profiles: list[str] = field(default_factory=list)
    supported_inference_tiers: list[str] = field(
        default_factory=lambda: ["standard"]
    )
    guardrail_compatible: bool = True
    distilled_from: str | None = None
    distilled_quality_delta: float = 0.0

    @property
    def is_cris_available(self) -> bool:
        return len(self.cris_profiles) > 0


@dataclass
class RequestAnalysis:
    """Result of analyzing an incoming request."""

    complexity: Complexity = Complexity.MODERATE
    complexity_score: float = 0.5
    estimated_input_tokens: int = 0
    estimated_output_tokens: int = 0
    requires_vision: bool = False
    requires_tool_use: bool = False
    requires_streaming: bool = False
    requires_long_context: bool = False
    requires_extended_thinking: bool = False
    is_code_task: bool = False
    is_conversational: bool = False
    is_multi_turn: bool = False
    conversation_turn_count: int = 0
    language: str = "en"
    content_sensitivity: str = "low"


@dataclass
class RoutingDecision:
    """The outcome of a routing decision."""

    selected_model: str
    strategy_used: str
    complexity_detected: str
    complexity_score: float
    candidates_evaluated: int
    candidate_scores: dict[str, dict[str, float]] = field(default_factory=dict)
    fallback_chain: list[str] = field(default_factory=list)
    estimated_cost: float = 0.0
    actual_cost: float | None = None
    latency_ms: float | None = None
    ttft_ms: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    fallback_used: bool = False
    fallback_model: str | None = None
    cache_hit: bool = False
    circuit_breaker_skipped: list[str] = field(default_factory=list)
    # Phase 3 fields
    inference_tier: str = "standard"
    cris_profile: str | None = None
    prompt_cache_savings: float = 0.0
    prompt_cache_read_tokens: int = 0
    prompt_cache_write_tokens: int = 0
    guardrail_checked: bool = False
    # Bedrock response metrics
    stop_reason: str = ""  # end_turn | max_tokens | tool_use | guardrail_intervened | content_filtered
    bedrock_latency_ms: float | None = None  # Server-side latency (excludes network)
    actual_service_tier: str = ""  # Tier that actually served the request
    metadata: dict[str, Any] = field(default_factory=dict)
