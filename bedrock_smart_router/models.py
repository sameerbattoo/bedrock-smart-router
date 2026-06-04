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
    streaming_tool_use: bool = True
    document_support: bool = False
    extended_thinking: bool = False
    prompt_caching: bool = False


# ── Inference tier pricing multipliers ───────────────────────────
# Standard = 1.0 (base price in models.json).
# Priority ≈ 1.75× Standard (up to 25% better OTPS latency).
# Flex ≈ 0.50× Standard (latency-tolerant, best-effort).
# These are approximate defaults — actual multipliers vary by model.
# Source: AWS Bedrock pricing page + Bedrock service tiers documentation.
TIER_PRICING_MULTIPLIER: dict[str, float] = {
    "standard": 1.0,
    "optimized": 1.75,
}


@dataclass(frozen=True)
class ModelPricing:
    """Per-1K-token pricing for a model.

    Prices represent **Standard tier** on-demand rates.  Use
    ``estimate_cost(tier=...)`` to apply the tier multiplier for
    Priority (~1.75×) or Flex (~0.50×) tiers.
    """

    input_per_1k: float = 0.0
    output_per_1k: float = 0.0
    cache_read_per_1k: float = 0.0
    cache_write_per_1k: float = 0.0

    def estimate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        tier: str = "standard",
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
    ) -> float:
        """Estimate the cost for a request in dollars.

        Formula::

            cost = (inputTokens / 1000 × inputRate)
                 + (cacheReadTokens / 1000 × cacheReadRate)
                 + (cacheWriteTokens / 1000 × cacheWriteRate)
                 + (outputTokens / 1000 × outputRate)

        All four token types are independent line items.
        Tier multiplier applies to the total.

        Args:
            input_tokens: Input tokens (from Bedrock ``usage.inputTokens``).
            output_tokens: Output tokens generated.
            tier: Inference tier — ``"standard"``, ``"optimized"``, or ``"standard"``.
            cache_read_tokens: Tokens read from prompt cache.
            cache_write_tokens: Tokens written to prompt cache.
        """
        multiplier = TIER_PRICING_MULTIPLIER.get(tier, 1.0)
        cost = (
            (input_tokens / 1000) * self.input_per_1k
            + (cache_read_tokens / 1000) * self.cache_read_per_1k
            + (cache_write_tokens / 1000) * self.cache_write_per_1k
            + (output_tokens / 1000) * self.output_per_1k
        )
        return cost * multiplier


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
    cris_profiles: list[str] = field(default_factory=list)
    regions: list[dict] = field(default_factory=list)
    supported_latency_modes: list[str] = field(
        default_factory=lambda: ["standard"]
    )
    guardrail_compatible: bool = True
    distilled_from: str | None = None
    distilled_quality_delta: float = 0.0
    quality_baseline: float = 0.0  # AA Intelligence Index score (0-60 scale)
    api_support: list[str] = field(default_factory=lambda: ["converse"])  # "converse", "chat_completions", "responses"

    @property
    def is_cris_available(self) -> bool:
        return any("cris_profiles" in r for r in self.regions)

    @property
    def base_model_id(self) -> str:
        """Model identity without geography prefix.

        ``"us.anthropic.claude-sonnet-4-6"``   → ``"anthropic.claude-sonnet-4-6"``
        ``"global.anthropic.claude-sonnet-4-6"`` → ``"anthropic.claude-sonnet-4-6"``
        """
        from bedrock_smart_router.model_registry import base_model_id
        return base_model_id(self.model_id)

    @property
    def is_global_profile(self) -> bool:
        """True if this model entry is a global CRIS profile."""
        return self.model_id.startswith("global.")


@dataclass
class RequestAnalysis:
    """Result of analyzing an incoming request."""

    complexity: Complexity = Complexity.MODERATE
    complexity_score: float = 0.5
    estimated_input_tokens: int = 0
    estimated_output_tokens: int = 0
    requires_vision: bool = False
    requires_document_support: bool = False
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
    tool_boost_applied: bool = False  # True when tool_config presence boosted complexity to moderate


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
    total_tokens: int = 0  # Bedrock's reported total (input + output)
    cache_details: list[dict[str, Any]] = field(default_factory=list)  # TTL breakdown of cache writes
    performance_config: dict[str, Any] = field(default_factory=dict)  # Latency optimization mode
    guardrail_trace: dict[str, Any] = field(default_factory=dict)  # Full guardrail assessment
    metadata: dict[str, Any] = field(default_factory=dict)
    routing_decision_ms: float | None = None  # Time spent on routing logic before API call
    explanation: dict[str, Any] | None = None  # Detailed explanation (when explain=True)

    @property
    def total_input_tokens(self) -> int:
        """Total input tokens including cached: input + cache_read + cache_write."""
        return (self.input_tokens or 0) + self.prompt_cache_read_tokens + self.prompt_cache_write_tokens

    @property
    def prompt_cache_hit_rate(self) -> float:
        """Bedrock prompt cache hit rate as a percentage (0.0–100.0).

        Formula: cacheReadTokens / (inputTokens + cacheReadTokens + cacheWriteTokens) × 100
        Returns 0.0 if no prompt caching occurred.
        """
        total = (self.input_tokens or 0) + self.prompt_cache_read_tokens + self.prompt_cache_write_tokens
        if total == 0 or self.prompt_cache_read_tokens == 0:
            return 0.0
        return (self.prompt_cache_read_tokens / total) * 100

    @property
    def network_overhead_ms(self) -> float | None:
        """Network overhead = wall-clock latency minus Bedrock server latency."""
        if self.latency_ms is not None and self.bedrock_latency_ms is not None:
            return round(self.latency_ms - self.bedrock_latency_ms, 1)
        return None
