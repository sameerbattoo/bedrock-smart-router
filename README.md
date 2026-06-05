# Bedrock Smart Router

Intelligent model routing for Amazon Bedrock. A lightweight Python SDK that sits between your application and Bedrock, automatically selecting the optimal model for each request based on cost, latency, quality, and task complexity.

The Smart Router unifies both Bedrock platforms — **bedrock-runtime** (Converse API) and **bedrock-mantle** (Chat Completions API) — into a single routing layer with **68 models**. It provides drop-in interfaces for both `boto3.client.converse()` and `openai.chat.completions.create()`, transparently routing to the correct backend regardless of which API surface you use.

Unlike generic LLM gateways (LiteLLM, Portkey, OpenRouter) that treat Bedrock as just another provider, the Bedrock Smart Router is purpose-built for Bedrock and understands CRIS profiles, latency optimization, prompt caching, guardrails, application inference profiles, and model distillation.

Unlike Bedrock's native prompt router, which only routes within a single model family, the Smart Router routes across all families (Anthropic, Amazon Nova, Meta, Mistral, OpenAI, DeepSeek, Qwen, NVIDIA, and more) with custom strategies and historical quality data.

## Table of Contents

- [Features](#features)
- [Quick Start](#quick-start)
- [Examples](#examples)
- [Reliability: Circuit Breakers, Fallbacks, and Retries](#reliability-circuit-breakers-fallbacks-and-retries)
- [Safe Model Rollouts: A/B Testing, Canary, and Shadow Mode](#safe-model-rollouts-ab-testing-canary-and-shadow-mode)
- [Budget Enforcement](#budget-enforcement)
- [Multimodal Routing: Images and Documents](#multimodal-routing-images-and-documents)
- [Boto Client Configuration](#boto-client-configuration)
- [Strands Agents SDK Integration](#strands-agents-sdk-integration)
- [Unified API Surface: Converse + Chat Completions](#unified-api-surface-converse--chat-completions)
- [Caching: Exact-Match, Semantic, and Auto-Extracting](#caching-exact-match-semantic-and-auto-extracting)
- [Architecture](#architecture)
- [How Routing Strategies Work](#how-routing-strategies-work)
- [Routing Decision Explainability](#routing-decision-explainability)
- [Model Catalog](#model-catalog)
- [Configuration Reference](#configuration-reference)
- [Configuration in Production](#configuration-in-production)
- [IAM Permissions](#iam-permissions)
- [Project Structure](#project-structure)
- [Building & Using in Your Project](#building--using-in-your-project)
- [Development](#development)
- [How It Compares](#how-it-compares)

## Features

**100% Bedrock Converse API Coverage**

The Smart Router is a true drop-in replacement for `bedrock-runtime.converse()` and `converse_stream()`. Every Bedrock Converse parameter is supported — either as a first-class parameter or via `**kwargs` passthrough. This includes `additionalModelRequestFields` (model-specific params like `top_k`, extended thinking), `guardrailConfig`, `performanceConfig`, `outputConfig`, `promptVariables`, and `requestMetadata`. Every response field is captured in the routing decision: token usage, prompt cache metrics, stop reason, server-side latency, service tier, cache details, performance config, and guardrail trace. Nothing is lost by using the router instead of calling Bedrock directly.

**OpenAI Chat Completions API Drop-in**

The router also exposes an OpenAI-compatible `router.chat.completions.create(...)` interface — a drop-in replacement for `openai.chat.completions.create()`. This unifies the Bedrock (`bedrock-runtime`) and Mantle (`bedrock-mantle`) platforms into a single 68-model pool. Users calling either API surface get transparent access to all models regardless of which backend they live on, with automatic format translation.

**Routing Strategies**
- Cost-optimized, latency-optimized, quality-optimized, and balanced (weighted composite)
- Named presets: `economy`, `speed`, `balanced`, `quality` — one-word shortcuts for common routing profiles
- Budget-constrained routing with per-request ceilings and rolling hourly/daily limits
- Tag-based routing for free/paid tiers and team access control
- Conditional routing based on request metadata
- Custom strategy plugins — subclass `RoutingStrategy` and register it
- `preferred_model` override — pin a specific model while keeping fallbacks and reliability

**Request Intelligence**
- 15-dimension zero-API-call complexity classifier (sub-millisecond overhead)
- Automatic complexity detection: simple, moderate, complex, reasoning
- Vision, tool use, long context, and code task detection
- Context window pre-validation before sending to Bedrock
- Multimodal-aware routing — automatically detects image and document content blocks, filters to capable models, and boosts complexity based on payload size

**Bedrock-Native Awareness**
- Cross-Region Inference (CRIS) profile selection — per-region awareness with global (`global.*`, ~10% cheaper), regional (`us.*`, `eu.*`), and direct invocation modes
- Inference tier auto-selection (Standard / Optimized) with tier-aware cost estimation
- Prompt cache benefit estimation — boosts cache-capable models (Claude and Nova) when savings are significant
- Bedrock Guardrails integration — pre-route and post-route checks via ApplyGuardrail API
- Application Inference Profile management for multi-tenant cost tracking
- Distilled model support with derived pricing and tier from teacher models

**Reliability**
- Circuit breakers (CLOSED/OPEN/HALF_OPEN) per model with separate throttle cooldowns
- Multi-level fallback chain: same-family downgrade, cross-family equivalent, direct invocation retry, safe default
- Configurable retry with exponential backoff for transient errors
- Content policy and context window fallbacks
- Graceful no-models-match errors with per-model rejection reasons and actionable suggestions

**Production Deployment**
- A/B testing with weighted variants and sticky user assignment
- Canary deployments with auto-rollback on error rate/latency thresholds
- Shadow mode — mirror traffic to a secondary model in background threads

**Caching**
- Response caching (in-memory LRU with TTL) — identical requests return instantly at zero cost
- Redis / Valkey / ElastiCache shared cache for multi-instance deployments
- Semantic caching via embeddings with pluggable vector stores (in-memory, FAISS, Redis/Valkey 8.2+, OpenSearch Serverless)
- Variable-aware semantic cache — same intent + different parameters = cache miss
- Auto-extracting semantic cache — LLM-based intent + variable extraction, no manual tagging needed
- Multi-turn semantic cache — resolves conversation history into a single query for cache matching
- Pluggable response store backends — inline, filesystem, S3, DynamoDB (or custom subclass)
- Cache filter — selective caching, app decides which responses are worth storing
- Semantic intent router — route queries to specialized models by meaning

**Observability**
- Structured routing decision logging on every request
- Custom callback hooks for Datadog, Splunk, or any analytics pipeline
- Cost tracking with breakdowns by model, strategy, complexity, and tenant
- Routing savings calculation (actual cost vs. most-expensive-model cost)
- Historical metrics store (in-memory or DynamoDB) for data-driven routing

**Async Support**
- `AsyncBedrockRouter` for async/await usage in FastAPI, aiohttp, etc.

**Strands Agents SDK Integration**
- `SmartRouterModel` — drop-in Strands `Model` provider backed by the smart router
- Every Strands agent call is automatically routed across Bedrock models by cost, latency, quality, and complexity
- Tool use, multi-turn conversations, streaming, and structured output all work transparently
- Per-request routing control via presets, strategies, cost limits, and metadata
- Routing decisions accessible after every call for observability

## Quick Start

### Installation

```bash
# Core SDK — only requires boto3, works in Lambda out of the box
pip install bedrock-smart-router

# With Strands Agents SDK integration
pip install bedrock-smart-router[strands]

# With Redis/Valkey/ElastiCache caching support
pip install bedrock-smart-router[redis]

# With OpenTelemetry tracing and metrics
pip install bedrock-smart-router[otel]

# With everything
pip install bedrock-smart-router[strands,redis,otel]

# For development (includes pytest, moto)
pip install bedrock-smart-router[dev]
```

| Extra | What it adds | When you need it |
|---|---|---|
| *(none)* | Core SDK, boto3 only | Lambda, single-instance, in-memory cache and metrics |
| `[strands]` | `strands-agents` package | Using the router as a Strands Agents model provider |
| `[redis]` | `redis` package | Shared cache + vector store via Redis 7+ (RediSearch), Valkey 8.2+, or ElastiCache |
| `[faiss]` | `faiss-cpu` package | Fast in-process vector search for semantic cache (~100K entries) |
| `[opensearch]` | `opensearch-py`, `requests-aws4auth` | OpenSearch Serverless vector store for semantic cache |
| `[otel]` | `opentelemetry-api`, `opentelemetry-sdk` | Distributed tracing and OTEL metrics export |
| `[dev]` | `pytest`, `pytest-cov`, `moto` | Running the test suite |

### Basic Usage

```python
from bedrock_smart_router import BedrockRouter

# All defaults — balanced strategy, in-memory metrics
router = BedrockRouter.create()

# Converse API (boto3 drop-in)
response = router.converse(
    messages=[{"role": "user", "content": [{"text": "Explain VPCs in AWS"}]}],
)

print(response["routing_decision"].selected_model)
# e.g. "amazon.nova-lite-v1:0" for a simple question

print(response["routing_decision"].actual_cost)
# e.g. 0.000012

# Chat Completions API (OpenAI SDK drop-in)
response = router.chat.completions.create(
    messages=[{"role": "user", "content": "Explain VPCs in AWS"}],
    max_tokens=500,
)

print(response["choices"][0]["message"]["content"])
print(response["model"])  # Selected model
```

### Configuration via Dict / YAML

```python
router = BedrockRouter.create({
    "region": "us-west-2",
    "strategy": "balanced",
    "weights": {"cost": 0.5, "latency": 0.2, "quality": 0.3},
    "cache": {"ttl_seconds": 1800, "max_entries": 5000},
    "metrics": {
        "backend": "dynamodb",
        "table_name": "MyRouterMetrics",
        "ttl_hours": 168,
    },
    "cris": {"preferred_geography": "us"},
    "inference_tier": {"allow_optimized": True, "optimized_for_complex": True},
    "guardrails": {
        "pre_route": {"guardrail_id": "gr-abc123", "action_on_block": "reject"},
    },
    "fallback": {"max_depth": 5},
    "circuit_breaker": {"failure_threshold": 10},
    "excluded_models": ["meta.*"],
})
```

### Per-Request Overrides

```python
from bedrock_smart_router import RoutingConfig

response = router.converse(
    messages=[{"role": "user", "content": [{"text": "Complex analysis..."}]}],
    routing=RoutingConfig(
        strategy="quality-optimized",
        preferred_family="anthropic",
        max_cost_per_request=0.05,
        tags=["paid-tier"],
        metadata={"user_id": "u123", "team": "engineering"},
    ),
)
```

### Named Presets

Presets are one-word shortcuts for common routing profiles. They set the strategy, weights, and constraints in a single parameter:

```python
from bedrock_smart_router import RoutingConfig

# Economy — cheapest model, max $0.002/request
response = router.converse(
    messages=[{"role": "user", "content": [{"text": "Classify this text"}]}],
    routing=RoutingConfig(preset="economy"),
)

# Speed — lowest latency model
response = router.converse(
    messages=[{"role": "user", "content": [{"text": "Translate: Hello"}]}],
    routing=RoutingConfig(preset="speed"),
)

# Quality — best model regardless of cost
response = router.converse(
    messages=[{"role": "user", "content": [{"text": "Analyze this contract"}]}],
    routing=RoutingConfig(preset="quality"),
)

# Presets can be overridden — use economy but allow Anthropic only
response = router.converse(
    messages=[{"role": "user", "content": [{"text": "Summarize"}]}],
    routing=RoutingConfig(preset="economy", preferred_family="anthropic"),
)
```

| Preset | Strategy | Cost Limit | Use Case |
|---|---|---|---|
| `economy` | cost-optimized | $0.002/req | Batch processing, classification, simple Q&A |
| `speed` | latency-optimized | — | Real-time chat, interactive UX |
| `balanced` | balanced (0.4/0.3/0.3) | — | General purpose (default) |
| `quality` | quality-optimized | — | Complex reasoning, analysis, code generation |

### Actionable Error Feedback

When no models satisfy the routing constraints, the router raises a `NoModelsMatchError` with structured, actionable feedback — not a generic error. It tells you exactly which models were checked, why each was excluded, and what to change:

```python
from bedrock_smart_router import RoutingConfig, NoModelsMatchError

try:
    response = router.converse(
        messages=[{"role": "user", "content": [{"text": "Hello"}]}],
        routing=RoutingConfig(
            preset="economy",
            preferred_family="nonexistent",
        ),
    )
except NoModelsMatchError as e:
    print(e)
    # No eligible models found for this request.
    #   Constraints: {complexity: simple, preferred_family: nonexistent, ...}
    #   Models checked:
    #     - Nova Micro (amazon.nova-micro-v1:0): family amazon != nonexistent
    #     - Claude Haiku 4.5 (...): family anthropic != nonexistent
    #   Suggestions:
    #     - Remove preferred_family='nonexistent' to consider all families

    # Structured access for API responses:
    print(e.constraints)      # dict of applied constraints
    print(e.rejections)       # list of ModelRejection(model_id, reasons)
    print(e.suggestions)      # list of actionable suggestions
    print(e.to_dict())        # full structured dict for JSON APIs
```

### Async Usage

```python
from bedrock_smart_router.async_router import AsyncBedrockRouter

router = AsyncBedrockRouter.create({"region": "us-west-2"})
response = await router.converse(
    messages=[{"role": "user", "content": [{"text": "Hello"}]}],
)
```

### Custom Strategy Plugin

Implement your own routing logic by subclassing `RoutingStrategy`. You only need to define two things: your scoring **weights** and a **score_model()** method for your custom dimensions. The base class handles cost/quality/latency scoring, ranking, fallback chains, and explanation assembly.

```python
from bedrock_smart_router import BedrockRouter
from bedrock_smart_router.custom_strategy import register_strategy
from bedrock_smart_router.strategy_engine import RoutingStrategy, StrategyContext
from bedrock_smart_router.models import BedrockModel, RequestAnalysis

# Compliance scores per model family (from your GRC system)
COMPLIANCE_SCORES = {
    "hipaa": {"anthropic": 0.95, "amazon": 0.98, "meta": 0.70, "mistral": 0.75},
    "pci":   {"anthropic": 0.90, "amazon": 0.95, "meta": 0.60, "mistral": 0.65},
    "general": {"anthropic": 1.0, "amazon": 1.0, "meta": 1.0, "mistral": 1.0},
}

class ComplianceStrategy(RoutingStrategy):
    name = "compliance"

    @property
    def weights(self) -> dict[str, float]:
        # Mix your custom dimension with built-in ones
        return {"compliance": 0.50, "quality": 0.30, "cost": 0.20}

    def score_model(self, model: BedrockModel, analysis: RequestAnalysis,
                    context: StrategyContext) -> dict[str, float]:
        # Score ONLY your custom dimensions (0.0 to 1.0)
        # Built-in "quality", "cost", "latency" are computed automatically
        tier = context.metadata.get("compliance_tier", "general")
        score = COMPLIANCE_SCORES.get(tier, {}).get(model.family, 0.5)
        return {"compliance": score}

    def filter_candidates(self, candidates, analysis, context: StrategyContext):
        # Optional: hard-gate filtering (approved models only)
        approved = set(context.metadata.get("approved_models", []))
        if not approved:
            return candidates, {}
        filtered = [m for m in candidates
                    if m.model_id in approved or m.base_model_id in approved]
        return filtered, {"rejected": len(candidates) - len(filtered),
                          "reason": "not in approved list"}

register_strategy("compliance", ComplianceStrategy)

router = BedrockRouter.create({
    "strategy": "compliance",
    "metadata": {
        "approved_models": [
            "anthropic.claude-sonnet-4-20250514",
            "anthropic.claude-haiku-4-20250514",
            "amazon.nova-pro-v1:0",
        ],
        "compliance_tier": "hipaa",
    },
})
```

**Interface contract:**

| Method | Required | What it does |
|---|---|---|
| `weights` (property) | **Yes** | Declares scoring dimensions + their weights |
| `score_model()` | **Yes** | Scores your custom dimensions per model (0–1) |
| `filter_candidates()` | No | Hard-gate filtering before scoring |
| `explain_metadata()` | No | Extra context for the decision JSON |
| `select()` | No | Full pipeline override (when scoring per-model isn't enough) |

The base class gives you for free: built-in `quality`, `cost`, `latency` scoring, composite calculation, ranking, fallback chain building, and explanation JSON assembly. If you subclass `RoutingStrategy` without implementing `weights` and `score_model()`, Python raises `TypeError` at instantiation — immediate, clear feedback.

### Inspecting Runtime State

```python
# Cache stats
print(router.cache.stats)
# {"hits": 42, "misses": 158, "hit_rate": 0.21, "size": 158}

# Cost tracking
print(router.observability.cost_tracker.stats)
# {"total_cost": 0.23, "cost_saved_by_routing": 0.87, ...}

# Historical metrics for a model
m = router.metrics.get_metrics("anthropic.claude-sonnet-4-6")
print(f"P95 latency: {m.p95_latency_ms}ms, error rate: {m.error_rate}")

# Last routing decision
d = router.last_routing_decision()
print(f"Model: {d.selected_model}, Tier: {d.inference_tier}, CRIS: {d.cris_profile}")
print(f"Stop: {d.stop_reason}, Bedrock latency: {d.bedrock_latency_ms}ms")
print(f"Prompt cache: {d.prompt_cache_read_tokens} read / {d.prompt_cache_write_tokens} write tokens")
print(f"Prompt cache hit rate: {d.prompt_cache_hit_rate:.0%}, Network overhead: {d.network_overhead_ms}ms")
```

## Examples

The [`examples/`](examples/) folder contains runnable code for every feature, with 2-3 examples each:

| Example | Feature |
|---|---|
| [`01_basic_routing.py`](examples/01_basic_routing.py) | Zero-config, YAML config, dict config |
| [`02_presets.py`](examples/02_presets.py) | Economy, speed, balanced, quality presets |
| [`03_strategies.py`](examples/03_strategies.py) | Cost, latency, quality, balanced, budget strategies |
| [`04_fallbacks_and_reliability.py`](examples/04_fallbacks_and_reliability.py) | Fallback chains, circuit breakers, retries |
| [`05_caching.py`](examples/05_caching.py) | In-memory cache, TTL, invalidation |
| [`06_observability.py`](examples/06_observability.py) | Callbacks, cost tracking, CloudWatch metrics |
| [`07_bedrock_native.py`](examples/07_bedrock_native.py) | CRIS profiles, latency optimization, guardrails |
| [`08_multi_tenant.py`](examples/08_multi_tenant.py) | Application Inference Profiles, per-tenant cost tracking |
| [`09_ab_testing_canary_shadow.py`](examples/09_ab_testing_canary_shadow.py) | A/B testing, canary rollouts, shadow mode |
| [`10_custom_strategy.py`](examples/10_custom_strategy.py) | Custom strategy plugins (code-aware, EU-only, time-of-day) |
| [`11_error_handling.py`](examples/11_error_handling.py) | NoModelsMatchError, rejection reasons, suggestions |
| [`12_metrics_and_dynamodb.py`](examples/12_metrics_and_dynamodb.py) | In-memory and DynamoDB metrics backends |
| [`13_async_usage.py`](examples/13_async_usage.py) | AsyncBedrockRouter, parallel requests |
| [`14_model_catalog.py`](examples/14_model_catalog.py) | List models, overlays, distilled models |
| [`15_redis_valkey_caching.py`](examples/15_redis_valkey_caching.py) | Redis, Valkey, ElastiCache shared caching |
| [`16_streaming.py`](examples/16_streaming.py) | Token-by-token streaming with TTFT tracking |
| [`17_advanced_bedrock_params.py`](examples/17_advanced_bedrock_params.py) | All Bedrock passthrough params (top_k, guardrails, structured output, etc.) |
| [`18_cross_region_data_residency.py`](examples/18_cross_region_data_residency.py) | CRIS profiles: US-only, EU-only (GDPR), global routing |
| [`19_opentelemetry.py`](examples/19_opentelemetry.py) | OTEL tracing and metrics (X-Ray, Jaeger, Datadog, etc.) |
| [`20_semantic_cache.py`](examples/20_semantic_cache.py) | Semantic cache: match by meaning, variable-aware caching |
| [`21_semantic_cache_deep_dive.py`](examples/21_semantic_cache_deep_dive.py) | Vector stores, response store backends, cache filter, threshold tuning |
| [`22_semantic_router.py`](examples/22_semantic_router.py) | Intent routing: route queries to specialized models by meaning |
| [`23_tag_and_conditional_routing.py`](examples/23_tag_and_conditional_routing.py) | Tag-based routing (free/paid tiers, teams) and metadata-driven conditions |
| [`24_budget_and_tier_pricing.py`](examples/24_budget_and_tier_pricing.py) | Per-request cost ceilings, rolling budgets, latency mode pricing (Standard/Optimized) |
| [`25_strands_integration.py`](examples/25_strands_integration.py) | Strands Agents SDK integration — use the smart router as a Strands Model provider |
| [`26_strands_first_agent.py`](examples/26_strands_first_agent.py) | Official Strands Agents SDK "First Agent" sample adapted to use smart routing |
| [`27_auto_semantic_cache.py`](examples/27_auto_semantic_cache.py) | Auto-extracting semantic cache: automatic intent + variable extraction, multi-turn resolution |
| [`28_strands_custom_tools_cached.py`](examples/28_strands_custom_tools_cached.py) | Official Strands "Custom Tools" sample adapted with smart routing + semantic cache |
| [`29_strands_streaming_multi_tenant.py`](examples/29_strands_streaming_multi_tenant.py) | Official Strands streaming sample adapted with multi-tenant routing (premium vs freemium) |
| [`30_strands_guardrails.py`](examples/30_strands_guardrails.py) | Official Strands guardrails sample adapted with router-level pre-route guardrails |

See [`examples/GUIDE.md`](examples/GUIDE.md) for a comprehensive walkthrough of every feature with explanations.

## Reliability: Circuit Breakers, Fallbacks, and Retries

The router is designed so that it never becomes the reason your application fails. Three mechanisms work together to ensure every request gets a response:

### Circuit Breakers

Each model has its own circuit breaker that prevents wasting time on a model that's failing. When too many failures happen, the breaker "trips" and the router skips that model entirely.

```
CLOSED (normal) ──5 failures in 60s──▶ OPEN (blocked)
                                          │
                                     wait 30s cooldown
                                          │
                                          ▼
                                     HALF-OPEN (testing)
                                          │
                              ┌───────────┴───────────┐
                         probe succeeds          probe fails
                              │                       │
                              ▼                       ▼
                         CLOSED                    OPEN
```

Without a circuit breaker, if Sonnet is throttled, every request would try Sonnet first, wait for the timeout, then fall back — wasted latency on every request. With the breaker open, the router skips Sonnet instantly and goes straight to the fallback.

Throttles (429s) get a shorter cooldown (10s) than hard errors (5xx = 30s), because throttles are transient.

```yaml
circuit_breaker:
  failure_threshold: 5           # Failures before tripping
  window_seconds: 60             # Count failures within this window
  cooldown_seconds: 30           # How long OPEN stays blocked
  throttle_cooldown_seconds: 10  # Shorter cooldown for 429s
```

### Fallback Chain

When the primary model fails (or its circuit breaker is open), the router walks an ordered fallback chain:

```
Primary: Claude Sonnet 4.6 (selected by strategy)
  │ fails or circuit breaker open
  ▼
Level 1: Claude Haiku 4.5 (same family, cheaper)
  │ fails
  ▼
Level 2: Nova Pro (different family, same capability tier)
  │ fails
  ▼
Level 3: Nova Lite (default safe model — cheap, fast, always available)
  │ fails
  ▼
Level 4: Error returned to caller with full trace
```

The chain is built automatically: same-family downgrade first (Sonnet → Haiku), then cross-family equivalent (Sonnet → Nova Pro), then the default safe model. The user never sees the failure — they get a response from the best available model.

```yaml
fallback:
  enabled: true
  max_depth: 5
  default_safe_model: "amazon.nova-micro-v1:0"
```

### Retries

Before falling back to a different model, the router retries the same model for transient errors (throttles, 503s) with exponential backoff:

```
Request → Sonnet (throttled, 429)
  → wait 0.5s → retry Sonnet (throttled again)
  → wait 1.0s → retry Sonnet (throttled again)
  → wait 2.0s → retry Sonnet (still failing)
  → max retries exhausted → fall back to Haiku
```

Non-retryable errors (ValidationException, AccessDeniedException) skip retries and go straight to fallback.

```yaml
retry:
  max_retries: 3
  backoff_base_seconds: 0.5
  backoff_max_seconds: 8.0
  backoff_multiplier: 2.0
```

### How They Work Together

```
Request arrives
  │
  ├─ Strategy picks: Sonnet 4.6
  ├─ Circuit breaker: is Sonnet OPEN?
  │   ├─ No → try Sonnet (with retries on transient errors)
  │   └─ Yes → skip Sonnet, go to fallback[0]
  │
  ├─ Sonnet call fails after retries
  │   ├─ Record failure (may trip circuit breaker)
  │   └─ Try fallback[0]: Haiku (with retries)
  │       ├─ Success → return response
  │       └─ Fail → try fallback[1]: Nova Pro
  │           └─ ... and so on
  │
  └─ Routing decision shows what happened:
       d.fallback_used = True
       d.fallback_model = "anthropic.claude-haiku-4-5-..."
       d.circuit_breaker_skipped = ["anthropic.claude-sonnet-4-6"]
```

The caller gets a response from the best available model, with full transparency about what happened behind the scenes.

## Safe Model Rollouts: A/B Testing, Canary, and Shadow Mode

Three mechanisms for safely introducing new models into production without risking user experience.

### A/B Testing

Split traffic between two (or more) models to compare quality, cost, and latency in production. Sticky mode hashes the user ID so the same user always sees the same variant — important for consistent UX and valid comparison.

```
                    ┌─── 70% ──→ Variant A: Claude Sonnet 4.6
Request arrives ────┤
                    └─── 30% ──→ Variant B: Claude Sonnet 4.5
```

```yaml
ab_test:
  enabled: true
  name: "sonnet-comparison"
  sticky: true                    # Same user_id → same variant
  variants:
    control:
      model: "anthropic.claude-sonnet-4-6"
      weight: 0.7                 # 70% of traffic
    challenger:
      model: "anthropic.claude-sonnet-4-5-20250929-v1:0"
      weight: 0.3                 # 30% of traffic
```

```python
# User ID drives sticky assignment
response = router.converse(
    messages=[...],
    routing=RoutingConfig(metadata={"user_id": "u-12345"}),
)
d = response["routing_decision"]
print(d.metadata)  # {"ab_variant": "control"} or {"ab_variant": "challenger"}

# Check split stats
print(router.ab_test.stats)
# {"test_name": "sonnet-comparison", "variant_counts": {"control": 700, "challenger": 300}}
```

The A/B test overrides the strategy engine — the variant's model is used directly, but fallbacks, circuit breakers, and metrics still apply.

### Canary Deployments

Gradually roll out a new model by sending a small percentage of traffic to it while monitoring error rate and latency. If the canary exceeds thresholds, it's automatically rolled back. If it performs well after enough requests, it's auto-promoted.

```
                    ┌─── 95% ──→ Baseline: Nova Pro (proven)
Request arrives ────┤
                    └───  5% ──→ Canary: Nova 2 Lite (new)
                                    │
                         ┌──────────┴──────────┐
                    error rate > 10%      100 requests, < 2% errors
                         │                     │
                    AUTO-ROLLBACK          AUTO-PROMOTE
                    (canary disabled)      (canary becomes baseline)
```

```yaml
canary:
  enabled: true
  baseline: "amazon.nova-pro-v1:0"
  canary_model: "amazon.nova-2-lite-v1:0"
  canary_percentage: 5              # Start with 5% of traffic
  auto_rollback:
    max_error_rate: 0.10            # Roll back if > 10% errors
    max_latency_p95_ms: 5000        # Roll back if P95 > 5 seconds
  auto_promote:
    min_requests: 100               # Need 100+ canary requests
    max_error_rate: 0.02            # Promote if < 2% errors
```

```python
# Check canary health
print(router.canary.stats)
# {"canary_requests": 47, "canary_error_rate": 0.02, "rolled_back": false, "promoted": false}

# After enough good requests:
# {"canary_requests": 150, "promoted": true}
```

### Shadow Mode

Mirror a sample of production traffic to a secondary model in background threads. The shadow response is logged but never returned to the caller — zero impact on latency or user experience. Use it to evaluate a new model's quality before any traffic shift.

```
Request arrives ──→ Primary: Claude Sonnet 4.6 ──→ Response to caller
                        │
                        └──→ Shadow (background thread): Nova Pro
                             │
                             └──→ Logged for offline comparison
                                  (never affects the response)
```

```yaml
shadow:
  enabled: true
  shadow_model: "amazon.nova-pro-v1:0"
  sample_rate: 0.1                  # Mirror 10% of traffic
```

```python
# Check shadow results
print(router.shadow.stats)
# {"shadow_model": "amazon.nova-pro-v1:0", "total": 42, "success_rate": 0.98, "avg_latency_ms": 320}

# Access individual results for quality comparison
for result in router.shadow.results[-5:]:
    print(f"  primary={result.primary_model} shadow={result.shadow_model} "
          f"latency={result.latency_ms:.0f}ms success={result.success}")
```

### Typical Rollout Workflow

```
Week 1: Shadow mode (10% sample) → compare quality offline
Week 2: Canary (5% traffic) → monitor error rate and latency
Week 3: Canary (25% traffic) → increase if metrics are good
Week 4: A/B test (50/50) → statistical comparison
Week 5: Full rollout → make the new model the default
```

## Budget Enforcement

The SDK provides three levels of cost control, from simple per-request caps to rolling budget windows with persistent tracking and automatic downgrade.

### Level 1: Per-Request Cost Ceiling

The simplest control. Set `max_cost_per_request` and the router excludes any model whose estimated cost exceeds it:

```python
response = router.converse(
    messages=[...],
    routing=RoutingConfig(max_cost_per_request=0.001),  # Max $0.001
)
```

The `economy` preset does this automatically with a $0.002 ceiling.

### Level 2: BudgetRule — Declarative Limits

Define per-request, hourly, and daily spend limits in a single rule. Choose what happens when the budget is exceeded — downgrade to a cheaper model or reject the request:

```python
from bedrock_smart_router.budget_strategy import BudgetRule

# Enterprise: generous limits, downgrade on exceed
enterprise = BudgetRule(
    max_cost_per_request=0.05,     # Max $0.05 per request
    max_hourly_spend=1.00,         # Max $1.00/hour (rolling window)
    max_daily_spend=10.00,         # Max $10.00/day (rolling 24h window)
    on_exceeded="downgrade",       # Switch to cheaper model (vs "reject")
    downgrade_to_tier="lite",      # Downgrade target tier
)

# Free tier: tight limits, hard reject
free = BudgetRule(
    max_cost_per_request=0.001,
    max_hourly_spend=0.10,
    max_daily_spend=0.50,
    on_exceeded="reject",          # Raise BudgetExceededError
)
```

| Field | What it does |
|---|---|
| `max_cost_per_request` | Excludes models whose estimated cost exceeds this |
| `max_hourly_spend` | Rolling 1-hour spend window per scope (resets naturally as records age out) |
| `max_daily_spend` | Rolling 24-hour spend window per scope (resets naturally as records age out) |
| `on_exceeded` | `"downgrade"` picks the cheapest model; `"reject"` raises `BudgetExceededError` |
| `downgrade_to_tier` | When downgrading, restrict to this tier or below |

### Level 3: BudgetTracker — Rolling Spend Tracking

Track actual spend per user, team, or tenant over sliding time windows. Pair it with `BudgetRule` to enforce rolling limits:

```python
from bedrock_smart_router.budget_strategy import BudgetTracker, BudgetRule

tracker = BudgetTracker()

# After each request, record the actual cost
tracker.record_spend("user-u123", decision.actual_cost)
tracker.record_spend("team-engineering", decision.actual_cost)

# Before the next request, check if the budget is exceeded
rule = BudgetRule(max_hourly_spend=1.00, max_daily_spend=10.00)
exceeded = tracker.check_budget("user-u123", rule)
if exceeded:
    print(f"Budget exceeded: {exceeded}")
    # "hourly spend $1.0234 >= $1.0000"

# Query spend for any time window
hourly = tracker.get_spend("user-u123", 3600)     # Last hour
daily  = tracker.get_spend("user-u123", 86400)    # Last 24 hours
weekly = tracker.get_spend("team-engineering", 604800)  # Last 7 days
```

### Level 4: Persistent Budget Tracking

By default, the `BudgetTracker` is in-memory — fast but resets on process restart. For production deployments that need spend tracking to survive restarts and work across multiple instances, add a persistent backend:

```python
from bedrock_smart_router.budget_strategy import BudgetTracker
from bedrock_smart_router.budget_store import SQLiteBudgetStore, DynamoDBBudgetStore

# SQLite — single instance, zero config, auto-creates table + indexes
tracker = BudgetTracker(
    store=SQLiteBudgetStore(path="/tmp/bsr_budget.db"),
    sync_interval=5.0,  # Flush to disk every 5 seconds
)

# DynamoDB — multi-instance, serverless, auto-TTL cleanup
tracker = BudgetTracker(
    store=DynamoDBBudgetStore(
        table_name="bsr-budget-tracking",
        region="us-west-2",
        ttl_seconds=86400,           # Records auto-expire after 24h
        auto_create_table=True,      # Create table if missing (dev/testing)
    ),
    sync_interval=5.0,
)
```

**DynamoDB table schema (for manual creation):**

If you set `auto_create_table=False` (the default for production), create the table with:

```bash
aws dynamodb create-table \
  --table-name bsr-budget-tracking \
  --attribute-definitions \
    AttributeName=scope,AttributeType=S \
    AttributeName=timestamp,AttributeType=N \
  --key-schema \
    AttributeName=scope,KeyType=HASH \
    AttributeName=timestamp,KeyType=RANGE \
  --billing-mode PAY_PER_REQUEST

# Enable TTL for automatic cleanup of old records
aws dynamodb update-time-to-live \
  --table-name bsr-budget-tracking \
  --time-to-live-specification Enabled=true,AttributeName=ttl
```

No GSI required — the partition key (`scope`) + sort key (`timestamp`) covers the primary query: "total spend for user X in the last N seconds."

**IAM permissions required:**

```json
{
  "Effect": "Allow",
  "Action": [
    "dynamodb:BatchWriteItem",
    "dynamodb:Query",
    "dynamodb:Scan"
  ],
  "Resource": "arn:aws:dynamodb:*:*:table/bsr-budget-tracking"
}
```

Add `dynamodb:CreateTable` and `dynamodb:UpdateTimeToLive` if using `auto_create_table=True`.

**How it works:**

1. **Hot path (every request):** `record_spend()` and `check_budget()` use in-memory data — no I/O, sub-microsecond
2. **Background sync:** Every `sync_interval` seconds, pending records are flushed to the persistent store
3. **On startup:** Recent spend is loaded from the store into memory (recovery after restart)
4. **Rolling windows:** `max_hourly_spend` and `max_daily_spend` use rolling time windows — spend naturally "resets" as old records age past the window boundary. No cron jobs or calendar resets needed

**Sample config (via router config dict):**

```python
router = BedrockRouter.create({
    "budget": {
        "tracker_backend": "sqlite",              # "memory" | "sqlite" | "dynamodb"
        "sqlite_path": "/tmp/bsr_budget.db",
        "scope_key": "user_id",                   # Track spend per user (from routing metadata)
        "rule_key": "tier",                       # Match rules by tier (from routing metadata)
        "sync_interval_seconds": 5,
        "rules": {
            "default": {"max_hourly_spend": 1.0, "max_daily_spend": 10.0, "on_exceeded": "downgrade"},
            "free": {"max_hourly_spend": 0.10, "max_daily_spend": 0.50, "on_exceeded": "reject"},
            "pro": {"max_hourly_spend": 2.0, "max_daily_spend": 20.0, "on_exceeded": "downgrade"},
            "enterprise": {"max_hourly_spend": 10.0, "max_daily_spend": 100.0, "on_exceeded": "downgrade"},
        },
    },
})

# The router automatically tracks spend and enforces budgets
# based on the user_id and tier from routing metadata:
response = router.converse(
    messages=[...],
    routing=RoutingConfig(metadata={"user_id": "alice", "tier": "free"}),
)
```

**Custom backends:** Subclass `BudgetStore` to implement any persistence layer (Postgres, Redis, etc.):

```python
from bedrock_smart_router.budget_store import BudgetStore, SpendRecord

class PostgresBudgetStore(BudgetStore):
    def write_batch(self, records: list[SpendRecord]) -> None: ...
    def get_spend(self, scope: str, window_seconds: float) -> float: ...
    def get_all_spend(self, window_seconds: float) -> dict[str, float]: ...
    def cleanup(self, older_than_seconds: float) -> int: ...
```

**Backend comparison:**

| Backend | Persistence | Multi-instance | Auto-cleanup | Dependencies | Best for |
|---|---|---|---|---|---|
| `memory` (default) | No | No | N/A | None | Dev, Lambda, single-instance |
| `sqlite` | Yes | No | Manual (via `cleanup()`) | None (stdlib) | Single-instance production |
| `dynamodb` | Yes | Yes | Auto (TTL) | None (`boto3` already required) | Multi-instance production |
| Custom (`BudgetStore`) | Yes | Depends | Depends | Your choice | Postgres, Redis, etc. |

**When to use each level:**

| Scenario | Use |
|---|---|
| Simple cost cap, no tracking needed | `RoutingConfig(max_cost_per_request=...)` |
| Different limits for free/paid tiers | `BudgetRule` per tier |
| Per-user or per-team rolling budgets | `BudgetTracker` + `BudgetRule` |
| Survive restarts (single instance) | `BudgetTracker` + `SQLiteBudgetStore` |
| Multi-instance with shared budgets | `BudgetTracker` + `DynamoDBBudgetStore` |
| SaaS with tenant cost isolation | `BudgetTracker` keyed by tenant ID + `BudgetRule` per plan |

## Multimodal Routing: Images and Documents

The router automatically detects image and document content blocks in your messages and makes intelligent routing decisions based on them.

### Capability Filtering

When your request contains multimodal content, the router automatically excludes models that can't handle it:

- **Image content** (`{"image": {...}}`) → only routes to models with `vision: true`
- **Document content** (`{"document": {...}}`) → only routes to models with `document_support: true`

This happens transparently — you don't need to specify which models support what. Just send your content and the router picks a capable model:

```python
# PDF extraction — automatically routes to Claude/Nova Pro (not Llama/DeepSeek)
response = router.converse(
    messages=[{
        "role": "user",
        "content": [
            {"document": {"format": "pdf", "name": "report", "source": {"bytes": pdf_bytes}}},
            {"text": "Extract all tables from this document"},
        ],
    }],
)
```

### Payload Size → Complexity Boost

The router uses the byte size of multimodal payloads to influence model selection. Larger payloads indicate more complex tasks that benefit from more capable (higher-tier) models:

| Payload Size | Complexity Boost | Typical Routing |
|---|---|---|
| < 100KB | +0.05 | Stays simple → lite tier (Haiku, Nova Lite) |
| 100KB – 1MB | +0.10 | Nudges to moderate → mid tier |
| 1MB – 5MB | +0.20 | Moderate → mid tier (Sonnet, Nova Pro) |
| > 5MB | +0.30 | Complex → heavy tier (Opus) |

This means:
- A small screenshot + "what is this?" → routes to a cheap, fast model
- A 50-page PDF + "summarize" → routes to a capable model with strong document understanding

### Token Estimation for Multimodal Content

The router estimates token consumption for images and documents so that context window pre-validation works correctly:

- **Documents:** ~1,500 tokens per estimated page (~3KB per page)
- **Images:** ~750 tokens per image

This prevents the router from picking a model whose context window is too small for the payload.

## Boto Client Configuration

You can configure the underlying Bedrock client's timeouts, retries, and connection settings. This is important for large document/image payloads where the default 60-second read timeout may not be enough.

### Option A: Pass a Config object

```python
from botocore.config import Config
from bedrock_smart_router import BedrockRouter

router = BedrockRouter.create(
    {"region": "us-west-2"},
    boto_config=Config(
        read_timeout=300,
        connect_timeout=10,
        retries={"max_attempts": 3, "mode": "adaptive"},
    ),
)
```

### Option B: Configure via dict/YAML

```python
router = BedrockRouter.create({
    "region": "us-west-2",
    "boto_config": {
        "read_timeout": 300,
        "connect_timeout": 10,
        "retries": {"max_attempts": 3, "mode": "adaptive"},
    },
})
```

Option A takes precedence if both are provided.

### Retry Conflict Prevention

The router has its own retry mechanism (exponential backoff on throttles and transient errors, then fallback to the next model). If you configure retries in `boto_config`, the router **automatically disables its native retry handler** to prevent multiplicative retry storms.

Without this protection, a single throttled request could be retried up to `boto_retries × router_retries` times. With it:

| Configuration | Who retries | Fallback |
|---|---|---|
| No `boto_config` retries | Router's RetryHandler (3 retries + backoff) | Then falls back to next model |
| `boto_config` with retries | boto3/botocore (SDK-level) | Router skips its retries, falls back to next model on failure |

If you want full control over retry behavior, set `boto_config` retries and the router will defer to boto3. The fallback chain (trying the next model) still works regardless.

## Strands Agents SDK Integration

The Smart Router integrates with the [Strands Agents SDK](https://strandsagents.com/) as a custom model provider. Instead of Strands calling Bedrock directly with a fixed model, every agent call flows through the router — getting automatic model selection, fallbacks, circuit breakers, cost tracking, and all other routing features.

### How It Works

```
Strands Agent
  │ agent("Explain quantum computing")
  ▼
SmartRouterModel (implements strands.models.Model)
  │ Converts Strands types → Bedrock Converse format
  │ Builds RoutingConfig from model config (preset, strategy, cost limits)
  ▼
BedrockRouter
  │ Analyzes complexity (15-dimension classifier)
  │ Selects optimal model via strategy engine
  │ Applies CRIS profile, latency mode, guardrails
  │ Invokes Bedrock converse_stream with fallback chain
  ▼
Bedrock converse_stream
  │ Returns stream events (messageStart, contentBlockDelta, ...)
  ▼
SmartRouterModel
  │ Passes events through (Bedrock events ARE Strands StreamEvents)
  │ Captures routing_decision for observability
  ▼
Strands Agent
  │ Processes events normally (text, tool calls, reasoning)
  │ Executes tools, feeds results back → next loop iteration
  ▼
Response returned to caller
```

The key insight: Bedrock's `converse_stream` event format is identical to Strands' `StreamEvent` format — the SDK was designed around Bedrock's Converse API. This means stream events pass through untouched with zero translation overhead.

### Installation

```bash
pip install bedrock-smart-router[strands]
```

### Basic Usage

```python
from strands import Agent
from bedrock_smart_router.strands_model import SmartRouterModel

# Create a Strands model backed by the smart router
model = SmartRouterModel(router_config={"region": "us-west-2"})
agent = Agent(model=model)

# Use it like any Strands agent — routing is automatic
response = agent("Explain quantum computing")

# Inspect the routing decision
d = model.last_routing_decision
print(f"Model: {d.selected_model}")      # e.g. "amazon.nova-lite-v1:0"
print(f"Strategy: {d.strategy_used}")     # e.g. "balanced"
print(f"Complexity: {d.complexity_detected}")  # e.g. "moderate"
print(f"Cost: ${d.actual_cost:.6f}")
```

### Routing Presets

Control the cost/quality/speed trade-off with a single parameter:

```python
# Economy — cheapest model for simple tasks
model = SmartRouterModel(
    router_config={"region": "us-west-2"},
    routing_preset="economy",
)

# Quality — best model for complex reasoning
model = SmartRouterModel(
    router_config={"region": "us-west-2"},
    routing_preset="quality",
)
```

### Tool Use

Strands handles the agent loop (call model → execute tools → feed results back). The router picks the best model that supports `tool_use`:

```python
from strands import Agent, tool

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"22°C and sunny in {city}"

model = SmartRouterModel(router_config={"region": "us-west-2"})
agent = Agent(model=model, tools=[get_weather])
response = agent("What's the weather in Seattle?")
```

### Runtime Config Changes

Switch routing behaviour mid-conversation:

```python
model.update_config(routing_preset="economy")
response = agent("Simple question")  # Routes to cheapest model

model.update_config(routing_preset="quality")
response = agent("Complex analysis")  # Routes to best model
```

### Bring Your Own Router

Pass a pre-configured `BedrockRouter` instance:

```python
from bedrock_smart_router import BedrockRouter

router = BedrockRouter.create({
    "region": "us-west-2",
    "strategy": "cost-optimized",
    "cache": {"enabled": True, "ttl": 300},
})

model = SmartRouterModel(router=router)
agent = Agent(model=model)
```

See [`examples/25_strands_integration.py`](examples/25_strands_integration.py) for the full set of examples.

## Unified API Surface: Converse + Chat Completions

The Smart Router unifies two Amazon Bedrock platforms — **bedrock-runtime** (Converse API) and **bedrock-mantle** (Chat Completions API) — into a single routing layer. Users get transparent access to **68 models** across both platforms regardless of which API surface they call from.

### The Problem

Amazon Bedrock exposes models through two separate endpoints:

| Endpoint | API Format | Models | Authentication |
|----------|-----------|--------|----------------|
| `bedrock-runtime` | Converse API (boto3) | Claude, Nova, Meta, Mistral, DeepSeek, etc. | SigV4 (IAM) or API key |
| `bedrock-mantle` | Chat Completions (OpenAI-compatible) | GPT-OSS, DeepSeek, Qwen, Mistral, NVIDIA, MiniMax, etc. | SigV4 or API key |

Some models are on both platforms. Some are exclusively on one. Users must know which endpoint to call for which model, manage two different auth patterns, and handle two response formats.

### How the Smart Router Solves This

The router exposes **both API surfaces** on a single `BedrockRouter` object. Internally, it routes to the correct backend based on the selected model's capabilities:

```
User calls router.converse(...)           User calls router.chat.completions.create(...)
         │                                              │
         ▼                                              ▼
   ┌─────────────────────────────────────────────────────────┐
   │              BedrockRouter (unified)                     │
   │                                                         │
   │  1. Classify complexity                                 │
   │  2. Select optimal model (from ALL 68 models)           │
   │  3. Dispatch to correct backend:                        │
   │     • Model supports Converse? → bedrock-runtime        │
   │     • Model is Mantle-only? → bedrock-mantle            │
   │  4. Translate format if needed (transparent)            │
   └─────────────────────────────────────────────────────────┘
         │                           │
         ▼                           ▼
   bedrock-runtime              bedrock-mantle
   (Converse API)               (Chat Completions)
```

### Usage: Converse API (boto3 drop-in)

```python
from bedrock_smart_router import BedrockRouter

router = BedrockRouter.create({"region": "us-west-2"})

# Same interface as boto3's bedrock-runtime.converse()
response = router.converse(
    messages=[{"role": "user", "content": [{"text": "Explain Kubernetes"}]}],
    inferenceConfig={"maxTokens": 500},
)

# If the router selects a Mantle-only model (e.g., DeepSeek V3.1),
# it transparently translates Converse → Chat Completions under the hood.
# The response is always in Converse format — the user doesn't know or care.
```

### Usage: Chat Completions API (OpenAI SDK drop-in)

```python
from bedrock_smart_router import BedrockRouter

router = BedrockRouter.create({"region": "us-west-2"})

# Same interface as OpenAI's client.chat.completions.create()
response = router.chat.completions.create(
    messages=[{"role": "user", "content": "Explain Kubernetes"}],
    max_tokens=500,
)

print(response["choices"][0]["message"]["content"])
print(response["model"])  # Shows which model was selected

# If the router selects a Converse-only model (e.g., Claude, Nova),
# it transparently translates Chat Completions → Converse under the hood.
# The response is always in Chat Completions format.
```

### Models API

```python
# List all available models (like OpenAI's client.models.list())
models = router.models.list()

# Get details for a specific model
model = router.models.retrieve("openai.gpt-oss-120b")
print(model["api_support"])  # ["converse", "chat_completions", "responses"]
print(model["tier"])         # "mid"
```

### How Backend Dispatch Works

For models available on **both** platforms (26 models), the router prefers `bedrock-runtime` (Converse) because it supports CRIS (Cross-Region Inference) for higher availability and data residency. Mantle has no CRIS equivalent.

For models that are **Mantle-only** (9 models), the router automatically translates the format and calls the Mantle endpoint. This is transparent — the user's code doesn't change.

| Model Category | Count | Backend Used | Notes |
|----------------|-------|-------------|-------|
| Converse-only (Claude, Nova, Meta) | 33 | bedrock-runtime | Full CRIS, guardrails support |
| Both platforms (Mistral, Qwen, NVIDIA, etc.) | 26 | bedrock-runtime (preferred) | CRIS advantage |
| Mantle-only (DeepSeek V3.1, Voxtral, GLM-4.6, etc.) | 9 | bedrock-mantle | Auto-translated |

### Authentication

Both endpoints are supported with a single configuration:

```python
# SigV4 (default) — uses your existing AWS credentials (IAM role, env vars, ~/.aws/config)
router = BedrockRouter.create({"region": "us-west-2"})

# Bedrock API key — works for both bedrock-runtime and bedrock-mantle
router = BedrockRouter.create({
    "region": "us-west-2",
    "api_key": "brk_xxxx...",
})
```

### Format Translation

The router handles bidirectional translation between Converse and Chat Completions formats:

| Feature | Converse → CC | CC → Converse |
|---------|:---:|:---:|
| Text messages | ✅ | ✅ |
| System prompts | ✅ | ✅ |
| Tool use / function calling | ✅ | ✅ |
| Tool results (parallel) | ✅ | ✅ |
| Images (base64) | ✅ | ✅ |
| Streaming | ✅ | ✅ |
| Inference parameters | ✅ | ✅ |

Translation is lossless for the common case. The only features that don't translate are Converse-specific `reasoningContent` blocks (stripped in CC output) and Responses API-specific features like `previous_response_id` (stateful chaining).

### Responses API (Planned)

The OpenAI Responses API — a newer, stateful API surface used by GPT-5.4 and GPT-5.5 — is planned for future support. Currently, these two models are in the catalog but not routable (they require the Responses API which operates differently from the stateless Converse/Chat Completions pattern).

Key Responses API features under evaluation:
- Server-side conversation state (`previous_response_id`)
- Built-in tools (web_search, code_interpreter, file_search)
- Background/async processing
- MCP server connections

For users who need GPT-5.4/5.5 today, the Mantle endpoint can be called directly or through the OpenAI Agents SDK pointed at `https://bedrock-mantle.<region>.api.aws/v1`.

## Caching: Exact-Match, Semantic, and Auto-Extracting

The router provides three layers of caching, each building on the previous. All are optional and can be used independently or together.

### Layer 1: Exact-Match Response Cache

The simplest cache. Stores responses keyed by a hash of the request (messages + system prompt + inference config). Identical requests return instantly at zero Bedrock cost.

```python
router = BedrockRouter.create({
    "cache": {"ttl_seconds": 1800, "max_entries": 5000},
})

r1 = router.converse(messages=msgs)  # Cache miss — calls Bedrock
r2 = router.converse(messages=msgs)  # Cache hit — instant, free
```

**Backends:** In-memory LRU (default), Redis, Valkey, ElastiCache. Use Redis/Valkey for shared cache across Lambda invocations or ECS tasks.

### Layer 2: Semantic Cache (Manual Variables)

Matches queries by meaning using embedding similarity. "How do I reset my password?" and "I forgot my password, help" are different strings but the same intent — the semantic cache catches this.

```python
from bedrock_smart_router.semantic_cache import SemanticCache, SemanticCacheConfig

cache = SemanticCache(config=SemanticCacheConfig(
    threshold=0.85,
    embedding_model="amazon.titan-embed-text-v2:0",
))

cache.put("How do I reset my password?", response)
cache.get("I forgot my password, help")  # HIT — same meaning
```

**Embedding model:** Amazon Titan Embed Text v2 (default, 1024 dimensions). Any Bedrock embedding model can be used.

**Vector store backends:**

| Backend | Install | Scales To | Shared Across Instances |
|---|---|---|---|
| `memory` (default) | *(none)* | ~500 entries | No |
| `faiss` | `pip install bedrock-smart-router[faiss]` | ~100K entries | No |
| `redis` | `pip install bedrock-smart-router[redis]` | Millions | Yes |
| `opensearch` | `pip install bedrock-smart-router[opensearch]` | Millions | Yes (AWS managed) |

**Variable-aware matching:** Queries like "top users for Electronics 2024" and "top users for Clothing 2025" are semantically identical but have different correct answers. Pass `variables` to distinguish them:

```python
cache.put("top users for Electronics 2024", response,
          variables={"category": "Electronics", "year": "2024"})

cache.get("show top users in Electronics 2024",
          variables={"category": "Electronics", "year": "2024"})  # HIT ✅

cache.get("top users for Clothing 2025",
          variables={"category": "Clothing", "year": "2025"})     # MISS ✅
```

The limitation: the caller must manually extract and pass the variables.

### Layer 3: Auto-Extracting Semantic Cache

Solves the manual extraction problem. Uses a cheap Bedrock model (Nova Micro, ~$0.00003/call) to automatically decompose each query into a canonical intent and variables. No manual tagging needed.

```python
cache = SemanticCache(config=SemanticCacheConfig(
    threshold=0.85,
    auto_extract=True,                            # Enable auto-extraction
    extraction_model="amazon.nova-micro-v1:0", # Cheapest model
))

# No variables needed — extracted automatically
cache.put("Count users by geo for 2026 with sales > $200", response)
cache.get("Show user distribution by geography, year 2026, sales over $200")  # HIT ✅
cache.get("Count users by geo for 2025 with sales > $100")                    # MISS ✅
```

The extractor calls Nova Micro with a structured prompt that returns:
- **Intent:** "Count users by geography for a year with sales above a threshold" (parameterised template)
- **Variables:** `{"year": "2026", "sales_threshold": "200"}` (extracted values)

The intent is embedded and stored in the vector store. The variables are hashed and compared exactly. Different wording with the same intent + same variables = HIT. Same intent + different variables = MISS.

### Multi-Turn Resolution

When `multi_turn_resolution=True`, the cache can resolve a multi-turn conversation into a single self-contained query before extraction. This means a cached single-turn response can match a multi-turn conversation with the same intent:

```python
cache = SemanticCache(config=SemanticCacheConfig(
    auto_extract=True,
))

# Store from single-turn
cache.put("Count users by geo for 2026 with sales > $200", response)

# Lookup from multi-turn conversation
cache.get(messages=[
    {"role": "user", "content": [{"text": "show me users by geo"}]},
    {"role": "assistant", "content": [{"text": "Here are users..."}]},
    {"role": "user", "content": [{"text": "now for 2026 with sales > $200"}]},
])
# → HIT! Conversation resolves to same intent + variables
```

### Using with Strands Agents

The semantic cache works alongside `SmartRouterModel`. Check the cache before calling the agent, and store the response after:

```python
from strands import Agent
from bedrock_smart_router.strands_model import SmartRouterModel
from bedrock_smart_router.semantic_cache import SemanticCache, SemanticCacheConfig

cache = SemanticCache(config=SemanticCacheConfig(
    auto_extract=True,
))

model = SmartRouterModel(router_config={"region": "us-west-2"})
agent = Agent(model=model)

def agent_with_cache(query: str) -> str:
    cached = cache.get(query)
    if cached:
        return cached["text"]
    response = agent(query)
    cache.put(query, {"text": str(response)})
    return str(response)

agent_with_cache("What is DynamoDB?")       # MISS → calls Bedrock
agent_with_cache("Tell me about DynamoDB")  # HIT → instant, free
```

### Response Store Backends

By default, responses are stored inline in the vector store payload. For large responses (SQL results, charts, full LLM outputs), configure an external response store to keep the vector store lean:

| Backend | Best For | Max Size | Auto-Expiry |
|---|---|---|---|
| `inline` (default) | Small responses, simple setups | Limited by vector store | Via cache TTL |
| `filesystem` | Dev/testing, Lambda /tmp, EFS | Unlimited | Manual |
| `s3` | Large responses, durability, multi-region | 5TB | S3 lifecycle rules |
| `dynamodb` | Serverless production, low-latency | 400KB | DynamoDB TTL |

```python
from bedrock_smart_router.semantic_response_store import (
    FilesystemResponseStore, S3ResponseStore, DynamoDBResponseStore,
)

# Filesystem (dev/testing, Lambda /tmp, EFS mounts)
cache = SemanticCache(
    config=SemanticCacheConfig(auto_extract=True, vector_store_backend="faiss"),
    response_store=FilesystemResponseStore(path="/tmp/cache_responses"),
)

# S3 (production, large payloads, durability)
cache = SemanticCache(
    config=SemanticCacheConfig(auto_extract=True, vector_store_backend="faiss"),
    response_store=S3ResponseStore(bucket="my-bucket", prefix="cache/"),
)

# DynamoDB (serverless, low-latency, auto-expiry via TTL)
cache = SemanticCache(
    config=SemanticCacheConfig(auto_extract=True, vector_store_backend="faiss"),
    response_store=DynamoDBResponseStore(table_name="cache-responses", ttl_seconds=3600),
)
```

Custom backends: subclass `ResponseStore` and implement `save()`, `load()`, `delete()`.

### Cache Filter (Selective Caching)

Not all responses should be cached. Use `cache_filter` to let the app decide which responses are worth storing:

```python
cache = SemanticCache(
    config=SemanticCacheConfig(auto_extract=True),
    # Only cache responses with actual data (skip errors/empty)
    cache_filter=lambda query, response: (
        response.get("row_count", 0) > 0
        and not response.get("error")
    ),
)

cache.put("top products", {"row_count": 5, "results": [...]})  # ✅ Stored
cache.put("bad query", {"error": "syntax error", "row_count": 0})  # ❌ Filtered
```

The filter is a callable `(query_text, response) -> bool`. Return `True` to cache, `False` to skip. Exceptions are caught and treated as `False` (safe default). Stats include a `filtered` count.

### Cost

| Component | Cost per call | Notes |
|---|---|---|
| Exact-match cache | Free | Hash comparison, no API calls |
| Embedding (semantic cache) | ~$0.00001 | Titan Embed v2 per query |
| Intent extraction (auto-extract) | ~$0.00003 | Nova Micro per query |
| **Total (auto-extract)** | ~$0.00004 | vs $0.001–$0.02 saved per cache hit |

Extraction and embedding results are both cached in-memory, so repeated identical queries incur zero additional API calls. All Bedrock calls (embedding and extraction) have built-in retry with exponential backoff for transient errors.

See [`examples/20_semantic_cache.py`](examples/20_semantic_cache.py), [`examples/21_semantic_cache_deep_dive.py`](examples/21_semantic_cache_deep_dive.py), and [`examples/27_auto_semantic_cache.py`](examples/27_auto_semantic_cache.py) for runnable examples.

## Architecture

```
Request arrives
  |
  +-- Step 1:  Pre-route guardrail check (ApplyGuardrail INPUT)
  +-- Step 2:  Request analysis (15-dimension complexity classifier)
  +-- Step 3:  Filter eligible models (tier, capabilities, family, exclusions)
  +-- Step 4:  Filter by context window
  +-- Step 5:  Filter by circuit breaker state
  +-- Step 6:  Run strategy (cost/latency/quality/balanced)
  |     +-- 6b: Prompt cache boost (swap to cache-capable model if within 10% score)
  |     +-- 6c: Select CRIS profile (us/eu/global geography preference)
  |     +-- 6d: Select latency mode (Standard/Optimized)
  +-- Step 7:  Check response cache (hit -> return immediately)
  +-- Step 8:  Build fallback chain
  +-- Step 9:  Invoke Bedrock (with AIP tenant resolution per model)
  |            +-- On failure: circuit breaker + fallback to next model
  +-- Step 10: Post-route guardrail check (ApplyGuardrail OUTPUT)
  +-- Step 11: Build RoutingDecision
  +-- Step 12: Record metrics to store
  +-- Step 13: Cache the response
  +-- Step 14: Emit observability event
```

## How Routing Strategies Work

Every strategy scores each eligible model on three dimensions — cost, latency, and quality — then picks the model with the highest composite score. The difference between strategies is which dimension drives the composite.

### Scoring Dimensions

**Cost score** (0.10–1.0, higher = cheaper) — ratio-based normalization against the cheapest candidate:

```
cost_score = max(0.10, cheapest_cost / model_cost)
```

The cheapest model in the eligible pool scores 1.0. More expensive models score proportionally lower but never below 0.10 (the floor). This prevents the "multiply by zero" problem where the most expensive model's quality becomes irrelevant in the composite. Global CRIS profiles score higher because they're ~10% cheaper.

**Latency score** (0.10–1.0, higher = faster) — blends real metrics with tier heuristics:

```
Day 1 (no data):   tier heuristic (MICRO=0.90, LITE=0.75, MID=0.50, HEAVY=0.25, REASONING=0.10)
                    + 0.05 bonus for CRIS availability
                    + 0.05 bonus for prompt caching on multi-turn

After 5+ requests: max(0.10, fastest_latency_in_pool / model_latency)
                   Ratio-based: fastest model = 1.0, 5x slower = 0.20
```

When real latency data exists for candidates, ratio normalization gives proper differentiation even among slow models (e.g. Opus 4.7 at 4.5s vs Kimi K2 at 22s → scores 1.0 vs 0.20). The 0.10 floor ensures even the slowest model contributes to the composite score.

**Quality score** (0–1, higher = better) — uses the model's `quality_baseline` from the catalog ([Artificial Analysis Intelligence Index](https://artificialanalysis.ai), normalized to 0–1):

```
quality_score = model.quality_baseline / 60.0

# If the model has high error rates, penalise:
quality_score *= (1.0 - error_rate × 0.5)
```

The `quality_baseline` is a static benchmark score from public evaluations — it doesn't require historical usage data. Error rate penalties from the metrics store are the only dynamic component.

### Strategy Comparison

| Strategy | Composite formula | Best for |
|---|---|---|
| `cost-optimized` | `composite = cost_score` | Batch processing, classification, high-volume |
| `latency-optimized` | `composite = latency_score` | Real-time chat, interactive UX |
| `quality-optimized` | `composite = quality_score` | Complex reasoning, analysis, code generation |
| `balanced` | `0.4×cost + 0.3×latency + 0.3×quality` | General purpose (default) |

All four strategies compute all three scores for every model — the difference is only which score(s) drive the final selection. The non-primary scores are still recorded in `routing_decision.candidate_scores` for observability.

### How Strategies Improve Over Time

```
Day 1:   Quality uses quality_baseline (static benchmark scores — always accurate)
         Cost uses real pricing from catalog (always accurate)
         Latency uses tier-based heuristics (sensible defaults)

Week 2:  5+ requests per model → latency scores switch to real P50 data
         Latency-optimized and balanced strategies get smarter

Ongoing: Error rates penalise unreliable models automatically
         Circuit breakers remove failing models from candidates
```

No cold-start problem — quality scores are based on public benchmarks (Artificial Analysis Intelligence Index), not historical data. The router gets better at latency estimation as it collects data, but quality and cost are accurate from day 1.

### Tuning Balanced Strategy Weights

The balanced strategy is the only strategy that accepts custom weights. The other strategies (cost-optimized, quality-optimized, latency-optimized) use a single dimension and ignore weights entirely.

```python
# Default balanced: cost=0.4, latency=0.3, quality=0.3
router.converse(messages=msgs, routing=RoutingConfig(strategy="balanced"))

# Prioritize quality (e.g., for customer-facing responses)
router.converse(messages=msgs, routing=RoutingConfig(
    strategy="balanced",
    weights={"cost": 0.2, "latency": 0.2, "quality": 0.6}
))

# Prioritize cost (e.g., for batch processing)
router.converse(messages=msgs, routing=RoutingConfig(
    strategy="balanced",
    weights={"cost": 0.7, "latency": 0.2, "quality": 0.1}
))
```

**How weights affect model selection:**

With default `{cost: 0.4, latency: 0.3, quality: 0.3}` — a cheap model with moderate quality wins:
```
Cheap model:   0.4×0.98 + 0.3×0.55 + 0.3×0.40 = 0.677 ← winner
Quality model: 0.4×0.65 + 0.3×0.55 + 0.3×0.74 = 0.647
```

With `{cost: 0.2, latency: 0.2, quality: 0.6}` — the quality model wins:
```
Cheap model:   0.2×0.98 + 0.2×0.55 + 0.6×0.40 = 0.546
Quality model: 0.2×0.65 + 0.2×0.55 + 0.6×0.74 = 0.684 ← winner
```

## Routing Decision Explainability

Enable `explain=True` to get a detailed breakdown of why the router selected a specific model. Useful for debugging, auditing, and building trust in routing decisions.

```python
response = router.converse(
    messages=[{"role": "user", "content": [{"text": "Design a fraud detection system"}]}],
    system=[{"text": "You are a principal engineer."}],
    routing=RoutingConfig(strategy="balanced", explain=True),
)

explanation = response["routing_decision"].explanation
```

The explanation includes:

```json
{
  "complexity": {
    "score": 0.162,
    "score_before_boost": 0.162,
    "classification": "moderate",
    "classification_thresholds": {
      "simple": "< 0.125",
      "moderate": "0.125 - 0.35",
      "complex": "0.35 - 0.5",
      "reasoning": ">= 0.5 OR reasoning_markers >= 4"
    },
    "tier_range": {"min": "lite", "max": "mid"},
    "markers_hit": {
      "reasoning": ["architect", "design a"],
      "complex_questions": ["architect", "design a"]
    },
    "marker_counts": {
      "reasoning": 2,
      "code": 0,
      "complex_questions": 2,
      "last_user_message_chars": 28,
      "full_context_chars": 850,
      "conversation_turns": 3,
      "structural_signals": 0
    },
    "dimension_scores": {
      "token_count": 0.03,
      "reasoning_markers": 0.70,
      "question_complexity": 0.40
    },
    "user_message_score": 0.0239,
    "system_prompt_floor": 0.162,
    "floor_applied": true,
    "system_floor_markers": {
      "reasoning": ["architect", "design a", "optimize", "trade-off"],
      "aws": ["aws", "s3", "ec2", "lambda", "bedrock"],
      "complex_questions": ["design a", "optimize"]
    },
    "multimodal_payload": null
  },
  "strategy": {
    "name": "balanced",
    "weights": {"cost": 0.4, "latency": 0.3, "quality": 0.3}
  },
  "top5_candidates": [
    {"model": "Claude Sonnet 4.6", "model_id": "anthropic.claude-sonnet-4-6", "composite": 0.72, "cost": 0.65, "latency": 0.68, "quality": 0.74},
    {"model": "MiniMax M2.5", "model_id": "minimax.minimax-m2.5", "composite": 0.70, "cost": 0.92, "latency": 0.55, "quality": 0.70},
    {"model": "Nova Pro", "model_id": "amazon.nova-pro-v1:0", "composite": 0.58, "cost": 0.88, "latency": 0.71, "quality": 0.23}
  ],
  "candidates_evaluated": 28,
  "reason": "Selected Claude Sonnet 4.6 (composite score: 0.720) for moderate complexity. Balanced across cost/latency/quality."
}
```

**What each section tells you:**
- **score** — Final complexity score: `max(user_message_score, system_prompt_floor)` + any multimodal boost
- **user_message_score** — Raw composite from 15 dimensions scored against the last user message only
- **system_prompt_floor** — Minimum complexity derived from system prompt keywords (30% of system prompt's raw keyword score)
- **floor_applied** — `true` when the system prompt floor determined the final score (user message was simpler than what the system prompt demands)
- **system_floor_markers** — Keywords in the system prompt that contributed to the floor (explains why the floor is what it is)
- **markers_hit** — Keywords from the user message that triggered scoring dimensions
- **dimension_scores** — Per-dimension scores (0–1) for the 15 scoring dimensions
- **classification / tier_range** — Why the prompt was classified at this level and which model tiers were eligible
- **multimodal_payload** — If a document/image was attached: byte size and how much it boosted the complexity score (null when no attachment)
- **top5_candidates** — Top 5 models ranked by composite score with full cost/latency/quality breakdown
- **candidates_evaluated** — Total number of models that were eligible (filtered by tier range + capabilities)
- **reason** — Human-readable one-liner explaining the choice

The explanation adds ~1KB to the response and negligible latency. It's opt-in — only computed when `explain=True`.

### Request Complexity Classification

Before strategy scoring, the router classifies each request to determine the minimum model tier. The classification uses a two-signal approach:

#### Signal 1: User Message Score (15 dimensions)

Only the **last user message** is scored — not the full conversation history or system prompt. This prevents multi-turn conversations from inflating simple follow-up messages.

| Dimension | Weight | What it detects |
|---|---|---|
| Text length | 0.378 | Log-scaled text length (strongest signal) |
| Code presence | 0.057 | `` ``` ``, `def`, `import`, language names |
| Reasoning markers | 0.081 | "analyze", "step by step", "trade-off", "prove" |
| Technical depth | 0.049 | Keyword density per 200 chars |
| Simple indicators | 0.007 | "hello", "what is", "translate" (inverted — presence lowers score) |
| Structural complexity | 0.001 | Tables, CSV data, code blocks, multi-paragraph |
| Tool use signals | 0.042 | "function call", "json schema" (keyword-based, not tool_config presence) |
| Domain specificity | 0.127 | AWS services, math, data analysis keywords |
| Conversation depth | 0.010 | Multi-turn message count |
| Multi-step patterns | 0.026 | "first", "then", "step 1" |
| Question complexity | 0.026 | "how would you design" vs "what is" |
| Creative/open-ended | 0.096 | "write a story", "brainstorm", "imagine" |
| Output format | 0.099 | "return as json", "format as table", structured output |
| Constraint density | 0.001 | "must be", "at least", "without using" |
| Context ratio | 0.001 | "based on the following", "the above document" |

#### Signal 2: System Prompt Floor

The system prompt establishes a **minimum complexity floor**. A complex system prompt (e.g. "You are a senior architect, analyze trade-offs, design well-architected solutions") means even short user messages like "analyse for X" require a capable model — because the system prompt defines what "analyse" means.

The floor is computed by scoring the system prompt's keywords across reasoning, code, AWS, math, creative, constraint, and complex question categories, then taking 30% of that raw score. This ensures:
- Simple system prompt ("You are a helpful assistant") → floor ≈ 0.0 (no effect)
- Complex system prompt (architect + AWS + trade-offs) → floor ≈ 0.15 (MODERATE minimum)

#### Final Score

```
final_score = max(user_message_score, system_prompt_floor) + multimodal_boost
```

The final score maps to a complexity level:

| Score Range | Classification | Eligible Tiers | Use Case |
|---|---|---|---|
| < 0.125 | Simple | MICRO → LITE | Greetings, definitions, translations |
| 0.125 – 0.350 | Moderate | LITE → MID | Code explanations, comparisons, SQL |
| 0.350 – 0.500 | Complex | MID → HEAVY | System design, algorithms, architecture |
| ≥ 0.500 OR 4+ reasoning markers | Reasoning | REASONING only | Math proofs, multi-step logic, formal analysis |

Models outside the eligible tier range are excluded before strategy scoring begins. This ensures simple questions never go to expensive models, and complex questions never go to models that can't handle them.

## Model Catalog

The router ships with a JSON catalog (`bedrock_smart_router/data/models.json`) containing all active Bedrock text-generation models with capabilities, pricing, quality baselines, and latency mode support. The catalog is auto-generated by `scripts/refresh_catalog.py`.

| Family | Tiers | Notes |
|---|---|---|
| Anthropic Claude | lite, mid, heavy, reasoning | Haiku, Sonnet, Opus variants + global CRIS profiles |
| Amazon Nova | micro, lite, mid | Micro, Lite, Nova 2 Lite, Pro |
| Meta Llama | micro, lite, mid | Llama 3.x, 4 Scout, 4 Maverick |
| DeepSeek | mid, reasoning | V3.1, V3.2, R1 |
| Mistral | micro, lite, mid | Ministral, Mistral Large 3, Pixtral, Devstral |
| MiniMax | mid | M2, M2.1, M2.5 |
| Qwen | lite, mid | Qwen3 32B–480B, Coder, VL |
| NVIDIA | micro, mid | Nemotron Nano, Super |
| Others | various | GLM, Kimi, Gemma, gpt-oss, Writer |

### Refreshing the Catalog with `refresh_catalog.py`

The catalog is a static file that ships with the SDK. As AWS launches new models, changes pricing, or retires old ones, the catalog needs updating. The `scripts/refresh_catalog.py` script fully automates this by combining multiple data sources:

**Data sources and what they provide:**

| Source | Data Retrieved | Method |
|---|---|---|
| AWS Bedrock `ListFoundationModels` (17 regions) | Model discovery, display names, vision/streaming capabilities | API calls across regions |
| AWS Bedrock `ListInferenceProfiles` (17 regions) | CRIS profiles (us.*, eu.*, ap.*, global.*) per region | API calls across regions |
| [LiteLLM](https://github.com/BerriAI/litellm) `model_prices_and_context_window.json` | Pricing (input/output/cache), max_input_tokens, max_output_tokens | GitHub download |
| [Artificial Analysis](https://artificialanalysis.ai) Intelligence Index API | Quality baseline scores (0–60 scale) | API call (free key) |
| Bedrock Converse API probing | tool_use, streaming_tool_use, extended_thinking, guardrails, latency optimization | Minimal API calls per model |

**Regional discovery:**

The script probes 17 AWS regions to build the `regions` array for each model. For each region where a model is found, it determines whether the model is available via CRIS profiles (with which prefixes) or direct invocation only. This replaces the old flat `cris_profiles` field with per-region granularity.

**How tier classification works:**

The tier (micro/lite/mid/heavy/reasoning) is derived automatically from multiple signals — no hardcoded model names:
- **Reasoning**: quality_baseline ≥ 50 OR name contains "r1", "thinking", "reasoning"
- **Heavy**: price ≥ $4/M input + supports prompt caching + extended thinking
- **Micro**: name contains "micro"/"nano" OR (small model ≤8B + cheap + low quality)
- **Lite**: name contains "lite"/"haiku"/"scout"/"mini" OR small model ≤14B + cheap
- **Mid**: name contains "pro"/"large"/"sonnet"/"maverick" OR model ≥70B OR quality ≥ 15

**Usage:**

```bash
# Quick refresh — skip probes, use cached AA data (~1s)
python scripts/refresh_catalog.py --skip-probes --aa-cache scripts/_aa_models.json --write

# Full refresh with capability probing (~60s, 5 models probed in parallel)
python scripts/refresh_catalog.py --aa-cache scripts/_aa_models.json --write

# Full refresh with fresh AA quality scores from API
python scripts/refresh_catalog.py --aa-key YOUR_AA_API_KEY --write

# Overwrite the production catalog
python scripts/refresh_catalog.py --aa-cache scripts/_aa_models.json --write \
  --output bedrock_smart_router/data/models.json
```

**Flags:**

| Flag | Default | Description |
|---|---|---|
| `--region` | `us-west-2` | AWS region for Bedrock API calls |
| `--write` | off | Write output to file (default: `scripts/_models.json`) |
| `--aa-key` | none | Artificial Analysis API key (free at artificialanalysis.ai) |
| `--aa-cache` | none | Path to cached AA JSON (use when rate-limited) |
| `--skip-probes` | off | Skip capability probing (faster but uses defaults) |
| `--output` | `scripts/_models.json` | Custom output path |

**Generated files (in `scripts/`):**
- `_models.json` — Generated catalog output
- `_litellm_models.json` — Cached LiteLLM data (re-downloaded each run)
- `_aa_models.json` — Cached Artificial Analysis data

**IAM permissions required:**

```json
{
  "Effect": "Allow",
  "Action": [
    "bedrock:ListFoundationModels",
    "bedrock:ListInferenceProfiles",
    "bedrock-runtime:Converse",
    "bedrock-runtime:ConverseStream"
  ],
  "Resource": "*"
}
```

**Recommended schedule:** Run monthly or after any AWS model launch. The probing step makes minimal API calls (~6 per model, most fail at validation with $0 cost).

### Global CRIS Profiles

Global cross-region inference profiles route requests to any commercial AWS Region worldwide for higher throughput and resilience. They are ~10% cheaper than regional profiles on both input and output tokens ([source](https://docs.aws.amazon.com/bedrock/latest/userguide/global-cross-region-inference.html)).

#### Regional Model Structure

Each model in the catalog has a `regions` array describing where it's available and how to invoke it:

```json
{
  "model_id": "anthropic.claude-sonnet-4-6",
  "regions": [
    {"name": "us-west-2", "cris_profiles": ["global", "us"]},
    {"name": "eu-west-1", "cris_profiles": ["global", "eu"]},
    {"name": "ap-northeast-1", "cris_profiles": ["global", "jp"]}
  ]
}
```

```json
{
  "model_id": "nvidia.nemotron-nano-12b-v2",
  "regions": [
    {"name": "us-west-2", "direct": true},
    {"name": "eu-west-1", "direct": true}
  ]
}
```

**Two invocation modes per region:**
- **`cris_profiles`** — Model is available via CRIS inference profiles. The array lists available prefixes (e.g. `["global", "us"]`). The router prepends the prefix to the model ID: `global.anthropic.claude-sonnet-4-6` or `us.anthropic.claude-sonnet-4-6`.
- **`direct`** — Model is invoked directly by its base model ID (no prefix). Typically newer or third-party models that don't yet have CRIS profiles.

**Profile selection priority:**
1. `global.*` — cheapest (~10% discount), highest availability (routes to any region)
2. Regional prefix (`us.*`, `eu.*`, `jp.*`) — stays within a geography (useful for data residency)
3. `direct` — no prefix, model invoked as-is

The `CrisManager.select_profile(model, region)` method handles this automatically. When `allow_global: false` is set in config, global profiles are skipped and the router uses regional prefixes or direct invocation.

### Latency Mode Pricing

Bedrock offers four on-demand service tiers. All prices in the catalog are **Standard tier** rates. The router applies tier multipliers at cost estimation time via `ModelPricing.estimate_cost(tier=...)`:

| Tier | Multiplier | Latency | Use Case |
|---|---|---|---|
| **Flex** | ~0.50× | Higher (best-effort) | Dev/test, model evals, batch-like workloads |
| **Standard** | 1.0× (base) | Normal | Default — everyday production workloads |
| **Priority** | ~1.75× | Up to 25% better OTPS | Mission-critical, customer-facing, latency-sensitive |
| **Reserved** | Fixed hourly | Guaranteed | Steady high-volume with 1–6 month commitment |

The `LatencyModeSelector` picks the tier automatically based on request complexity and budget constraints. Not all models support all tiers — the catalog tracks which tiers each model supports in `supported_latency_modes`.

```python
from bedrock_smart_router.models import TIER_PRICING_MULTIPLIER

# Check the multipliers
print(TIER_PRICING_MULTIPLIER)
# {"standard": 1.0, "optimized": 1.75, "standard": 0.50}

# Estimate cost for a specific tier
model = router.registry.get("amazon.nova-pro-v1:0")
cost_standard = model.pricing.estimate_cost(1000, 500)                    # $0.003
cost_priority = model.pricing.estimate_cost(1000, 500, tier="optimized")   # $0.00525
cost_flex     = model.pricing.estimate_cost(1000, 500, tier="standard")       # $0.0015
```

### Prompt Caching

Both Anthropic Claude and Amazon Nova models support prompt caching on Bedrock, but with different pricing models:

| Provider | Cache Reads | Cache Writes | Mechanism |
|---|---|---|---|
| Anthropic Claude | ~10% of input price | ~125% of input price | Explicit — you mark cache breakpoints |
| Amazon Nova | ~25% of input price | Free ($0.00) | Automatic — Bedrock caches repeated prefixes |
| Meta, Mistral, DeepSeek | N/A | N/A | Not supported |

The router's prompt cache advisor estimates savings and can boost cache-capable models in the balanced/cost strategy scoring when the savings are significant.

### Updating the Catalog

To add custom models or override values at runtime without modifying the shipped catalog:

```python
router.registry.load_overlay("my-custom-models.json")
```

## Configuration Reference

All configuration is driven through a single `RouterConfig` object, constructable from a dict or YAML:

| Section | Key Fields | Default |
|---|---|---|
| `region` | AWS region | `us-west-2` |
| `strategy` | `balanced`, `cost-optimized`, `latency-optimized`, `quality-optimized` | `balanced` |
| `weights` | `{cost, latency, quality}` weights for balanced strategy | `{0.4, 0.3, 0.3}` |
| `cache` | `enabled`, `backend`, `ttl_seconds`, `max_entries`, `redis_url`, `key_prefix` | enabled, memory, 3600s, 10K |
| `metrics` | `backend` (`memory`/`dynamodb`), `table_name`, `ttl_hours` | memory |
| `observability` | `log_decisions` | true |
| `cris` | `enabled`, `preferred_geography`, `allow_global`, `blocked_prefixes`, `allowed_prefixes` | enabled, no pref, global allowed |
| `inference_tier` | `allow_optimized`, `optimized_for_complex` | all enabled |
| `guardrails` | `pre_route`, `post_route` with `guardrail_id` and `action_on_block` | disabled |
| `aip` | `enabled`, `auto_create`, `tag_keys` | disabled |
| `fallback` | `enabled`, `max_depth`, `default_safe_model` | enabled, depth 5 |
| `circuit_breaker` | `failure_threshold`, `window_seconds`, `cooldown_seconds`, `throttle_cooldown_seconds`, `half_open_max_requests` | 5 failures, 60s, 30s, 10s, 1 req |
| `retry` | `max_retries`, `backoff_base_seconds`, `backoff_max_seconds`, `backoff_multiplier` | 3 retries, 0.5s base, 8.0s max, 2.0× |

See [BEDROCK_SMART_ROUTER_DETAILED_DESIGN.md](BEDROCK_SMART_ROUTER_DETAILED_DESIGN.md) for the full configuration schema.

## Configuration in Production

The router reads its config once at creation time. Changing a YAML file or dict after `BedrockRouter.create()` has no effect on the running instance. This is by design — the router is cheap to create and stateless enough to swap.

How you handle config changes depends on your deployment model:

### Lambda

Not a problem. Lambda instances are short-lived. Store your config in S3, Parameter Store, or AppConfig and read it at the module level:

```python
import json, boto3, yaml
from bedrock_smart_router import BedrockRouter

ssm = boto3.client("ssm")
config_str = ssm.get_parameter(Name="/myapp/router-config", WithDecryption=True)["Parameter"]["Value"]
router = BedrockRouter.create(yaml.safe_load(config_str))

def handler(event, context):
    return router.converse(messages=event["messages"])
```

Change the parameter → next cold start picks it up. Force a cold start by deploying a no-op change or updating the function's environment variable.

### ECS / Fargate

Store config in S3 or Parameter Store. When you change the config, trigger a rolling deployment — ECS drains old tasks and starts new ones with the fresh config. This is the standard ECS pattern for any config change.

```python
# startup.py — runs once when the container starts
import boto3, yaml
from bedrock_smart_router import BedrockRouter

s3 = boto3.client("s3")
obj = s3.get_object(Bucket="my-config-bucket", Key="router-config.yaml")
config = yaml.safe_load(obj["Body"].read())
router = BedrockRouter.create(config)
```

For zero-downtime config changes without redeployment, use **AWS AppConfig**:

```python
import time, threading, yaml
from bedrock_smart_router import BedrockRouter

_router = BedrockRouter.create(initial_config)

def _poll_appconfig():
    """Background thread that checks for config changes every 60s."""
    while True:
        time.sleep(60)
        new_config = fetch_from_appconfig()  # Your AppConfig client
        if new_config != current_config:
            global _router
            _router = BedrockRouter.create(new_config)  # Swap atomically

threading.Thread(target=_poll_appconfig, daemon=True).start()
```

Creating a new router is fast (~10ms, no API calls). The only things lost on swap are the in-memory response cache and in-memory metrics — both rebuild within minutes. If you use DynamoDB metrics and Redis cache, nothing is lost.

### EKS (Kubernetes)

Mount the config as a ConfigMap. Kubernetes updates the mounted file when the ConfigMap changes. Use a file watcher to detect changes:

```python
import os, time, threading, yaml
from bedrock_smart_router import BedrockRouter

CONFIG_PATH = "/etc/config/router-config.yaml"
_last_mtime = 0
_router = None

def _load():
    global _router, _last_mtime
    with open(CONFIG_PATH) as f:
        _router = BedrockRouter.create(yaml.safe_load(f))
    _last_mtime = os.path.getmtime(CONFIG_PATH)

def _watch():
    while True:
        time.sleep(30)
        mtime = os.path.getmtime(CONFIG_PATH)
        if mtime != _last_mtime:
            _load()

_load()
threading.Thread(target=_watch, daemon=True).start()
```

### What's preserved vs lost on router swap

| Component | Preserved? | Notes |
|---|---|---|
| DynamoDB metrics | ✅ Yes | Stored externally, new router reads them |
| Redis/Valkey cache | ✅ Yes | Stored externally, new router connects |
| In-memory metrics | ❌ Lost | Rebuilds from incoming requests |
| In-memory cache | ❌ Lost | Rebuilds from cache misses |
| Circuit breaker state | ❌ Lost | Resets to CLOSED — re-trips within seconds if model is still failing |
| A/B test counts | ❌ Lost | Counters restart from zero |
| Canary health data | ❌ Lost | Monitoring restarts |

For production services, use DynamoDB metrics (`metrics.backend: "dynamodb"`) — the strategy engine reads historical latency, quality, and error rate data from DynamoDB on startup, so routing decisions are informed immediately after a swap. Circuit breaker state resets to CLOSED, but re-trips within seconds if a model is still failing — the fallback chain handles those requests gracefully.

## IAM Permissions

See [docs/iam-permissions.md](docs/iam-permissions.md) for the complete IAM reference including:
- Bedrock inference permissions (always required)
- DynamoDB metrics store permissions (when using `dynamodb` backend)
- Pricing API permissions (optional, for dynamic pricing refresh)
- Guardrails permissions (when guardrails are configured)
- Least-privilege production policy example

## Project Structure

```
bedrock_smart_router/
  __init__.py                  # Public API exports
  models.py                    # Data models (BedrockModel, RoutingDecision, enums)
  config.py                    # Consolidated RouterConfig from dict/YAML
  router.py                    # BedrockRouter — main entry point (14-step request flow)
  async_router.py              # AsyncBedrockRouter for async/await
  data/models.json             # JSON model catalog (65 models, pricing, capabilities, quality baselines)
  # Core routing
  model_registry.py            # JSON-driven model catalog with filtering and overlays
  request_analyzer.py          # 15-dimension zero-API-call complexity classifier
  strategy_engine.py           # Cost, latency, balanced strategies + plugin base
  quality_strategy.py          # Quality-optimized strategy using quality_baseline scores
  context_validator.py         # Pre-call context window validation
  fallback_handler.py          # Multi-level fallback chain (ordered by quality_baseline)
  circuit_breaker.py           # CLOSED/OPEN/HALF_OPEN per model
  retry_handler.py             # Exponential backoff with error classification
  # Intelligence & metrics
  metrics_store.py             # In-memory sliding-window metrics store
  dynamodb_metrics_store.py    # DynamoDB-backed persistent metrics store
  cache_layer.py               # LRU response cache with TTL
  budget_strategy.py           # Per-request and rolling budget enforcement
  tag_strategy.py              # Glob-pattern tag-based routing
  conditional_strategy.py      # Metadata-based conditional routing
  observability.py             # Structured logging, callbacks, CostTracker
  # Bedrock-native features
  cris_manager.py              # CRIS profile selection by geography
  inference_tier.py            # Standard/Optimized auto-selection
  prompt_cache_advisor.py      # Prompt caching benefit estimation
  guardrails_integration.py    # Pre/post-route guardrail checks
  aip_manager.py               # Application Inference Profile management
  distilled_models.py          # Distilled model registration
  pricing_refresh.py           # Dynamic pricing from AWS Pricing API (runtime)
  # Advanced features
  ab_testing.py                # A/B testing with sticky sessions
  canary.py                    # Canary deployments with auto-rollback
  shadow_mode.py               # Traffic mirroring to shadow model (with quality_baseline comparison)
  custom_strategy.py           # Strategy plugin registration
  strands_model.py             # Strands Agents SDK Model provider (SmartRouterModel)
  semantic_cache.py            # Embedding-based semantic cache (optional)
  semantic_response_store.py   # Pluggable response storage backends (filesystem, S3, DynamoDB)
  intent_extractor.py          # Auto-extraction of intent + variables for semantic cache
  opensearch_vector_store.py   # OpenSearch Serverless vector store backend
  semantic_router.py           # Intent routing via embeddings (optional)

scripts/
  refresh_catalog.py           # Auto-refresh models.json (Bedrock API + LiteLLM + AA + probing)
  _aa_models.json              # Cached Artificial Analysis quality scores
  _litellm_models.json         # Cached LiteLLM pricing + context windows
  _models.json                 # Generated catalog output (review before promoting)

demo/                          # React + FastAPI comparison demo app
benchmarks/                    # Heuristic classifier accuracy benchmarks + ONNX model
tests/                         # 451 unit tests + integration tests (gated)
docs/
  iam-permissions.md           # IAM policy reference (Bedrock, DynamoDB, Guardrails)
```

## Building & Using in Your Project

### Install from source

```bash
git clone https://github.com/sameerbattoo/bedrock-smart-router.git
cd bedrock-smart-router
pip install .
```

With optional extras:
```bash
pip install ".[strands]"              # Strands Agents integration
pip install ".[strands,faiss,redis]"  # Multiple extras
pip install ".[opensearch]"           # OpenSearch Serverless vector store
```

### Build distributable packages

```bash
pip install build
python -m build
```

This produces two files in `dist/`:
- `bedrock_smart_router-0.1.0-py3-none-any.whl` — wheel (fast install)
- `bedrock_smart_router-0.1.0.tar.gz` — source distribution

### Install the built package in another project

```bash
# From the wheel
pip install dist/bedrock_smart_router-0.1.0-py3-none-any.whl

# Or from the tarball
pip install dist/bedrock_smart_router-0.1.0.tar.gz

# Or point pip at the directory
pip install /path/to/bedrock-smart-router/dist/bedrock_smart_router-0.1.0-py3-none-any.whl
```

### Use in a Lambda layer or Docker image

```dockerfile
# In your Dockerfile
COPY dist/bedrock_smart_router-0.1.0-py3-none-any.whl /tmp/
RUN pip install /tmp/bedrock_smart_router-0.1.0-py3-none-any.whl
```

For Lambda, add the wheel to your deployment package or create a Lambda layer:
```bash
mkdir -p layer/python
pip install dist/bedrock_smart_router-0.1.0-py3-none-any.whl -t layer/python/
cd layer && zip -r ../bedrock-smart-router-layer.zip python/
```

## Development

```bash
# Clone
git clone https://github.com/YOUR_ORG/bedrock-smart-router.git
cd bedrock_smart_router

# Setup
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,redis,otel]"   # All extras for full test coverage

# Run unit tests (420 tests, no AWS calls)
pytest tests/ -v

# Run ALL integration tests (53 tests, requires AWS credentials)
INTEGRATION_TEST=1 pytest tests/ -v -s

# Run specific integration test suites
INTEGRATION_TEST=1 pytest tests/test_bedrock_converse_integration.py -v -s  # Bedrock Converse
INTEGRATION_TEST=1 pytest tests/test_streaming_integration.py -v -s         # Streaming + TTFT
INTEGRATION_TEST=1 pytest tests/test_dynamodb_integration.py -v -s          # DynamoDB metrics
INTEGRATION_TEST=1 pytest tests/test_cloudwatch_integration.py -v -s        # CloudWatch metrics
INTEGRATION_TEST=1 pytest tests/test_aip_integration.py -v -s               # Application Inference Profiles
INTEGRATION_TEST=1 pytest tests/test_guardrails_real_integration.py -v -s   # Bedrock Guardrails
INTEGRATION_TEST=1 pytest tests/test_pricing_refresh_integration.py -v -s   # Pricing API
INTEGRATION_TEST=1 VALKEY_URL=rediss://... pytest tests/test_valkey_cache_integration.py -v -s  # ElastiCache (VPC)

# Validate model catalog pricing against live AWS Pricing API
python scripts/refresh_catalog.py --aa-cache scripts/_aa_models.json --skip-probes
```

## How It Compares

| Feature | LiteLLM | OpenRouter | Portkey | Bedrock Native | **Smart Router** |
|---|---|---|---|---|---|
| Bedrock-specific | No | No | No | Yes | **Yes** |
| Cross-family routing | Generic | Generic | Generic | No (single family) | **Yes** |
| CRIS awareness | No | No | No | Yes | **Yes** |
| Global CRIS profiles | No | No | No | Manual | **Auto (separate entries, ~10% cheaper)** |
| Inference tier routing | No | No | No | Manual | **Auto (Standard/Optimized)** |
| Tier-aware cost estimation | No | No | No | No | **Yes (0.5×/1.0×/1.75× multipliers)** |
| Prompt cache-aware | No | No | No | No | **Yes (Claude + Nova)** |
| Circuit breaker | No | No | Yes | No | **Yes** |
| A/B + canary + shadow | Mirror only | No | Canary only | No | **Yes** |
| Historical quality routing | No | No | No | No | **Yes** |
| Budget enforcement | Yes | No | No | No | **Yes** |
| Multi-tenant AIPs | No | No | No | Manual | **Auto** |
| Lambda-friendly | Partial | No | No | Yes | **Yes** |
| Strands Agents SDK | No | No | No | No | **Yes (SmartRouterModel)** |
| Zero-dependency core | No (Redis) | N/A | No | N/A | **Yes (boto3 only)** |
| Pricing validation script | No | No | No | No | **Yes (vs AWS Pricing API)** |

## Design Document

See [BEDROCK_SMART_ROUTER_DETAILED_DESIGN.md](BEDROCK_SMART_ROUTER_DETAILED_DESIGN.md) for the full design including competitive landscape analysis, gap analysis, architecture diagrams, and implementation details.

## License

MIT
