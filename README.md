# Bedrock Smart Router

Intelligent model routing for Amazon Bedrock. A lightweight Python SDK that sits between your application and Bedrock, automatically selecting the optimal model for each request based on cost, latency, quality, and task complexity.

Unlike generic LLM gateways (LiteLLM, Portkey, OpenRouter) that treat Bedrock as just another provider, the Bedrock Smart Router is purpose-built for Bedrock and understands CRIS profiles, inference tiers, prompt caching, guardrails, application inference profiles, and model distillation.

Unlike Bedrock's native prompt router, which only routes within a single model family, the Smart Router routes across all families (Anthropic, Amazon Nova, Meta, Mistral) with custom strategies and historical quality data.

## Features

**100% Bedrock Converse API Coverage**

The Smart Router is a true drop-in replacement for `bedrock-runtime.converse()` and `converse_stream()`. Every Bedrock Converse parameter is supported — either as a first-class parameter or via `**kwargs` passthrough. This includes `additionalModelRequestFields` (model-specific params like `top_k`, extended thinking), `guardrailConfig`, `performanceConfig`, `outputConfig`, `promptVariables`, and `requestMetadata`. Every response field is captured in the routing decision: token usage, prompt cache metrics, stop reason, server-side latency, service tier, cache details, performance config, and guardrail trace. Nothing is lost by using the router instead of calling Bedrock directly.

**Routing Strategies**
- Cost-optimized, latency-optimized, quality-optimized, and balanced (weighted composite)
- Named presets: `economy`, `speed`, `balanced`, `quality` — one-word shortcuts for common routing profiles
- Budget-constrained routing with per-request ceilings and rolling hourly/daily limits
- Tag-based routing for free/paid tiers and team access control
- Conditional routing based on request metadata
- Custom strategy plugins — subclass `RoutingStrategy` and register it
- `preferred_model` override — pin a specific model while keeping fallbacks and reliability

**Request Intelligence**
- 12-dimension zero-API-call complexity classifier (sub-millisecond overhead)
- Automatic complexity detection: simple, moderate, complex, reasoning
- Vision, tool use, long context, and code task detection
- Context window pre-validation before sending to Bedrock

**Bedrock-Native Awareness**
- Cross-Region Inference (CRIS) profile selection — regional (`us.*`) and global (`global.*`, ~10% cheaper) profiles as separate catalog entries
- Inference tier auto-selection (Standard / Priority / Flex) with tier-aware cost estimation
- Prompt cache benefit estimation — boosts cache-capable models (Claude and Nova) when savings are significant
- Provisioned throughput detection — prefers already-paid capacity
- Bedrock Guardrails integration — pre-route and post-route checks via ApplyGuardrail API
- Application Inference Profile management for multi-tenant cost tracking
- Distilled model support with derived pricing and tier from teacher models

**Reliability**
- Circuit breakers (CLOSED/OPEN/HALF_OPEN) per model with separate throttle cooldowns
- Multi-level fallback chain: same-family downgrade, cross-family equivalent, CRIS retry, safe default
- Configurable retry with exponential backoff for transient errors
- Content policy and context window fallbacks
- Graceful no-models-match errors with per-model rejection reasons and actionable suggestions

**Production Deployment**
- A/B testing with weighted variants and sticky user assignment
- Canary deployments with auto-rollback on error rate/latency thresholds
- Shadow mode — mirror traffic to a secondary model in background threads
- Response caching (in-memory LRU with TTL)
- Semantic caching via embeddings with pluggable vector stores (in-memory, FAISS, Redis)
- Variable-aware semantic cache — same intent + different parameters = cache miss
- Semantic intent router — route queries to specialized models by meaning

**Observability**
- Structured routing decision logging on every request
- Custom callback hooks for Datadog, Splunk, or any analytics pipeline
- Cost tracking with breakdowns by model, strategy, complexity, and tenant
- Routing savings calculation (actual cost vs. most-expensive-model cost)
- Historical metrics store (in-memory or DynamoDB) for data-driven routing

**Async Support**
- `AsyncBedrockRouter` for async/await usage in FastAPI, aiohttp, etc.

## Quick Start

### Installation

```bash
# Core SDK — only requires boto3, works in Lambda out of the box
pip install bedrock-smart-router

# With Redis/Valkey/ElastiCache caching support
pip install bedrock-smart-router[redis]

# With OpenTelemetry tracing and metrics
pip install bedrock-smart-router[otel]

# With everything
pip install bedrock-smart-router[redis,otel]

# For development (includes pytest, moto)
pip install bedrock-smart-router[dev]
```

| Extra | What it adds | When you need it |
|---|---|---|
| *(none)* | Core SDK, boto3 only | Lambda, single-instance, in-memory cache and metrics |
| `[redis]` | `redis` package | Shared cache + vector store via Redis, Valkey, or ElastiCache |
| `[faiss]` | `faiss-cpu` package | Fast in-process vector search for semantic cache (~100K entries) |
| `[otel]` | `opentelemetry-api`, `opentelemetry-sdk` | Distributed tracing and OTEL metrics export |
| `[dev]` | `pytest`, `pytest-cov`, `moto` | Running the test suite |

### Basic Usage

```python
from bedrock_smart_router import BedrockRouter

# All defaults — balanced strategy, in-memory metrics
router = BedrockRouter.create()

response = router.converse(
    messages=[{"role": "user", "content": [{"text": "Explain VPCs in AWS"}]}],
)

print(response["routing_decision"].selected_model)
# e.g. "us.amazon.nova-lite-v1:0" for a simple question

print(response["routing_decision"].actual_cost)
# e.g. 0.000012
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
    "inference_tier": {"allow_priority": True, "flex_for_batch": True},
    "guardrails": {
        "pre_route": {"guardrail_id": "gr-abc123", "action_on_block": "reject"},
    },
    "fallback": {"max_depth": 5},
    "circuit_breaker": {"failure_threshold": 10},
    "excluded_models": ["us.meta.*"],
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
    #     - Nova Micro (us.amazon.nova-micro-v1:0): family amazon != nonexistent
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

```python
from bedrock_smart_router.custom_strategy import register_strategy
from bedrock_smart_router.strategy_engine import RoutingStrategy, StrategyResult

class PreferAnthropicForCode(RoutingStrategy):
    name = "anthropic-code"

    def select(self, candidates, analysis):
        if analysis.is_code_task:
            candidates = [c for c in candidates if c.family == "anthropic"] or candidates
        best = max(candidates, key=lambda m: m.pricing.input_per_1k)  # highest quality
        return StrategyResult(
            selected_model=best,
            scores={best.model_id: {"composite": 1.0}},
            fallback_chain=candidates[:3],
        )

register_strategy("anthropic-code", PreferAnthropicForCode)
router = BedrockRouter.create({"strategy": "anthropic-code"})
```

### Inspecting Runtime State

```python
# Cache stats
print(router.cache.stats)
# {"hits": 42, "misses": 158, "hit_rate": 0.21, "size": 158}

# Cost tracking
print(router.observability.cost_tracker.stats)
# {"total_cost": 0.23, "cost_saved_by_routing": 0.87, ...}

# Historical metrics for a model
m = router.metrics.get_metrics("us.anthropic.claude-sonnet-4-6")
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
| [`07_bedrock_native.py`](examples/07_bedrock_native.py) | CRIS profiles, inference tiers, guardrails |
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
| [`21_semantic_cache_deep_dive.py`](examples/21_semantic_cache_deep_dive.py) | Vector stores (memory/FAISS/Redis), threshold tuning |
| [`22_semantic_router.py`](examples/22_semantic_router.py) | Intent routing: route queries to specialized models by meaning |
| [`23_tag_and_conditional_routing.py`](examples/23_tag_and_conditional_routing.py) | Tag-based routing (free/paid tiers, teams) and metadata-driven conditions |
| [`24_budget_and_tier_pricing.py`](examples/24_budget_and_tier_pricing.py) | Per-request cost ceilings, rolling budgets, inference tier pricing (Flex/Standard/Priority) |

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
  default_safe_model: "us.amazon.nova-lite-v1:0"
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
       d.fallback_model = "us.anthropic.claude-haiku-4-5-..."
       d.circuit_breaker_skipped = ["us.anthropic.claude-sonnet-4-6"]
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
      model: "us.anthropic.claude-sonnet-4-6"
      weight: 0.7                 # 70% of traffic
    challenger:
      model: "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
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
  baseline: "us.amazon.nova-pro-v1:0"
  canary_model: "us.amazon.nova-2-lite-v1:0"
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
  shadow_model: "us.amazon.nova-pro-v1:0"
  sample_rate: 0.1                  # Mirror 10% of traffic
```

```python
# Check shadow results
print(router.shadow.stats)
# {"shadow_model": "us.amazon.nova-pro-v1:0", "total": 42, "success_rate": 0.98, "avg_latency_ms": 320}

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

The SDK provides three levels of cost control, from simple per-request caps to rolling budget windows with automatic downgrade.

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
    max_hourly_spend=1.00,         # Max $1.00/hour
    max_daily_spend=10.00,         # Max $10.00/day
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
| `max_hourly_spend` | Rolling 1-hour spend window per scope |
| `max_daily_spend` | Rolling 24-hour spend window per scope |
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

**When to use each level:**

| Scenario | Use |
|---|---|
| Simple cost cap, no tracking needed | `RoutingConfig(max_cost_per_request=...)` |
| Different limits for free/paid tiers | `BudgetRule` per tier |
| Per-user or per-team rolling budgets | `BudgetTracker` + `BudgetRule` |
| SaaS with tenant cost isolation | `BudgetTracker` keyed by tenant ID + `BudgetRule` per plan |

The `BudgetTracker` is in-memory — it resets on process restart and doesn't share across instances. This works well for single-instance deployments and Lambda (where each invocation is independent anyway).

For multi-instance persistent budget tracking, the DynamoDB metrics store (`metrics.backend: "dynamodb"`) records every request with `cost` and `tenant_id`, but there's no built-in query for "total spend by tenant in the last hour." You'd need to add a GSI on `tenant_id` or build a separate spend-tracking table. This is on the roadmap but not yet implemented.

## Architecture

```
Request arrives
  |
  +-- Step 1:  Pre-route guardrail check (ApplyGuardrail INPUT)
  +-- Step 2:  Request analysis (12-dimension complexity classifier)
  +-- Step 3:  Filter eligible models (tier, capabilities, family, exclusions)
  +-- Step 4:  Filter by context window
  +-- Step 5:  Filter by circuit breaker state
  +-- Step 6:  Run strategy (cost/latency/quality/balanced)
  |     +-- 6b: Prompt cache boost (swap to cache-capable model if within 10% score)
  |     +-- 6c: Select CRIS profile (us/eu/global geography preference)
  |     +-- 6d: Select inference tier (Standard/Priority/Flex)
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

## Model Catalog

The router ships with a JSON catalog (`bedrock_smart_router/data/models.json`) containing 39 Bedrock models — 27 regional and 12 global CRIS profiles — with capabilities, pricing, and inference tier support:

| Family | Models | Tiers |
|---|---|---|
| Amazon Nova (5 regional + 5 global) | Micro 1.0, Lite 1.0, Nova 2 Lite, Pro 1.0, Premier 1.0 | micro, lite, mid, heavy |
| Anthropic Claude (9 regional + 7 global) | Haiku 4.5, 3.7 Sonnet, Sonnet 4, Sonnet 4.5, Sonnet 4.6, Opus 4.1, Opus 4.5, Opus 4.6, Opus 4.7 | lite, mid, heavy, reasoning |
| Meta Llama (10) | 3.2 1B, 3.2 3B, 3.1 8B, 3.2 11B, 3.1 70B, 3.3 70B, 3.2 90B, 4 Scout 17B, 4 Maverick 17B | micro, lite, mid, heavy |
| DeepSeek (1) | R1 | reasoning |
| Mistral (3) | Small, Large 2, Pixtral Large | lite, mid |

Pricing is validated against the live AWS Pricing API using `scripts/refresh_pricing.py`. Run it periodically to catch price changes:

```bash
# Dry run — show mismatches
python scripts/refresh_pricing.py

# Auto-fix models.json and regenerate global entries
python scripts/refresh_pricing.py --fix --regen-global --check-tiers
```

### Global CRIS Profiles

Global cross-region inference profiles route requests to any commercial AWS Region worldwide for higher throughput and resilience. They are ~10% cheaper than regional (`us.*`) profiles on both input and output tokens ([source](https://docs.aws.amazon.com/bedrock/latest/userguide/global-cross-region-inference.html)).

The catalog includes global profiles as separate model entries with their own discounted pricing. The router's cost-optimized strategy naturally prefers them when `allow_global` is enabled in the CRIS config. Example: `global.anthropic.claude-sonnet-4-6` is the same model as `us.anthropic.claude-sonnet-4-6` but at 90% of the price.

### Inference Tier Pricing

Bedrock offers four on-demand service tiers. All prices in the catalog are **Standard tier** rates. The router applies tier multipliers at cost estimation time via `ModelPricing.estimate_cost(tier=...)`:

| Tier | Multiplier | Latency | Use Case |
|---|---|---|---|
| **Flex** | ~0.50× | Higher (best-effort) | Dev/test, model evals, batch-like workloads |
| **Standard** | 1.0× (base) | Normal | Default — everyday production workloads |
| **Priority** | ~1.75× | Up to 25% better OTPS | Mission-critical, customer-facing, latency-sensitive |
| **Reserved** | Fixed hourly | Guaranteed | Steady high-volume with 1–6 month commitment |

The `InferenceTierSelector` picks the tier automatically based on request complexity and budget constraints:
- Simple/moderate requests with tight budgets → Flex (if the model supports it)
- Complex/reasoning requests → Priority (if the model supports it)
- Everything else → Standard

Not all models support all tiers. The catalog tracks which tiers each model supports in `supported_inference_tiers`. The actual cost is always captured from the Bedrock response in `RoutingDecision.actual_cost` — the multipliers are used only for pre-call cost estimation during strategy scoring.

```python
from bedrock_smart_router.models import TIER_PRICING_MULTIPLIER

# Check the multipliers
print(TIER_PRICING_MULTIPLIER)
# {"standard": 1.0, "priority": 1.75, "flex": 0.50}

# Estimate cost for a specific tier
model = router.registry.get("us.amazon.nova-pro-v1:0")
cost_standard = model.pricing.estimate_cost(1000, 500)                    # $0.003
cost_priority = model.pricing.estimate_cost(1000, 500, tier="priority")   # $0.00525
cost_flex     = model.pricing.estimate_cost(1000, 500, tier="flex")       # $0.0015
```

### Prompt Caching

Both Anthropic Claude and Amazon Nova models support prompt caching on Bedrock, but with different pricing models:

| Provider | Cache Reads | Cache Writes | Mechanism |
|---|---|---|---|
| Anthropic Claude | ~10% of input price | ~125% of input price | Explicit — you mark cache breakpoints |
| Amazon Nova | ~25% of input price | Free ($0.00) | Automatic — Bedrock caches repeated prefixes |
| Meta, Mistral, DeepSeek | N/A | N/A | Not supported |

The router's prompt cache advisor estimates savings and can boost cache-capable models in the strategy scoring when the savings are significant (within 10% of the primary model's score).

### Updating the Catalog

To update pricing or add models, edit the JSON file directly or use an overlay at runtime:

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
| `cache` | `enabled`, `ttl_seconds`, `max_entries` | enabled, 3600s, 10K |
| `metrics` | `backend` (`memory`/`dynamodb`), `table_name`, `ttl_hours` | memory |
| `observability` | `log_decisions` | true |
| `cris` | `enabled`, `preferred_geography`, `allow_global` | enabled, no pref |
| `inference_tier` | `allow_priority`, `allow_flex`, `flex_for_batch` | all enabled |
| `guardrails` | `pre_route`, `post_route` with `guardrail_id` and `action_on_block` | disabled |
| `aip` | `enabled`, `auto_create`, `tag_keys` | disabled |
| `fallback` | `enabled`, `max_depth`, `default_safe_model` | enabled, depth 5 |
| `circuit_breaker` | `failure_threshold`, `window_seconds`, `cooldown_seconds` | 5 failures, 60s, 30s |
| `retry` | `max_retries`, `backoff_base_seconds`, `backoff_multiplier` | 3 retries, 0.5s base |

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
| Circuit breaker state | ❌ Lost | All circuits reset to CLOSED |
| A/B test counts | ❌ Lost | Counters restart from zero |
| Canary health data | ❌ Lost | Monitoring restarts |

For production services where losing circuit breaker state is a concern, use DynamoDB metrics — the quality-optimized strategy reads historical data from DynamoDB, so it survives router swaps.

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
  data/models.json             # JSON model catalog (39 models, pricing, capabilities)
  # Phase 1: Core
  model_registry.py            # JSON-driven model catalog with filtering and overlays
  request_analyzer.py          # 12-dimension zero-API-call complexity classifier
  strategy_engine.py           # Cost, latency, balanced strategies + plugin base
  context_validator.py         # Pre-call context window validation
  fallback_handler.py          # Multi-level fallback chain
  circuit_breaker.py           # CLOSED/OPEN/HALF_OPEN per model
  retry_handler.py             # Exponential backoff with error classification
  # Phase 2: Intelligence
  metrics_store.py             # In-memory sliding-window metrics store
  dynamodb_metrics_store.py    # DynamoDB-backed persistent metrics store
  quality_strategy.py          # Historical quality routing with heuristic blending
  cache_layer.py               # LRU response cache with TTL
  budget_strategy.py           # Per-request and rolling budget enforcement
  tag_strategy.py              # Glob-pattern tag-based routing
  conditional_strategy.py      # Metadata-based conditional routing
  observability.py             # Structured logging, callbacks, CostTracker
  pricing_refresh.py           # Dynamic pricing from AWS Pricing API
  # Phase 3: Bedrock-Native
  cris_manager.py              # CRIS profile selection by geography
  inference_tier.py            # Standard/Priority/Flex auto-selection
  prompt_cache_advisor.py      # Prompt caching benefit estimation
  provisioned_throughput.py    # Detect and prefer provisioned capacity
  guardrails_integration.py    # Pre/post-route guardrail checks
  aip_manager.py               # Application Inference Profile management
  distilled_models.py          # Distilled model registration
  # Phase 4: Advanced
  ab_testing.py                # A/B testing with sticky sessions
  canary.py                    # Canary deployments with auto-rollback
  shadow_mode.py               # Traffic mirroring to shadow model
  custom_strategy.py           # Strategy plugin registration
  semantic_cache.py            # Embedding-based semantic cache (optional)
  semantic_router.py           # Intent routing via embeddings (optional)

scripts/
  refresh_pricing.py           # Validate & refresh models.json from AWS Pricing API

tests/                         # 387 unit tests + 53 integration tests (gated)
docs/
  iam-permissions.md           # IAM policy reference (Bedrock, DynamoDB, Pricing, Guardrails)
```

## Development

```bash
# Clone
git clone git@ssh.gitlab.aws.dev:sbattoo/bedrock_smart_router.git
cd bedrock_smart_router

# Setup
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,redis,otel]"   # All extras for full test coverage

# Run unit tests (387 tests, no AWS calls)
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
python scripts/refresh_pricing.py --check-tiers
```

## How It Compares

| Feature | LiteLLM | OpenRouter | Portkey | Bedrock Native | **Smart Router** |
|---|---|---|---|---|---|
| Bedrock-specific | No | No | No | Yes | **Yes** |
| Cross-family routing | Generic | Generic | Generic | No (single family) | **Yes** |
| CRIS awareness | No | No | No | Yes | **Yes** |
| Global CRIS profiles | No | No | No | Manual | **Auto (separate entries, ~10% cheaper)** |
| Inference tier routing | No | No | No | Manual | **Auto (Flex/Standard/Priority)** |
| Tier-aware cost estimation | No | No | No | No | **Yes (0.5×/1.0×/1.75× multipliers)** |
| Prompt cache-aware | No | No | No | No | **Yes (Claude + Nova)** |
| Circuit breaker | No | No | Yes | No | **Yes** |
| A/B + canary + shadow | Mirror only | No | Canary only | No | **Yes** |
| Historical quality routing | No | No | No | No | **Yes** |
| Budget enforcement | Yes | No | No | No | **Yes** |
| Multi-tenant AIPs | No | No | No | Manual | **Auto** |
| Lambda-friendly | Partial | No | No | Yes | **Yes** |
| Zero-dependency core | No (Redis) | N/A | No | N/A | **Yes (boto3 only)** |
| Pricing validation script | No | No | No | No | **Yes (vs AWS Pricing API)** |

## Design Document

See [BEDROCK_SMART_ROUTER_DETAILED_DESIGN.md](BEDROCK_SMART_ROUTER_DETAILED_DESIGN.md) for the full design including competitive landscape analysis, gap analysis, architecture diagrams, and implementation details.

## License

MIT
