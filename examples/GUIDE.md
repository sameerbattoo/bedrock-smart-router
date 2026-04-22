# Bedrock Smart Router — Feature Guide & Examples

This guide covers every feature of the SDK with runnable code examples. Each example file in this folder is self-contained and can be run directly.

**100% Converse API Coverage:** The router is a true drop-in replacement for `bedrock-runtime.converse()` and `converse_stream()`. Every request parameter is supported (first-class or via `**kwargs`). Every response field is captured in the routing decision. You lose nothing by routing through the SDK instead of calling Bedrock directly.

## Prerequisites

```bash
pip install bedrock-smart-router
# AWS credentials configured (via ~/.aws/credentials, IAM role, or env vars)
```

---

## 1. Basic Routing (`01_basic_routing.py`)

The router sits between your application and Bedrock. It analyzes each request, picks the optimal model, calls Bedrock, and returns the response with routing metadata attached.

**Three ways to configure:**

| Method | When to use |
|---|---|
| `BedrockRouter.create()` | Quick start with sensible defaults |
| `BedrockRouter.create(yaml.safe_load(file))` | Team-shared config, no code changes to switch strategies |
| `BedrockRouter.create({"strategy": "..."})` | Config loaded from DynamoDB, S3, Parameter Store, etc. |

The key design principle: **application code never changes**. All routing behavior is driven by the config file.

```python
router = BedrockRouter.create()
response = router.converse(
    messages=[{"role": "user", "content": [{"text": "What is S3?"}]}],
)
print(response["routing_decision"].selected_model)  # e.g. "us.amazon.nova-lite-v1:0"
```

Every response includes a `routing_decision` with: selected model, strategy used, complexity detected, cost, latency, fallback chain, inference tier, CRIS profile, and more.

---

## 2. Named Presets (`02_presets.py`)

Presets are one-word shortcuts for common routing profiles. Instead of configuring strategy + weights + constraints manually, use a preset:

| Preset | What it does | Best for |
|---|---|---|
| `economy` | Cost-optimized, max $0.002/request | Batch, classification, simple Q&A |
| `speed` | Latency-optimized | Real-time chat, interactive UX |
| `balanced` | Weighted composite (40% cost, 30% latency, 30% quality) | General purpose |
| `quality` | Quality-optimized using historical data | Complex reasoning, code gen, analysis |

```python
response = router.converse(
    messages=[...],
    routing=RoutingConfig(preset="economy"),
)
```

Presets can be overridden — use economy but restrict to Anthropic:
```python
routing=RoutingConfig(preset="economy", preferred_family="anthropic")
```

---

## 3. Routing Strategies (`03_strategies.py`)

Five built-in strategies, each scoring models differently:

**Cost-optimized** — picks the cheapest model that meets the complexity requirement. Simple questions go to Nova Micro ($0.035/1M), complex ones to Sonnet ($3/1M).

**Latency-optimized** — picks the fastest model. Considers historical P50 latency, CRIS availability (cross-region = less queue time), and prompt caching support (faster TTFT).

**Quality-optimized** — picks the model with the highest historical quality scores from your own evaluation data. Falls back to tier heuristics when insufficient data exists. The more you use it, the better it gets.

**Balanced** — weighted composite of cost, latency, and quality. Default weights: 40/30/30. Fully configurable per request.

**Budget-constrained** — like balanced, but enforces a hard cost ceiling per request. If no model fits the budget, it either downgrades or rejects.

---

## 4. Fallbacks & Reliability (`04_fallbacks_and_reliability.py`)

The router never fails silently. When the primary model fails, it walks a multi-level fallback chain:

```
Primary model (selected by strategy)
  → Same-family downgrade (Sonnet → Haiku)
  → Cross-family equivalent (Sonnet → Nova Pro)
  → CRIS profile retry (try cross-region)
  → Default safe model (Nova Lite)
```

**Circuit breakers** prevent cascading failures. Each model has its own breaker:
- CLOSED → normal operation
- OPEN → model is failing, skip to fallback immediately
- HALF_OPEN → after cooldown, send one probe request to test recovery

**Retries** with exponential backoff handle transient errors (throttles, 503s) without bothering the caller.

---

## 5. Response Caching (`05_caching.py`)

The cache stores responses keyed by the user's request (messages + system prompt + config). Cache hits bypass Bedrock entirely — zero cost, sub-millisecond latency.

Key design: the cache key does NOT include the model ID. This means if request A fell back from model X to model Y, repeating request A still hits the cache.

```python
r1 = router.converse(messages=msgs)  # Cache miss — calls Bedrock
r2 = router.converse(messages=msgs)  # Cache hit — instant, free
```

Configure TTL and max entries:
```yaml
cache:
  ttl_seconds: 1800
  max_entries: 5000
```

---

## 6. Observability (`06_observability.py`)

Every routing decision is observable through three channels:

**Structured logging** — every request logs model, strategy, complexity, cost, latency, cache hit, and fallback status via Python's `logging` module.

**Custom callbacks** — register functions that receive a `RoutingEvent` on every request. Send to Datadog, Splunk, or your own analytics pipeline.

**CloudWatch metrics** — when enabled, publishes 7 custom metrics (RoutingDecisions, Latency, Cost, CacheHits, FallbacksUsed, CircuitBreakerSkips, CostSavings) with Model/Strategy/Complexity dimensions.

**Cost tracking** — the built-in `CostTracker` accumulates spend by model, strategy, and complexity, and calculates how much routing saved vs. always using the most expensive model.

---

## 7. Bedrock-Native Features (`07_bedrock_native.py`)

Features that no other router provides because they're specific to Bedrock:

**CRIS profiles** — Cross-Region Inference routes requests across AWS regions for higher throughput. The router selects the optimal profile by geography preference (us/eu/global).

**Inference tiers** — Bedrock offers Standard, Priority (25% faster, premium), and Flex (cheaper, latency-tolerant). The router auto-selects based on request complexity and budget.

**Prompt cache awareness** — Bedrock caches the prefix (system prompt + conversation history) server-side. The router estimates the dollar savings and boosts cache-capable models when the benefit is significant.

**Guardrails** — pre-route checks screen user input via `ApplyGuardrail` before model selection. Post-route checks screen model output. Blocked requests either raise an error or return sanitized text.

---

## 8. Multi-Tenant Support (`08_multi_tenant.py`)

Application Inference Profiles (AIPs) are Bedrock's mechanism for per-tenant cost tracking. The router auto-creates AIPs per tenant+model combination and invokes Bedrock using the AIP ARN, so Cost Explorer automatically attributes costs to the right tenant.

```python
routing=RoutingConfig(
    metadata={"tenant": "acme-corp", "team": "engineering"},
)
```

No manual AIP management needed — the router handles creation, caching, and ARN resolution.

---

## 9. A/B Testing, Canary, Shadow (`09_ab_testing_canary_shadow.py`)

**A/B testing** — split traffic between models to compare quality, cost, and latency in production. Sticky mode ensures the same user always sees the same variant.

**Canary deployments** — gradually roll out a new model at X% of traffic. The router monitors error rate and latency, auto-rolling back if thresholds are exceeded, or auto-promoting if the canary performs well.

**Shadow mode** — mirror a sample of production traffic to a secondary model in background threads. The shadow response is logged for offline comparison but never returned to the caller.

All three are config-driven and no-op when disabled — zero overhead.

---

## 10. Custom Strategy Plugins (`10_custom_strategy.py`)

Subclass `RoutingStrategy`, implement `select()`, and register it:

```python
class MyStrategy(RoutingStrategy):
    name = "my-custom"
    def select(self, candidates, analysis):
        best = candidates[0]
        return StrategyResult(selected_model=best, ...)

register_strategy("my-custom", MyStrategy)
router = BedrockRouter.create({"strategy": "my-custom"})
```

Examples in the file: code-aware routing, EU-only routing, time-of-day routing.

---

## 11. Error Handling (`11_error_handling.py`)

When no models satisfy the constraints, the router raises `NoModelsMatchError` with:
- **Per-model rejection reasons** — why each model was excluded (tier, cost, context, capability, family, pattern)
- **Constraints applied** — the full set of filters that were active
- **Actionable suggestions** — "Remove preferred_family", "Increase max_cost_per_request", etc.
- **`to_dict()`** — structured dict for JSON API responses

---

## 12. Historical Metrics (`12_metrics_and_dynamodb.py`)

Two backends:

| Backend | Persistence | Use case |
|---|---|---|
| `memory` (default) | Resets on restart | Lambda, single-instance, dev |
| `dynamodb` | Persistent, shared | Multi-instance, cross-Lambda, production |

The quality-optimized strategy reads from the metrics store. After 20+ requests with quality scores, it trusts historical data over tier heuristics.

---

## 13. Async Usage (`13_async_usage.py`)

`AsyncBedrockRouter` wraps the sync router and runs Bedrock calls in a thread pool executor. All routing logic runs synchronously (sub-millisecond), only the Bedrock API call is offloaded.

```python
router = AsyncBedrockRouter.create({"strategy": "balanced"})
response = await router.converse(messages=[...])
```

Works with FastAPI, aiohttp, or any async framework.

---

## 14. Model Catalog (`14_model_catalog.py`)

The router ships with a JSON catalog (`data/models.json`) containing 39 Bedrock models (27 regional + 12 global CRIS profiles). You can:

- **List and filter** by family, tier, capability, context window
- **Load overlays** to add custom/imported models or fix stale pricing
- **Register distilled models** with derived pricing and tier from the teacher model
- **Refresh pricing** from the AWS Pricing API at runtime

---

## Configuration Reference

All features are configurable from a single YAML/dict:

```yaml
region: us-west-2
strategy: balanced
weights: {cost: 0.4, latency: 0.3, quality: 0.3}
cache: {ttl_seconds: 1800, max_entries: 5000}
metrics: {backend: dynamodb, table_name: MyMetrics, ttl_hours: 168}
observability: {log_decisions: true, cloudwatch_enabled: true, cloudwatch_namespace: MyApp}
cris: {preferred_geography: us}
inference_tier: {allow_priority: true, flex_for_batch: true}
guardrails:
  pre_route: {guardrail_id: gr-abc, action_on_block: reject}
aip: {enabled: true, auto_create: true, tag_keys: [tenant, team]}
ab_test:
  enabled: true
  name: sonnet-vs-nova
  variants:
    control: {model: us.anthropic.claude-sonnet-4-6, weight: 0.5}
    treatment: {model: us.amazon.nova-pro-v1:0, weight: 0.5}
canary:
  enabled: true
  baseline: us.anthropic.claude-sonnet-4-6
  canary_model: us.anthropic.claude-opus-4-7
  canary_percentage: 5
shadow: {enabled: true, shadow_model: us.amazon.nova-pro-v1:0, sample_rate: 0.1}
fallback: {max_depth: 5}
circuit_breaker: {failure_threshold: 5, cooldown_seconds: 30}
retry: {max_retries: 3}
excluded_models: ["us.meta.*"]
```

---

## 15. Redis / Valkey Caching (`15_redis_valkey_caching.py`)

The in-memory cache (default) works for single-process deployments but doesn't share across instances. For production with multiple Lambda invocations or ECS tasks, use Redis or Valkey to share the cache.

**Compatible with:** Redis, Valkey, Amazon ElastiCache (Redis or Valkey engine), ElastiCache Serverless, Amazon MemoryDB.

```bash
pip install bedrock-smart-router[redis]
```

```python
router = BedrockRouter.create({
    "cache": {
        "backend": "valkey",  # or "redis" — same protocol
        "redis_url": "rediss://master.my-cluster.abc123.usw2.cache.amazonaws.com:6379",
        "ttl_seconds": 1800,
        "key_prefix": "bsr:prod:",
    },
})
```

Use `redis://` for unencrypted connections (local dev) and `rediss://` (double s) for TLS (ElastiCache requires TLS by default).

**Cache invalidation** works the same as in-memory — per-model or full flush:
```python
router.cache.invalidate("us.anthropic.claude-sonnet-4-6")  # One model
router.cache.invalidate()  # Everything
```

**Recommended production setup** — Valkey cache + DynamoDB metrics + CloudWatch:
```yaml
cache:
  backend: valkey
  redis_url: "rediss://master.my-cluster.abc123.usw2.cache.amazonaws.com:6379"
  ttl_seconds: 1800
  key_prefix: "bsr:prod:"
metrics:
  backend: dynamodb
  table_name: BedrockRouterMetrics
observability:
  cloudwatch_enabled: true
  cloudwatch_namespace: MyApp/BedrockRouter
```

---

## 16. Streaming (`16_streaming.py`)

`converse_stream()` routes the request through the same pipeline as `converse()` (complexity analysis, strategy, CRIS, tier, guardrails, fallbacks) but returns tokens as they arrive instead of waiting for the full response.

```python
for event in router.converse_stream(
    messages=[{"role": "user", "content": [{"text": "Write a haiku about clouds."}]}],
):
    if "contentBlockDelta" in event:
        print(event["contentBlockDelta"]["delta"]["text"], end="", flush=True)
    elif "routing_decision" in event:
        d = event["routing_decision"]
        print(f"\n[Model: {d.selected_model}, TTFT: {d.ttft_ms:.0f}ms, Total: {d.latency_ms:.0f}ms]")
```

**TTFT (Time to First Token)** is automatically measured — the time from stream start to the first `contentBlockDelta` event. This is the key latency metric for streaming:

| Metric | What it measures | Typical values |
|---|---|---|
| `ttft_ms` | Time until first token arrives | 2–600ms depending on model |
| `latency_ms` | Total time from start to stream end | 500–5000ms depending on output length |

For non-streaming `converse()`, TTFT equals total latency (the entire response arrives at once).

**Presets work with streaming:**
```python
for event in router.converse_stream(
    messages=[...],
    routing=RoutingConfig(preset="speed"),  # Lowest TTFT
):
    ...
```

**Async streaming** for FastAPI / aiohttp:
```python
async for event in async_router.converse_stream(messages=[...]):
    if "contentBlockDelta" in event:
        yield event["contentBlockDelta"]["delta"]["text"]
```

**Note:** Streaming responses are not cached (they're consumed once). The routing decision, metrics, and observability events are still recorded after the stream completes.

### Bedrock Response Metrics

Every routing decision (both `converse()` and `converse_stream()`) captures the full set of metrics from the Bedrock response:

```python
d = response["routing_decision"]

# Latency
print(f"Wall-clock latency: {d.latency_ms:.0f}ms")
print(f"Bedrock server latency: {d.bedrock_latency_ms}ms")
print(f"Network overhead: {d.network_overhead_ms}ms")  # Convenience property
print(f"TTFT (streaming only): {d.ttft_ms}ms")

# Tokens and cost
print(f"Input tokens: {d.input_tokens}")
print(f"Output tokens: {d.output_tokens}")
print(f"Cost: ${d.actual_cost:.6f}")

# Bedrock prompt cache (server-side prefix caching)
print(f"Prompt cache read: {d.prompt_cache_read_tokens} tokens")
print(f"Prompt cache write: {d.prompt_cache_write_tokens} tokens")
# Bedrock prompt cache hit ratio
# Convenience property: d.prompt_cache_hit_rate (0.0–1.0)
print(f"Prompt cache hit rate: {d.prompt_cache_hit_rate:.0%}")
print(f"Total input tokens (incl cached): {d.total_input_tokens}")
# e.g. 1200 read / (300 input + 1200 read + 0 write) = 80%

# Stop reason
print(f"Stop reason: {d.stop_reason}")
# end_turn = normal completion
# max_tokens = response truncated (consider increasing maxTokens)
# tool_use = model wants to call a tool
# guardrail_intervened = guardrail blocked the output
# content_filtered = content filter triggered

# Service tier verification
print(f"Requested tier: {d.inference_tier}")
print(f"Actual tier served: {d.actual_service_tier}")
```

---

## Updated Configuration Reference

All features including the new ones are configurable from a single YAML/dict:

```yaml
region: us-west-2
strategy: balanced
weights: {cost: 0.4, latency: 0.3, quality: 0.3}

# Caching — memory (default), redis, or valkey
cache:
  backend: valkey              # "memory" | "redis" | "valkey"
  redis_url: "rediss://my-cluster.abc123.usw2.cache.amazonaws.com:6379"
  ttl_seconds: 1800
  max_entries: 10000           # For memory backend only
  key_prefix: "bsr:"

# Metrics — memory (default) or dynamodb
metrics:
  backend: dynamodb
  table_name: MyRouterMetrics
  ttl_hours: 168

# Observability
observability:
  log_decisions: true
  cloudwatch_enabled: true
  cloudwatch_namespace: MyApp/BedrockRouter

# Bedrock-native
cris: {preferred_geography: us}
inference_tier: {allow_priority: true, flex_for_batch: true}
guardrails:
  pre_route: {guardrail_id: gr-abc, action_on_block: reject}
aip: {enabled: true, auto_create: true, tag_keys: [tenant, team]}

# Deployment features
ab_test:
  enabled: true
  name: sonnet-vs-nova
  variants:
    control: {model: us.anthropic.claude-sonnet-4-6, weight: 0.5}
    treatment: {model: us.amazon.nova-pro-v1:0, weight: 0.5}
canary:
  enabled: true
  baseline: us.anthropic.claude-sonnet-4-6
  canary_model: us.anthropic.claude-opus-4-7
  canary_percentage: 5
shadow: {enabled: true, shadow_model: us.amazon.nova-pro-v1:0, sample_rate: 0.1}

# Reliability
fallback: {max_depth: 5}
circuit_breaker: {failure_threshold: 5, cooldown_seconds: 30}
retry: {max_retries: 3}
excluded_models: ["us.meta.*"]
```

---

## 17. Advanced Bedrock Parameters (`17_advanced_bedrock_params.py`)

The router is a 100% drop-in replacement for `bedrock-runtime.converse()` and `converse_stream()`. Every Bedrock Converse parameter is supported — either as a first-class parameter or via `**kwargs` passthrough.

**First-class parameters** (explicit in the method signature):
- `messages`, `system`, `tool_config`, `inference_config`

**Passthrough via `**kwargs`** (forwarded unchanged to Bedrock):

| Parameter | What it does | Example |
|---|---|---|
| `additionalModelRequestFields` | Model-specific params (top_k, extended thinking) | `additionalModelRequestFields={"top_k": 50}` |
| `additionalModelResponseFieldPaths` | Request extra response fields | `additionalModelResponseFieldPaths=["/stop_sequence"]` |
| `guardrailConfig` | Native Bedrock guardrail on the call | `guardrailConfig={"guardrailIdentifier": "gr-abc"}` |
| `promptVariables` | Prompt Management integration | `promptVariables={"topic": {"text": "AI"}}` |
| `outputConfig` | Structured JSON output | `outputConfig={"textFormat": {"type": "json", ...}}` |
| `performanceConfig` | Latency-optimized inference | `performanceConfig={"latency": "optimized"}` |

`requestMetadata` is automatically forwarded from `routing.metadata` for CloudWatch invocation log filtering.

**All response fields captured** in `RoutingDecision`:
- `stop_reason`, `bedrock_latency_ms`, `actual_service_tier`
- `total_tokens`, `prompt_cache_read_tokens`, `prompt_cache_write_tokens`
- `cache_details`, `performance_config`, `guardrail_trace`
- Convenience properties: `prompt_cache_hit_rate`, `total_input_tokens`, `network_overhead_ms`

---

## 18. Cross-Region Inference & Data Residency (`18_cross_region_data_residency.py`)

Bedrock Cross-Region Inference (CRIS) automatically routes requests across AWS regions for higher throughput. Geography-specific profiles ensure data residency compliance.

Your boto3 client connects to ONE region, but the CRIS profile controls where inference runs:

| Profile Prefix | Routes To | Use Case |
|---|---|---|
| `us.*` | US regions only | ITAR, FedRAMP, US financial services |
| `eu.*` | EU regions only | GDPR compliance |
| `ap.*` | Asia-Pacific only | APAC data residency |
| `global.*` | Any commercial region | Maximum throughput |
| *(no prefix)* | Configured region only | Single-region locked |

```yaml
# EU-only — GDPR, data never leaves Europe
cris:
  preferred_geography: eu
  allow_global: false

# US preferred, global fallback for capacity
cris:
  preferred_geography: us
  allow_global: true

# Maximum throughput, no residency requirement
cris:
  allow_global: true
```

The router selects the profile automatically. You never manage region endpoints or create multiple clients.

---

## 19. OpenTelemetry (`19_opentelemetry.py`)

The router emits OTEL spans and metrics when enabled. Works with any OTEL-compatible backend: AWS X-Ray (via ADOT), Jaeger, Datadog, Grafana Tempo, Honeycomb, etc.

```bash
pip install bedrock-smart-router[otel]
```

```yaml
observability:
  otel_enabled: true
  otel_service_name: "my-app"
```

**Metrics emitted per request** (with `{model, strategy, complexity}` labels):

| Metric | Type | Unit |
|---|---|---|
| `bedrock_router.requests` | Counter | 1 |
| `bedrock_router.latency` | Histogram | ms |
| `bedrock_router.ttft` | Histogram | ms |
| `bedrock_router.cost` | Counter | USD |
| `bedrock_router.cache_hits` | Counter | 1 |
| `bedrock_router.fallbacks` | Counter | 1 |
| `bedrock_router.errors` | Counter | 1 |

**All three observability channels work simultaneously:**
```yaml
observability:
  log_decisions: true           # Python logging
  cloudwatch_enabled: true      # CloudWatch PutMetricData
  otel_enabled: true            # OpenTelemetry spans + metrics
```

**For AWS production**, use the AWS Distro for OpenTelemetry (ADOT) collector as a sidecar to send traces to X-Ray and metrics to CloudWatch or Amazon Managed Prometheus.

OTEL is disabled by default — zero overhead when not configured. The `opentelemetry` package is only imported when `otel_enabled: true`.


---

## 20. Semantic Cache (`20_semantic_cache.py`)

The semantic cache matches queries by meaning using embedding similarity. Unlike the exact-match cache (which only hits on identical requests), it catches rephrased questions:

```python
cache.put("How do I reset my password?", response)
cache.get("I forgot my password, help")  # HIT — same meaning
```

**Variable-aware caching** prevents false hits on parameterized queries:

```python
cache.put("Top users for Electronics 2024", response,
          variables={"category": "Electronics", "year": "2024"})

cache.get("Top users for Electronics 2024",
          variables={"category": "Electronics", "year": "2024"})  # HIT

cache.get("Top users for Clothing 2025",
          variables={"category": "Clothing", "year": "2025"})     # MISS
```

Same intent + same variables = HIT. Same intent + different variables = MISS.

---

## 21. Semantic Cache Deep Dive (`21_semantic_cache_deep_dive.py`)

Covers all vector store backends and configuration options:

| Backend | Install | Scales To | Shared |
|---|---|---|---|
| `memory` (default) | *(none)* | ~500 entries | No |
| `faiss` | `pip install bedrock-smart-router[faiss]` | ~100K entries | No |
| `redis` | `pip install bedrock-smart-router[redis]` | Millions | Yes |

```python
SemanticCache(config=SemanticCacheConfig(
    enabled=True,
    threshold=0.90,
    vector_store_backend="faiss",  # or "memory" or "redis"
    embedding_model="amazon.titan-embed-text-v2:0",
))
```

**Threshold tuning guide:**
- FAQ / customer support: 0.88–0.92
- General knowledge: 0.90–0.95
- Code questions: 0.93–0.97 (code is more specific)

---

## 22. Semantic Router (`22_semantic_router.py`)

Routes queries to specialized models by intent using embedding similarity:

```python
intent_router = SemanticRouter(routes=[
    SemanticRoute(name="code", model="us.anthropic.claude-sonnet-4-6",
                  examples=["Write a function", "Debug this code"]),
    SemanticRoute(name="creative", model="us.anthropic.claude-opus-4-7",
                  examples=["Write a story", "Compose a poem"]),
])

match = intent_router.route("Help me fix this Python bug")
# match.route_name = "code", match.model = "us.anthropic.claude-sonnet-4-6"
```

**Combining with the smart router** — use `preferred_model` to pin the intent router's model while keeping the smart router's reliability features:

```python
match = intent_router.route(query)
response = router.converse(
    messages=[...],
    routing=RoutingConfig(preferred_model=match.model),
)
```

The smart router uses the intent router's exact model as primary, builds a fallback chain around it, and provides retries, circuit breakers, caching, metrics, and observability. If the model is unavailable, it falls back gracefully.

---

## 23. Tag-Based & Conditional Routing (`23_tag_and_conditional_routing.py`)

Route requests based on metadata rather than content. Useful for multi-tier products, team access control, and compliance.

**Tag-based routing** — restrict model pools per tier:

```python
# Free-tier: cheap models only
response = router.converse(
    messages=[...],
    routing=RoutingConfig(
        tags=["free-tier"],
        exclude_models=["us.anthropic.*"],
        max_cost_per_request=0.001,
    ),
)

# Paid-tier: full access, quality-optimized
response = router.converse(
    messages=[...],
    routing=RoutingConfig(
        tags=["paid-tier"],
        strategy="quality-optimized",
    ),
)
```

**Conditional routing** — drive strategy from metadata:

```python
# Enterprise user → quality strategy
response = router.converse(
    messages=[...],
    routing=RoutingConfig(
        strategy="quality-optimized",
        metadata={"user_tier": "enterprise", "user_id": "u-001"},
    ),
)

# EU user → Anthropic only (GDPR), EU CRIS profile
response = router.converse(
    messages=[...],
    routing=RoutingConfig(
        preferred_family="anthropic",
        metadata={"region": "eu", "compliance": "gdpr"},
    ),
)
```

Tags, metadata, presets, and strategy overrides can all be combined in a single `RoutingConfig`.

---

## 24. Budget Enforcement & Inference Tier Pricing (`24_budget_and_tier_pricing.py`)

Control costs at every level — from per-request ceilings to rolling daily budgets.

**Inference tier pricing multipliers:**

| Tier | Multiplier | Latency | Use Case |
|---|---|---|---|
| Flex | 0.50× | Higher (best-effort) | Dev/test, batch, model evals |
| Standard | 1.0× (base) | Normal | Default production workloads |
| Priority | 1.75× | Up to 25% better OTPS | Mission-critical, customer-facing |

```python
from bedrock_smart_router import TIER_PRICING_MULTIPLIER

# Estimate cost across tiers
model = router.registry.get("us.amazon.nova-pro-v1:0")
cost_std = model.pricing.estimate_cost(1000, 500)                    # Standard
cost_pri = model.pricing.estimate_cost(1000, 500, tier="priority")   # 1.75× more
cost_flx = model.pricing.estimate_cost(1000, 500, tier="flex")       # 0.50× less
```

**Auto tier selection** — the router picks the tier based on complexity and budget:
- Simple + tight budget → Flex (if model supports it)
- Complex/reasoning → Priority (if model supports it)
- Everything else → Standard

**Per-request cost ceiling:**

```python
response = router.converse(
    messages=[...],
    routing=RoutingConfig(max_cost_per_request=0.001),
)
```

**Rolling budget tracking** — track spend per user/team with hourly and daily limits using `BudgetTracker` and `BudgetRule`. When exceeded, the strategy either downgrades to a cheaper model or rejects the request.
