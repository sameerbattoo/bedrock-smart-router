# Bedrock Smart Router — Feature Guide & Examples

This guide covers every feature of the SDK with runnable code examples. Each example file in this folder is self-contained and can be run directly.

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

The router ships with a JSON catalog (`data/models.json`) containing 16 Bedrock models. You can:

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
