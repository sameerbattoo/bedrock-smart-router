# Bedrock Smart Router — One Pager

## The Problem

When building applications on Amazon Bedrock, teams face a recurring set of challenges:

- **One model doesn't fit all.** A simple "What is S3?" costs the same as a complex architecture question when you use a single model. You're either overpaying for simple tasks or under-serving complex ones.
- **No cross-family routing.** Bedrock's native prompt router only routes within a single model family (e.g., Haiku ↔ Sonnet). It can't route a simple question to Nova Micro and a complex one to Claude Opus.
- **Bedrock-specific features are invisible to generic gateways.** Tools like LiteLLM and Portkey treat Bedrock as just another provider — they don't understand CRIS profiles, inference tiers (Standard/Priority/Flex), prompt caching, guardrails, or application inference profiles.
- **No built-in reliability.** A single model throttle or outage means your application fails. There's no automatic fallback, circuit breaking, or retry logic.
- **Cost is unpredictable.** Without per-request cost tracking, budget enforcement, and routing optimization, Bedrock spend grows unchecked.

## The Solution

The **Bedrock Smart Router** is a lightweight Python SDK that sits between your application and Amazon Bedrock. It automatically selects the optimal model for each request based on cost, latency, quality, and task complexity — while leveraging every Bedrock-native feature.

```python
pip install bedrock-smart-router
```

```python
from bedrock_smart_router import BedrockRouter

router = BedrockRouter.create()
response = router.converse(
    messages=[{"role": "user", "content": [{"text": "What is S3?"}]}],
)
# → Automatically routes to the cheapest capable model
# → Falls back if the model fails
# → Tracks cost, latency, and routing decisions
```

## Key Features

### Intelligent Routing
- **12-dimension complexity classifier** analyzes each request in sub-millisecond time (zero API calls) and routes simple questions to cheap models, complex reasoning to powerful ones
- **4 built-in strategies:** cost-optimized, latency-optimized, quality-optimized, and balanced — plus named presets (`economy`, `speed`, `balanced`, `quality`)
- **Cross-family routing** across Anthropic Claude, Amazon Nova, Meta Llama, Mistral, and DeepSeek

### Bedrock-Native Awareness
- **CRIS profiles** — automatic cross-region inference selection (regional and global, ~10% cheaper)
- **Inference tiers** — auto-selects Standard, Priority, or Flex based on complexity and budget
- **Prompt caching** — boosts cache-capable models (Claude, Nova) when savings are significant
- **Guardrails** — pre-route and post-route content screening via ApplyGuardrail API
- **Application Inference Profiles** — automatic per-tenant cost tracking

### Semantic Caching with Auto-Extraction
- **Exact-match cache** — identical requests return instantly at zero cost
- **Semantic cache** — matches queries by meaning using embedding similarity ("How do I reset my password?" matches "I forgot my password, help")
- **Auto-extracting mode** — a cheap LLM (Nova Micro, ~$0.00003/call) automatically decomposes queries into canonical intent + variables, so "Count users by geo for 2026 with sales > $200" and "Show user distribution by geography, year 2026, sales over $200" are correctly matched — while "Count users by geo for 2025 with sales > $100" correctly misses
- **Multi-turn resolution** — resolves conversation history into a single query, so a cached single-turn response matches a multi-turn conversation with the same intent
- **4 vector store backends:** in-memory, FAISS, Redis/Valkey, and OpenSearch Serverless

### Production Reliability
- **Circuit breakers** per model with separate throttle cooldowns (10s for 429s, 30s for hard errors)
- **Multi-level fallback chain** — same-family downgrade → cross-family equivalent → default safe model
- **Retry with exponential backoff** for transient errors
- **A/B testing, canary deployments, and shadow mode** for safe model rollouts

### Strands Agents SDK Integration
- **`SmartRouterModel`** — drop-in replacement for Strands' `BedrockModel` that brings routing intelligence to any agent
- Every agent call is automatically routed, with fallbacks, cost tracking, and observability
- Tool use, multi-turn conversations, streaming, and structured output all work transparently
- Runtime config changes via `model.update_config()` — switch presets mid-conversation

### Cost Control
- Per-request cost ceilings, rolling hourly/daily budgets per user/team/tenant
- Multi-tenant routing with different strategies per tier (premium → quality, freemium → economy)
- Cost tracking with breakdowns by model, strategy, complexity, and tenant

### Observability
- Structured routing decision logging on every request
- CloudWatch metrics, OpenTelemetry spans, and custom callback hooks
- Full routing decision metadata: model selected, strategy used, complexity detected, cost, latency, fallback chain, CRIS profile, inference tier, prompt cache metrics

## How It Compares

| Capability | LiteLLM | Portkey | Bedrock Native | **Smart Router** |
|---|---|---|---|---|
| Cross-family routing | Generic | Generic | No | **Yes (Bedrock-aware)** |
| CRIS / Inference tiers | No | No | Manual | **Automatic** |
| Prompt cache-aware | No | No | No | **Yes** |
| Semantic caching + auto-extract | Basic | Basic | No | **Yes (intent + variables)** |
| Multi-turn cache resolution | No | No | No | **Yes** |
| Circuit breakers + fallbacks | No | Yes | No | **Yes** |
| Strands Agents SDK | No | No | No | **Yes** |
| Budget enforcement | Yes | No | No | **Yes** |
| Zero-dependency core | No | No | N/A | **Yes (boto3 only)** |
| Lambda-friendly | Partial | No | Yes | **Yes** |

## Quick Start

```bash
# Core SDK
pip install bedrock-smart-router

# With Strands Agents
pip install bedrock-smart-router[strands]

# With FAISS vector store for semantic cache
pip install bedrock-smart-router[faiss]
```

```python
# Strands Agent with smart routing
from strands import Agent
from bedrock_smart_router.strands_model import SmartRouterModel

model = SmartRouterModel(router_config={"region": "us-west-2"})
agent = Agent(model=model)
response = agent("Explain quantum computing")

print(model.last_routing_decision.selected_model)  # e.g. "us.amazon.nova-lite-v1:0"
print(model.last_routing_decision.actual_cost)      # e.g. $0.000012
```

## Links

- **GitHub:** [github.com/sameerbattoo/bedrock-smart-router](https://github.com/sameerbattoo/bedrock-smart-router)
- **30 runnable examples** covering every feature
- **445 unit tests + 56 integration tests**
- **MIT License**
