# Bedrock Smart Router — One Pager

## The Problem

When building applications on Amazon Bedrock, teams face a recurring set of challenges:

- **One model doesn't fit all.** A simple "What is S3?" costs the same as a complex architecture question when you use a single model. You're either overpaying for simple tasks or under-serving complex ones.
- **No cross-family routing.** Bedrock's native prompt router only routes within a single model family (e.g., Haiku ↔ Sonnet). It can't route a simple question to Nova Micro and a complex one to Claude Opus.
- **Bedrock-specific features are invisible to generic gateways.** Tools like LiteLLM and Portkey treat Bedrock as just another provider — they don't understand CRIS profiles, inference tiers (Standard/Priority/Flex), prompt caching, guardrails, or application inference profiles.
- **No built-in reliability.** A single model throttle or outage means your application fails. There's no automatic fallback, circuit breaking, or retry logic.
- **Cost is unpredictable.** Without per-request cost tracking, budget enforcement, and routing optimization, Bedrock spend grows unchecked.

## The Solution

The **Bedrock Smart Router** was created after a deep competitive analysis of 7 existing routing solutions — LiteLLM, OpenRouter, Portkey, Inworld Router, NVIDIA AI Blueprint, RouteLLM, and Bedrock's native prompt router. The study revealed that every existing option either treats Bedrock as just another generic provider (missing CRIS, inference tiers, prompt caching, guardrails, AIPs) or only routes within a single model family. None were purpose-built for Bedrock. The full analysis is documented in [BEDROCK_SMART_ROUTER_DETAILED_DESIGN.md](../BEDROCK_SMART_ROUTER_DETAILED_DESIGN.md).

The Bedrock Smart Router fills this gap — a lightweight Python SDK that sits between your application and Amazon Bedrock, automatically selecting the optimal model for each request based on cost, latency, quality, and task complexity, while leveraging every Bedrock-native feature.

## Design Philosophy: Drop-in Replacement

A core design goal of the SDK is **zero friction adoption**. The router provides 100% API coverage for Bedrock's `Converse` and `ConverseStream` APIs — the same method signatures, the same request format, the same response format. Upgrading existing code is a two-line change:

```python
# Before — direct Bedrock client
client = boto3.client("bedrock-runtime", region_name="us-west-2")
response = client.converse(
    modelId="us.anthropic.claude-sonnet-4-6",
    messages=[{"role": "user", "content": [{"text": "What is S3?"}]}],
    inferenceConfig={"maxTokens": 1024},
)

# After — smart router (same parameters, same response)
from bedrock_smart_router import BedrockRouter
router = BedrockRouter.create({"region": "us-west-2"})
response = router.converse(
    messages=[{"role": "user", "content": [{"text": "What is S3?"}]}],
    inference_config={"maxTokens": 1024},
)
```

Everything the caller already passes — `messages`, `system`, `toolConfig`, `inferenceConfig`, `guardrailConfig` — flows through unchanged. The response is a standard Bedrock Converse response with an additional `routing_decision` field. No new abstractions to learn, no request format to translate, no response parsing to rewrite.

The same applies to streaming. Replace `client.converse_stream(...)` with `router.converse_stream(...)` and the stream events are identical — `contentBlockDelta`, `messageStop`, `metadata` — because the router passes Bedrock's native events through untouched.

**Why Converse and not InvokeModel?** The Converse API is Bedrock's unified interface — one format across all model families (Claude, Nova, Llama, Mistral, DeepSeek). InvokeModel requires model-specific request/response schemas, making transparent cross-family routing impossible without format translation that would break the drop-in contract. By building on Converse, the router can route a request from Nova Micro to Claude Haiku without the caller knowing or caring — the format is the same either way.

For Strands Agents users, the same principle applies: `SmartRouterModel` is a drop-in replacement for Strands' `BedrockModel`. Swap the model provider, and every agent call is automatically routed with zero code changes to the agent logic.

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
- **4 vector store backends:** in-memory, FAISS, Redis/Valkey (8.2+), and OpenSearch Serverless

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
| Drop-in for Converse API | No (own format) | No (own format) | N/A | **Yes (same signature + response)** |
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
# Core SDK (only dependency: boto3)
pip install bedrock-smart-router

# With Strands Agents
pip install bedrock-smart-router[strands]

# With FAISS vector store for semantic cache
pip install bedrock-smart-router[faiss]
```

```python
# Drop-in replacement — same Converse API, automatic routing
from bedrock_smart_router import BedrockRouter

router = BedrockRouter.create()
response = router.converse(
    messages=[{"role": "user", "content": [{"text": "What is S3?"}]}],
)
# → Automatically routes to the cheapest capable model
# → Falls back if the model fails
# → Tracks cost, latency, and routing decisions
print(response["output"]["message"]["content"][0]["text"])
```

```python
# Strands Agent — drop-in replacement for BedrockModel
from strands import Agent
from bedrock_smart_router.strands_model import SmartRouterModel

model = SmartRouterModel(router_config={"region": "us-west-2"})
agent = Agent(model=model)
response = agent("Explain quantum computing")

print(model.last_routing_decision.selected_model)  # e.g. "us.amazon.nova-lite-v1:0"
print(model.last_routing_decision.actual_cost)      # e.g. $0.000012
```

## Future: Native TypeScript SDK

The current SDK is Python-only. To serve Node.js and TypeScript teams, a **native TypeScript port** should be planned as the next major milestone.

**Why a native port instead of a proxy?**

A proxy mode (HTTP API wrapping the Python SDK) was considered and deliberately not implemented. While it would serve any language, it introduces significant operational complexity that contradicts the SDK's core design principle of being lightweight and embeddable:

- **Authentication overhead** — the proxy needs its own auth layer (API keys, Cognito, IAM federation), adding infrastructure the SDK was designed to avoid
- **Extra network hop** — adds latency and a new failure point between the application and Bedrock
- **Deployment burden** — another service to deploy, monitor, scale, and secure (VPC, TLS, rate limiting)
- **Credential management** — the proxy must manage or federate AWS credentials, whereas the SDK simply uses the caller's existing IAM role

A native TypeScript SDK avoids all of this. Node.js developers would `npm install bedrock-smart-router` and use their own AWS credentials — the same pattern as `@aws-sdk/client-bedrock-runtime`. No proxy, no auth layer, no extra infrastructure.

**What carries over from the Python SDK:**
- `models.json` catalog — shared across both SDKs
- Routing algorithms — same math, different syntax
- Config schema — same YAML/JSON structure
- Strands TypeScript SDK integration — the Strands Agents SDK already has a TypeScript version

**Estimated effort:** 4–6 weeks for core routing + Strands TS integration.

## Links

- **GitHub:** [github.com/sameerbattoo/bedrock-smart-router](https://github.com/sameerbattoo/bedrock-smart-router)
- **30 runnable examples** covering every feature
- **445 unit tests + 56 integration tests**
- **MIT License**
