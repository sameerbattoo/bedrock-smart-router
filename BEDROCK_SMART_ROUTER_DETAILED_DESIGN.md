# Bedrock Smart Router — Detailed Design Document

> A lightweight, SDK-based smart routing layer purpose-built for Amazon Bedrock.
> Any Python project can incorporate it to get intelligent model selection, cost optimization, and production-grade reliability for Bedrock inference.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Competitive Landscape Deep Dive](#2-competitive-landscape-deep-dive)
3. [Gap Analysis](#3-gap-analysis)
4. [Design Goals and Principles](#4-design-goals-and-principles)
5. [Delivery Modes: SDK vs Proxy](#5-delivery-modes-sdk-vs-proxy)
6. [Architecture](#6-architecture)
7. [Core Components — Detailed Design](#7-core-components--detailed-design)
8. [Routing Strategies](#8-routing-strategies)
9. [Reliability and Fallback System](#9-reliability-and-fallback-system)
10. [Caching Layer](#10-caching-layer)
11. [Observability and Metrics](#11-observability-and-metrics)
12. [A/B Testing and Canary Deployments](#12-ab-testing-and-canary-deployments)
13. [Multi-Tenant Support](#13-multi-tenant-support)
14. [Guardrails Integration](#14-guardrails-integration)
15. [Security Considerations](#15-security-considerations)
16. [SDK API Design](#16-sdk-api-design)
17. [Proxy Mode API Design](#17-proxy-mode-api-design)
18. [Configuration Schema](#18-configuration-schema)
19. [Implementation Plan](#19-implementation-plan)
20. [Differentiation Matrix](#20-differentiation-matrix)
21. [Strands Agents SDK Integration](#21-strands-agents-sdk-integration)
22. [References](#22-references)

---

## 1. Executive Summary

The Bedrock Smart Router is a lightweight, open-source routing layer that sits between your application and Amazon Bedrock. It intelligently selects the optimal model for each request based on cost, latency, quality, and task complexity — while leveraging Bedrock-native features that no existing router understands.

**Why build this?**

Existing solutions fall into two camps:
- **Generic gateways** (LiteLLM, Portkey, OpenRouter) that treat Bedrock as just another provider, missing CRIS profiles, inference tiers, prompt caching, guardrails, application inference profiles, and model distillation.
- **Bedrock's native prompt router** which only routes within a single model family (e.g., Haiku ↔ Sonnet) and cannot route across families, apply custom strategies, or integrate historical quality data.

The Bedrock Smart Router fills this gap with:
- **Zero-API-call request classification** for sub-millisecond routing overhead
- **Cross-family routing** (e.g., Nova Micro for simple tasks, Claude Sonnet for complex ones)
- **Bedrock-native awareness** of CRIS, inference tiers (Standard/Priority/Flex), prompt caching, guardrails, and application inference profiles
- **Two delivery modes**: embeddable Python SDK (single `pip install`) or standalone proxy server *(proxy mode planned, not yet implemented)*
- **Production-grade reliability**: circuit breakers, multi-level fallbacks, cooldown tracking
- **Built-in A/B testing and canary deployment** for safe model rollouts
- **Strands Agents SDK integration**: drop-in `Model` provider that brings routing intelligence to any Strands agent

---

## 2. Competitive Landscape Deep Dive

### 2.1 LiteLLM

LiteLLM is an open-source Python SDK and proxy server providing a unified OpenAI-compatible interface to 140+ providers and 2,500+ models.

**Routing capabilities:**
- Weighted round-robin, RPM/TPM-aware shuffling, latency-based, least-busy, and cost-based routing strategies
- Semantic/intent routing via embedding-based matching (100–500ms overhead per request)
- Rule-based complexity router: zero-API-call classification into SIMPLE/MEDIUM/COMPLEX/REASONING tiers using 7 scoring dimensions
- Tag-based routing for free/paid tiers and team-based access control
- Traffic mirroring / shadow mode for A/B evaluation
- Budget tracking and enforcement per user, team, API key, and organization

**Reliability:**
- Automatic retries with exponential backoff
- Cooldown system (deployments removed after >50% failure rate)
- Three fallback types: general, context-window, and content-policy fallbacks
- Configurable `RetryPolicy` and `AllowedFailsPolicy` per error type

**Caching:**
- In-memory or Redis exact-match caching
- Semantic caching via vector similarity (requires embedding model)
- Cross-model-group cache sharing

**What LiteLLM lacks for Bedrock users:**
- No awareness of Bedrock inference tiers (Standard / Priority / Flex)
- No CRIS profile integration or cross-region routing awareness
- No prompt caching-aware routing (Bedrock's native cache, not response caching)
- No application inference profile support for multi-tenant cost tracking
- No Bedrock Guardrails integration
- Pricing data from community-maintained JSON (often stale for Bedrock)
- Requires Redis for distributed routing strategies — not Lambda-friendly
- No model distillation awareness

### 2.2 OpenRouter

OpenRouter is a managed marketplace-style gateway with 300+ models from 60+ providers.

**Routing capabilities:**
- Provider-level routing sorted by price, latency, uptime, or throughput
- Explicit provider ordering and inclusion/exclusion rules
- BYOK (Bring Your Own Keys) with priority routing for own-key requests
- Presets: named configurations encapsulating model, provider, and parameter choices
- "Auto" model that opaquely selects a model per request

**What OpenRouter lacks:**
- No user-defined semantic or complexity routing
- No budget enforcement per user/team
- No custom routing strategy plugins
- No pre-call context window validation
- No A/B testing or traffic mirroring
- No self-hosted option — SaaS only with 5% markup
- No Bedrock-specific awareness whatsoever
- No open-source component

### 2.3 Portkey

Portkey is an AI gateway focused on observability, guardrails, and governance. Its gateway is now fully open-source, processing 1T+ tokens daily.

**Routing capabilities:**
- Conditional routing based on metadata, user attributes, or content policies
- Load balancing with weighted distribution
- Canary testing: split traffic (e.g., 5% to new model) for safe rollouts
- Circuit breaker: temporarily blocks requests to failing targets
- Simple and semantic caching
- Automatic retries with configurable policies
- Fallback chains across providers

**What Portkey lacks for Bedrock users:**
- Routing is rule-based and compliance-driven, not optimization-driven
- No dynamic cost-per-quality optimization
- No task complexity classification
- No Bedrock-native feature awareness (CRIS, inference tiers, prompt caching, guardrails)
- Enterprise pricing not published; free tier is limited

### 2.4 Inworld Router

Inworld Router (Research Preview, 2026) routes on business-level metrics: cost per output quality, latency targets, and task complexity.

**Notable capabilities:**
- Context-aware routing analyzing semantic content of each request
- Automatic failover across providers
- Built-in A/B testing with sticky user assignment
- Multimodal support (text, image, audio)

**What Inworld lacks:**
- Not open source — no self-hosting
- Currently in Research Preview with no published post-preview pricing
- No Bedrock-specific awareness
- Closed routing logic — no custom strategies

### 2.5 NVIDIA AI Blueprint for LLM Router

NVIDIA's blueprint provides a Rust-based, high-performance router using Triton Inference Server for classification.

**Notable capabilities:**
- Task classification routing (code generation → large model, summarization → small model)
- Complexity classification routing
- Multi-turn conversation routing (different model per turn based on task shift)
- Fine-tunable classification models via NeMo Curator
- OpenAI API-compliant, acts as drop-in replacement
- Grafana-based monitoring dashboard

**What NVIDIA's blueprint lacks:**
- Requires GPU (V100+) for the classification model — not serverless-friendly
- No cost-aware routing or budget management
- No fallback chains or circuit breakers
- No caching layer
- No Bedrock integration — designed for NIM endpoints
- Heavy infrastructure footprint (Docker Compose, Triton, GPU)

### 2.6 RouteLLM (lm-sys)

Academic framework from LMSYS for training and evaluating LLM routers.

**Notable capabilities:**
- Trained router models using preference data from Chatbot Arena
- Multiple router architectures: similarity-weighted ranking, matrix factorization, BERT classifier, causal LLM classifier
- Configurable cost-quality threshold
- OpenAI-compatible server

**What RouteLLM lacks:**
- Research-oriented, not production-ready
- Binary routing only (strong model vs weak model) — no multi-model selection
- Requires training data and model fine-tuning
- No fallbacks, caching, observability, or budget management
- No Bedrock awareness

### 2.7 Amazon Bedrock Native Routing

**Intelligent Prompt Routing:**
- Routes within a single model family based on predicted response quality
- Supports Anthropic (Haiku ↔ Sonnet), Meta (Llama 8B ↔ 70B), Amazon Nova (Lite ↔ Pro)
- Configurable quality-difference threshold
- English-only, cannot learn from application-specific data

**Cross-Region Inference (CRIS):**
- Automatically routes across AWS regions for higher throughput
- Geography-specific profiles (us., eu.) and global profiles
- Transparent to the caller

**Inference Tiers (new in late 2025):**
- **Standard**: everyday workloads with reliable performance
- **Priority**: up to 25% better OTPS latency for mission-critical workloads
- **Flex**: cost-optimized for latency-tolerant batch workloads

**Application Inference Profiles:**
- Logical wrapper around a model for per-tenant cost tracking via custom tags
- Enables Cost Explorer integration for granular cost allocation

**Model Distillation:**
- Teacher-student distillation producing models up to 500% faster and 75% cheaper with <2% accuracy loss

**What Bedrock native routing lacks:**
- Cannot route across model families
- No custom routing strategies
- No historical quality-based routing from your own evaluation data
- No budget enforcement
- No A/B testing or canary deployments
- No semantic or complexity-based classification
- No unified routing decision observability

---

## 3. Gap Analysis

### 3.1 Features Missing from ALL Existing Solutions (for Bedrock Users)

| Gap | Impact | Our Solution |
|---|---|---|
| No cross-family routing on Bedrock | Users stuck within one family or must build custom logic | Cross-family strategy engine with tier mapping |
| No inference tier awareness | Users can't auto-select Standard/Priority/Flex per request | Tier-aware routing based on urgency and budget |
| No CRIS-aware routing | External routers don't leverage cross-region profiles | CRIS profile registry with automatic selection |
| No prompt caching-aware routing | Missed cost savings when cacheable models aren't preferred | Cache benefit estimator influences model selection |
| No application inference profile integration | No per-tenant cost tracking through the router | Automatic AIP creation and tag propagation |
| No model distillation awareness | Distilled models not considered as routing candidates | Registry includes distilled variants with quality metadata |
| No Bedrock Guardrails pre-routing | Content safety checked after model selection, not before | Pre-route guardrail check to select appropriate model |
| No multi-turn complexity adaptation | Same model used for entire conversation regardless of turn complexity | Per-turn re-evaluation (inspired by NVIDIA blueprint) |
| No historical quality routing from own data | Routing based on generic benchmarks, not your workload | Integration with evaluation/judge scores from DynamoDB or any store |
| No agentic framework integration | Routing intelligence not available to agent SDKs (Strands, LangChain) | SmartRouterModel — drop-in Strands Model provider with full routing |

### 3.2 Features in Competitors We Should Also Support

| Feature | Found In | Priority | Notes |
|---|---|---|---|
| Circuit breaker pattern | Portkey | **P0** | Prevents cascading failures; missing from initial proposal |
| Semantic/exact response caching | LiteLLM, Portkey | **P0** | Major cost saver; should work with DynamoDB or ElastiCache |
| A/B testing / canary deployments | Portkey, Inworld, LiteLLM (mirror) | **P0** | Critical for safe model rollouts in production |
| Tag-based routing | LiteLLM | **P1** | Useful for free/paid tiers, team access control |
| Conditional routing (metadata-based) | Portkey | **P1** | Route based on user attributes, request metadata |
| Context window pre-validation | LiteLLM | **P1** | Reject or re-route before hitting model limits |
| Content policy fallbacks | LiteLLM | **P1** | Auto-fallback when model refuses content |
| Budget enforcement (per-user/team) | LiteLLM | **P1** | Hard caps on spend with automatic downgrade |
| Request timeout management | Portkey | **P2** | Configurable timeouts with automatic fallback |
| Multi-turn conversation routing | NVIDIA Blueprint | **P2** | Re-classify complexity per turn |
| Embedding-based semantic routing | LiteLLM | **P2** | Intent matching for specialized model selection |
| Custom routing strategy plugins | LiteLLM | **P2** | User-defined strategy classes |

---

## 4. Design Goals and Principles

1. **Bedrock-native**: Built specifically for Bedrock's API surface, model families, pricing, inference tiers, and features. Not a generic multi-provider gateway.

2. **Lightweight and embeddable**: Single `pip install bedrock-smart-router`. No Redis, no Docker, no GPU required. Works in Lambda, ECS, EC2, or local development.

3. **Two delivery modes**: Use as a Python SDK (import and call) or deploy as a standalone proxy server (OpenAI-compatible endpoint). Same routing engine powers both. *(Note: Proxy mode is designed but not yet implemented. The SDK is fully functional.)*

4. **Zero-overhead classification**: Request analysis uses local heuristics (no API calls) for sub-millisecond routing decisions. Optional embedding-based classification for higher accuracy at ~100ms cost.

5. **Data-driven improvement**: Uses your own historical metrics (latency, cost, quality scores) to improve routing over time. No cold-start problem — sensible defaults from model tier heuristics.

6. **Production-grade reliability**: Circuit breakers, multi-level fallbacks, cooldown tracking, and automatic retries. The router should never be the reason your application fails.

7. **Observable by default**: Every routing decision is logged with full context (why this model, what alternatives were scored, estimated vs actual cost). Integrates with CloudWatch, OpenTelemetry, or custom callbacks.

8. **Minimal dependencies**: Core SDK depends only on `boto3`. Optional extras for caching (`[redis]`), semantic routing (`[embeddings]`), and proxy mode (`[proxy]` — planned, not yet implemented).

9. **Agentic framework integration**: First-class support for the Strands Agents SDK via `SmartRouterModel`, so agent developers get routing intelligence without changing their agent code.

---

## 5. Delivery Modes: SDK vs Proxy

### 5.1 SDK Mode (Primary)

```python
from bedrock_smart_router import BedrockRouter

router = BedrockRouter(
    strategy="balanced",
    region="us-west-2",
)

# Drop-in replacement for bedrock-runtime converse()
response = router.converse(
    messages=[{"role": "user", "content": [{"text": "Summarize this document..."}]}],
    system=[{"text": "You are a helpful assistant."}],
)

# Access routing metadata
print(response["routing_decision"])
# {
#   "selected_model": "us.amazon.nova-lite-v1:0",
#   "strategy": "balanced",
#   "complexity": "simple",
#   "estimated_cost": 0.0002,
#   "alternatives_scored": 6,
#   "fallback_chain": ["us.anthropic.claude-haiku-4-5-20251001-v1:0", ...]
# }
```

**When to use SDK mode:**
- Lambda functions (no sidecar needed)
- Applications already using boto3
- When you want full control over the routing lifecycle
- Minimal latency overhead (in-process, no network hop)

### 5.2 Proxy Mode

> **⚠️ NOT YET IMPLEMENTED** — Proxy mode is designed but not built. The SDK (Section 5.1) is fully functional and covers all routing, reliability, and observability features. Proxy mode is planned for a future release if there is demand from multi-language teams.

```bash
# Planned — not yet available
pip install bedrock-smart-router[proxy]
bedrock-router serve --port 8080 --config router-config.yaml
```

Would expose an OpenAI-compatible `/v1/chat/completions` endpoint that internally routes to Bedrock models.

**When proxy mode would be useful:**
- Multi-language applications (Node.js, Go, Java calling the proxy)
- Team-wide shared routing configuration
- Centralized observability and cost tracking
- Drop-in replacement for OpenAI API calls

### 5.3 Feature Parity

| Feature | SDK Mode | Proxy Mode *(planned)* |
|---|---|---|
| All routing strategies | ✅ Yes | Planned |
| Fallbacks and circuit breakers | ✅ Yes | Planned |
| Response caching | ✅ In-memory + Redis | Planned (+ Redis/ElastiCache) |
| A/B testing | ✅ Yes | Planned (+ sticky sessions via headers) |
| Multi-tenant cost tracking | ✅ Via AIP tags | Planned (+ API keys) |
| Budget enforcement | ✅ Per-instance | Planned (per-key, per-team) |
| OpenAI API compatibility | No (Bedrock Converse API) | Planned |
| Observability | ✅ Callbacks + CloudWatch + OTEL | Planned (+ Prometheus) |

---

## 6. Architecture

```
┌───────────────────────────────────────────────────────────────────────┐
│                        Bedrock Smart Router                           │
│                                                                       │
│  ┌──────────────┐  ┌───────────────┐  ┌──────────────┐  ┌───────────┐ │
│  │   Request    │  │   Strategy    │  │    Model     │  │  Response │ │
│  │   Pipeline   │  │   Engine      │  │   Registry   │  │  Pipeline │ │
│  │              │  │               │  │              │  │           │ │
│  │ • Pre-route  │  │ • Cost-opt    │  │ • Families   │  │ • Cache   │ │
│  │   guardrails │  │ • Latency-opt │  │ • Tiers      │  │   store   │ │
│  │ • Complexity │  │ • Quality-opt │  │ • Pricing    │  │ • Metrics │ │
│  │   analyzer   │  │ • Balanced    │  │ • Caps/Quotas│  │   collect │ │
│  │ • Token est  │  │ • Budget      │  │ • CRIS       │  │ • Quality │ │
│  │ • Context    │  │ • A/B split   │  │ • Inf. Tiers │  │   scoring │ │
│  │   window chk │  │ • Canary      │  │ • Prov. TPut │  │ • Cost    │ │
│  │ • Cache      │  │ • Tag-based   │  │ • Guardrails │  │   tracking│ │
│  │   lookup     │  │ • Conditional │  │ • Distilled  │  │ • Fallback│ │
│  │ • Tag/meta   │  │ • Custom      │  │   variants   │  │   handler │ │
│  │   extraction │  └───────────────┘  └──────────────┘  └───────────┘ │
│  └──────────────┘         │                   │                       │
│         │          ┌──────────────┐    ┌──────────────┐               │
│         │          │  Circuit     │    │  Historical  │               │
│         │          │  Breaker     │    │  Metrics     │               │
│         │          │  Registry    │    │  Store       │               │
│         │          └──────────────┘    └──────────────┘               │
│         │                                                             │
│  ┌──────────────────────────────────────────────────────────────────┐ │
│  │                    Observability Layer                           │ │
│  │  • Routing decision logs  • Cost tracking  • CloudWatch/OTel     │ │
│  │  • Latency histograms     • Error rates    • Custom callbacks    │ │
│  └──────────────────────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
     ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
     │   Bedrock    │ │   Bedrock    │ │   Bedrock    │
     │   Standard   │ │   Priority   │ │   Flex/Batch │
     │   Tier       │ │   Tier       │ │   Tier       │
     └──────────────┘ └──────────────┘ └──────────────┘
```

### Request Flow

```
1. Request arrives (SDK call or proxy HTTP)
         │
2. REQUEST PIPELINE
   ├─ Cache lookup (exact match → return cached response)
   ├─ Pre-route guardrail check (optional, via ApplyGuardrail API)
   ├─ Request analysis (complexity, token estimate, capabilities needed)
   ├─ Context window pre-validation
   └─ Tag/metadata extraction
         │
3. STRATEGY ENGINE
   ├─ Filter eligible models (capabilities, context window, tags, budget)
   ├─ Score candidates (cost, latency, quality — weighted by strategy)
   ├─ Apply A/B split or canary rules
   ├─ Check circuit breaker state
   └─ Select model + inference tier + CRIS profile
         │
4. INVOKE BEDROCK
   ├─ Call Converse/ConverseStream API with selected model
   ├─ On success → response pipeline
   └─ On failure → fallback handler
         │
5. RESPONSE PIPELINE
   ├─ Cache store (if cacheable)
   ├─ Collect metrics (latency, tokens, cost)
   ├─ Record routing decision for observability
   └─ Return response + routing metadata
```

---

## 7. Core Components — Detailed Design

### 7.1 Model Registry

The Model Registry is the foundation of all routing decisions. It maintains a comprehensive catalog of every Bedrock model with capabilities, pricing, quotas, and historical performance data.

```python
@dataclass
class BedrockModel:
    model_id: str                    # e.g., "us.anthropic.claude-sonnet-4-6"
    family: str                      # "anthropic" | "amazon" | "meta" | "mistral" | "deepseek"
    tier: str                        # "micro" | "lite" | "mid" | "heavy" | "reasoning"
    display_name: str                # Human-readable name

    # Capabilities
    capabilities: ModelCapabilities  # tool_use, vision, streaming, etc.
    max_input_tokens: int
    max_output_tokens: int

    # Pricing (per 1K tokens)
    pricing: ModelPricing            # input, output, cache_read, cache_write per tier

    # Bedrock-specific
    supports_prompt_caching: bool
    supports_extended_thinking: bool
    cris_profiles: list[str]         # Available cross-region inference profiles
    supported_inference_tiers: list[str]  # ["standard", "priority", "flex"]
    guardrail_compatible: bool
    distilled_from: str | None       # Parent model if this is a distilled variant
    distilled_quality_delta: float   # Quality loss vs parent (e.g., -0.02 for 2% loss)

    # Runtime state (updated periodically)
    current_health: HealthStatus     # healthy | degraded | down
    circuit_breaker_state: str       # closed | open | half-open
```

**Bedrock Model Tier Mapping (expanded from initial proposal):**

| Tier | Anthropic | Amazon Nova | Meta Llama | Mistral | DeepSeek | Typical Use Case |
|---|---|---|---|---|---|---|
| `micro` | — | Nova Micro | Llama 3.1 8B | — | — | Classification, extraction, yes/no, simple Q&A |
| `lite` | Haiku 4.5 | Nova Lite, Nova 2 Lite | 4 Scout 17B | — | — | Summarization, chat, moderate tasks |
| `mid` | Sonnet 4.5, 4.6 | Nova Pro | Llama 3.1 70B, 3.3 70B, 4 Maverick 17B | Pixtral Large | — | General-purpose, coding, analysis |
| `heavy` | Opus 4.1, 4.5, 4.6 | — | — | — | — | Complex reasoning, long documents |
| `reasoning` | Opus 4.7 | — | — | — | DeepSeek R1 | Multi-step reasoning, math, planning |

**Registry population strategy:**
1. **Static defaults**: Ship with a built-in catalog of all current Bedrock models (updated with each SDK release)
2. **Dynamic refresh**: Optionally call `bedrock:ListFoundationModels` and `pricing:GetProducts` on startup to get latest models and pricing
3. **User overrides**: Allow users to add custom/imported models or override tier assignments
4. **Historical enrichment**: Merge in historical latency, error rate, and quality scores from the metrics store

### 7.2 Request Analyzer

Classifies incoming requests to determine routing requirements. Runs locally with zero API calls for sub-millisecond overhead.

```python
@dataclass
class RequestAnalysis:
    complexity: str              # "simple" | "moderate" | "complex" | "reasoning"
    complexity_score: float      # 0.0 - 1.0 continuous score
    estimated_input_tokens: int
    estimated_output_tokens: int
    requires_vision: bool
    requires_tool_use: bool
    requires_streaming: bool
    requires_long_context: bool  # Estimated tokens > 32K
    requires_extended_thinking: bool
    is_code_task: bool
    is_conversational: bool
    is_multi_turn: bool          # Has conversation history
    conversation_turn_count: int
    language: str                # Detected language (for future multi-language routing)
    urgency: str                 # "real-time" | "near-real-time" | "batch-eligible"
    cache_benefit_score: float   # 0.0 - 1.0, how much this request benefits from caching
    content_sensitivity: str     # "low" | "medium" | "high" (for guardrails routing)
```

**Scoring dimensions (15 dimensions, extended from LiteLLM's 7):**

| # | Dimension | Weight | Detection Method |
|---|---|---|---|
| 1 | Token count | 0.07 | tiktoken estimation on input |
| 2 | Code presence | 0.12 | Backticks, language keywords, import statements |
| 3 | Reasoning markers | 0.14 | "step by step", "analyze", "compare", "evaluate", "prove" |
| 4 | Technical depth | 0.10 | Domain-specific terminology density |
| 5 | Simple indicators | 0.05 | Greetings, definitions, yes/no patterns |
| 6 | Multi-step patterns | 0.08 | Numbered lists, "first...then", sequential instructions |
| 7 | Tool use signals | 0.09 | Function calling patterns, JSON schema references |
| 8 | Document analysis | 0.08 | References to documents, "attached", long context indicators |
| 9 | Conversation depth | 0.06 | History length, follow-up patterns, pronoun references |
| 10 | AWS specificity | 0.06 | AWS service names, ARNs, CloudFormation, CDK, IAM patterns |
| 11 | Mathematical/logical | 0.08 | Equations, formal logic, proof requests, optimization problems |
| 12 | Creative/open-ended | 0.07 | Story writing, brainstorming, "imagine", "create" |

**Complexity thresholds (configurable):**
- `simple`: score < 0.25
- `moderate`: 0.25 <= score < 0.55
- `complex`: 0.55 <= score < 0.80
- `reasoning`: score >= 0.80 OR 2+ reasoning markers detected

**Multi-turn adaptation (inspired by NVIDIA Blueprint):**
For multi-turn conversations, the analyzer re-evaluates complexity on each turn. A conversation that starts with a simple question may shift to complex reasoning. The router can switch models mid-conversation when the complexity tier changes, while maintaining conversation context through the message history.

### 7.3 Context Window Validator

Pre-validates that the request fits within the candidate model's context window before sending it. This prevents wasted API calls and latency from context-too-long errors.

```python
class ContextWindowValidator:
    def validate(self, messages, model: BedrockModel) -> ValidationResult:
        estimated_tokens = self.estimate_tokens(messages)
        if estimated_tokens > model.max_input_tokens:
            return ValidationResult(
                valid=False,
                estimated_tokens=estimated_tokens,
                model_limit=model.max_input_tokens,
                suggestion="truncate" | "switch_model"  # Suggest a model with larger context
            )
        return ValidationResult(valid=True, estimated_tokens=estimated_tokens)
```

If validation fails, the strategy engine automatically considers only models with sufficient context windows, or triggers a `context_window_fallback` if configured.

### 7.4 Historical Metrics Store

Stores and retrieves historical performance data for routing decisions. Pluggable backend — works with DynamoDB, local SQLite, or in-memory for Lambda.

```python
@dataclass
class ModelMetrics:
    model_id: str
    window: str                  # "1h" | "24h" | "7d"
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    avg_ttft_ms: float
    avg_output_tps: float        # Tokens per second
    error_rate: float            # 0.0 - 1.0
    throttle_rate: float         # 429 rate
    avg_cost_per_request: float
    avg_quality_score: float     # From judge evaluations (0.0 - 1.0)
    cache_hit_rate: float
    sample_count: int
```

**Store backends:**
- **In-memory** (default): Sliding window of last N requests per model. Good for Lambda (resets on cold start, warms up quickly).
- **DynamoDB**: Persistent storage with TTL. Good for shared state across instances.
- **SQLite**: Local persistent storage. Good for single-instance deployments.
- **Custom**: Implement the `MetricsStore` protocol to plug in any backend.

## 8. Routing Strategies

### 8.1 Cost-Optimized Strategy

Routes to the cheapest model that meets the complexity requirement.

```
simple    -> Nova Micro ($0.000035/1K in) or Llama 8B
moderate  -> Nova Lite ($0.00006/1K in) or Haiku 4.5
complex   -> Sonnet 4.5 or Nova Pro
reasoning -> Opus 4.7 or DeepSeek R1
```

**Inference tier selection:** Always uses Standard tier. Falls back to Flex tier for batch-eligible requests (50% cheaper). Never selects Priority tier.

**CRIS profile selection:** Prefers CRIS profiles (cross-region) for higher availability at no extra cost.

**Prompt caching:** If the request has a long system prompt or conversation history, prefers cache-capable models even if slightly more expensive, when estimated cache savings exceed the price delta.

### 8.2 Latency-Optimized Strategy

Routes to the model with lowest expected latency that meets complexity.

Factors considered:
- Historical P50 latency from metrics store
- Historical TTFT (time to first token)
- Model size (smaller models = faster inference)
- CRIS profile availability (cross-region = less queue time)
- Prompt caching eligibility (cached prefix = faster TTFT)
- Inference tier: Prefers Priority tier (up to 25% better OTPS) for real-time requests

### 8.3 Quality-Optimized Strategy

Routes to the model with highest historical quality scores from your own evaluation data.

```python
def score_quality(model, analysis, metrics):
    if metrics.sample_count >= MIN_SAMPLES:
        return metrics.avg_quality_score  # From judge evaluations
    else:
        return TIER_QUALITY_HEURISTIC[model.tier]  # Fallback heuristic
```

Falls back to tier-based heuristic when insufficient historical data exists:
- micro: 0.55, lite: 0.70, mid: 0.82, heavy: 0.90, reasoning: 0.93

### 8.4 Balanced Strategy (recommended for production)

Composite score combining cost, latency, and quality:

```
composite = w_cost * cost_score + w_latency * latency_score + w_quality * quality_score
```

Default weights: cost=0.4, latency=0.3, quality=0.3

User-configurable per request or globally. The strategy normalizes each dimension to 0-1 before weighting.

### 8.5 Budget-Constrained Strategy

Like balanced, but enforces a per-request cost ceiling:

```python
def select(self, analysis, budget_cents):
    candidates = self.filter_by_budget(analysis, budget_cents)
    if not candidates:
        raise BudgetExceededError(f"No model under ${budget_cents/100:.4f}")
    return self.balanced.select_from(candidates, analysis)
```

Supports rolling budget windows: "max $5/hour per user", "max $100/day per team".

### 8.6 Tag-Based Strategy

Routes based on request tags for access control and tiering:

```python
router.converse(
    messages=[...],
    routing={"tags": ["paid-tier", "team-alpha"]}
)
```

Configuration maps tags to allowed models:
```yaml
tag_routing:
  paid-tier: ["us.anthropic.claude-sonnet-4-6", "us.amazon.nova-pro-v1:0"]
  free-tier: ["us.amazon.nova-micro-v1:0", "us.amazon.nova-lite-v1:0"]
  team-alpha: ["us.anthropic.*"]  # Glob patterns supported
```

### 8.7 Conditional Strategy

Routes based on arbitrary metadata conditions (inspired by Portkey):

```yaml
conditional_routing:
  - condition: "metadata.user_tier == 'enterprise'"
    strategy: quality-optimized
  - condition: "metadata.region == 'eu'"
    models: ["eu.anthropic.*", "eu.amazon.*"]  # EU CRIS profiles only
  - condition: "metadata.department == 'finance'"
    guardrail_id: "gr-finance-pii"  # Enforce specific guardrail
  - default:
    strategy: cost-optimized
```

### 8.8 Custom Strategy Plugin

Users can implement their own strategy by subclassing:

```python
from bedrock_smart_router import RoutingStrategy, RequestAnalysis, BedrockModel

class MyCustomStrategy(RoutingStrategy):
    def score(self, model: BedrockModel, analysis: RequestAnalysis) -> float:
        # Your custom scoring logic
        if analysis.is_code_task and "anthropic" in model.family:
            return 0.95  # Prefer Anthropic for code
        return 0.5

router = BedrockRouter(strategy=MyCustomStrategy())
```

### 8.9 Named Presets

One-word shortcuts for common routing profiles.  A preset bundles a strategy, weights, and constraints into a single parameter:

```python
response = router.converse(
    messages=[...],
    routing=RoutingConfig(preset="economy"),
)
```

| Preset | Strategy | Cost Limit | Use Case |
|---|---|---|---|
| `economy` | cost-optimized | $0.002/req | Batch processing, classification, simple Q&A |
| `speed` | latency-optimized | — | Real-time chat, interactive UX |
| `balanced` | balanced (0.4/0.3/0.3) | — | General purpose (default) |
| `quality` | quality-optimized | — | Complex reasoning, analysis, code generation |

Presets are defined in ``ROUTING_PRESETS`` and can be extended by users.  Explicit fields in ``RoutingConfig`` override the preset defaults:

```python
# Economy preset but restricted to Anthropic models
routing=RoutingConfig(preset="economy", preferred_family="anthropic")
```


## 9. Reliability and Fallback System

### 9.1 Circuit Breaker (new — not in initial proposal)

Implements the circuit breaker pattern (inspired by Portkey) to prevent cascading failures:

```
States:
  CLOSED  -> Normal operation. Requests flow through.
  OPEN    -> Model is failing. Requests immediately skip to fallback.
  HALF-OPEN -> After cooldown, allow one probe request to test recovery.

Transitions:
  CLOSED -> OPEN:     When failure_count >= threshold within window
  OPEN -> HALF-OPEN:  After cooldown_seconds elapsed
  HALF-OPEN -> CLOSED: If probe request succeeds
  HALF-OPEN -> OPEN:   If probe request fails (reset cooldown)
```

Configuration:
```python
circuit_breaker:
  failure_threshold: 5        # Failures before opening
  window_seconds: 60          # Sliding window for counting failures
  cooldown_seconds: 30        # How long to stay open before probing
  throttle_cooldown_seconds: 10  # Shorter cooldown for 429s (transient)
  half_open_max_requests: 1   # Probe requests in half-open state
```

Separate circuit breakers per model AND per inference tier. A model might be throttled on Standard tier but available on Priority tier.

### 9.2 Multi-Level Fallback Chain

```
Level 1: Primary model (selected by strategy)
    | fails or circuit breaker open
    v
Level 2: Same-family downgrade (e.g., Sonnet -> Haiku)
    | fails
    v
Level 3: Cross-family equivalent tier (e.g., Sonnet -> Nova Pro)
    | fails
    v
Level 4: CRIS profile retry (try cross-region inference profile)
    | fails
    v
Level 5: Inference tier switch (Standard -> Priority if available)
    | fails
    v
Level 6: Default safe model (Nova Lite - cheap, fast, always available)
    | fails
    v
Level 7: Return error to caller with full fallback trace
```

### 9.3 Specialized Fallbacks

**Context window fallback:** When estimated tokens exceed model limit, automatically select a model with larger context window.

**Content policy fallback:** When a model refuses content (content filter triggered), fall back to a model with different content policies or route through Bedrock Guardrails for pre-screening.

**Timeout fallback:** When a request exceeds the configured timeout, cancel and retry on a faster (smaller) model.

### 9.4 Retry Policy

```python
@dataclass
class RetryPolicy:
    max_retries: int = 3
    backoff_base_seconds: float = 0.5
    backoff_max_seconds: float = 8.0
    backoff_multiplier: float = 2.0
    retryable_errors: list[str] = field(default_factory=lambda: [
        "ThrottlingException",       # 429 - retry with backoff
        "ServiceUnavailableException",  # 503 - retry immediately
        "ModelTimeoutException",     # Timeout - retry or fallback
    ])
    non_retryable_errors: list[str] = field(default_factory=lambda: [
        "ValidationException",       # Bad request - don't retry
        "AccessDeniedException",     # Auth error - don't retry
    ])
```

### 9.5 Graceful No-Models-Match Error

When no models satisfy the routing constraints, the router raises a ``NoModelsMatchError`` instead of a generic exception.  The error includes structured feedback so the caller knows exactly what to fix:

```python
from bedrock_smart_router import NoModelsMatchError

try:
    response = router.converse(
        messages=[...],
        routing=RoutingConfig(preset="economy", preferred_family="nonexistent"),
    )
except NoModelsMatchError as e:
    print(e.constraints)    # {"complexity": "simple", "preferred_family": "nonexistent", ...}
    print(e.rejections)     # [ModelRejection("nova-micro", ["family amazon != nonexistent"]), ...]
    print(e.suggestions)    # ["Remove preferred_family='nonexistent' to consider all families"]
    print(e.to_dict())      # Full structured dict for JSON API responses
```

Each ``ModelRejection`` lists the specific reasons a model was excluded: tier too low, cost too high, missing capability, context window too small, excluded by pattern, or wrong family.  The ``suggestions`` list provides actionable fixes based on which constraints were most restrictive.


## 10. Caching Layer

### 10.1 Response Caching (new — not in initial proposal)

Two-tier caching system inspired by LiteLLM and Portkey:

**Tier 1: Exact-match cache**
- Key: hash(model_id + messages + inference_params)
- Storage: in-memory LRU (SDK mode) or Redis/ElastiCache (proxy mode)
- TTL: configurable, default 1 hour
- Hit rate: typically 15-30% for repetitive workloads

**Tier 2: Semantic cache (optional, requires `[embeddings]` extra)**
- Key: embedding vector of the user message
- Similarity threshold: configurable (default 0.95 cosine similarity)
- Storage: Redis 7+ with RediSearch, ElastiCache Valkey 8.2+, or in-memory FAISS
- Hit rate: typically 40-60% for customer support / FAQ workloads
- Adds ~50-100ms latency for embedding computation

```python
router = BedrockRouter(
    cache="memory",           # "memory" | "redis" | "none"
    cache_ttl_seconds=3600,
    semantic_cache=True,      # Requires [embeddings] extra
    semantic_threshold=0.95,
)
```

**Cache-aware routing:** The router checks the cache BEFORE running the strategy engine. Cache hits bypass model selection entirely, returning the cached response with zero Bedrock API cost.

### 10.2 Bedrock Prompt Caching Integration

Distinct from response caching — this leverages Bedrock's native prompt caching feature where the model caches the prefix (system prompt + early conversation turns) server-side.

```python
def estimate_cache_benefit(messages, system_prompt, model):
    if not model.supports_prompt_caching:
        return 0.0
    cacheable_tokens = estimate_tokens(system_prompt) + estimate_prefix_tokens(messages)
    cache_read_savings = cacheable_tokens * (model.pricing.input - model.pricing.cache_read) / 1000
    return cache_read_savings  # Dollar savings per request
```

When the cache benefit exceeds a threshold, the strategy engine boosts the score of cache-capable models.


## 11. Observability and Metrics

### 11.1 Routing Decision Log

Every routing decision is logged as a structured event:

```json
{
  "timestamp": "2026-04-20T10:30:00Z",
  "request_id": "req_abc123",
  "strategy": "balanced",
  "complexity_detected": "complex",
  "complexity_score": 0.72,
  "selected_model": "us.anthropic.claude-sonnet-4-6",
  "inference_tier": "standard",
  "cris_profile": "us.anthropic.claude-sonnet-4-6",
  "candidates_evaluated": 8,
  "candidate_scores": {
    "us.anthropic.claude-sonnet-4-6": {"cost": 0.6, "latency": 0.7, "quality": 0.9, "composite": 0.73},
    "us.amazon.nova-pro-v1:0": {"cost": 0.8, "latency": 0.8, "quality": 0.7, "composite": 0.76}
  },
  "fallback_chain": ["us.anthropic.claude-haiku-4-5-20251001-v1:0", "us.amazon.nova-lite-v1:0"],
  "cache_hit": false,
  "estimated_cost": 0.0045,
  "actual_cost": 0.0042,
  "latency_ms": 1340,
  "ttft_ms": 420,
  "input_tokens": 1250,
  "output_tokens": 380,
  "fallback_used": false,
  "circuit_breaker_skipped": [],
  "tags": ["paid-tier"],
  "tenant_id": "tenant_xyz"
}
```

### 11.2 Metrics Export

**CloudWatch (native):**
- Custom metrics: `BedrockRouter/RoutingDecisions`, `BedrockRouter/CostSavings`, `BedrockRouter/FallbackRate`, `BedrockRouter/CacheHitRate`
- Dimensions: model, strategy, complexity, tenant

**OpenTelemetry (optional):**
- Traces: span per routing decision + span per Bedrock invocation
- Metrics: histograms for latency, counters for requests/errors/fallbacks

**Custom callbacks:**
```python
def my_callback(event: RoutingEvent):
    # Send to your own analytics pipeline
    send_to_datadog(event)

router = BedrockRouter(callbacks=[my_callback])
```

### 11.3 Cost Tracking Dashboard

The router tracks cumulative cost with breakdowns by:
- Model, strategy, complexity tier
- Tenant / team / user (when tags or AIPs are used)
- Actual cost vs. "would-have-cost" (what it would have cost without routing)
- Cache savings (requests served from cache at zero Bedrock cost)


## 12. A/B Testing and Canary Deployments

### 12.1 A/B Testing (new — not in initial proposal)

Split traffic between models to compare quality, cost, and latency in production:

```python
router = BedrockRouter(
    ab_test={
        "name": "sonnet-vs-nova-pro",
        "variants": {
            "control": {"model": "us.anthropic.claude-sonnet-4-6", "weight": 0.5},
            "treatment": {"model": "us.amazon.nova-pro-v1:0", "weight": 0.5},
        },
        "sticky": True,  # Same user always gets same variant (via hash of user_id)
    }
)
```

When A/B testing is active, the strategy engine is bypassed — the A/B splitter selects the variant first. Results are tagged with the variant name for analysis.

### 12.2 Canary Deployments (new — not in initial proposal)

Gradually roll out a new model with automatic rollback:

```python
router = BedrockRouter(
    canary={
        "baseline": "us.anthropic.claude-sonnet-4-6",
        "canary_model": "us.anthropic.claude-sonnet-4-20260401-v1:0",  # New version
        "canary_percentage": 5,       # Start with 5% traffic
        "auto_promote_threshold": {
            "min_requests": 100,
            "max_error_rate": 0.02,
            "max_latency_p95_ms": 3000,
            "min_quality_score": 0.80,
        },
        "auto_rollback_threshold": {
            "error_rate": 0.10,        # Rollback if >10% errors
            "latency_p95_ms": 5000,    # Rollback if P95 > 5s
        },
    }
)
```

### 12.3 Shadow Mode / Traffic Mirroring

Send a copy of production traffic to a secondary model without affecting the primary response:

```python
router = BedrockRouter(
    shadow={
        "primary": "us.anthropic.claude-sonnet-4-6",
        "shadow_model": "us.amazon.nova-pro-v1:0",
        "sample_rate": 0.1,  # Mirror 10% of traffic
    }
)
```

Shadow responses are logged for offline comparison but never returned to the caller. Useful for evaluating a new model before any traffic shift.


## 13. Multi-Tenant Support

### 13.1 Application Inference Profiles (new — not in initial proposal)

Bedrock Application Inference Profiles (AIPs) allow per-tenant cost tracking via custom tags. The router automatically creates and manages AIPs:

```python
router = BedrockRouter(
    multi_tenant={
        "enabled": True,
        "tenant_header": "X-Tenant-ID",       # Extract tenant from request header
        "auto_create_profiles": True,          # Auto-create AIPs per tenant
        "cost_allocation_tags": ["team", "project", "environment"],
    }
)
```

When a request arrives with tenant metadata, the router:
1. Looks up or creates an Application Inference Profile for that tenant+model combination
2. Invokes Bedrock using the AIP ARN instead of the raw model ID
3. Cost Explorer and CloudWatch automatically attribute costs to the tenant via tags

### 13.2 Per-Tenant Budget Enforcement

```yaml
tenant_budgets:
  tenant_alpha:
    max_daily_spend: 50.00
    max_monthly_spend: 1000.00
    on_budget_exceeded: "downgrade"  # "downgrade" | "reject" | "alert"
    downgrade_to_tier: "lite"
  tenant_beta:
    max_daily_spend: 10.00
    on_budget_exceeded: "reject"
```

### 13.3 Per-Tenant Strategy Override

Different tenants can have different routing strategies:

```yaml
tenant_strategies:
  enterprise_tenants:
    strategy: quality-optimized
    allowed_tiers: ["mid", "heavy", "reasoning"]
  free_tier_tenants:
    strategy: cost-optimized
    allowed_tiers: ["micro", "lite"]
    max_cost_per_request: 0.001
```


## 14. Guardrails Integration

### 14.1 Pre-Route Guardrail Check (new — not in initial proposal)

Use Bedrock's ApplyGuardrail API to screen requests BEFORE model selection:

```python
router = BedrockRouter(
    guardrails={
        "pre_route": {
            "guardrail_id": "gr-abc123",
            "guardrail_version": "DRAFT",
            "action_on_block": "reject",  # "reject" | "fallback" | "sanitize"
        },
        "post_route": {
            "guardrail_id": "gr-abc123",
            "action_on_block": "retry_with_sanitized",
        }
    }
)
```

**Pre-route flow:**
1. Run ApplyGuardrail on the input
2. If blocked: reject request or sanitize and continue
3. If PII detected: route to a model with stricter data handling
4. If content sensitivity is high: prefer models with guardrails attached

**Post-route flow:**
1. Run ApplyGuardrail on the model output
2. If blocked: retry with a different model or return sanitized output

### 14.2 Content-Sensitivity-Aware Routing

The request analyzer detects content sensitivity signals (PII patterns, financial data, health information). High-sensitivity requests are automatically routed to models with Bedrock Guardrails attached, even if a cheaper unguarded model would otherwise be selected.


## 15. Security Considerations

- **No credential storage**: The router uses the caller's boto3 session / IAM role. It never stores or manages AWS credentials.
- **No data persistence by default**: Request/response content is not logged or stored unless the user explicitly enables caching or logging.
- **Cache encryption**: When Redis caching is enabled, supports TLS in transit and encryption at rest via ElastiCache encryption.
- **Tenant isolation**: Multi-tenant mode uses Bedrock AIPs for cost isolation. The router does not provide data isolation between tenants — that remains the application's responsibility.
- **Proxy mode authentication**: *(Not yet implemented.)* The planned proxy server would support API key authentication and integration with AWS IAM, Cognito, or custom auth via middleware.
- **Guardrails as security layer**: Pre-route guardrails can enforce content policies, block prompt injection attempts, and redact PII before any model sees the data.


## 16. SDK API Design

### 16.1 Core API

```python
from bedrock_smart_router import BedrockRouter, RoutingConfig

# Initialize with defaults
router = BedrockRouter(region="us-west-2")

# Initialize with full config
router = BedrockRouter(
    region="us-west-2",
    strategy="balanced",
    weights={"cost": 0.4, "latency": 0.3, "quality": 0.3},
    fallback_enabled=True,
    cache="memory",
    metrics_store="memory",
    callbacks=[my_logger],
)

# Basic usage - drop-in for bedrock converse()
response = router.converse(
    messages=[{"role": "user", "content": [{"text": "Hello"}]}],
)

# With routing overrides per request
response = router.converse(
    messages=[{"role": "user", "content": [{"text": "Complex analysis..."}]}],
    routing=RoutingConfig(
        strategy="quality-optimized",
        preferred_family="anthropic",
        required_capabilities=["tool_use"],
        max_cost_per_request=0.01,
        exclude_models=["us.meta.*"],
        tags=["paid-tier"],
        metadata={"user_id": "u123", "team": "engineering"},
    ),
)

# Streaming
for event in router.converse_stream(
    messages=[{"role": "user", "content": [{"text": "Write a story..."}]}],
):
    if "contentBlockDelta" in event:
        print(event["contentBlockDelta"]["delta"]["text"], end="")

# Access routing decision from last call
decision = router.last_routing_decision()
print(f"Model: {decision.selected_model}, Cost: ${decision.actual_cost:.4f}")
```

### 16.2 Async API

```python
from bedrock_smart_router import AsyncBedrockRouter

router = AsyncBedrockRouter(region="us-west-2")

response = await router.converse(
    messages=[{"role": "user", "content": [{"text": "Hello"}]}],
)
```

### 16.3 Model Registry API

```python
# List available models
models = router.list_models()
models = router.list_models(family="anthropic", tier="mid")

# Get model details
model = router.get_model("us.anthropic.claude-sonnet-4-6")
print(model.pricing, model.capabilities)

# Register a custom/imported model
router.register_model(
    model_id="my-custom-model",
    tier="mid",
    capabilities={"tool_use": True, "vision": False},
    pricing={"input_per_1k": 0.002, "output_per_1k": 0.008},
)

# Refresh pricing from AWS Pricing API
router.refresh_pricing()
```

### 16.4 Strands Agents SDK API

`SmartRouterModel` implements the Strands `Model` interface, allowing any Strands `Agent` to use the smart router as its model provider. All routing intelligence — complexity analysis, strategy selection, fallbacks, circuit breakers, CRIS, inference tiers, caching, guardrails — is applied transparently on every agent call.

```python
from strands import Agent, tool
from bedrock_smart_router.strands_model import SmartRouterModel

# Basic usage — all routing is automatic
model = SmartRouterModel(router_config={"region": "us-west-2"})
agent = Agent(model=model)
response = agent("Explain quantum computing")

# Inspect routing decision after each call
d = model.last_routing_decision
print(f"Model: {d.selected_model}, Cost: ${d.actual_cost:.6f}")

# Routing presets
model = SmartRouterModel(
    router_config={"region": "us-west-2"},
    routing_preset="quality",       # economy | speed | balanced | quality
    preferred_family="anthropic",   # Optional: restrict to a family
    max_cost_per_request=0.05,      # Optional: cost ceiling
)

# Tool use — Strands handles the agent loop, router picks tool-capable models
@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"22°C and sunny in {city}"

agent = Agent(model=model, tools=[get_weather])
response = agent("What's the weather in Seattle?")

# Bring your own router — pass a pre-configured BedrockRouter
from bedrock_smart_router import BedrockRouter
router = BedrockRouter.create({
    "region": "us-west-2",
    "strategy": "cost-optimized",
    "cache": {"enabled": True, "ttl": 300},
})
model = SmartRouterModel(router=router)

# Runtime config changes — switch routing mid-conversation
model.update_config(routing_preset="economy")
response = agent("Simple question")
model.update_config(routing_preset="quality")
response = agent("Complex analysis")
```

**How it works internally:**

1. Strands calls `model.stream(messages, tool_specs, system_prompt)` on each agent loop iteration
2. `SmartRouterModel` converts Strands types to Bedrock Converse format (messages pass through as-is, tool_specs are wrapped in `toolConfig`, system_prompt becomes `system`)
3. The sync `BedrockRouter.converse_stream()` runs in a background thread via `asyncio.to_thread` with a callback queue — the same pattern Strands' own `BedrockModel` uses
4. Bedrock stream events pass through untouched — they're already valid Strands `StreamEvent`s (the formats are identical by design)
5. The router's `routing_decision` event is captured but not forwarded to Strands — it's stored on `model.last_routing_decision` for observability
6. Error mapping converts Bedrock `ThrottlingException` → Strands `ModelThrottledException` and context overflow → `ContextWindowOverflowException`, so the Strands agent loop handles retries correctly

**Supported Strands features:**

| Feature | Supported | Notes |
|---|---|---|
| Text generation | ✅ | Streaming and non-streaming |
| Tool use / function calling | ✅ | Router picks tool-capable models automatically |
| Multi-turn conversations | ✅ | Each turn independently routed by complexity |
| Structured output | ✅ | Via `agent.structured_output(PydanticModel, ...)` |
| System prompts | ✅ | String or content block format |
| Reasoning content | ✅ | Extended thinking blocks pass through |
| Guardrail redaction | ✅ | Bedrock guardrail events forwarded to Strands |


## 17. Proxy Mode API Design

> **⚠️ NOT YET IMPLEMENTED** — This section describes the planned proxy mode API. It has not been built. All functionality below is a design proposal for a future release.

### 17.1 OpenAI-Compatible Endpoint

```
POST /v1/chat/completions
Authorization: Bearer <api-key>

{
  "model": "auto",                    // "auto" = use strategy engine
  "messages": [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Explain quantum computing."}
  ],
  "stream": true,
  "extra_body": {
    "routing": {
      "strategy": "balanced",
      "tags": ["paid-tier"],
      "max_cost": 0.01,
      "preferred_family": "anthropic"
    }
  }
}
```

The proxy translates OpenAI format to Bedrock Converse API internally.

### 17.2 Native Bedrock Endpoint

```
POST /v1/bedrock/converse
Authorization: Bearer <api-key>

{
  "messages": [...],
  "system": [...],
  "routing": {
    "strategy": "cost-optimized"
  }
}
```

Passes through to Bedrock Converse API format directly.

### 17.3 Admin Endpoints

```
GET  /admin/models          # List registered models with health status
GET  /admin/metrics         # Prometheus-format metrics
GET  /admin/circuit-breakers # Circuit breaker states
POST /admin/config          # Hot-reload routing configuration
GET  /admin/ab-tests        # Active A/B test results
```


## 18. Configuration Schema

```yaml
# bedrock-router-config.yaml

region: us-west-2

# Default routing strategy
strategy: balanced
weights:
  cost: 0.4
  latency: 0.3
  quality: 0.3

# Model registry overrides
models:
  tier_overrides:
    "us.amazon.nova-micro-v1:0": "micro"
    "my-distilled-model": "lite"
  excluded_models:
    - "us.meta.llama3-1-8b-instruct-v1:0"  # Exclude specific models globally

# Fallback configuration
fallback:
  enabled: true
  max_fallback_depth: 5
  default_safe_model: "us.amazon.nova-lite-v1:0"
  context_window_fallback: true
  content_policy_fallback: true
  timeout_fallback:
    enabled: true
    timeout_seconds: 30
    fallback_to_tier: "lite"

# Circuit breaker
circuit_breaker:
  enabled: true
  failure_threshold: 5
  window_seconds: 60
  cooldown_seconds: 30
  throttle_cooldown_seconds: 10

# Retry policy
retry:
  max_retries: 3
  backoff_base: 0.5
  backoff_max: 8.0
  backoff_multiplier: 2.0

# Caching
cache:
  type: memory          # memory | redis | none
  ttl_seconds: 3600
  max_entries: 10000    # For memory cache
  redis_url: null       # For redis cache
  semantic_cache:
    enabled: false
    threshold: 0.95
    embedding_model: "amazon.titan-embed-text-v2:0"

# Observability
observability:
  log_routing_decisions: true
  cloudwatch:
    enabled: true
    namespace: "BedrockSmartRouter"
  opentelemetry:
    enabled: false
    endpoint: null
  callbacks: []

# A/B testing
ab_test: null

# Canary deployment
canary: null

# Multi-tenant
multi_tenant:
  enabled: false
  tenant_header: "X-Tenant-ID"
  auto_create_profiles: false

# Guardrails
guardrails:
  pre_route: null
  post_route: null

# Tag routing
tag_routing: {}

# Conditional routing
conditional_routing: []

# Budget enforcement
budgets: {}

# Metrics store
metrics_store:
  type: memory          # memory | dynamodb | sqlite
  dynamodb_table: null
  retention_hours: 168  # 7 days
```


## 19. Implementation Plan

### Phase 1: Core Foundation

| Component | Description | Priority |
|---|---|---|
| `model_registry.py` | Model catalog with capabilities, pricing, tier mapping | P0 |
| `request_analyzer.py` | Zero-API-call complexity classifier (15 dimensions) | P0 |
| `strategy_engine.py` | Cost-optimized, latency-optimized, balanced strategies | P0 |
| `context_validator.py` | Pre-call context window validation | P0 |
| `fallback_handler.py` | Multi-level fallback chain | P0 |
| `circuit_breaker.py` | Circuit breaker with per-model state tracking | P0 |
| `retry_handler.py` | Configurable retry with exponential backoff | P0 |
| `router.py` | Main BedrockRouter class (SDK mode) | P0 |
| `config.py` | YAML/dict configuration loader | P0 |

**Milestone:** SDK mode works end-to-end with `pip install bedrock-smart-router`. Users can route requests across Bedrock models with cost/latency/quality optimization, fallbacks, and circuit breakers.

### Phase 2: Intelligence Layer

| Component | Description | Priority |
|---|---|---|
| `metrics_store.py` | Historical metrics with in-memory + DynamoDB backends | P0 |
| `quality_strategy.py` | Quality-optimized routing using historical judge scores | P0 |
| `cache_layer.py` | Exact-match response caching (in-memory) | P0 |
| `budget_strategy.py` | Budget-constrained strategy with per-user/team limits | P1 |
| `tag_strategy.py` | Tag-based routing for access control | P1 |
| `conditional_strategy.py` | Metadata-based conditional routing | P1 |
| `observability.py` | Routing decision logging + CloudWatch integration | P1 |
| `pricing_refresh.py` | Dynamic pricing from AWS Pricing API | P1 |

**Milestone:** Router learns from historical data, enforces budgets, supports tag-based access control, and provides full observability.

### Phase 3: Bedrock-Native Features

| Component | Description | Priority |
|---|---|---|
| `cris_manager.py` | CRIS profile detection and selection | P1 |
| `inference_tier.py` | Standard/Priority/Flex tier-aware routing | P1 |
| `prompt_cache_advisor.py` | Prompt caching benefit estimation | P1 |
| `guardrails_integration.py` | Pre/post-route guardrail checks via ApplyGuardrail API | P1 |
| `aip_manager.py` | Application Inference Profile management for multi-tenant | P1 |
| `distilled_models.py` | Registry support for distilled model variants | P2 |

**Milestone:** Full Bedrock-native awareness. The router leverages every Bedrock feature that affects cost, latency, and quality.

### Phase 4: Advanced Features

| Component | Description | Priority |
|---|---|---|
| `ab_testing.py` | A/B testing with sticky sessions and auto-analysis | P1 |
| `canary.py` | Canary deployments with auto-promote/rollback | P1 |
| `shadow_mode.py` | Traffic mirroring for offline evaluation | P2 |
| `semantic_cache.py` | Embedding-based semantic caching (optional extra) | P2 |
| `semantic_router.py` | Embedding-based intent routing (optional extra) | P2 |
| `custom_strategy.py` | Plugin interface for user-defined strategies | P2 |
| `async_router.py` | AsyncBedrockRouter for async/await usage | P2 |
| `strands_model.py` | Strands Agents SDK Model provider (SmartRouterModel) | P1 |

**Milestone:** Production-grade deployment features. Teams can safely roll out new models with A/B tests and canary deployments. Strands agent developers get routing intelligence via `SmartRouterModel`.

### Phase 5: Proxy Mode ⚠️ *NOT YET IMPLEMENTED*

> Proxy mode is designed but not built. Phases 1–4 (SDK) are complete and production-ready. Proxy mode will be implemented if there is demand from multi-language teams that need an HTTP endpoint.

| Component | Description | Priority | Status |
|---|---|---|---|
| `proxy_server.py` | FastAPI-based proxy with OpenAI-compatible endpoints | P1 | Not started |
| `proxy_auth.py` | API key authentication + IAM integration | P1 | Not started |
| `proxy_admin.py` | Admin endpoints (metrics, config reload, health) | P1 | Not started |
| `redis_cache.py` | Redis/ElastiCache cache backend for proxy mode | P2 | Not started |
| `prometheus_metrics.py` | Prometheus metrics endpoint | P2 | Not started |

**Milestone:** Proxy mode available for multi-language teams. Single deployment serves all applications.


## 20. Differentiation Matrix

| Feature | LiteLLM | OpenRouter | Portkey | Inworld | NVIDIA Blueprint | Bedrock Native | **Bedrock Smart Router** |
|---|---|---|---|---|---|---|---|
| **Bedrock-specific** | No | No | No | No | No | Yes | **Yes** |
| **Open source** | Yes | No | Partial | No | Yes | N/A | **Yes** |
| **SDK mode (no server)** | Yes | No | No | No | No | N/A | **Yes** |
| **Proxy mode** | Yes | Yes (SaaS) | Yes | Yes (SaaS) | Yes | N/A | **Planned (not yet built)** |
| **Zero-dependency core** | No (Redis) | N/A | No | N/A | No (GPU) | N/A | **Yes (boto3 only)** |
| **Lambda-friendly** | Partial | No | No | No | No | Yes | **Yes** |
| Cross-family routing | Generic | Generic | Generic | Yes | Yes | No (single family) | **Yes (Bedrock-aware)** |
| Complexity classification | 7 dimensions | No | No | Yes | Yes (GPU model) | Yes (ML model) | **15 dimensions, zero-API** |
| CRIS profile awareness | No | No | No | No | No | Yes | **Yes** |
| Inference tier routing | No | No | No | No | No | Manual | **Auto (Std/Priority/Flex)** |
| Prompt cache-aware routing | No | No | No | No | No | No | **Yes** |
| Application inference profiles | No | No | No | No | No | Manual | **Yes (auto-manage)** |
| Model distillation awareness | No | No | No | No | No | Separate | **Yes (registry)** |
| Bedrock Guardrails integration | No | No | Own guardrails | No | No | Separate | **Yes (pre/post route)** |
| Circuit breaker | No | No | Yes | Yes | No | No | **Yes** |
| A/B testing | Mirror only | No | Canary only | Yes | No | No | **Yes (A/B + canary + shadow)** |
| Response caching | Yes (Redis) | No | Yes | No | No | No | **Yes (memory + Redis)** |
| Semantic caching | Yes | No | Yes | No | No | No | **Yes (optional)** |
| Budget enforcement | Yes | No | No | No | No | No | **Yes (per-user/team/tenant)** |
| Tag-based routing | Yes | No | Conditional | No | No | No | **Yes** |
| Multi-tenant cost tracking | No | No | No | No | No | Via AIPs | **Yes (auto AIP management)** |
| Historical quality routing | No | No | No | Opaque | No | No | **Yes (your own judge data)** |
| Multi-turn re-routing | No | No | No | No | Yes | No | **Yes** |
| Custom strategy plugins | Yes | No | No | No | Yes (fine-tune) | No | **Yes** |
| Real-time AWS pricing | Community JSON | Markup | No | No | No | N/A | **Yes (AWS Pricing API)** |
| Named presets | No | No | No | No | No | No | **Yes (economy/speed/balanced/quality)** |
| Strands Agents SDK integration | No | No | No | No | No | No | **Yes (SmartRouterModel)** |
| Graceful no-match errors | No | No | No | No | No | No | **Yes (per-model rejections + suggestions)** |


## 21. Strands Agents SDK Integration

### 21.1 Overview

The Bedrock Smart Router provides first-class integration with the [Strands Agents SDK](https://strandsagents.com/) via `SmartRouterModel` — a custom `Model` provider that brings the full routing pipeline to any Strands agent. This means agent developers get automatic model selection, fallbacks, circuit breakers, cost tracking, and all other routing features without changing their agent code.

**Problem:** When building a Strands agent with the default `BedrockModel`, you pick one model and every call goes to it regardless of complexity. A simple "Hello" costs the same as a complex architecture question. There's no fallback if the model is throttled, no cost tracking, and no way to route different turns to different models.

**Solution:** `SmartRouterModel` replaces `BedrockModel` as the Strands model provider. Every agent loop iteration flows through the smart router's 14-step pipeline — complexity analysis, strategy selection, CRIS profile, inference tier, guardrails, fallback chain — before hitting Bedrock.

### 21.2 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Strands Agent                               │
│                                                                   │
│  agent("Explain quantum computing")                               │
│    ↓                                                              │
│  Agent Loop: model.stream(messages, tool_specs, system_prompt)    │
│    ↓                                                              │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │              SmartRouterModel (strands_model.py)             │  │
│  │                                                              │  │
│  │  1. Convert Strands types → Bedrock Converse format          │  │
│  │     • messages: pass through (identical format)              │  │
│  │     • tool_specs → toolConfig: {"tools": [{toolSpec: ...}]}  │  │
│  │     • system_prompt → system: [{"text": "..."}]              │  │
│  │     • tool_choice → toolConfig.toolChoice                    │  │
│  │                                                              │  │
│  │  2. Build RoutingConfig from model config                    │  │
│  │     • preset, strategy, preferred_model, cost limits, etc.   │  │
│  │                                                              │  │
│  │  3. Delegate to BedrockRouter.converse_stream()              │  │
│  │     (runs in background thread via asyncio.to_thread)        │  │
│  │                                                              │  │
│  │  4. Forward Bedrock stream events as Strands StreamEvents    │  │
│  │     (identical format — zero translation overhead)           │  │
│  │                                                              │  │
│  │  5. Capture routing_decision → model.last_routing_decision   │  │
│  │                                                              │  │
│  │  6. Map errors: ThrottlingException → ModelThrottledException│  │
│  │                  context overflow → ContextWindowOverflow     │  │
│  └─────────────────────────────────────────────────────────────┘  │
│    ↓                                                              │
│  Agent Loop: process events, execute tools, feed results back     │
│    ↓                                                              │
│  Response returned to caller                                      │
└─────────────────────────────────────────────────────────────────┘
```

### 21.3 Design Decisions

**Why Bedrock events pass through as-is:** The Strands SDK was designed around Bedrock's Converse API. Strands' `StreamEvent` type uses the exact same event names and structure as Bedrock's `converse_stream` response — `messageStart`, `contentBlockStart`, `contentBlockDelta`, `contentBlockStop`, `messageStop`, `metadata`. This means the adapter has zero translation overhead for stream events.

**Why async-to-sync bridging:** The Strands `Model.stream()` method must be `async`. The `BedrockRouter` is synchronous (it wraps a boto3 client). We use `asyncio.to_thread` with a callback queue — the same pattern Strands' own `BedrockModel` uses internally. The sync router runs in a background thread and pushes events through a queue that the async generator consumes.

**Why filter Strands-internal kwargs:** Strands passes internal kwargs like `invocation_state` through to the model's `stream()` method. These must be filtered out before reaching the Bedrock API, which strictly validates its parameters. The adapter maintains a blocklist (`_STRANDS_ONLY_KWARGS`) of Strands-internal keys.

**Why store routing_decision on the model:** The router appends a `{"routing_decision": ...}` event at the end of the stream. Strands doesn't know about this event type, so the adapter captures it and stores it on `model.last_routing_decision` instead of forwarding it. This gives users observability without breaking the Strands event loop.

### 21.4 Supported Features

| Strands Feature | Support | How It Works |
|---|---|---|
| Text generation (streaming) | ✅ | `router.converse_stream()` → events pass through |
| Text generation (non-streaming) | ✅ | `router.converse()` → converted to streaming events |
| Tool use / function calling | ✅ | `tool_specs` wrapped in `toolConfig`, router picks tool-capable models |
| Multi-turn conversations | ✅ | Each turn independently routed by complexity |
| Structured output | ✅ | `structured_output()` uses tool calling with Pydantic model |
| System prompts (string) | ✅ | Converted to `system: [{"text": "..."}]` |
| System prompts (content blocks) | ✅ | Passed through as `system_prompt_content` |
| Reasoning / extended thinking | ✅ | `reasoningContent` blocks pass through in events |
| Guardrail redaction events | ✅ | Bedrock guardrail events forwarded to Strands |
| Runtime config changes | ✅ | `model.update_config()` changes routing on next call |
| `tool_choice` parameter | ✅ | Forwarded as `toolConfig.toolChoice` |

| Router Feature | Available via SmartRouterModel |
|---|---|
| Routing presets (economy/speed/balanced/quality) | ✅ via `routing_preset` config |
| Per-request cost ceilings | ✅ via `max_cost_per_request` config |
| Preferred model / family | ✅ via `preferred_model` / `preferred_family` config |
| Fallback chains | ✅ automatic — transparent to the agent |
| Circuit breakers | ✅ automatic — failing models skipped |
| CRIS profile selection | ✅ automatic — based on router config |
| Inference tier selection | ✅ automatic — based on complexity and budget |
| Response caching | ✅ automatic — cache hits bypass Bedrock |
| Metrics and observability | ✅ via `model.last_routing_decision` and router callbacks |
| A/B testing / canary / shadow | ✅ via router config — transparent to the agent |
| Tags and metadata | ✅ via `tags` / `metadata` config |
| Guardrails (pre/post route) | ✅ via router config |
| Budget enforcement | ✅ via `max_cost_per_request` and router-level budget rules |

### 21.5 Installation and Dependencies

```bash
pip install bedrock-smart-router[strands]
```

This installs `strands-agents` as an optional dependency. The `SmartRouterModel` class is only importable when `strands-agents` is installed — the core SDK has no dependency on it. The `__init__.py` conditionally exports `SmartRouterModel` when the import succeeds.

### 21.6 Configuration

`SmartRouterModel` accepts two categories of configuration:

**Router-level config** (passed to `BedrockRouter.create()`):
- `router_config`: dict or `RouterConfig` — region, strategy, weights, cache, metrics, CRIS, guardrails, etc.
- `router`: pre-built `BedrockRouter` instance (overrides `router_config`)

**Per-model config** (controls routing behaviour per call):

| Parameter | Type | Default | Description |
|---|---|---|---|
| `streaming` | bool | `True` | Use `converse_stream` (True) or `converse` (False) |
| `routing_preset` | str | `None` | Named preset: `"economy"`, `"speed"`, `"balanced"`, `"quality"` |
| `routing_strategy` | str | `None` | Explicit strategy name (overrides preset) |
| `preferred_model` | str | `None` | Pin a specific Bedrock model ID |
| `preferred_family` | str | `None` | Prefer a model family (e.g. `"anthropic"`) |
| `max_cost_per_request` | float | `None` | Cost ceiling in dollars |
| `max_tokens` | int | `None` | Maximum output tokens (forwarded as `inferenceConfig.maxTokens`) |
| `temperature` | float | `None` | Sampling temperature |
| `top_p` | float | `None` | Nucleus sampling parameter |
| `stop_sequences` | list[str] | `None` | Stop sequences |
| `exclude_models` | list[str] | `None` | Glob patterns of models to exclude |
| `tags` | list[str] | `None` | Tags forwarded to the routing decision |
| `metadata` | dict | `None` | Arbitrary metadata forwarded to the router |

All parameters can be changed at runtime via `model.update_config(**kwargs)`.

---

## 22. References

### Competitor Documentation
- [LiteLLM Router - Load Balancing](https://docs.litellm.ai/docs/routing)
- [LiteLLM Fallbacks and Reliability](https://docs.litellm.ai/docs/proxy/reliability)
- [LiteLLM Tag Routing](https://docs.litellm.ai/docs/proxy/tag_routing)
- [LiteLLM Complexity Router / Auto Routing](https://docs.litellm.ai/docs/proxy/auto_routing)
- [OpenRouter Provider Routing](https://openrouter.ai/docs/guides/features/presets)
- [Portkey AI Gateway](https://portkey.ai/docs/product/ai-gateway)
- [Portkey Circuit Breaker](https://portkey.ai/docs/product/ai-gateway/circuit-breaker)
- [Portkey Conditional Routing](https://portkey.ai/docs/product/ai-gateway/conditional-routing)
- [Portkey Canary Testing](https://portkey.ai/docs/product/ai-gateway/canary-testing)
- [Inworld Router](https://inworld.ai/resources/best-llm-router-ai-gateway) (Source: Inworld AI, March 2026)
- [NVIDIA AI Blueprint for LLM Router](https://developer.nvidia.com/blog/deploying-the-nvidia-ai-blueprint-for-cost-efficient-llm-routing/) (Source: NVIDIA, March 2025)
- [RouteLLM Framework](https://github.com/lm-sys/RouteLLM)

### Amazon Bedrock Documentation
- [Bedrock Intelligent Prompt Routing](https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-routing.html)
- [Bedrock Cross-Region Inference](https://docs.aws.amazon.com/bedrock/latest/userguide/cross-region-inference.html)
- [Bedrock Inference Profiles](https://docs.aws.amazon.com/bedrock/latest/userguide/inference-profiles.html)
- [Bedrock Inference Tiers (Standard/Priority/Flex)](https://aws.amazon.com/bedrock/service-tiers/)
- [Bedrock Prompt Caching](https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-caching.html)
- [Bedrock Guardrails - ApplyGuardrail API](https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails-use-independent-api.html)
- [Bedrock Application Inference Profiles for Multi-Tenant Cost Tracking](https://aws.amazon.com/blogs/machine-learning/manage-multi-tenant-amazon-bedrock-costs-using-application-inference-profiles/)
- [Bedrock Model Distillation](https://aws.amazon.com/bedrock/model-distillation/)
- [Bedrock Latency-Optimized Inference](https://docs.aws.amazon.com/bedrock/latest/userguide/latency-optimized-inference.html)
- [Bedrock CloudWatch Metrics](https://docs.aws.amazon.com/bedrock/latest/userguide/monitoring-cw.html)

### Academic Research on LLM Routing
- [RouteLLM: Learning to Route LLMs with Preference Data](https://arxiv.org/html/2406.18665) (Source: arXiv, 2024)
- [LLMRank: Understanding LLM Strengths for Model Routing](https://arxiv.org/html/2510.01234v1) (Source: arXiv, 2025)
- [Adaptive LLM Routing under Budget Constraints](https://arxiv.org/html/2508.21141v1) (Source: arXiv, 2025)
- [Doing More with Less - Routing Strategies in LLM Systems](https://arxiv.org/html/2502.00409v2) (Source: arXiv, 2025)
- [GMTRouter: Personalized LLM Router over Multi-turn Interactions](https://arxiv.org/html/2511.08590) (Source: arXiv, 2025)
- [Leveraging Uncertainty Estimation for Efficient LLM Routing](https://arxiv.org/html/2502.11021) (Source: arXiv, 2025)
- [Budget and Performance Controllable Multi-LLM Routing](https://arxiv.org/html/2502.20576) (Source: arXiv, 2025)
- [Semantic Caching for LLMs with Domain-Specific Embeddings](https://arxiv.org/html/2504.02268) (Source: arXiv, 2025)

### Industry Analysis
- [Best LLM Router and AI Gateway 2026](https://inworld.ai/resources/best-llm-router-ai-gateway) (Source: Inworld AI, March 2026)
- [Best AI Gateways 2026](https://blog.lavapayments.com/blog/best-ai-gateways) (Source: Lava Payments, February 2026)
- [LLM Gateway Routing Production Multi-Model 2026](https://fordelstudios.com/research/llm-gateway-routing-production-multi-model-2026) (Source: Fordel Studios, 2026)

Content was rephrased for compliance with licensing restrictions.
