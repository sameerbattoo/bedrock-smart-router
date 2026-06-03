# Bedrock Smart Router — Interactive Demo

A real-time comparison UI demonstrating intelligent model routing for Amazon Bedrock. See cost savings, latency improvements, and quality-aware model selection in action across multiple use cases.

## Quick Start

```bash
bash demo/start.sh
```

That's it. The script handles everything:
- Checks prerequisites (Python 3.9+, Node.js 18+, AWS credentials, uvx, graphviz)
- Installs the `bedrock-smart-router` package (editable)
- Installs backend Python dependencies
- Sets up databases (Text2SQL SQLite, Usage Tracking SQLite)
- Configures Bedrock Guardrail
- Pre-warms MCP server packages
- Installs frontend npm packages
- Builds the frontend
- Kills any existing processes on ports 8000/5173
- Starts backend + frontend
- Runs health checks
- Opens your browser

## Use Cases

### 1. Baseline vs Smart Router (Compare)

Side-by-side comparison of a fixed model (Claude Sonnet 4.6) against the Smart Router with configurable strategies. Includes an LLM judge (Claude Opus 4.7) that scores both responses.

- Template prompts across 4 difficulty levels (simple → reasoning)
- Custom prompt input with file upload (PDF, images)
- Real-time streaming with TTFT, latency, tokens, cost metrics
- Routing decision explainer (complexity scoring, tier mapping, candidate ranking)
- Baseline model selector (Haiku 4.5, Sonnet 4.6, Opus 4.7, Nova Pro)
- Strategy selector (balanced, cost, latency, quality)
- Classifier toggle (heuristic vs ML)

### 2. Throttle Handling

Demonstrates automatic retry and fallback when models are throttled. Fires concurrent requests to trigger rate limits and shows how the router gracefully degrades.

- Concurrent request bursts to trigger throttling
- Visual fallback chain progression
- Circuit breaker state display
- Retry with exponential backoff

### 3. Strands Agents (Multi-turn Chat)

Two Strands SDK agents with MCP tools (AWS Documentation, AWS Diagrams) running side-by-side:
- **Baseline agent**: Fixed model via boto3
- **Smart Router agent**: Auto-routed via `SmartRouterModel`

Features:
- Multi-turn conversation with session persistence
- MCP tool usage (search docs, read docs, generate diagrams)
- Per-turn metrics comparison (TTFT, latency, tokens, cost, accuracy)
- Send to both, baseline only, or router only
- Strategy switching mid-conversation (balanced, cost, latency, quality, preferred model)
- **Voice input** — browser-based speech-to-text using Whisper (tiny.en, ~40MB model loaded on first use)

### 4. Multi-Tenant Routing

Per-tenant model segregation and routing policies. Shows how different tenants get different models based on their configuration (tier, budget, preferred models).

### 5. Semantic Caching (Text2SQL)

Text-to-SQL with semantic cache, FAISS vectors, and chart generation. Demonstrates embedding-based similarity matching for cache hits across paraphrased queries.

### 6. Pre-Route Guardrails

Content safety check BEFORE model selection. Shows how blocked requests cost $0 (model is never invoked). Demonstrates:
- PII anonymization (input sanitization)
- Topic filtering (investment advice, medical diagnosis)
- Content filters (hate, insults, sexual, violence)
- Side-by-side: native boto3 (server-side, model always invoked) vs Smart Router (pre-route, $0 on block)

### 7. Usage & Cost Tracking

Per-user and per-tenant budget enforcement with automatic strategy downgrade or request rejection. Demonstrates the core library's `BudgetTracker` + `SQLiteBudgetStore`.

Features:
- **Scope toggle**: Per User or Per Tenant (Department) tracking
- **Budget rules per tier**: free (reject on exceed), pro (downgrade), enterprise (generous)
- **Rolling 1-hour window**: spend naturally resets as records age out
- **Parallel simulation**: all users fire requests concurrently per round
- **Live dashboard** (always visible): per-entity spend, budget bars, status badges
- **Request log**: prompt, model, complexity, latency, cost, expandable answers
- **Budget enforcement events**: downgrade/reject notifications
- **Persistent tracking**: `SQLiteBudgetStore` survives restarts, auto-creates table + indexes

How it works:
1. Each request tagged with `user_id` and `tier` via routing metadata
2. Cost tracked per-scope via `BudgetTracker` + `SQLiteBudgetStore` (in-memory hot path, async persistence)
3. `BudgetRule` per tier defines `max_hourly_spend` and `on_exceeded` action
4. `on_exceeded: "downgrade"` → switches to cost-optimized strategy (cheapest model for detected complexity)
5. `on_exceeded: "reject"` → request blocked entirely, raises `BudgetExceededError`, $0 cost

### 8. Safe Model Rollouts (A/B Testing, Canary, Shadow)

Full safe rollout lifecycle for model changes. Each mode loads from a JSONC config file (production pattern).

**A/B Testing:**
- Split traffic between control (Sonnet 4.6) and treatment (Haiku 4.5) by weight (50/50)
- Sticky user assignment — same user always gets the same variant (deterministic hash)
- Live traffic split bar and per-variant cost/latency comparison
- Sticky verification panel proves consistent assignment

**Canary Deployment:**
- Roll out a new model at 20% traffic with auto-rollback thresholds
- Baseline model receives 80% of traffic
- Auto-rollback if error rate > 10% or P95 latency exceeds threshold
- Auto-promote after N successful requests with low error rate
- Canary health panel shows status, error rate, P95 latency, and rollback reason

**Shadow Mode:**
- Mirror 30% of traffic to a shadow model asynchronously (thread pool)
- User always sees the primary model's response — zero latency impact
- Shadow responses logged for offline quality comparison
- Side-by-side request log: primary (left) + shadow (right)
- Shadow calls are fire-and-forget — no impact on user-facing latency

Config-driven (JSONC with comments):
```jsonc
{
  "ab_test": {
    "name": "sonnet-vs-haiku",
    "sticky": true,
    "variants": {
      "control": { "model": "anthropic.claude-sonnet-4-6", "weight": 0.5 },
      "treatment": { "model": "anthropic.claude-haiku-4-5-20251001-v1:0", "weight": 0.5 }
    }
  }
}
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  React Frontend (Vite + Tailwind)                           │
│  - Use-case selector (sidebar navigation)                   │
│  - Side-by-side response display with streaming             │
│  - Real-time metrics, charts, and analytics panel           │
│  - Voice-to-text (Whisper via @huggingface/transformers)    │
└──────────────────────────┬──────────────────────────────────┘
                           │ /api/*
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  FastAPI Backend                                            │
│  - routes_compare.py    → Use-case 1 (compare)             │
│  - routes_throttle.py   → Use-case 2 (throttle)            │
│  - routes_strands.py    → Use-case 3 (agents + MCP)        │
│  - routes_multi_tenant  → Use-case 4 (multi-tenant)        │
│  - routes_text2sql.py   → Use-case 5 (semantic cache)      │
│  - routes_guardrails.py → Use-case 6 (guardrails)          │
│  - routes_usage.py      → Use-case 7 (budget tracking)     │
│  - routes_rollout.py    → Use-case 8 (A/B, canary, shadow) │
│  - shared.py            → Config, clients, judge, helpers   │
└─────────────────────────────────────────────────────────────┘
         │                              │
         ▼                              ▼
┌─────────────────┐          ┌─────────────────────┐
│  boto3 Bedrock  │          │  BedrockRouter      │
│  (fixed model)  │          │  (smart routing)    │
└─────────────────┘          └─────────────────────┘
```

## Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.9+ | With pip |
| Node.js | 18+ | With npm |
| AWS credentials | — | Must have Bedrock access in us-west-2 |
| uv/uvx | Latest | For MCP servers (auto-installed by start.sh) |
| Graphviz | Any | For diagram generation (auto-installed by start.sh) |

## Manual Setup (if not using start.sh)

### Backend

```bash
# From project root
pip install -e .
pip install -r demo/backend/requirements.txt

# Start
cd demo/backend
uvicorn app:app --host 0.0.0.0 --port 8000
```

### Frontend

```bash
cd demo/frontend
npm install
npm run dev
```

Open http://localhost:5173

## Metrics Compared

| Metric | Description |
|--------|-------------|
| TTFT | Time to first token (ms) |
| Total Latency | End-to-end response time (ms) |
| Model Used | Which model was selected by the router |
| Complexity | Detected prompt complexity (simple/moderate/complex/reasoning) |
| Input Tokens | Tokens in the prompt |
| Output Tokens | Tokens in the response |
| Cache Tokens | Prompt cache read/write tokens |
| Cost | Estimated cost ($) using registry pricing |
| Accuracy | LLM judge rating (1-10) by Claude Opus 4.7 |
| Routing Overhead | Time spent on routing decision (ms) |

## Voice Input (Use-case 3)

The Strands Agents chat includes browser-based voice-to-text:

- Powered by [Whisper tiny.en](https://huggingface.co/Xenova/whisper-tiny.en) (~40MB, downloaded on first use)
- Runs entirely in-browser via WebGPU/WASM — no server calls for transcription
- Silence detection auto-stops recording after 2 seconds of quiet
- Transcript is auto-submitted to whichever agent target is selected (both/baseline/router)

## Project Structure

```
demo/
├── start.sh                 # One-command setup & launch
├── README.md                # This file
├── prerequisite/
│   ├── setup_all.py         # Runs all prerequisite setup
│   ├── setup_database.py    # Text2SQL SQLite database
│   ├── setup_guardrail.py   # Bedrock Guardrail configuration
│   └── setup_usage_tracking.py  # Usage tracking SQLite table + indexes
├── backend/
│   ├── app.py               # FastAPI app, mounts all routers
│   ├── shared.py            # Config, clients, judge, helpers
│   ├── routes_compare.py    # Use-case 1: Baseline vs Router
│   ├── routes_throttle.py   # Use-case 2: Throttle handling
│   ├── routes_strands.py    # Use-case 3: Strands Agents + MCP
│   ├── routes_multi_tenant.py # Use-case 4: Multi-tenant routing
│   ├── routes_text2sql.py   # Use-case 5: Semantic caching
│   ├── routes_guardrails.py # Use-case 6: Pre-route guardrails
│   ├── routes_usage.py      # Use-case 7: Usage & cost tracking
│   ├── routes_rollout.py    # Use-case 8: Safe model rollouts
│   └── rollout_configs/     # JSONC config files for rollout modes
│       ├── ab_test.jsonc    # A/B test: Sonnet vs Haiku (50/50)
│       ├── canary.jsonc     # Canary: Sonnet baseline + Haiku canary (20%)
│       └── shadow.jsonc     # Shadow: balanced primary + Nova Pro mirror (30%)
└── frontend/
    ├── package.json          # React + Vite + Tailwind + HuggingFace
    ├── src/
    │   ├── App.jsx           # Main app with navigation
    │   ├── components/       # Page components per use-case
    │   │   ├── ComparePage.jsx
    │   │   ├── ThrottlePage.jsx
    │   │   ├── StrandsPage.jsx
    │   │   ├── MultiTenantPage.jsx
    │   │   ├── Text2SQLPage.jsx
    │   │   ├── GuardrailsPage.jsx
    │   │   ├── UsagePage.jsx
    │   │   ├── RolloutPage.jsx
    │   │   └── shared.jsx    # Shared components, constants, API helpers
    │   └── hooks/
    │       └── useSpeechToText.js  # Whisper speech-to-text hook
    └── dist/                 # Built frontend (served by Vite)
```
