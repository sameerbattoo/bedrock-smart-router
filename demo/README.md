# Bedrock Smart Router — Interactive Demo

A real-time comparison UI demonstrating intelligent model routing for Amazon Bedrock. See cost savings, latency improvements, and quality-aware model selection in action across multiple use cases.

## Quick Start

```bash
bash demo/start.sh
```

That's it. The script handles everything:
- Checks prerequisites (Python 3.9+, Node.js 18+, AWS credentials)
- Installs the `bedrock-smart-router` package (editable)
- Installs backend Python dependencies
- Installs frontend npm packages (including `@huggingface/transformers` for voice input)
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

### 2. Throttle Handling

Demonstrates automatic retry and fallback when models are throttled. Fires concurrent requests to trigger rate limits and shows how the router gracefully degrades.

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
- **Voice input** — browser-based speech-to-text using Whisper (tiny.en, ~40MB model loaded on first use). Click the mic, speak, and it auto-submits on silence detection.

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
│  - routes_compare.py  → Use-case 1 (compare)               │
│  - routes_throttle.py → Use-case 2 (throttle)              │
│  - routes_strands.py  → Use-case 3 (agents + MCP)          │
│  - shared.py          → Config, clients, judge, helpers     │
│  - LLM Judge (Opus 4.7) scores responses with date context  │
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
├── backend/
│   ├── app.py               # FastAPI app, mounts all routers
│   ├── shared.py            # Config, clients, judge, helpers
│   ├── routes_compare.py    # Use-case 1: Baseline vs Router
│   ├── routes_throttle.py   # Use-case 2: Throttle handling
│   └── routes_strands.py    # Use-case 3: Strands Agents + MCP
└── frontend/
    ├── package.json          # React + Vite + Tailwind + HuggingFace
    ├── src/
    │   ├── App.jsx           # Main app with navigation
    │   ├── components/       # Page components per use-case
    │   └── hooks/
    │       └── useSpeechToText.js  # Whisper speech-to-text hook
    └── dist/                 # Built frontend (served by Vite)
```
