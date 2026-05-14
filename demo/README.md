# Bedrock Smart Router — Interactive Demo

A side-by-side comparison UI showing the Smart Router vs a baseline model (Claude Sonnet 4.6). Demonstrates cost savings, latency improvements, and intelligent model selection in real-time.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  React Frontend (Vite + Tailwind)                       │
│  - Template prompt selector (easy/medium/complex)       │
│  - Custom prompt input                                  │
│  - Side-by-side response display                        │
│  - Real-time metrics comparison                         │
└────────────────────────┬────────────────────────────────┘
                         │ POST /api/compare
                         ▼
┌─────────────────────────────────────────────────────────┐
│  FastAPI Backend                                        │
│  - Runs baseline (boto3 → Sonnet 4.6) in parallel      │
│  - Runs Smart Router (balanced strategy) in parallel    │
│  - LLM Judge scores both responses                     │
│  - Returns metrics: TTFT, latency, tokens, cost, score │
└─────────────────────────────────────────────────────────┘
```

## Quick Start

### Backend

```bash
cd demo/backend
pip install fastapi uvicorn
uvicorn app:app --reload --port 8000
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
| Model Used | Which model was selected |
| Input Tokens | Tokens in the prompt |
| Output Tokens | Tokens in the response |
| Cost | Estimated cost ($) |
| Quality Score | LLM judge rating (1-10) |
