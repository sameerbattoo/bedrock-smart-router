"""FastAPI backend for the Smart Router demo.

Uses Server-Sent Events (SSE) to stream results back to the frontend:
1. Streams baseline tokens as they arrive
2. Streams router tokens as they arrive
3. Sends metrics once each completes
4. LLM judge runs in background, sends scores when ready (non-blocking)

Supports: text prompts, file uploads (PDF/images), MCP tool config.
"""
from __future__ import annotations

import asyncio
import json
import time
import uuid
import mimetypes
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, AsyncGenerator

import boto3
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from bedrock_smart_router import BedrockRouter, RoutingConfig

app = FastAPI(title="Bedrock Smart Router Demo")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Configuration ───────────────────────────────────────────────────

# Temp file storage for history re-runs (TTL: 30 minutes)
_TEMP_DIR = Path(tempfile.gettempdir()) / "bsr_demo_uploads"
_TEMP_DIR.mkdir(exist_ok=True)
_TEMP_FILES: dict[str, dict] = {}  # file_id -> {path, type, name, expires}
_TEMP_TTL = 1800  # 30 minutes


def _store_temp_file(file_bytes: bytes, file_type: str, file_name: str) -> str:
    """Store uploaded file temporarily and return a file_id."""
    file_id = str(uuid.uuid4())
    path = _TEMP_DIR / file_id
    path.write_bytes(file_bytes)
    _TEMP_FILES[file_id] = {
        "path": str(path),
        "type": file_type,
        "name": file_name,
        "expires": time.time() + _TEMP_TTL,
    }
    return file_id


def _get_temp_file(file_id: str) -> tuple[bytes | None, str | None, str | None]:
    """Retrieve a temp file by ID. Returns (bytes, type, name) or (None, None, None)."""
    entry = _TEMP_FILES.get(file_id)
    if not entry:
        # Check if file exists on disk (survives backend restarts)
        path = _TEMP_DIR / file_id
        if path.exists():
            # Re-register with default TTL (we don't know original type/name)
            file_bytes = path.read_bytes()
            # Try to guess type from magic bytes
            file_type = "application/pdf" if file_bytes[:4] == b'%PDF' else "image/png"
            _TEMP_FILES[file_id] = {"path": str(path), "type": file_type, "name": "attachment", "expires": time.time() + _TEMP_TTL}
            return file_bytes, file_type, "attachment"
        return None, None, None
    if time.time() > entry["expires"]:
        _cleanup_temp_file(file_id)
        return None, None, None
    path = Path(entry["path"])
    if not path.exists():
        return None, None, None
    return path.read_bytes(), entry["type"], entry["name"]


def _cleanup_temp_file(file_id: str):
    entry = _TEMP_FILES.pop(file_id, None)
    if entry:
        Path(entry["path"]).unlink(missing_ok=True)


def _cleanup_expired():
    """Remove expired temp files (called periodically)."""
    now = time.time()
    expired = [fid for fid, e in _TEMP_FILES.items() if now > e["expires"]]
    for fid in expired:
        _cleanup_temp_file(fid)


# Run cleanup every 5 minutes
def _cleanup_loop():
    while True:
        time.sleep(300)
        _cleanup_expired()

threading.Thread(target=_cleanup_loop, daemon=True).start()

REGION = "us-west-2"
BASELINE_MODEL = "global.anthropic.claude-sonnet-4-6"
JUDGE_MODEL = "global.anthropic.claude-opus-4-7"
# Pricing per 1M tokens for the default baseline model (Sonnet 4.6) — used by legacy endpoint only
BASELINE_PRICING = {"input": 3.0, "output": 15.0}

# Available baseline models for the UI (these are actual boto3 model IDs with geo prefix)
BASELINE_MODELS = {
    "haiku": {"model_id": "global.anthropic.claude-haiku-4-5-20251001-v1:0", "label": "Haiku 4.5"},
    "sonnet": {"model_id": "global.anthropic.claude-sonnet-4-6", "label": "Sonnet 4.6"},
    "opus": {"model_id": "global.anthropic.claude-opus-4-7", "label": "Opus 4.7"},
    "nova-pro": {"model_id": "us.amazon.nova-pro-v1:0", "label": "Nova Pro"},
}

# Available router strategies
ROUTER_STRATEGIES = ["balanced", "cost-optimized", "latency-optimized", "quality-optimized"]

# ── MCP Tool Definitions ───────────────────────────────────────────

MCP_TOOLS = {
    "aws-docs": {
        "id": "aws-docs",
        "name": "AWS Documentation Search",
        "description": "Search and read AWS documentation pages",
        "icon": "📚",
        "tools": [
            {
                "name": "search_documentation",
                "description": "Search AWS documentation for a topic",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "search_phrase": {"type": "string", "description": "Search query"},
                    },
                    "required": ["search_phrase"],
                },
            },
            {
                "name": "read_documentation",
                "description": "Read an AWS documentation page by URL",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "AWS docs URL ending in .html"},
                    },
                    "required": ["url"],
                },
            },
        ],
    },
    "aws-sentral": {
        "id": "aws-sentral",
        "name": "AWS Sentral (Account Search)",
        "description": "Search SFDC accounts, opportunities, and spend data",
        "icon": "🔍",
        "tools": [
            {
                "name": "search_accounts",
                "description": "Search for AWS customer accounts by name",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Account name to search"},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "get_account_spend",
                "description": "Get spend summary for an account",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "account_id": {"type": "string", "description": "SFDC account ID"},
                    },
                    "required": ["account_id"],
                },
            },
        ],
    },
}

# ── Clients ─────────────────────────────────────────────────────────

def _display_model_name(model_id: str) -> str:
    """Get display name from the router's model registry."""
    model = router.registry.get(model_id)
    if model:
        return model.display_name
    return model_id

session = boto3.Session(region_name=REGION)
bedrock_client = session.client("bedrock-runtime")
router = BedrockRouter.create({"region": REGION, "excluded_models": ["deepseek.*", "global.*"]})
executor = ThreadPoolExecutor(max_workers=6)

# ── Pre-seed latency metrics for strategy differentiation ───────────
# Seed realistic latency data so the balanced strategy can factor in
# latency differences between models (quality comes from quality_baseline).

def _seed_metrics():
    """Seed the router's in-memory metrics store with realistic latency data."""
    import time as _time
    import random
    from bedrock_smart_router.metrics_store import RequestRecord

    store = router._metrics_store
    if store is None:
        return

    # Realistic latency per model (based on actual benchmark runs)
    # Haiku and Sonnet have good latency — makes them competitive in balanced strategy
    seed_data = [
        # model_id, avg_latency_ms, error_rate, samples
        ("amazon.nova-micro-v1:0", 350, 0.02, 50),
        ("amazon.nova-lite-v1:0", 800, 0.01, 50),
        ("amazon.nova-2-lite-v1:0", 600, 0.01, 30),
        ("amazon.nova-pro-v1:0", 1500, 0.01, 40),
        ("anthropic.claude-haiku-4-5-20251001-v1:0", 900, 0.002, 60),
        ("anthropic.claude-sonnet-4-6", 1800, 0.003, 60),
        ("anthropic.claude-sonnet-4-5-20250929-v1:0", 2200, 0.005, 40),
        ("meta.llama3-3-70b-instruct-v1:0", 2000, 0.03, 30),
        ("meta.llama4-maverick-17b-instruct-v1:0", 1000, 0.02, 20),
        ("anthropic.claude-opus-4-5-20251101-v1:0", 12000, 0.01, 15),
        ("anthropic.claude-opus-4-6-v1", 10000, 0.01, 15),
        ("anthropic.claude-opus-4-7", 15000, 0.01, 10),
    ]

    random.seed(42)
    base_time = _time.monotonic()

    for model_id, avg_latency, error_rate, samples in seed_data:
        for i in range(samples):
            latency = avg_latency * random.uniform(0.7, 1.3)
            success = random.random() > error_rate
            store.record(RequestRecord(
                model_id=model_id,
                timestamp=base_time - (samples - i) * 60,
                latency_ms=latency,
                ttft_ms=latency * 0.3,
                input_tokens=random.randint(50, 500),
                output_tokens=random.randint(100, 1000),
                cost=random.uniform(0.0001, 0.01),
                success=success,
                is_throttle=not success and random.random() > 0.5,
                strategy="balanced",
                complexity=random.choice(["simple", "moderate", "complex"]),
            ))

_seed_metrics()

# ── Template Prompts ────────────────────────────────────────────────

TEMPLATES = [
    {"id": "simple-1", "label": "Simple: Greeting", "difficulty": "simple", "system_prompt": "Respond in markdown format.", "prompt": "What is Amazon S3 in one sentence?"},
    {"id": "simple-2", "label": "Simple: Definition", "difficulty": "simple", "system_prompt": "Respond in markdown format.", "prompt": "Define serverless computing."},
    {"id": "simple-3", "label": "Simple: Translation", "difficulty": "simple", "system_prompt": "You are a professional translator. Provide accurate translations. Format your response using markdown.", "prompt": "Translate 'Hello, how are you?' to French and Spanish."},
    {"id": "simple-4", "label": "Simple: Fact", "difficulty": "simple", "system_prompt": "Respond in markdown format.", "prompt": "What are the three main types of cloud computing services?"},
    {"id": "medium-1", "label": "Medium: Code Explanation", "difficulty": "medium", "system_prompt": "You are a senior Python developer. Use markdown with code blocks. Keep under 500 words.", "prompt": "Explain how Python decorators work. Include a practical example of a retry decorator with exponential backoff."},
    {"id": "medium-2", "label": "Medium: Architecture", "difficulty": "medium", "system_prompt": "You are a solutions architect. Use markdown with tables and lists. Keep under 500 words.", "prompt": "Compare REST and GraphQL APIs. When would you choose one over the other? Give specific use cases."},
    {"id": "medium-3", "label": "Medium: SQL Query", "difficulty": "medium", "system_prompt": "You are a data engineer. Use markdown code blocks. Keep under 500 words.", "prompt": "Write a SQL query to find the top 5 customers by total revenue in the last 90 days, including their most purchased product category. Use CTEs for clarity."},
    {"id": "medium-4", "label": "Medium: DevOps", "difficulty": "medium", "system_prompt": "You are a DevOps engineer. Use markdown with code blocks. Keep under 500 words.", "prompt": "Write a Dockerfile for a Python FastAPI application with multi-stage build, non-root user, and health check endpoint."},
    {"id": "complex-1", "label": "Complex: System Design", "difficulty": "complex", "system_prompt": "You are a principal engineer. Use markdown with headings and bullet points. Keep under 800 words.", "prompt": "Design a real-time fraud detection system that processes 1 million transactions per second with sub-100ms latency. Include the data pipeline architecture, ML model serving strategy, feature store design, and alerting system."},
    {"id": "complex-2", "label": "Complex: Algorithm", "difficulty": "complex", "system_prompt": "You are a computer science professor. Use markdown code blocks. Keep under 150 lines.", "prompt": "Implement a B-tree in Python with insert, search, and range query operations. The tree should support configurable order (minimum degree). Include proper node splitting and rebalancing. Add type hints and docstrings."},
    {"id": "complex-3", "label": "Complex: Analysis", "difficulty": "complex", "system_prompt": "You are a CTO advisor. Use markdown with tables and lists. Keep under 800 words.", "prompt": "Analyze the trade-offs between microservices and monolithic architecture across these dimensions: team scalability, deployment complexity, data consistency, operational overhead, testing strategy, and cost at different scales. Provide a decision framework."},
    {"id": "complex-4", "label": "Complex: Security", "difficulty": "complex", "system_prompt": "You are a cloud security architect. Use markdown with headings and lists. Keep under 800 words.", "prompt": "Design a zero-trust security architecture for a multi-account AWS organization. Cover identity federation, network segmentation, data encryption, secrets management, and incident response automation."},
    {"id": "reasoning-1", "label": "Reasoning: Math Proof", "difficulty": "reasoning", "system_prompt": "You are a mathematics professor. Think through each step systematically. Prove your answer rigorously using formal logic. Show all intermediate steps. Keep under 1000 words.", "prompt": "Prove that for every positive integer n, the sum 1² + 2² + 3² + ... + n² equals n(n+1)(2n+1)/6. Then derive the closed-form formula for the sum of cubes 1³ + 2³ + ... + n³ and prove it by induction step by step."},
    {"id": "reasoning-2", "label": "Reasoning: Algorithm Design", "difficulty": "reasoning", "system_prompt": "You are an algorithms researcher. Analyze each approach systematically, evaluate trade-offs, and reason through the complexity analysis step by step. Prove correctness. Keep under 1000 words.", "prompt": "Design an algorithm to find the longest increasing subsequence in an array of n integers. Compare and contrast the brute force O(2^n), dynamic programming O(n²), and patience sorting O(n log n) approaches. For each, prove the time complexity, explain why it works, and analyze the space trade-offs."},
    {"id": "reasoning-3", "label": "Reasoning: System Tradeoffs", "difficulty": "reasoning", "system_prompt": "You are a distributed systems architect. Reason through each design decision systematically, analyze the pros and cons of each approach, and explain why certain trade-offs are unavoidable. Think step by step. Keep under 1000 words.", "prompt": "A global e-commerce platform needs to handle flash sales with 10x traffic spikes while maintaining strong consistency for inventory counts. Analyze step by step: Why can't you have both strong consistency and high availability during a network partition? Evaluate three approaches (pessimistic locking, optimistic concurrency with CRDTs, saga pattern) and prove which guarantees each provides."},
    {"id": "reasoning-4", "label": "Reasoning: Logic Puzzle", "difficulty": "reasoning", "system_prompt": "You are a logic and reasoning expert. Work through this problem step by step, showing your deductive reasoning at each stage. Explain why you can eliminate each possibility. Keep under 1000 words.", "prompt": "Five houses in a row are painted different colors. Each owner has a different nationality, drinks a different beverage, smokes a different brand, and keeps a different pet. Given: The Brit lives in the red house. The Swede keeps dogs. The Dane drinks tea. The green house is left of the white house. The green house owner drinks coffee. The Pall Mall smoker keeps birds. The yellow house owner smokes Dunhill. The middle house owner drinks milk. The Norwegian lives in the first house. The Blend smoker lives next to the cat owner. The horse owner lives next to the Dunhill smoker. The Blue Master smoker drinks beer. The German smokes Prince. The Norwegian lives next to the blue house. The Blend smoker has a neighbor who drinks water. Who keeps the fish? Show your complete reasoning."},
]

# ── File & Tool Helpers ─────────────────────────────────────────────

SUPPORTED_IMAGE_TYPES = {"image/png", "image/jpeg", "image/gif", "image/webp"}
SUPPORTED_DOC_TYPES = {"application/pdf"}


def build_content_blocks(prompt: str, file_bytes: bytes | None = None, file_type: str | None = None) -> list[dict]:
    blocks = []
    if file_bytes and file_type:
        if file_type in SUPPORTED_IMAGE_TYPES:
            fmt = {"image/png": "png", "image/jpeg": "jpeg", "image/gif": "gif", "image/webp": "webp"}
            blocks.append({"image": {"format": fmt.get(file_type, "png"), "source": {"bytes": file_bytes}}})
        elif file_type in SUPPORTED_DOC_TYPES:
            blocks.append({"document": {"format": "pdf", "name": "uploaded_document", "source": {"bytes": file_bytes}}})
    blocks.append({"text": prompt})
    return blocks


def build_tool_config(selected_tools: list[str]) -> dict | None:
    if not selected_tools:
        return None
    tools = []
    for tool_id in selected_tools:
        mcp_server = MCP_TOOLS.get(tool_id)
        if not mcp_server:
            continue
        for tool in mcp_server["tools"]:
            tools.append({"toolSpec": {"name": f"{tool_id}__{tool['name']}", "description": tool["description"], "inputSchema": {"json": tool["input_schema"]}}})
    return {"tools": tools} if tools else None


# ── Streaming Runners ───────────────────────────────────────────────

def run_baseline_stream(prompt: str, file_bytes: bytes | None, file_type: str | None, tool_config: dict | None) -> dict:
    """Run baseline with converse_stream, yield chunks via a shared list, return full result."""
    content_blocks = build_content_blocks(prompt, file_bytes, file_type)
    messages = [{"role": "user", "content": content_blocks}]

    kwargs: dict[str, Any] = {"modelId": BASELINE_MODEL, "messages": messages}
    if tool_config:
        kwargs["toolConfig"] = tool_config

    t_start = time.perf_counter()
    response = bedrock_client.converse_stream(**kwargs)

    output_text = ""
    ttft_ms = None
    input_tokens = output_tokens = 0
    stop_reason = ""

    for event in response.get("stream", []):
        if "contentBlockDelta" in event:
            delta = event["contentBlockDelta"].get("delta", {})
            if "text" in delta:
                if ttft_ms is None:
                    ttft_ms = (time.perf_counter() - t_start) * 1000
                output_text += delta["text"]
        elif "metadata" in event:
            usage = event["metadata"].get("usage", {})
            input_tokens = usage.get("inputTokens", 0)
            output_tokens = usage.get("outputTokens", 0)
        elif "messageStop" in event:
            stop_reason = event["messageStop"].get("stopReason", "")

    latency_ms = (time.perf_counter() - t_start) * 1000
    if ttft_ms is None:
        ttft_ms = latency_ms

    cost = (input_tokens * BASELINE_PRICING["input"] + output_tokens * BASELINE_PRICING["output"]) / 1_000_000

    return {
        "response_text": output_text,
        "model_used": _display_model_name(BASELINE_MODEL),
        "latency_ms": round(latency_ms, 1),
        "ttft_ms": round(ttft_ms, 1),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "cost": round(cost, 6),
        "stop_reason": stop_reason,
        "inference_tier": "standard",
        "strategy_used": "direct (boto3)",
        "has_multimodal": file_bytes is not None,
    }


def run_router_stream(prompt: str, file_bytes: bytes | None, file_type: str | None, tool_config: dict | None) -> dict:
    """Run router with converse_stream, return full result with TTFT."""
    content_blocks = build_content_blocks(prompt, file_bytes, file_type)
    messages = [{"role": "user", "content": content_blocks}]

    kwargs: dict[str, Any] = {"messages": messages, "routing": RoutingConfig(strategy="balanced")}
    if tool_config:
        kwargs["tool_config"] = tool_config

    t_start = time.perf_counter()
    output_text = ""
    ttft_ms = None
    chunks = []
    decision = None

    # converse_stream is a generator that yields events
    for event in router.converse_stream(**kwargs):
        if "contentBlockDelta" in event:
            delta = event["contentBlockDelta"].get("delta", {})
            if "text" in delta:
                if ttft_ms is None:
                    ttft_ms = (time.perf_counter() - t_start) * 1000
                text_chunk = delta["text"]
                output_text += text_chunk
                chunks.append(text_chunk)
        elif "routing_decision" in event:
            decision = event["routing_decision"]

    latency_ms = (time.perf_counter() - t_start) * 1000
    if ttft_ms is None:
        ttft_ms = latency_ms

    if decision:
        model_used = decision.selected_model
        input_tokens = decision.input_tokens or 0
        output_tokens = decision.output_tokens or 0
        total_tokens = decision.total_tokens or (input_tokens + output_tokens)
        cost = decision.actual_cost or 0
        complexity = decision.complexity_detected
        complexity_score = decision.complexity_score
        strategy_used = decision.strategy_used
        inference_tier = decision.inference_tier
        stop_reason = decision.stop_reason
        fallback_used = decision.fallback_used
        cache_hit = decision.cache_hit
        candidates_evaluated = decision.candidates_evaluated
        bedrock_latency = decision.bedrock_latency_ms
        # Router overhead = time spent on routing logic (analysis + model selection)
        # Approximated as: TTFT minus what a direct call's TTFT would be
        # Since we can't know direct TTFT, use: total_latency - bedrock_latency - estimated_network
        # Simpler: just report the analysis time which is sub-millisecond
        # For now, use decision's own overhead calculation if bedrock_latency available
        if bedrock_latency and decision.ttft_ms:
            # overhead = ttft - (network estimate ~50ms for same-region)
            # Better: overhead = total - bedrock - network, but network is unknown
            # Most accurate available: total - bedrock gives network+routing combined
            # Since baseline has same network, the TRUE routing cost is:
            # (router_total - router_bedrock) - (baseline_total - baseline_bedrock)
            # But we don't have baseline here. Just show the sub-ms analysis time.
            routing_overhead_ms = None  # Will be computed in frontend from both sides
        else:
            routing_overhead_ms = None
        # Keep our locally measured ttft_ms (includes routing overhead)
        # Don't override with decision.ttft_ms which is Bedrock-internal only
    else:
        model_used = "unknown"
        input_tokens = output_tokens = total_tokens = 0
        cost = 0
        complexity = complexity_score = None
        strategy_used = inference_tier = stop_reason = ""
        fallback_used = cache_hit = False
        candidates_evaluated = 0
        bedrock_latency = None
        routing_overhead_ms = None

    return {
        "response_text": output_text,
        "chunks": chunks,
        "model_used": _display_model_name(model_used),
        "latency_ms": round(latency_ms, 1),
        "ttft_ms": round(ttft_ms, 1),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "cost": round(cost, 6),
        "complexity_detected": complexity,
        "complexity_score": complexity_score,
        "strategy_used": strategy_used,
        "inference_tier": inference_tier,
        "stop_reason": stop_reason,
        "fallback_used": fallback_used,
        "cache_hit": cache_hit,
        "candidates_evaluated": candidates_evaluated,
        "routing_overhead_ms": routing_overhead_ms,
        "has_multimodal": file_bytes is not None,
    }


def judge_response(prompt: str, response_text: str, file_bytes: bytes | None = None, file_type: str | None = None) -> dict:
    """Score a response using LLM-as-judge. Returns {score, reasoning}."""
    judge_prompt = f"""Score the following AI response on a scale of 1-10.
Consider: accuracy, completeness, relevance, clarity, and helpfulness.

User prompt: {prompt}

AI Response: {response_text}

Respond with ONLY a JSON object: {{"score": <1-10>, "reasoning": "<one sentence explanation>"}}"""
    try:
        content_blocks = [{"text": judge_prompt}]
        # Include the attached document/image so the judge can verify accuracy
        if file_bytes and file_type:
            if file_type in SUPPORTED_IMAGE_TYPES:
                fmt = {"image/png": "png", "image/jpeg": "jpeg", "image/gif": "gif", "image/webp": "webp"}
                content_blocks.insert(0, {"image": {"format": fmt.get(file_type, "png"), "source": {"bytes": file_bytes}}})
            elif file_type in SUPPORTED_DOC_TYPES:
                content_blocks.insert(0, {"document": {"format": "pdf", "name": "uploaded_document", "source": {"bytes": file_bytes}}})
        response = bedrock_client.converse(
            modelId=JUDGE_MODEL,
            messages=[{"role": "user", "content": content_blocks}],
        )
        text = response["output"]["message"]["content"][0]["text"].strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0]
        parsed = json.loads(text)
        return {"score": float(parsed.get("score", 5)), "reasoning": parsed.get("reasoning", "")}
    except Exception as e:
        return {"score": 5.0, "reasoning": f"Judge error"}


# ── SSE Streaming Endpoint ──────────────────────────────────────────

@app.post("/api/compare-stream")
async def compare_stream(
    prompt: str = Form(...),
    system_prompt: str = Form(""),
    run_judge: bool = Form(True),
    selected_tools: str = Form("[]"),
    baseline_model: str = Form("sonnet"),
    router_strategy: str = Form("balanced"),
    preferred_model: str = Form(""),
    file_id: str = Form(""),
    file: UploadFile | None = File(None),
):
    """Stream comparison results via Server-Sent Events.

    Event types:
      - baseline_complete: baseline metrics + full response
      - router_complete: router metrics + full response
      - judge_scores: quality scores (arrives later, non-blocking)
    """
    file_bytes = None
    file_type = None
    file_name = None

    # Try to load from file_id (history re-run) first
    if file_id:
        file_bytes, file_type, file_name = _get_temp_file(file_id)

    # Otherwise use the uploaded file
    if not file_bytes and file and file.filename:
        file_bytes = await file.read()
        file_type = file.content_type or mimetypes.guess_type(file.filename)[0]
        file_name = file.filename
        if file_type not in SUPPORTED_IMAGE_TYPES | SUPPORTED_DOC_TYPES:
            raise HTTPException(400, f"Unsupported file type: {file_type}")
        if len(file_bytes) > 20_000_000:
            raise HTTPException(400, "File too large. Maximum 20MB.")
        # Store for future re-runs
        file_id = _store_temp_file(file_bytes, file_type, file_name)

    try:
        tool_ids = json.loads(selected_tools)
    except json.JSONDecodeError:
        tool_ids = []
    tool_config = build_tool_config(tool_ids)

    async def event_stream() -> AsyncGenerator[str, None]:
        loop = asyncio.get_event_loop()
        import queue

        # Send file_id so frontend can store it for re-runs
        if file_id:
            yield f"event: file_stored\ndata: {json.dumps({'file_id': file_id, 'file_name': file_name})}\n\n"
        baseline_q = queue.Queue()
        router_q = queue.Queue()

        def run_baseline_streaming():
            """Run baseline, putting chunks into queue as they arrive."""
            bl_config = BASELINE_MODELS.get(baseline_model, BASELINE_MODELS["sonnet"])
            bl_model_id = bl_config["model_id"]
            # Get pricing from the router's registry
            bl_model = router.registry.get(bl_model_id)
            content_blocks = build_content_blocks(prompt, file_bytes, file_type)
            messages = [{"role": "user", "content": content_blocks}]
            kwargs_bl: dict[str, Any] = {"modelId": bl_model_id, "messages": messages}
            if system_prompt:
                kwargs_bl["system"] = [{"text": system_prompt}]
            if tool_config:
                kwargs_bl["toolConfig"] = tool_config

            t_start = time.perf_counter()
            response = bedrock_client.converse_stream(**kwargs_bl)
            output_text = ""
            ttft_ms = None
            input_tokens = output_tokens = 0
            stop_reason = ""

            for event in response.get("stream", []):
                if "contentBlockDelta" in event:
                    delta = event["contentBlockDelta"].get("delta", {})
                    if "text" in delta:
                        if ttft_ms is None:
                            ttft_ms = (time.perf_counter() - t_start) * 1000
                        output_text += delta["text"]
                        baseline_q.put(("chunk", delta["text"]))
                elif "metadata" in event:
                    usage = event["metadata"].get("usage", {})
                    input_tokens = usage.get("inputTokens", 0)
                    output_tokens = usage.get("outputTokens", 0)
                elif "messageStop" in event:
                    stop_reason = event["messageStop"].get("stopReason", "")

            latency_ms = (time.perf_counter() - t_start) * 1000
            if ttft_ms is None:
                ttft_ms = latency_ms
            cost = bl_model.pricing.estimate_cost(input_tokens, output_tokens) if bl_model else 0
            result = {
                "response_text": output_text, "model_used": _display_model_name(bl_model_id),
                "latency_ms": round(latency_ms, 1), "ttft_ms": round(ttft_ms, 1),
                "input_tokens": input_tokens, "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens, "cost": round(cost, 6),
                "stop_reason": stop_reason, "inference_tier": "standard",
                "strategy_used": "direct (boto3)", "has_multimodal": file_bytes is not None,
            }
            baseline_q.put(("done", result))

        def run_router_streaming():
            """Run router, putting chunks into queue as they arrive."""
            content_blocks = build_content_blocks(prompt, file_bytes, file_type)
            messages = [{"role": "user", "content": content_blocks}]
            strategy = router_strategy if router_strategy in ROUTER_STRATEGIES else "balanced"
            kwargs_rt: dict[str, Any] = {"messages": messages, "routing": RoutingConfig(strategy=strategy, preferred_model=preferred_model if preferred_model else None, explain=True)}
            if system_prompt:
                kwargs_rt["system"] = [{"text": system_prompt}]
            if tool_config:
                kwargs_rt["tool_config"] = tool_config

            t_start = time.perf_counter()
            output_text = ""
            ttft_ms = None
            decision = None
            first_event_time = None

            for event in router.converse_stream(**kwargs_rt):
                if first_event_time is None:
                    first_event_time = time.perf_counter()
                if "contentBlockDelta" in event:
                    delta = event["contentBlockDelta"].get("delta", {})
                    if "text" in delta:
                        if ttft_ms is None:
                            ttft_ms = (time.perf_counter() - t_start) * 1000
                        output_text += delta["text"]
                        router_q.put(("chunk", delta["text"]))
                elif "routing_decision" in event:
                    decision = event["routing_decision"]

            latency_ms = (time.perf_counter() - t_start) * 1000
            if ttft_ms is None:
                ttft_ms = latency_ms

            # Router overhead: computed in frontend as (router_ttft - baseline_ttft)
            routing_overhead = None

            latency_ms = (time.perf_counter() - t_start) * 1000
            if ttft_ms is None:
                ttft_ms = latency_ms

            if decision:
                result = {
                    "response_text": output_text,
                    "model_used": _display_model_name(decision.selected_model),
                    "latency_ms": round(latency_ms, 1),
                    "ttft_ms": round(ttft_ms, 1),
                    "input_tokens": decision.input_tokens or 0,
                    "output_tokens": decision.output_tokens or 0,
                    "total_tokens": decision.total_tokens or 0,
                    "cost": round(decision.actual_cost or 0, 6),
                    "complexity_detected": decision.complexity_detected,
                    "complexity_score": decision.complexity_score,
                    "strategy_used": decision.strategy_used,
                    "inference_tier": decision.inference_tier,
                    "stop_reason": decision.stop_reason,
                    "fallback_used": decision.fallback_used,
                    "cache_hit": decision.cache_hit,
                    "candidates_evaluated": decision.candidates_evaluated,
                    "routing_overhead_ms": decision.routing_decision_ms,
                    "has_multimodal": file_bytes is not None,
                    "strategy_used": decision.strategy_used,
                    "explanation": decision.explanation,
                }
            else:
                result = {"response_text": output_text, "model_used": "unknown", "latency_ms": round(latency_ms, 1), "ttft_ms": round(ttft_ms, 1), "input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "cost": 0, "strategy_used": "balanced"}
            router_q.put(("done", result))

        # Start both in threads
        loop.run_in_executor(executor, run_baseline_streaming)
        loop.run_in_executor(executor, run_router_streaming)

        # Poll queues and yield SSE events
        baseline_done = False
        router_done = False
        baseline_result = None
        router_result = None

        while not (baseline_done and router_done):
            await asyncio.sleep(0.02)  # 20ms poll interval for responsive streaming

            # Drain baseline queue
            while not baseline_q.empty():
                msg_type, data = baseline_q.get_nowait()
                if msg_type == "chunk":
                    yield f"event: baseline_chunk\ndata: {json.dumps({'text': data})}\n\n"
                elif msg_type == "done":
                    baseline_result = data
                    baseline_done = True
                    yield f"event: baseline_complete\ndata: {json.dumps(data, default=str)}\n\n"

            # Drain router queue
            while not router_q.empty():
                msg_type, data = router_q.get_nowait()
                if msg_type == "chunk":
                    yield f"event: router_chunk\ndata: {json.dumps({'text': data})}\n\n"
                elif msg_type == "done":
                    router_result = data
                    router_done = True
                    # Calculate savings
                    if baseline_result and baseline_result["cost"] > 0:
                        data["savings_pct"] = round((1 - data["cost"] / baseline_result["cost"]) * 100, 1)
                        data["latency_improvement_pct"] = round((1 - data["latency_ms"] / baseline_result["latency_ms"]) * 100, 1) if baseline_result["latency_ms"] > 0 else 0
                    yield f"event: router_complete\ndata: {json.dumps(data, default=str)}\n\n"

        # If router finished before baseline, send updated savings
        if router_result and baseline_result and "savings_pct" not in router_result:
            savings = round((1 - router_result["cost"] / baseline_result["cost"]) * 100, 1) if baseline_result["cost"] > 0 else 0
            yield f"event: metrics_update\ndata: {json.dumps({'savings_pct': savings})}\n\n"

        # Judge (non-blocking, after both done)
        if run_judge and baseline_result and router_result:
            judge_bl = loop.run_in_executor(executor, judge_response, prompt, baseline_result["response_text"], file_bytes, file_type)
            judge_rt = loop.run_in_executor(executor, judge_response, prompt, router_result["response_text"], file_bytes, file_type)
            bj, rj = await asyncio.gather(judge_bl, judge_rt)
            yield f"event: judge_scores\ndata: {json.dumps({'baseline_score': bj['score'], 'baseline_reasoning': bj['reasoning'], 'router_score': rj['score'], 'router_reasoning': rj['reasoning']})}\n\n"

        yield f"event: done\ndata: {{}}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"})


# ── Legacy non-streaming endpoint (kept for compatibility) ──────────

@app.post("/api/compare")
async def compare(
    prompt: str = Form(...),
    run_judge: bool = Form(True),
    selected_tools: str = Form("[]"),
    file: UploadFile | None = File(None),
):
    """Non-streaming comparison (legacy). Use /api/compare-stream for better UX."""
    file_bytes = None
    file_type = None
    if file and file.filename:
        file_bytes = await file.read()
        file_type = file.content_type or mimetypes.guess_type(file.filename)[0]
        if file_type not in SUPPORTED_IMAGE_TYPES | SUPPORTED_DOC_TYPES:
            raise HTTPException(400, f"Unsupported file type: {file_type}")
        if len(file_bytes) > 20_000_000:
            raise HTTPException(400, "File too large. Maximum 20MB.")

    try:
        tool_ids = json.loads(selected_tools)
    except json.JSONDecodeError:
        tool_ids = []
    tool_config = build_tool_config(tool_ids)

    loop = asyncio.get_event_loop()
    baseline_future = loop.run_in_executor(executor, run_baseline_stream, prompt, file_bytes, file_type, tool_config)
    router_future = loop.run_in_executor(executor, run_router_stream, prompt, file_bytes, file_type, tool_config)
    baseline_result, router_result = await asyncio.gather(baseline_future, router_future)

    if run_judge:
        judge_b = loop.run_in_executor(executor, judge_response, prompt, baseline_result["response_text"], file_bytes, file_type)
        judge_r = loop.run_in_executor(executor, judge_response, prompt, router_result["response_text"], file_bytes, file_type)
        bj, rj = await asyncio.gather(judge_b, judge_r)
        baseline_result["quality_score"] = bj["score"]
        baseline_result["quality_reasoning"] = bj["reasoning"]
        router_result["quality_score"] = rj["score"]
        router_result["quality_reasoning"] = rj["reasoning"]

    savings_pct = round((1 - router_result["cost"] / baseline_result["cost"]) * 100, 1) if baseline_result["cost"] > 0 else 0
    latency_imp = round((1 - router_result["latency_ms"] / baseline_result["latency_ms"]) * 100, 1) if baseline_result["latency_ms"] > 0 else 0

    return {"baseline": baseline_result, "router": router_result, "savings_pct": savings_pct, "latency_improvement_pct": latency_imp}


# ── Other Endpoints ─────────────────────────────────────────────────

@app.get("/api/templates")
def get_templates():
    return TEMPLATES

@app.get("/api/mcp-tools")
def get_mcp_tools():
    return list(MCP_TOOLS.values())

@app.get("/api/options")
def get_options():
    """Return available baseline models, router strategies, and preferred models from the registry."""
    # Get all models from the router's registry
    all_models = router._registry.all_models
    # Deduplicate by display name (prefer us. over global.)
    seen = {}
    for m in all_models:
        if m.display_name not in seen or not m.model_id.startswith("global."):
            seen[m.display_name] = m

    preferred = [{"id": "", "label": "Auto (router decides)"}]
    for m in seen.values():
        preferred.append({"id": m.model_id, "label": m.display_name})

    return {
        "baseline_models": [{"id": k, "label": v["label"]} for k, v in BASELINE_MODELS.items()],
        "router_strategies": ROUTER_STRATEGIES,
        "preferred_models": preferred,
    }

@app.get("/api/health")
def health():
    return {"status": "ok", "region": REGION, "baseline_model": BASELINE_MODEL}


@app.get("/api/check-file")
def check_file(file_id: str = ""):
    """Check if a temp file still exists (not expired)."""
    if not file_id:
        return {"exists": False}
    fb, _, _ = _get_temp_file(file_id)
    return {"exists": fb is not None}
