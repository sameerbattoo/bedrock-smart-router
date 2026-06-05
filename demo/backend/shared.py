"""Shared configuration, clients, and helpers for the demo backend."""
from __future__ import annotations

import json
import mimetypes
import threading
import time
import uuid
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import boto3

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from bedrock_smart_router import BedrockRouter, RoutingConfig

# ── Configuration ───────────────────────────────────────────────────

REGION = "us-west-2"
BASELINE_MODEL = "global.anthropic.claude-sonnet-4-6"
JUDGE_MODEL = "global.anthropic.claude-opus-4-8"
BASELINE_PRICING = {"input": 3.0, "output": 15.0}

BASELINE_MODELS = {
    "haiku": {"model_id": "global.anthropic.claude-haiku-4-5-20251001-v1:0", "label": "Haiku 4.5"},
    "sonnet": {"model_id": "global.anthropic.claude-sonnet-4-6", "label": "Sonnet 4.6"},
    "opus": {"model_id": "global.anthropic.claude-opus-4-8", "label": "Opus 4.8"},
    "nova-pro": {"model_id": "us.amazon.nova-pro-v1:0", "label": "Nova Pro"},
}

ROUTER_STRATEGIES = ["balanced", "cost-optimized", "latency-optimized", "quality-optimized"]

SUPPORTED_IMAGE_TYPES = {"image/png", "image/jpeg", "image/gif", "image/webp"}
SUPPORTED_DOC_TYPES = {"application/pdf"}

# ── Clients ─────────────────────────────────────────────────────────

session = boto3.Session(region_name=REGION)
bedrock_client = session.client("bedrock-runtime")
router = BedrockRouter.create({"region": REGION, "excluded_models": ["deepseek.*"], "prompt_cache_boost": False, "aip": {"enabled": True, "auto_create": True, "tag_keys": ["tenant", "tier"]}})
executor = ThreadPoolExecutor(max_workers=6)

# ── Temp File Storage ───────────────────────────────────────────────

_TEMP_DIR = Path(tempfile.gettempdir()) / "bsr_demo_uploads"
_TEMP_DIR.mkdir(exist_ok=True)
_TEMP_FILES: dict[str, dict] = {}
_TEMP_TTL = 1800
_TEMP_MAX_FILES = 100  # Maximum number of temp files to prevent unbounded growth


def store_temp_file(file_bytes: bytes, file_type: str, file_name: str) -> str:
    # Evict oldest files if at capacity
    if len(_TEMP_FILES) >= _TEMP_MAX_FILES:
        oldest = sorted(_TEMP_FILES.items(), key=lambda x: x[1]["expires"])[:10]
        for fid, _ in oldest:
            _cleanup_temp_file(fid)
    file_id = str(uuid.uuid4())
    path = _TEMP_DIR / file_id
    path.write_bytes(file_bytes)
    _TEMP_FILES[file_id] = {"path": str(path), "type": file_type, "name": file_name, "expires": time.time() + _TEMP_TTL}
    return file_id


def get_temp_file(file_id: str) -> tuple[bytes | None, str | None, str | None]:
    entry = _TEMP_FILES.get(file_id)
    if not entry:
        path = _TEMP_DIR / file_id
        if path.exists():
            file_bytes = path.read_bytes()
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
    now = time.time()
    expired = [fid for fid, e in _TEMP_FILES.items() if now > e["expires"]]
    for fid in expired:
        _cleanup_temp_file(fid)


def _cleanup_loop():
    while True:
        time.sleep(300)
        _cleanup_expired()

threading.Thread(target=_cleanup_loop, daemon=True).start()

# ── Helpers ─────────────────────────────────────────────────────────

def display_model_name(model_id: str) -> str:
    """Get display name from the router's model registry."""
    model = router.registry.get(model_id)
    if model:
        return model.display_name
    base = model_id.split(".", 1)[1] if "." in model_id and model_id.split(".")[0] in ("us", "eu", "apac", "global", "au", "jp", "ca") else model_id
    model = router.registry.get(base)
    if model:
        return model.display_name
    return model_id


def compute_cost(model_id: str, input_tokens: int, output_tokens: int,
                 cache_read_tokens: int = 0, cache_write_tokens: int = 0) -> float:
    """Compute cost for a model using the registry's pricing.

    Single source of truth for cost calculation across all use-cases.
    Uses the core library's estimate_cost with all 4 token types.
    """
    model = router.registry.get(model_id)
    if not model:
        return 0.0
    return model.pricing.estimate_cost(
        input_tokens, output_tokens,
        cache_read_tokens=cache_read_tokens,
        cache_write_tokens=cache_write_tokens,
    )


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


def judge_response(prompt: str, response_text: str, file_bytes: bytes | None = None, file_type: str | None = None) -> dict:
    """Score a response using LLM-as-judge."""
    from datetime import datetime
    today = datetime.now().strftime("%B %d, %Y")

    judge_prompt = f"""Score the following AI response on a scale of 1-10.
Consider: accuracy, completeness, relevance, clarity, and helpfulness.

Today's date is {today}. Use this to evaluate whether any dates mentioned in the response are reasonable.

IMPORTANT: If the context indicates the agent used documentation tools (search_documentation, read_documentation) to retrieve information, the response is grounded in official sources. Do NOT penalize it for containing information you are unfamiliar with — the agent has access to more current documentation than your training data.

User prompt: {prompt}

AI Response: {response_text}

Respond with ONLY a JSON object: {{"score": <1-10>, "reasoning": "<one sentence explanation>"}}"""
    try:
        content_blocks = [{"text": judge_prompt}]
        if file_bytes and file_type:
            if file_type in SUPPORTED_IMAGE_TYPES:
                fmt = {"image/png": "png", "image/jpeg": "jpeg", "image/gif": "gif", "image/webp": "webp"}
                content_blocks.insert(0, {"image": {"format": fmt.get(file_type, "png"), "source": {"bytes": file_bytes}}})
            elif file_type in SUPPORTED_DOC_TYPES:
                content_blocks.insert(0, {"document": {"format": "pdf", "name": "uploaded_document", "source": {"bytes": file_bytes}}})
        response = bedrock_client.converse(modelId=JUDGE_MODEL, messages=[{"role": "user", "content": content_blocks}])
        text = response["output"]["message"]["content"][0]["text"].strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0]
        parsed = json.loads(text)
        return {"score": float(parsed.get("score", 5)), "reasoning": parsed.get("reasoning", "")}
    except Exception:
        return {"score": 5.0, "reasoning": "Judge error"}


# ── Seed Metrics ────────────────────────────────────────────────────

def seed_metrics():
    """Seed the router's in-memory metrics store with realistic latency data."""
    import random
    from bedrock_smart_router.metrics_store import RequestRecord

    store = router._metrics_store
    if store is None:
        return

    seed_data = [
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
        ("anthropic.claude-opus-4-7", 4500, 0.01, 10),
        # Reasoning models — Kimi and DeepSeek are ~5x slower than Opus 4.7
        ("moonshot.kimi-k2-thinking", 22000, 0.02, 10),
        ("deepseek.r1-v1:0", 25000, 0.03, 10),
    ]

    random.seed(42)
    base_time = time.monotonic()

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

seed_metrics()

# ── Templates ───────────────────────────────────────────────────────

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


# ── Common LLM Call Function ────────────────────────────────────────

def stream_converse(
    client,
    messages: list[dict],
    system_prompt: str = "",
    model_id: str | None = None,
    routing=None,
    on_chunk=None,
    **extra_kwargs,
) -> dict:
    """Call converse_stream on any client and collect metrics.

    This is the core function demonstrating drop-in replacement:
    - For boto3 bedrock client: pass model_id
    - For BedrockRouter: pass routing config (model auto-selected)

    Both use the exact same converse_stream API.

    Returns a dict with: response_text, model_used, latency_ms, ttft_ms,
    input_tokens, output_tokens, cost, and routing-specific fields.
    """
    import time

    # Build kwargs — same structure for both clients
    kwargs: dict[str, Any] = {"messages": messages}
    if model_id:
        kwargs["modelId"] = model_id
    if routing:
        kwargs["routing"] = routing
    if system_prompt:
        kwargs["system"] = [{"text": system_prompt}]
    # Pass through any extra kwargs (e.g., guardrailConfig)
    kwargs.update(extra_kwargs)

    t_start = time.perf_counter()
    output_text = ""
    ttft_ms = None
    input_tokens = output_tokens = 0
    cache_read_tokens = cache_write_tokens = 0
    stop_reason = ""
    decision = None

    # Call converse_stream — handle both boto3 (returns dict with "stream" key)
    # and SmartRouter (returns generator directly)
    response = client.converse_stream(**kwargs)

    # boto3 returns {"stream": EventStream, ...}
    # SmartRouter yields events directly (it's a generator)
    if isinstance(response, dict) and "stream" in response:
        stream = response["stream"]
    else:
        stream = response

    for event in stream:
        # boto3 returns {"stream": EventStream} — unwrap if needed
        if "stream" in event and hasattr(event["stream"], "__iter__"):
            # This is the boto3 response wrapper — iterate the stream
            for stream_event in event["stream"]:
                if "contentBlockDelta" in stream_event:
                    delta = stream_event["contentBlockDelta"].get("delta", {})
                    if "text" in delta:
                        if ttft_ms is None:
                            ttft_ms = (time.perf_counter() - t_start) * 1000
                        output_text += delta["text"]
                        if on_chunk:
                            on_chunk(delta["text"])
                elif "metadata" in stream_event:
                    usage = stream_event["metadata"].get("usage", {})
                    input_tokens = usage.get("inputTokens", 0)
                    output_tokens = usage.get("outputTokens", 0)
                    cache_read_tokens = usage.get("cacheReadInputTokens", 0)
                    cache_write_tokens = usage.get("cacheWriteInputTokens", 0)
                elif "messageStop" in stream_event:
                    stop_reason = stream_event["messageStop"].get("stopReason", "")
            break  # Only one "stream" key in the response
        elif "contentBlockDelta" in event:
            delta = event["contentBlockDelta"].get("delta", {})
            if "text" in delta:
                if ttft_ms is None:
                    ttft_ms = (time.perf_counter() - t_start) * 1000
                output_text += delta["text"]
                if on_chunk:
                    on_chunk(delta["text"])
        elif "metadata" in event:
            usage = event["metadata"].get("usage", {})
            input_tokens = usage.get("inputTokens", 0)
            output_tokens = usage.get("outputTokens", 0)
            cache_read_tokens = usage.get("cacheReadInputTokens", 0)
            cache_write_tokens = usage.get("cacheWriteInputTokens", 0)
        elif "messageStop" in event:
            stop_reason = event["messageStop"].get("stopReason", "")
        elif "routing_decision" in event:
            decision = event["routing_decision"]

    latency_ms = (time.perf_counter() - t_start) * 1000
    if ttft_ms is None:
        ttft_ms = latency_ms

    # Build result
    result = {
        "response_text": output_text,
        "latency_ms": round(latency_ms, 1),
        "ttft_ms": round(ttft_ms, 1),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "cache_read_tokens": cache_read_tokens,
        "cache_write_tokens": cache_write_tokens,
        "stop_reason": stop_reason,
    }

    if decision:
        # Smart Router response — rich metadata
        result.update({
            "model_used": display_model_name(decision.selected_model),
            "model_id_full": decision.cris_profile or decision.selected_model,
            "cost": round(decision.actual_cost or 0, 6),
            "complexity_detected": decision.complexity_detected,
            "complexity_score": decision.complexity_score,
            "strategy_used": decision.strategy_used,
            "inference_tier": decision.inference_tier,
            "fallback_used": decision.fallback_used,
            "fallback_model": decision.fallback_model,
            "cache_hit": decision.cache_hit,
            "candidates_evaluated": decision.candidates_evaluated,
            "routing_overhead_ms": decision.routing_decision_ms,
            "has_multimodal": False,
            "explanation": decision.explanation,
            "cache_read_tokens": decision.prompt_cache_read_tokens,
            "cache_write_tokens": decision.prompt_cache_write_tokens,
        })
    else:
        # Boto3 direct response — compute cost from registry
        cost = compute_cost(model_id or "", input_tokens, output_tokens,
                            cache_read_tokens, cache_write_tokens)
        result.update({
            "model_used": display_model_name(model_id or ""),
            "cost": round(cost, 6),
            "strategy_used": "direct (boto3)",
            "inference_tier": "standard",
            "has_multimodal": False,
        })

    return result


# ── Non-Streaming Converse (for throttle demo) ──────────────────────

def call_converse(
    client,
    messages: list[dict],
    system_prompt: str = "",
    model_id: str | None = None,
    routing=None,
) -> dict:
    """Call converse (non-streaming) on any client and collect metrics.

    Drop-in replacement demo:
    - For boto3 bedrock client: pass model_id
    - For BedrockRouter: pass routing config (uses retry + fallback internally)

    Returns a dict with: response_text, model_used, latency_ms,
    input_tokens, output_tokens, cost, and routing-specific fields.
    """
    import time

    kwargs: dict[str, Any] = {"messages": messages}
    if model_id:
        kwargs["modelId"] = model_id
    if routing:
        kwargs["routing"] = routing
    if system_prompt:
        kwargs["system"] = [{"text": system_prompt}]

    t_start = time.perf_counter()
    response = client.converse(**kwargs)
    latency_ms = (time.perf_counter() - t_start) * 1000

    # Extract response text
    output_text = ""
    for block in response.get("output", {}).get("message", {}).get("content", []):
        if "text" in block:
            output_text += block["text"]

    # Extract usage (includes cache tokens)
    usage = response.get("usage", {})
    input_tokens = usage.get("inputTokens", 0)
    output_tokens = usage.get("outputTokens", 0)
    cache_read_tokens = usage.get("cacheReadInputTokens", 0)
    cache_write_tokens = usage.get("cacheWriteInputTokens", 0)
    stop_reason = response.get("stopReason", "")

    # Build result
    result = {
        "response_text": output_text,
        "latency_ms": round(latency_ms, 1),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "cache_read_tokens": cache_read_tokens,
        "cache_write_tokens": cache_write_tokens,
        "stop_reason": stop_reason,
    }

    # Check for routing decision (Smart Router attaches it)
    decision = response.get("routing_decision")
    if decision:
        result.update({
            "model_used": display_model_name(decision.selected_model),
            "model_id_full": decision.cris_profile or decision.selected_model,
            "cost": round(decision.actual_cost or 0, 6),
            "complexity_detected": decision.complexity_detected,
            "strategy_used": decision.strategy_used,
            "inference_tier": decision.inference_tier,
            "fallback_used": decision.fallback_used,
            "fallback_model": decision.fallback_model,
            "candidates_evaluated": decision.candidates_evaluated,
            "routing_overhead_ms": decision.routing_decision_ms,
            "explanation": decision.explanation,
            "cache_read_tokens": decision.prompt_cache_read_tokens,
            "cache_write_tokens": decision.prompt_cache_write_tokens,
        })
    else:
        cost = compute_cost(model_id or "", input_tokens, output_tokens,
                            cache_read_tokens, cache_write_tokens)
        result.update({
            "model_used": display_model_name(model_id or ""),
            "cost": round(cost, 6),
            "strategy_used": "direct (boto3)",
            "inference_tier": "standard",
        })

    return result
