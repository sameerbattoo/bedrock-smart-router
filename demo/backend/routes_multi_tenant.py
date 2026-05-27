"""Use-case 4: Multi-Tenant Routing — same prompt, different routing per tenant.

Demonstrates how a single router instance serves multiple tenants with
different SLAs using RoutingConfig metadata, strategies, and budgets.
"""
from __future__ import annotations

import asyncio
import json
import queue
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, AsyncGenerator

from fastapi import APIRouter, Form
from fastapi.responses import StreamingResponse

from shared import (
    router as smart_router, executor, display_model_name,
    judge_response, stream_converse,
)

router = APIRouter()

# ── Tenant Configurations ───────────────────────────────────────────

TENANTS = {
    "enterprise": {
        "name": "Acme Corp",
        "tier": "Enterprise",
        "icon": "🏢",
        "color": "purple",
        "strategy": "quality-optimized",
        "preferred_model": "anthropic.claude-opus-4-7",
        "max_cost_per_request": None,
        "exclude_models": None,
        "tags": None,
        "description": "Premium tier — best quality, preferred model, no cost limits",
        "code": '''# Enterprise tenant — best quality, preferred model
router.converse(
    messages=messages,
    routing=RoutingConfig(
        strategy="quality-optimized",
        preferred_model="anthropic.claude-opus-4-7",
        metadata={"tenant": "acme-corp", "tier": "enterprise"},
        explain=True,
    ),
)''',
    },
    "free": {
        "name": "FreeUser42",
        "tier": "Free",
        "icon": "🎁",
        "color": "green",
        "strategy": "cost-optimized",
        "preferred_model": None,
        "max_cost_per_request": 0.001,
        "exclude_models": [".*opus.*"],
        "tags": None,
        "description": "Free tier — cheapest models, excludes Opus, strict budget",
        "code": '''# Free tenant — cheapest models, strict budget
router.converse(
    messages=messages,
    routing=RoutingConfig(
        strategy="cost-optimized",
        max_cost_per_request=0.001,
        exclude_models=[".*opus.*"],
        metadata={"tenant": "free-user-42", "tier": "free"},
        explain=True,
    ),
)''',
    },
}


def _build_routing_config(tenant_id: str, classifier: str = "heuristic") -> dict[str, Any]:
    """Build routing kwargs for a tenant."""
    tenant = TENANTS[tenant_id]
    config: dict[str, Any] = {
        "strategy": tenant["strategy"],
        "metadata": {"tenant": tenant["name"], "tier": tenant["tier"].lower()},
        "tags": tenant["tags"],
        "classifier": classifier,
    }
    if tenant["preferred_model"]:
        config["preferred_model"] = tenant["preferred_model"]
    if tenant["max_cost_per_request"]:
        config["max_cost_per_request"] = tenant["max_cost_per_request"]
    if tenant["exclude_models"]:
        config["exclude_models"] = tenant["exclude_models"]
    return config


# ── API Endpoints ───────────────────────────────────────────────────

ROUTER_SETUP_CODE = '''from bedrock_smart_router import BedrockRouter, RoutingConfig

# Single router instance — serves ALL tenants
router = BedrockRouter.create({
    "region": "us-west-2",
    "classifier": "heuristic",        # or "ml" for ML-based complexity detection
    "aip": {
        "enabled": True,              # Application Inference Profiles
        "auto_create": True,          # Auto-create per tenant on first request
        "tag_keys": ["tenant", "tier"],  # Tags → Cost Explorer attribution
    },
})'''


@router.get("/multi-tenant/tenants")
async def get_tenants():
    """Return tenant configurations for the UI."""
    return {
        "router_setup_code": ROUTER_SETUP_CODE,
        "tenants": [
            {
                "id": tid,
                "name": t["name"],
                "tier": t["tier"],
                "icon": t["icon"],
                "color": t["color"],
                "strategy": t["strategy"],
                "preferred_model": t["preferred_model"],
                "max_cost_per_request": t["max_cost_per_request"],
                "exclude_models": t["exclude_models"],
                "tags": t["tags"],
                "description": t["description"],
                "code": t["code"],
            }
            for tid, t in TENANTS.items()
        ]
    }


@router.post("/multi-tenant/run")
async def multi_tenant_run(
    prompt: str = Form(...),
    system_prompt: str = Form(""),
    classifier: str = Form("heuristic"),
):
    """Run the same prompt through all tenant configurations with token streaming."""
    import queue

    async def event_stream() -> AsyncGenerator[str, None]:
        loop = asyncio.get_event_loop()
        result_queue: queue.Queue = queue.Queue()

        # Build messages
        messages = [{"role": "user", "content": [{"text": prompt}]}]

        # Fire all tenants in parallel
        for tenant_id in TENANTS:
            routing_config = _build_routing_config(tenant_id, classifier=classifier)
            loop.run_in_executor(
                executor,
                _run_tenant,
                tenant_id,
                messages,
                system_prompt,
                routing_config,
                result_queue,
            )

        # Stream events from queue
        tenants_done = set()
        results = {}

        while len(tenants_done) < len(TENANTS):
            await asyncio.sleep(0.03)
            while not result_queue.empty():
                item = result_queue.get_nowait()
                msg_type = item[0]
                tenant_id = item[1]

                if msg_type == "token":
                    text = item[2]
                    yield f"event: tenant_token\ndata: {json.dumps({'tenant_id': tenant_id, 'text': text})}\n\n"
                elif msg_type == "done":
                    data = item[2]
                    tenants_done.add(tenant_id)
                    results[tenant_id] = data
                    yield f"event: tenant_complete\ndata: {json.dumps({'tenant_id': tenant_id, **data}, default=str)}\n\n"
                elif msg_type == "error":
                    error = item[2]
                    tenants_done.add(tenant_id)
                    yield f"event: tenant_error\ndata: {json.dumps({'tenant_id': tenant_id, 'error': error})}\n\n"

        yield f"event: all_complete\ndata: {{}}\n\n"

        # Judge all responses
        judge_futures = {}
        for tenant_id, result in results.items():
            if result.get("response_text"):
                judge_futures[tenant_id] = loop.run_in_executor(
                    executor, judge_response, prompt, result["response_text"], None, None,
                )

        for tenant_id, future in judge_futures.items():
            try:
                score = await future
                yield f"event: judge_score\ndata: {json.dumps({'tenant_id': tenant_id, 'score': score['score'], 'reasoning': score['reasoning']})}\n\n"
            except Exception:
                pass

        yield f"event: done\ndata: {{}}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


def _run_tenant(
    tenant_id: str,
    messages: list[dict],
    system_prompt: str,
    routing_config: dict[str, Any],
    result_queue: Any,
) -> None:
    """Execute a single tenant's request with token streaming."""
    from bedrock_smart_router.config import RoutingConfig
    import time

    routing = RoutingConfig(**routing_config, explain=True)

    t_start = time.perf_counter()
    ttft_ms = None
    output_text = ""
    input_tokens = output_tokens = 0
    cache_read_tokens = cache_write_tokens = 0

    kwargs: dict[str, Any] = {"messages": messages, "routing": routing}
    if system_prompt:
        kwargs["system"] = [{"text": system_prompt}]

    try:
        response = smart_router.converse_stream(**kwargs)

        for event in response:
            if "contentBlockDelta" in event:
                delta = event["contentBlockDelta"].get("delta", {})
                if "text" in delta:
                    if ttft_ms is None:
                        ttft_ms = (time.perf_counter() - t_start) * 1000
                    output_text += delta["text"]
                    result_queue.put(("token", tenant_id, delta["text"]))
            elif "metadata" in event:
                usage = event["metadata"].get("usage", {})
                input_tokens = usage.get("inputTokens", 0)
                output_tokens = usage.get("outputTokens", 0)
                cache_read_tokens = usage.get("cacheReadInputTokens", 0)
                cache_write_tokens = usage.get("cacheWriteInputTokens", 0)
            elif "routing_decision" in event:
                decision = event["routing_decision"]
                latency_ms = (time.perf_counter() - t_start) * 1000
                cost = decision.actual_cost or 0

                result_queue.put(("done", tenant_id, {
                    "response_text": output_text,
                    "model_used": display_model_name(decision.selected_model),
                    "latency_ms": round(latency_ms, 1),
                    "ttft_ms": round(ttft_ms, 1) if ttft_ms else round(latency_ms, 1),
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "cost": round(cost, 6),
                    "complexity_detected": decision.complexity_detected,
                    "strategy_used": decision.strategy_used,
                    "fallback_used": decision.fallback_used,
                    "explanation": decision.explanation,
                }))
                return

        # If no routing_decision event (shouldn't happen), still send done
        latency_ms = (time.perf_counter() - t_start) * 1000
        result_queue.put(("done", tenant_id, {
            "response_text": output_text,
            "model_used": "unknown",
            "latency_ms": round(latency_ms, 1),
            "ttft_ms": round(ttft_ms or latency_ms, 1),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost": 0,
        }))
    except Exception as e:
        result_queue.put(("error", tenant_id, str(e)[:200]))
