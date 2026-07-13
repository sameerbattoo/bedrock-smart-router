# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Use-case 6: Guardrails — Pre-Route Content Safety.

Demonstrates the difference between:
- Native boto3: guardrail applied server-side (model is still invoked)
- Smart Router: guardrail applied PRE-ROUTE (blocked requests cost $0)
"""
from __future__ import annotations

import asyncio
import json
import queue
import time
from pathlib import Path
from typing import AsyncGenerator

from fastapi import APIRouter, Form
from fastapi.responses import StreamingResponse

from shared import (
    REGION,
    BASELINE_MODEL,
    BASELINE_MODELS,
    bedrock_client,
    executor,
    display_model_name,
    compute_cost,
    stream_converse,
)
from bedrock_smart_router import BedrockRouter, RoutingConfig, GuardrailBlockedError

router = APIRouter()

# ── Guardrail Config ────────────────────────────────────────────────

GUARDRAIL_CONFIG_PATH = Path(__file__).parent.parent / "prerequisite" / ".guardrail_config.json"


def _load_guardrail_config() -> dict | None:
    """Load guardrail config from the prerequisite directory."""
    if not GUARDRAIL_CONFIG_PATH.exists():
        return None
    with open(GUARDRAIL_CONFIG_PATH) as f:
        return json.load(f)


def _get_guardrail_router() -> BedrockRouter | None:
    """Create a Smart Router instance with pre-route guardrails configured."""
    config = _load_guardrail_config()
    if not config:
        return None
    return BedrockRouter.create({
        "region": REGION,
        "excluded_models": ["deepseek.*"],
        "prompt_cache_boost": False,
        "guardrails": {
            "pre_route": {
                "guardrail_id": config["guardrail_id"],
                "guardrail_version": config["guardrail_version"],
                "action_on_block": "reject",
            }
        },
    })


# Initialize guardrail router at module load (avoids cold start on first request)
_guardrail_router: BedrockRouter | None = _get_guardrail_router()


def _get_or_create_guardrail_router() -> BedrockRouter | None:
    global _guardrail_router
    if _guardrail_router is None:
        _guardrail_router = _get_guardrail_router()
    return _guardrail_router


# ── Info Endpoint ───────────────────────────────────────────────────

@router.get("/guardrails-config")
def guardrails_config_info():
    """Return the current guardrail configuration for the UI."""
    config = _load_guardrail_config()
    if not config:
        return {"configured": False}

    return {
        "configured": True,
        "guardrail_id": config.get("guardrail_id", ""),
        "guardrail_version": config.get("guardrail_version", ""),
        "guardrail_name": config.get("guardrail_name", ""),
        "pii_entities": ["US_SOCIAL_SECURITY_NUMBER", "EMAIL", "PHONE", "CREDIT_DEBIT_CARD_NUMBER", "NAME", "ADDRESS"],
        "content_filters": ["HATE", "INSULTS", "SEXUAL", "VIOLENCE"],
        "topics_denied": ["investment_advice", "medical_diagnosis"],
    }


# ── Execution Functions ─────────────────────────────────────────────

def _run_baseline_with_guardrail(
    prompt: str,
    guardrail_id: str,
    guardrail_version: str,
    model_id: str,
    on_chunk,
) -> dict:
    """Baseline: converse_stream with server-side guardrailConfig.

    The model IS invoked — guardrail is applied server-side.
    """
    t_start = time.perf_counter()
    output_text = ""
    input_tokens = output_tokens = 0
    guardrail_action = "NONE"
    guardrail_trace = None
    ttft_ms = None

    try:
        response = bedrock_client.converse_stream(
            modelId=model_id,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            guardrailConfig={
                "guardrailIdentifier": guardrail_id,
                "guardrailVersion": guardrail_version,
                "trace": "enabled",
            },
        )

        stream = response.get("stream", response)
        for event in stream:
            if "contentBlockDelta" in event:
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
                # Check for guardrail trace in metadata
                trace = event["metadata"].get("trace", {})
                if trace.get("guardrail"):
                    guardrail_trace = trace["guardrail"]
            elif "messageStop" in event:
                stop_reason = event["messageStop"].get("stopReason", "")
                if stop_reason == "guardrail_intervened":
                    guardrail_action = "GUARDRAIL_INTERVENED"

    except bedrock_client.exceptions.ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "")
        if "guardrail" in error_code.lower() or "guardrail" in str(e).lower():
            guardrail_action = "GUARDRAIL_INTERVENED"
            output_text = f"Blocked by guardrail: {str(e)}"
        else:
            raise
    except Exception as e:
        # Some guardrail blocks come as exceptions
        if "guardrail" in str(e).lower():
            guardrail_action = "GUARDRAIL_INTERVENED"
            output_text = str(e)
        else:
            raise

    latency_ms = (time.perf_counter() - t_start) * 1000
    # When guardrail blocks server-side, Bedrock may report 0 tokens in metadata
    # but still charges for input processing. Estimate from prompt length.
    if input_tokens == 0 and guardrail_action == "GUARDRAIL_INTERVENED":
        input_tokens = max(1, len(prompt) // 4)  # ~4 chars per token
    cost = compute_cost(model_id, input_tokens, output_tokens)

    # Detect if server-side guardrail anonymized PII in the output.
    # When stopReason is "guardrail_intervened" but the response contains PII
    # markers, it means the guardrail anonymized (not blocked). Bedrock uses
    # the same stopReason for both block and anonymize actions.
    pii_markers = [
        "{US_SOCIAL_SECURITY_NUMBER}", "{EMAIL}", "{PHONE}",
        "{CREDIT_DEBIT_CARD_NUMBER}", "{NAME}", "{ADDRESS}",
    ]
    has_pii_markers = any(m in output_text for m in pii_markers)

    if has_pii_markers:
        # Guardrail anonymized PII — not a hard block
        guardrail_action = "PII_ANONYMIZED_OUTPUT"
    elif guardrail_action == "NONE":
        # No intervention detected, but check if this is a PII prompt where
        # the model was smart enough not to echo PII back
        pass  # Will be handled by frontend based on category

    return {
        "response_text": output_text,
        "model_used": display_model_name(model_id),
        "cost": round(cost, 6),
        "latency_ms": round(latency_ms, 1),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "guardrail_action": guardrail_action,
        "guardrail_trace": guardrail_trace,
    }


def _run_router_with_pre_route_guardrail(
    prompt: str,
    guardrail_id: str,
    guardrail_version: str,
    strategy: str,
    classifier: str,
    on_chunk,
) -> dict:
    """Smart Router: pre-route guardrail is built into the router.

    The router's GuardrailsManager runs apply_guardrail(source="INPUT")
    automatically before model selection. If blocked, raises
    GuardrailBlockedError with full trace ($0 cost, model never called).
    If passed, routes normally with guardrailConfig for output PII masking.
    """
    t_start = time.perf_counter()

    guardrail_router = _get_or_create_guardrail_router()
    if guardrail_router is None:
        return {
            "response_text": "Guardrail router not configured.",
            "model_used": "None",
            "cost": 0.0,
            "latency_ms": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "guardrail_action": "ERROR",
            "guardrail_trace": None,
            "cost_saved": False,
        }

    try:
        # The router handles pre-route guardrail automatically.
        # We also pass guardrailConfig for server-side PII masking on output.
        result = stream_converse(
            client=guardrail_router,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            routing=RoutingConfig(strategy=strategy, classifier=classifier, explain=True),
            on_chunk=on_chunk,
            guardrailConfig={
                "guardrailIdentifier": guardrail_id,
                "guardrailVersion": guardrail_version,
                "trace": "enabled",
            },
        )

        latency_ms = (time.perf_counter() - t_start) * 1000

        # Detect if server-side guardrail anonymized PII in the output
        response_text = result.get("response_text", "")
        pii_markers = [
            "{US_SOCIAL_SECURITY_NUMBER}", "{EMAIL}", "{PHONE}",
            "{CREDIT_DEBIT_CARD_NUMBER}", "{NAME}", "{ADDRESS}",
        ]
        found_markers = [m for m in pii_markers if m in response_text]
        pii_anonymized = len(found_markers) > 0

        # Determine effective guardrail action
        effective_action = "PII_ANONYMIZED_OUTPUT" if pii_anonymized else "NONE"

        return {
            "response_text": response_text,
            "model_used": result.get("model_used", "Unknown"),
            "cost": result.get("cost", 0.0),
            "latency_ms": round(latency_ms, 1),
            "input_tokens": result.get("input_tokens", 0),
            "output_tokens": result.get("output_tokens", 0),
            "guardrail_action": effective_action,
            "guardrail_trace": {"action": "NONE", "assessments": [], "latency_ms": None},
            "cost_saved": False,
            "explanation": result.get("explanation"),
            "complexity_detected": result.get("complexity_detected"),
            "fallback_used": result.get("fallback_used", False),
        }

    except GuardrailBlockedError as e:
        # Pre-route guardrail blocked the request — $0 cost
        # Use the guardrail's own latency (not wall clock which includes router init)
        latency_ms = e.latency_ms or (time.perf_counter() - t_start) * 1000
        blocked_message = e.output_text or "Content blocked by guardrail."

        guardrail_trace = {
            "action": "GUARDRAIL_INTERVENED",
            "assessments": e.assessments,
            "latency_ms": e.latency_ms,
        }

        return {
            "response_text": blocked_message,
            "model_used": "None (blocked pre-route)",
            "cost": 0.0,
            "latency_ms": round(latency_ms, 1),
            "input_tokens": 0,
            "output_tokens": 0,
            "guardrail_action": "BLOCKED",
            "guardrail_trace": guardrail_trace,
            "cost_saved": True,
            "original_prompt": prompt,
        }


# ── API Endpoint ────────────────────────────────────────────────────

@router.post("/guardrails-compare")
async def guardrails_compare(
    prompt: str = Form(...),
    mode: str = Form("block"),
    baseline_model: str = Form("sonnet"),
    strategy: str = Form("balanced"),
    classifier: str = Form("heuristic"),
):
    """Compare native guardrails (server-side) vs Smart Router (pre-route).

    Streams results via SSE with events:
    - baseline_chunk / router_chunk: streaming tokens
    - baseline_complete / router_complete: final results
    - done: stream finished
    """
    config = _load_guardrail_config()
    if not config:
        async def error_stream():
            yield f"event: error\ndata: {json.dumps({'message': 'Guardrail not configured. Run: cd demo/prerequisite && python setup_guardrail.py'})}\n\n"
        return StreamingResponse(error_stream(), media_type="text/event-stream")

    guardrail_id = config["guardrail_id"]
    guardrail_version = config["guardrail_version"]

    async def event_stream() -> AsyncGenerator[str, None]:
        loop = asyncio.get_event_loop()
        baseline_q: queue.Queue = queue.Queue()
        router_q: queue.Queue = queue.Queue()

        def _baseline_task():
            bl_config = BASELINE_MODELS.get(baseline_model, BASELINE_MODELS["sonnet"])
            bl_model_id = bl_config["model_id"]
            try:
                result = _run_baseline_with_guardrail(
                    prompt=prompt,
                    guardrail_id=guardrail_id,
                    guardrail_version=guardrail_version,
                    model_id=bl_model_id,
                    on_chunk=lambda text: baseline_q.put(("chunk", text)),
                )
                baseline_q.put(("done", result))
            except Exception as e:
                baseline_q.put(("done", {
                    "response_text": f"Error: {str(e)}",
                    "model_used": display_model_name(bl_model_id),
                    "cost": 0.0,
                    "latency_ms": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "guardrail_action": "ERROR",
                    "guardrail_trace": None,
                }))

        def _router_task():
            try:
                result = _run_router_with_pre_route_guardrail(
                    prompt=prompt,
                    guardrail_id=guardrail_id,
                    guardrail_version=guardrail_version,
                    strategy=strategy,
                    classifier=classifier,
                    on_chunk=lambda text: router_q.put(("chunk", text)),
                )
                router_q.put(("done", result))
            except Exception as e:
                router_q.put(("done", {
                    "response_text": f"Error: {str(e)}",
                    "model_used": "Smart Router",
                    "cost": 0.0,
                    "latency_ms": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "guardrail_action": "ERROR",
                    "guardrail_trace": None,
                    "cost_saved": False,
                }))

        # Run both in parallel
        loop.run_in_executor(executor, _baseline_task)
        loop.run_in_executor(executor, _router_task)

        baseline_done = False
        router_done = False
        baseline_result = None
        router_result = None

        while not (baseline_done and router_done):
            await asyncio.sleep(0.02)

            while not baseline_q.empty():
                msg_type, data = baseline_q.get_nowait()
                if msg_type == "chunk":
                    yield f"event: baseline_chunk\ndata: {json.dumps({'text': data})}\n\n"
                elif msg_type == "done":
                    baseline_result = data
                    baseline_done = True
                    yield f"event: baseline_complete\ndata: {json.dumps(data, default=str)}\n\n"

            while not router_q.empty():
                msg_type, data = router_q.get_nowait()
                if msg_type == "chunk":
                    yield f"event: router_chunk\ndata: {json.dumps({'text': data})}\n\n"
                elif msg_type == "done":
                    router_result = data
                    router_done = True
                    # Calculate cost saved
                    if baseline_result and router_result.get("cost_saved"):
                        router_result["cost_saved_amount"] = baseline_result.get("cost", 0)
                    yield f"event: router_complete\ndata: {json.dumps(router_result, default=str)}\n\n"

        yield f"event: done\ndata: {{}}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )
