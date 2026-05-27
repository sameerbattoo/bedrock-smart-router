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
    BASELINE_MODEL,
    bedrock_client,
    router as smart_router,
    executor,
    display_model_name,
    compute_cost,
    stream_converse,
)
from bedrock_smart_router import RoutingConfig

router = APIRouter()

# ── Guardrail Config ────────────────────────────────────────────────

GUARDRAIL_CONFIG_PATH = Path(__file__).parent.parent / "prerequisite" / ".guardrail_config.json"


def _load_guardrail_config() -> dict | None:
    """Load guardrail config from the prerequisite directory."""
    if not GUARDRAIL_CONFIG_PATH.exists():
        return None
    with open(GUARDRAIL_CONFIG_PATH) as f:
        return json.load(f)


# ── Execution Functions ─────────────────────────────────────────────

def _run_baseline_with_guardrail(
    prompt: str,
    guardrail_id: str,
    guardrail_version: str,
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
            modelId=BASELINE_MODEL,
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
    cost = compute_cost(BASELINE_MODEL, input_tokens, output_tokens)

    return {
        "response_text": output_text,
        "model_used": display_model_name(BASELINE_MODEL),
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
    on_chunk,
) -> dict:
    """Smart Router: apply_guardrail BEFORE routing.

    If blocked: return immediately with $0 cost (model never called).
    If anonymized: route the sanitized text.
    If none: route normally.
    """
    t_start = time.perf_counter()

    # Step 1: Pre-route guardrail check
    guardrail_response = bedrock_client.apply_guardrail(
        guardrailIdentifier=guardrail_id,
        guardrailVersion=guardrail_version,
        source="INPUT",
        content=[{"text": {"text": prompt}}],
    )

    guardrail_action = guardrail_response.get("action", "NONE")
    assessments = guardrail_response.get("assessments", [])
    outputs = guardrail_response.get("outputs", [])

    guardrail_latency_ms = (time.perf_counter() - t_start) * 1000

    # Build trace info
    guardrail_trace = {
        "action": guardrail_action,
        "assessments": assessments,
        "outputs": outputs,
        "latency_ms": round(guardrail_latency_ms, 1),
    }

    # Case 1: BLOCKED — don't call the model at all
    if guardrail_action == "GUARDRAIL_INTERVENED":
        # Get the blocked message from outputs
        blocked_message = ""
        if outputs:
            blocked_message = outputs[0].get("text", "Content blocked by guardrail.")
        else:
            blocked_message = "Content blocked by guardrail."

        latency_ms = (time.perf_counter() - t_start) * 1000
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

    # Case 2: Content passed or was anonymized — route via smart router
    # Check if text was modified (anonymized)
    routed_prompt = prompt
    anonymized = False
    if outputs:
        output_text = outputs[0].get("text", "")
        if output_text and output_text != prompt:
            routed_prompt = output_text
            anonymized = True
            guardrail_trace["action"] = "ANONYMIZED"

    # Route via smart router
    router_q: queue.Queue = queue.Queue()

    def _on_chunk(text):
        if on_chunk:
            on_chunk(text)

    result = stream_converse(
        client=smart_router,
        messages=[{"role": "user", "content": [{"text": routed_prompt}]}],
        routing=RoutingConfig(strategy="balanced", explain=True),
        on_chunk=_on_chunk,
    )

    latency_ms = (time.perf_counter() - t_start) * 1000

    return {
        "response_text": result.get("response_text", ""),
        "model_used": result.get("model_used", "Unknown"),
        "cost": result.get("cost", 0.0),
        "latency_ms": round(latency_ms, 1),
        "input_tokens": result.get("input_tokens", 0),
        "output_tokens": result.get("output_tokens", 0),
        "guardrail_action": "ANONYMIZED" if anonymized else "NONE",
        "guardrail_trace": guardrail_trace,
        "cost_saved": False,
        "original_prompt": prompt if anonymized else None,
        "sanitized_prompt": routed_prompt if anonymized else None,
    }


# ── API Endpoint ────────────────────────────────────────────────────

@router.post("/guardrails-compare")
async def guardrails_compare(
    prompt: str = Form(...),
    mode: str = Form("block"),
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
            try:
                result = _run_baseline_with_guardrail(
                    prompt=prompt,
                    guardrail_id=guardrail_id,
                    guardrail_version=guardrail_version,
                    on_chunk=lambda text: baseline_q.put(("chunk", text)),
                )
                baseline_q.put(("done", result))
            except Exception as e:
                baseline_q.put(("done", {
                    "response_text": f"Error: {str(e)}",
                    "model_used": display_model_name(BASELINE_MODEL),
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
