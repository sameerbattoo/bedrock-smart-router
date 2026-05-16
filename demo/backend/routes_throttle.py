"""Use-case 2: Throttle Handling demo.

Patches the shared bedrock_client at the botocore level so that both
the baseline (direct boto3 call) and the Smart Router (which uses the
same client internally) see ThrottlingException for the selected model.

The baseline retries and fails. The Smart Router retries, exhausts,
and falls back to the next model in the chain — which succeeds.
"""
from __future__ import annotations

import asyncio
import json
import time
import uuid
import queue
import threading
from typing import Any, AsyncGenerator

from fastapi import APIRouter, Form
from fastapi.responses import StreamingResponse
from botocore.exceptions import ClientError

from shared import (
    bedrock_client, router as smart_router, executor,
    display_model_name, build_content_blocks, call_converse,
)
from bedrock_smart_router import RoutingConfig

router = APIRouter()

# Global lock to prevent concurrent throttle demos from corrupting each other's patches
_throttle_lock = threading.Lock()

def _make_throttle_error():
    """Create a realistic ThrottlingException with proper request ID."""
    return ClientError(
        error_response={
            "Error": {
                "Code": "ThrottlingException",
                "Message": "Too many requests, please wait before trying again.",
            },
            "ResponseMetadata": {
                "RequestId": str(uuid.uuid4()),
                "HTTPStatusCode": 429,
                "HTTPHeaders": {"x-amzn-requestid": str(uuid.uuid4())},
            },
        },
        operation_name="ConverseStream",
    )


def _strip_geo_prefix(model_id: str) -> str:
    """Remove geo prefix from a model ID."""
    for prefix in ("us.", "eu.", "apac.", "global.", "au.", "jp.", "ca."):
        if model_id.startswith(prefix):
            return model_id[len(prefix):]
    return model_id


# ══════════════════════════════════════════════════════════════════════
# Module-level execution functions (run in ThreadPoolExecutor)
# ══════════════════════════════════════════════════════════════════════

def _run_throttle_baseline(
    client,
    messages: list[dict],
    system_prompt: str,
    model_id: str,
    result_queue: queue.Queue,
) -> None:
    """Baseline makes ONE direct call — fails immediately on throttle.

    Puts ("success", result) or ("failed", error_info) into result_queue.
    """
    try:
        result = call_converse(
            client=client,
            messages=messages,
            system_prompt=system_prompt,
            model_id=model_id,
        )
        result_queue.put(("success", result))
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "")
        error_msg = e.response.get("Error", {}).get("Message", "")
        request_id = e.response.get("ResponseMetadata", {}).get("RequestId", "")
        result_queue.put(("failed", {
            "error": f"{error_code}: {error_msg}",
            "model": display_model_name(model_id),
            "model_id": model_id,
            "request_id": request_id,
        }))


def _run_throttle_router(
    client,
    messages: list[dict],
    system_prompt: str,
    throttle_model: str,
    attempt_log: list[dict],
    result_queue: queue.Queue,
) -> None:
    """Router with preferred_model — retries 3x then falls back.

    Puts ("done", result) or ("error", error_str) into result_queue.
    """
    try:
        result = call_converse(
            client=client,
            messages=messages,
            system_prompt=system_prompt,
            routing=RoutingConfig(
                strategy="balanced",
                preferred_model=throttle_model,
                explain=True,
            ),
        )
        # Enrich with attempt timeline from the patch
        throttled = [a for a in attempt_log if a["status"] == "throttled"]
        fallback = next((a for a in attempt_log if a["status"] == "fallback"), None)
        result["throttled_attempts"] = len(throttled)
        result["fallback_from"] = display_model_name(throttle_model)
        if fallback:
            result["fallback_to"] = fallback["model"]
            result["fallback_model_id"] = fallback["model_id"]
        result_queue.put(("done", result))
    except Exception as e:
        # Include attempt log in error for debugging
        throttled = [a for a in attempt_log if a["status"] == "throttled"]
        fallback = next((a for a in attempt_log if a["status"] == "fallback"), None)
        result_queue.put(("error", f"{type(e).__name__}: {str(e)[:200]}. Throttled {len(throttled)} times. Fallback: {fallback['model'] if fallback else 'none'}"))


# ══════════════════════════════════════════════════════════════════════
# API Endpoint
# ══════════════════════════════════════════════════════════════════════

@router.post("/throttle-demo")
async def throttle_demo(
    prompt: str = Form(...),
    system_prompt: str = Form(""),
    throttle_model: str = Form(...),
):
    """Stream throttle demo results via SSE.

    Patches bedrock_client.converse_stream to throw ThrottlingException
    for the selected model. Both baseline and router use the same client.
    """
    throttle_base = _strip_geo_prefix(throttle_model)

    # Track attempts for timeline reporting
    attempt_log: list[dict] = []
    log_lock = threading.Lock()

    # Patch the shared client (both converse and converse_stream)
    original_converse = bedrock_client.converse
    original_converse_stream = bedrock_client.converse_stream
    original_router_converse = smart_router._bedrock.converse
    original_router_converse_stream = smart_router._bedrock.converse_stream

    def throttled_converse(**kwargs):
        model_id = kwargs.get("modelId", "")
        call_base = _strip_geo_prefix(model_id)

        if call_base == throttle_base:
            with log_lock:
                attempt_log.append({
                    "timestamp": time.time(),
                    "model": display_model_name(model_id),
                    "model_id": model_id,
                    "status": "throttled",
                })
            raise _make_throttle_error()
        else:
            with log_lock:
                attempt_log.append({
                    "timestamp": time.time(),
                    "model": display_model_name(model_id),
                    "model_id": model_id,
                    "status": "fallback_attempt",
                })
            try:
                result = original_converse(**kwargs)
                with log_lock:
                    attempt_log.append({
                        "timestamp": time.time(),
                        "model": display_model_name(model_id),
                        "model_id": model_id,
                        "status": "fallback_success",
                    })
                return result
            except Exception as e:
                with log_lock:
                    attempt_log.append({
                        "timestamp": time.time(),
                        "model": display_model_name(model_id),
                        "model_id": model_id,
                        "status": "fallback_failed",
                        "error": str(e)[:100],
                    })
                raise

    def throttled_converse_stream(**kwargs):
        model_id = kwargs.get("modelId", "")
        call_base = _strip_geo_prefix(model_id)

        if call_base == throttle_base:
            with log_lock:
                attempt_log.append({
                    "timestamp": time.time(),
                    "model": display_model_name(model_id),
                    "model_id": model_id,
                    "status": "throttled",
                })
            raise _make_throttle_error()
        else:
            with log_lock:
                attempt_log.append({
                    "timestamp": time.time(),
                    "model": display_model_name(model_id),
                    "model_id": model_id,
                    "status": "fallback",
                })
            return original_converse_stream(**kwargs)

    async def event_stream() -> AsyncGenerator[str, None]:
        loop = asyncio.get_event_loop()
        baseline_q: queue.Queue = queue.Queue()
        router_q: queue.Queue = queue.Queue()

        # Install the patches on BOTH the shared client AND the router's internal client
        # Protected by _throttle_lock to prevent concurrent demos from corrupting patches
        _throttle_lock.acquire()
        bedrock_client.converse = throttled_converse
        bedrock_client.converse_stream = throttled_converse_stream
        smart_router._bedrock.converse = throttled_converse
        smart_router._bedrock.converse_stream = throttled_converse_stream

        try:
            # Build messages (same for both)
            content_blocks = build_content_blocks(prompt)
            messages = [{"role": "user", "content": content_blocks}]

            # Resolve the model ID for baseline (needs geo prefix for boto3)
            bl_model_id = throttle_model
            model = smart_router.registry.get(throttle_model)
            if model and model.regions:
                for r in model.regions:
                    if "global" in r.get("cris_profiles", []):
                        bl_model_id = f"global.{throttle_base}"
                        break
                    elif r.get("cris_profiles"):
                        bl_model_id = f"{r['cris_profiles'][0]}.{throttle_base}"
                        break

            # Run both in parallel via module-level functions
            loop.run_in_executor(
                executor,
                _run_throttle_baseline,
                bedrock_client, messages, system_prompt, bl_model_id, baseline_q,
            )
            loop.run_in_executor(
                executor,
                _run_throttle_router,
                smart_router, messages, system_prompt, throttle_model, attempt_log, router_q,
            )

            # Stream events
            baseline_done = False
            router_done = False
            router_attempts_sent = 0

            while not (baseline_done and router_done):
                await asyncio.sleep(0.05)

                # Send router attempts in real-time as they happen
                with log_lock:
                    while router_attempts_sent < len(attempt_log):
                        entry = attempt_log[router_attempts_sent]
                        # Compute delay from previous attempt
                        delay_ms = 0
                        if router_attempts_sent > 0:
                            prev_ts = attempt_log[router_attempts_sent - 1]["timestamp"]
                            delay_ms = int((entry["timestamp"] - prev_ts) * 1000)
                        if entry["status"] == "throttled":
                            yield f"event: router_attempt\ndata: {json.dumps({'attempt': router_attempts_sent + 1, 'status': 'throttled', 'model': entry['model'], 'model_id': entry['model_id'], 'error': 'ThrottlingException: Rate exceeded', 'backoff_ms': delay_ms})}\n\n"
                        elif entry["status"] == "fallback_attempt":
                            yield f"event: router_fallback\ndata: {json.dumps({'from_model': display_model_name(throttle_model), 'to_model': entry['model'], 'to_model_id': entry['model_id'], 'total_retry_time_ms': delay_ms})}\n\n"
                        elif entry["status"] == "fallback_failed":
                            yield f"event: router_attempt\ndata: {json.dumps({'attempt': router_attempts_sent + 1, 'status': 'fallback_failed', 'model': entry['model'], 'model_id': entry['model_id'], 'error': entry.get('error', 'Model error'), 'backoff_ms': 0})}\n\n"
                        elif entry["status"] == "fallback_success":
                            yield f"event: router_fallback\ndata: {json.dumps({'from_model': display_model_name(throttle_model), 'to_model': entry['model'], 'to_model_id': entry['model_id'], 'total_retry_time_ms': delay_ms, 'success': True})}\n\n"
                        router_attempts_sent += 1

                while not baseline_q.empty():
                    msg_type, data = baseline_q.get_nowait()
                    if msg_type == "failed":
                        baseline_done = True
                        yield f"event: baseline_failed\ndata: {json.dumps(data)}\n\n"
                    elif msg_type == "success":
                        baseline_done = True
                        yield f"event: baseline_success\ndata: {json.dumps(data, default=str)}\n\n"

                while not router_q.empty():
                    msg_type, data = router_q.get_nowait()
                    if msg_type == "done":
                        router_done = True
                        yield f"event: router_complete\ndata: {json.dumps(data, default=str)}\n\n"
                    elif msg_type == "error":
                        router_done = True
                        yield f"event: router_error\ndata: {json.dumps({'error': data})}\n\n"

            yield f"event: done\ndata: {{}}\n\n"

        finally:
            bedrock_client.converse = original_converse
            bedrock_client.converse_stream = original_converse_stream
            smart_router._bedrock.converse = original_router_converse
            smart_router._bedrock.converse_stream = original_router_converse_stream
            _throttle_lock.release()

    return StreamingResponse(event_stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"})
