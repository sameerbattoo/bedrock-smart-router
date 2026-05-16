"""Use-case 1: Baseline vs Smart Router comparison."""
from __future__ import annotations

import asyncio
import json
import mimetypes
import queue
from typing import AsyncGenerator

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse

from shared import (
    BASELINE_MODELS, ROUTER_STRATEGIES, SUPPORTED_IMAGE_TYPES, SUPPORTED_DOC_TYPES,
    bedrock_client, router as smart_router, executor,
    build_content_blocks, judge_response, stream_converse,
    store_temp_file, get_temp_file,
)
from bedrock_smart_router import RoutingConfig

router = APIRouter()


# ══════════════════════════════════════════════════════════════════════
# Module-level execution functions (run in ThreadPoolExecutor)
# ══════════════════════════════════════════════════════════════════════

def _run_compare_baseline(
    client,
    messages: list[dict],
    system_prompt: str,
    model_id: str,
    on_chunk,
) -> dict:
    """Baseline: direct boto3 converse_stream call. Returns result dict."""
    return stream_converse(
        client=client,
        messages=messages,
        system_prompt=system_prompt,
        model_id=model_id,
        on_chunk=on_chunk,
    )


def _run_compare_router(
    client,
    messages: list[dict],
    system_prompt: str,
    strategy: str,
    preferred_model: str,
    on_chunk,
) -> dict:
    """Smart Router: same API, just swap the client. Returns result dict."""
    return stream_converse(
        client=client,
        messages=messages,
        system_prompt=system_prompt,
        routing=RoutingConfig(
            strategy=strategy,
            preferred_model=preferred_model if preferred_model else None,
            explain=True,
        ),
        on_chunk=on_chunk,
    )


# ══════════════════════════════════════════════════════════════════════
# API Endpoint
# ══════════════════════════════════════════════════════════════════════

@router.post("/compare-stream")
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
    """Stream comparison results via Server-Sent Events."""
    file_bytes = None
    file_type = None
    file_name = None

    if file_id:
        file_bytes, file_type, file_name = get_temp_file(file_id)

    if not file_bytes and file and file.filename:
        file_bytes = await file.read()
        file_type = file.content_type or mimetypes.guess_type(file.filename)[0]
        file_name = file.filename
        if file_type not in SUPPORTED_IMAGE_TYPES | SUPPORTED_DOC_TYPES:
            raise HTTPException(400, f"Unsupported file type: {file_type}")
        if len(file_bytes) > 20_000_000:
            raise HTTPException(400, "File too large. Maximum 20MB.")
        file_id = store_temp_file(file_bytes, file_type, file_name)

    async def event_stream() -> AsyncGenerator[str, None]:
        loop = asyncio.get_event_loop()
        baseline_q: queue.Queue = queue.Queue()
        router_q: queue.Queue = queue.Queue()

        if file_id:
            yield f"event: file_stored\ndata: {json.dumps({'file_id': file_id, 'file_name': file_name})}\n\n"

        # Build messages (same for both — drop-in replacement demo)
        content_blocks = build_content_blocks(prompt, file_bytes, file_type)
        messages = [{"role": "user", "content": content_blocks}]

        # Resolve params
        bl_config = BASELINE_MODELS.get(baseline_model, BASELINE_MODELS["sonnet"])
        bl_model_id = bl_config["model_id"]
        strategy = router_strategy if router_strategy in ROUTER_STRATEGIES else "balanced"

        def _baseline_task():
            result = _run_compare_baseline(
                client=bedrock_client,
                messages=messages,
                system_prompt=system_prompt,
                model_id=bl_model_id,
                on_chunk=lambda text: baseline_q.put(("chunk", text)),
            )
            result["has_multimodal"] = file_bytes is not None
            baseline_q.put(("done", result))

        def _router_task():
            result = _run_compare_router(
                client=smart_router,
                messages=messages,
                system_prompt=system_prompt,
                strategy=strategy,
                preferred_model=preferred_model,
                on_chunk=lambda text: router_q.put(("chunk", text)),
            )
            result["has_multimodal"] = file_bytes is not None
            router_q.put(("done", result))

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
                    if baseline_result and baseline_result.get("cost", 0) > 0:
                        data["savings_pct"] = round((1 - data["cost"] / baseline_result["cost"]) * 100, 1)
                    yield f"event: router_complete\ndata: {json.dumps(data, default=str)}\n\n"

        # Judge
        if run_judge and baseline_result and router_result:
            judge_bl = loop.run_in_executor(executor, judge_response, prompt, baseline_result["response_text"], file_bytes, file_type)
            judge_rt = loop.run_in_executor(executor, judge_response, prompt, router_result["response_text"], file_bytes, file_type)
            bj, rj = await asyncio.gather(judge_bl, judge_rt)
            yield f"event: judge_scores\ndata: {json.dumps({'baseline_score': bj['score'], 'baseline_reasoning': bj['reasoning'], 'router_score': rj['score'], 'router_reasoning': rj['reasoning']})}\n\n"

        yield f"event: done\ndata: {{}}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"})
