# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Use-case 5: Text2SQL with Semantic Cache — chat-style agent.

Single Strands agent that queries an e-commerce SQLite database,
generates charts, and uses FAISS-based semantic cache for repeat queries.
"""
from __future__ import annotations

import asyncio
import json
import queue
import threading
import time
from pathlib import Path
from typing import AsyncGenerator

from fastapi import APIRouter, Form
from fastapi.responses import StreamingResponse, FileResponse

from shared import router as smart_router, executor, display_model_name
from bedrock_smart_router.strands_model import SmartRouterModel

router = APIRouter()

# ── Session management ──────────────────────────────────────────────
_sessions: dict[str, "Text2SQLSession"] = {}
_session_model: SmartRouterModel | None = None

CHART_DIR = Path("/tmp/text2sql_charts")


def _get_model() -> SmartRouterModel:
    """Get or create the shared SmartRouterModel for Text2SQL."""
    global _session_model
    if _session_model is None:
        _session_model = SmartRouterModel(
            router=smart_router,
            routing_preset="balanced",
            explain=True,
        )
    return _session_model


def _get_session(session_id: str):
    """Get or create a Text2SQL session."""
    from text2sql.orch_agent import Text2SQLSession
    if session_id not in _sessions:
        model = _get_model()
        _sessions[session_id] = Text2SQLSession(router_model=model, region="us-west-2")
    return _sessions[session_id]


# ── Endpoints ───────────────────────────────────────────────────────

@router.get("/text2sql/status")
async def text2sql_status():
    """Check if the Text2SQL database is ready."""
    from text2sql.db import get_table_list
    tables = get_table_list()
    return {"ready": len(tables) > 0, "tables": tables}


@router.get("/text2sql/system-prompt")
async def text2sql_system_prompt():
    """Return the orchestrator system prompt."""
    from text2sql.db import get_table_list
    tables = get_table_list()
    return {"system_prompt": f"E-Commerce Data Assistant\nTables: {', '.join(tables)}\nTools: query_database, get_sample_questions\nFeatures: Text2SQL generation, chart visualization, semantic cache with auto-extract + variable hashing, multi-turn intent resolution, prompt caching via cachePoint"}


@router.post("/text2sql/chat")
async def text2sql_chat(
    message: str = Form(...),
    session_id: str = Form("default"),
    strategy: str = Form("balanced"),
    classifier: str = Form("heuristic"),
):
    """Chat with the Text2SQL agent. Streams response via SSE."""

    async def event_stream() -> AsyncGenerator[str, None]:
        loop = asyncio.get_event_loop()
        result_queue: queue.Queue = queue.Queue()

        def _run():
            from text2sql.orch_agent import set_active_session
            session = _get_session(session_id)
            set_active_session(session)
            session.reset_metrics()

            # Update strategy and classifier on all 3 agent models
            session.update_strategy(strategy)
            session.update_classifier(classifier)

            # Wire status callback to stream progress events
            session._status_callback = lambda msg: result_queue.put(("status", msg))

            t_start = time.perf_counter()
            ttft = [None]

            # Pre-orchestrator semantic cache check
            # (The orchestrator may skip the tool call if it has conversation context)
            result_queue.put(("status", "🔍 Extracting intent & checking cache..."))
            import logging as _logging
            _log = _logging.getLogger("text2sql.route")
            print(f"[CACHE] Checking: '{message}' (session={session_id})", flush=True)
            cached = session.cache.get(query_text=message)
            print(f"[CACHE] Result: {'HIT' if cached else 'MISS'} (entries={session.cache.stats.get('entries', 0)})", flush=True)
            if cached is not None:
                session.metrics["cache_hits"] += 1
                result_queue.put(("status", "⚡ Cache hit — skipping DB query & chart generation"))

                # Feed cached data to the orchestrator so it can generate
                # key insights and followup questions (we skip DB + chart, not the LLM)
                import json as _json
                # Inject the cached result as if the tool returned it
                session._last_tool_result = cached
                cache_context = _json.dumps(cached, default=str)
                cache_message = (
                    f"[SEMANTIC CACHE HIT] The following data was retrieved from cache "
                    f"(original query: '{message}'). Format this data with key insights "
                    f"and followup questions as usual:\n\n{cache_context}"
                )

                def _callback(**kwargs):
                    nonlocal ttft
                    if "data" in kwargs:
                        if ttft[0] is None:
                            ttft[0] = (time.perf_counter() - t_start) * 1000
                        result_queue.put(("token", kwargs["data"]))
                    elif "current_tool_use" in kwargs and kwargs["current_tool_use"].get("name"):
                        result_queue.put(("tool", kwargs["current_tool_use"]["name"]))

                agent = session.orchestrator
                agent.callback_handler = _callback
                try:
                    response = agent(cache_message)
                    agent.callback_handler = None
                    elapsed_ms = (time.perf_counter() - t_start) * 1000

                    # Capture orchestrator token usage (same as normal path)
                    if hasattr(response, 'metrics') and response.metrics:
                        usage = getattr(response.metrics, 'accumulated_usage', {}) or {}
                        orch_input = usage.get('inputTokens', 0)
                        orch_output = usage.get('outputTokens', 0)
                        session.metrics["total_input_tokens"] += orch_input
                        session.metrics["total_output_tokens"] += orch_output

                    model = session._orch_model
                    decision = model.last_routing_decision

                    metrics = session.get_metrics()
                    metrics["latency_ms"] = round(elapsed_ms, 1)
                    metrics["ttft_ms"] = round(ttft[0], 1) if ttft[0] else round(elapsed_ms, 1)
                    metrics["model_used"] = display_model_name(decision.selected_model) if decision else "unknown"
                    metrics["complexity"] = "cached"
                    metrics["cost"] = round(decision.actual_cost or 0, 6) if decision else 0
                    metrics["prompt_cache_read"] = 0
                    metrics["prompt_cache_write"] = 0
                    metrics["routing_overhead_ms"] = decision.routing_decision_ms if decision else None
                    metrics["fallback_used"] = decision.fallback_used if decision else False
                    metrics["explanation"] = decision.explanation if decision else None
                    metrics["cache_hit"] = True
                    result_queue.put(("metrics", metrics))
                    result_queue.put(("done", None))
                except Exception as exc:
                    import traceback
                    traceback.print_exc()
                    result_queue.put(("error", str(exc)[:300]))
                return
            last_decision = [None]  # Fix #7: capture per-call

            def _callback(**kwargs):
                if "data" in kwargs:
                    if ttft[0] is None:
                        ttft[0] = (time.perf_counter() - t_start) * 1000
                    result_queue.put(("token", kwargs["data"]))
                elif "current_tool_use" in kwargs and kwargs["current_tool_use"].get("name"):
                    result_queue.put(("tool", kwargs["current_tool_use"]["name"]))
                elif "tool_result" in kwargs:
                    result_queue.put(("status", "🧠 Generating insights..."))

            agent = session.orchestrator
            agent.callback_handler = _callback

            try:
                response = agent(message)
                agent.callback_handler = None
                elapsed_ms = (time.perf_counter() - t_start) * 1000

                # Fix #5: Capture orchestrator's own token usage from response metrics
                if hasattr(response, 'metrics') and response.metrics:
                    usage = getattr(response.metrics, 'accumulated_usage', {}) or {}
                    orch_input = usage.get('inputTokens', 0)
                    orch_output = usage.get('outputTokens', 0)
                    orch_cache_read = usage.get('cacheReadInputTokens', 0)
                    orch_cache_write = usage.get('cacheWriteInputTokens', 0)
                    session.metrics["total_input_tokens"] += orch_input
                    session.metrics["total_output_tokens"] += orch_output
                    session.metrics["cache_read_tokens"] += orch_cache_read
                    session.metrics["cache_write_tokens"] += orch_cache_write

                # Fix #7: Get routing decision from the orchestrator's model
                model = session._orch_model
                decision = model.last_routing_decision

                metrics = session.get_metrics()
                metrics["latency_ms"] = round(elapsed_ms, 1)
                metrics["ttft_ms"] = round(ttft[0], 1) if ttft[0] else round(elapsed_ms, 1)
                metrics["model_used"] = display_model_name(decision.selected_model) if decision else "unknown"
                metrics["complexity"] = decision.complexity_detected if decision else "unknown"
                metrics["cost"] = round(decision.actual_cost or 0, 6) if decision else 0
                metrics["prompt_cache_read"] = session.metrics.get("cache_read_tokens", 0)
                metrics["prompt_cache_write"] = session.metrics.get("cache_write_tokens", 0)
                metrics["routing_overhead_ms"] = decision.routing_decision_ms if decision else None
                metrics["fallback_used"] = decision.fallback_used if decision else False
                metrics["explanation"] = decision.explanation if decision else None

                # Cache the result — the cache_filter decides if it's worth storing
                # (only caches responses with actual results, skips errors/empty)
                tool_result = getattr(session, '_last_tool_result', None)
                if tool_result:
                    session.cache.put(query_text=message, response=tool_result)

                result_queue.put(("metrics", metrics))
                result_queue.put(("done", None))
            except Exception as exc:
                import traceback
                traceback.print_exc()
                result_queue.put(("error", str(exc)[:300]))
                result_queue.put(("done", None))

        loop.run_in_executor(executor, _run)

        # Stream events
        done = False
        while not done:
            await asyncio.sleep(0.03)
            while not result_queue.empty():
                msg_type, data = result_queue.get_nowait()
                if msg_type == "token":
                    yield f"event: token\ndata: {json.dumps({'text': data})}\n\n"
                elif msg_type == "status":
                    yield f"event: status\ndata: {json.dumps({'message': data})}\n\n"
                elif msg_type == "tool":
                    yield f"event: tool_use\ndata: {json.dumps({'name': data})}\n\n"
                elif msg_type == "metrics":
                    yield f"event: metrics\ndata: {json.dumps(data, default=str)}\n\n"
                elif msg_type == "error":
                    yield f"event: error\ndata: {json.dumps({'error': data})}\n\n"
                elif msg_type == "done":
                    yield f"event: done\ndata: {{}}\n\n"
                    done = True

    return StreamingResponse(event_stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.post("/text2sql/reset")
async def text2sql_reset(session_id: str = Form("default")):
    """Reset a Text2SQL session."""
    if session_id in _sessions:
        del _sessions[session_id]
    return {"status": "ok"}


@router.get("/text2sql/charts/{filename}")
async def text2sql_chart(filename: str):
    """Serve a generated chart image."""
    import os
    safe = os.path.basename(filename)
    path = CHART_DIR / safe
    if path.exists():
        return FileResponse(path, media_type="image/png")
    return {"error": "Chart not found"}
