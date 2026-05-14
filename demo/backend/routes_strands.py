"""Use-case 3: Strands Agents — AWS Tech Assistant with MCP tools.

Two agents side-by-side:
- Baseline: Strands Agent with fixed Bedrock model (boto3)
- Smart Router: Strands Agent with SmartRouterModel (auto-routing)

Both share the same MCP tools (aws-docs, aws-diagram) and system prompt.
Multi-turn conversation with per-turn metrics.
"""
from __future__ import annotations

import asyncio
import json
import time
import uuid
import queue
import tempfile
import threading
from pathlib import Path
from typing import Any, AsyncGenerator

from fastapi import APIRouter, Form
from fastapi.responses import StreamingResponse, FileResponse
from strands import Agent
from strands.models import BedrockModel
from mcp import stdio_client, StdioServerParameters
from strands.tools.mcp import MCPClient

from shared import (
    REGION, BASELINE_MODELS, ROUTER_STRATEGIES,
    router as smart_router, executor, display_model_name,
)
from bedrock_smart_router.strands_model import SmartRouterModel

router = APIRouter()

# ── System Prompt ───────────────────────────────────────────────────

AWS_SYSTEM_PROMPT = """You are an AWS Technical Assistant — a senior solutions architect with deep expertise across all AWS services.

Your capabilities:
- Search and reference official AWS documentation for accurate, up-to-date information
- Generate architecture diagrams using AWS service icons
- Provide best practices for cost optimization, security, performance, and reliability
- Help design well-architected solutions following the AWS Well-Architected Framework
- Explain complex AWS concepts clearly with examples

Guidelines:
- Always cite AWS documentation when providing technical details
- Use diagrams when explaining architectures (generate them with the diagram tool)
- Provide cost estimates when relevant
- Suggest alternatives and trade-offs
- Keep responses concise but thorough (under 800 words unless asked for more)
- Use markdown formatting with code blocks for CLI commands and configurations
"""

# ── MCP Clients (shared between agents) ────────────────────────────

def _create_aws_docs_client():
    return MCPClient(lambda: stdio_client(
        StdioServerParameters(
            command="uvx",
            args=["awslabs.aws-documentation-mcp-server@latest"],
            env={"FASTMCP_LOG_LEVEL": "ERROR"},
        )
    ))

def _create_aws_diagram_client():
    return MCPClient(lambda: stdio_client(
        StdioServerParameters(
            command="uvx",
            args=["awslabs.aws-diagram-mcp-server@1.0.23"],
            env={"FASTMCP_LOG_LEVEL": "ERROR"},
        )
    ))

# ── MCP client state ─────────────────────────────────────────────────
# MCP clients are created per-session (stdio pipes are thread-bound).
# After first run, uvx caches the packages so startup is fast (~1-2s).
_mcp_ready = False


@router.get("/strands-status")
async def strands_status():
    """Check MCP server readiness status."""
    # After first session is created, tools are ready
    if _mcp_ready:
        return {"docs": "ready", "diagram": "ready"}
    return {"docs": "starting", "diagram": "starting"}


# ── Session Storage ─────────────────────────────────────────────────
_sessions: dict[str, dict] = {}
_DIAGRAM_DIR = Path(tempfile.gettempdir()) / "bsr_demo_diagrams"
_DIAGRAM_DIR.mkdir(exist_ok=True)


def _get_or_create_session(session_id: str, agent_type: str, baseline_model: str | None, router_strategy: str | None) -> dict:
    """Get or create a single agent session."""
    if session_id in _sessions:
        return _sessions[session_id]

    # Create fresh MCP clients for this session (stdio pipes are thread-bound)
    global _mcp_ready
    docs_client = _create_aws_docs_client()
    diagram_client = None
    try:
        diagram_client = _create_aws_diagram_client()
    except Exception:
        pass

    tool_list = [docs_client]
    if diagram_client:
        tool_list.append(diagram_client)
    _mcp_ready = True

    if agent_type == "baseline":
        bl_config = BASELINE_MODELS.get(baseline_model or "sonnet", BASELINE_MODELS["sonnet"])
        model = BedrockModel(
            model_id=bl_config["model_id"],
            region_name=REGION,
        )
        agent = Agent(
            model=model,
            tools=tool_list,
            system_prompt=AWS_SYSTEM_PROMPT,
        )
        session = {
            "id": session_id,
            "type": "baseline",
            "agent": agent,
            "model_id": bl_config["model_id"],
            "smart_model": None,
            "docs_client": docs_client,
            "diagram_client": diagram_client, "tool_list": tool_list,
        }
    else:
        smart_model = SmartRouterModel(
            router=smart_router,
            routing_preset=router_strategy if router_strategy and router_strategy != "balanced" else None,
        )
        agent = Agent(
            model=smart_model,
            tools=tool_list,
            system_prompt=AWS_SYSTEM_PROMPT,
        )
        session = {
            "id": session_id,
            "type": "router",
            "agent": agent,
            "model_id": None,
            "smart_model": smart_model,
            "docs_client": docs_client,
            "diagram_client": diagram_client, "tool_list": tool_list,
        }

    _sessions[session_id] = session
    return session


@router.post("/strands-chat")
async def strands_chat(
    message: str = Form(...),
    baseline_session_id: str = Form(""),
    router_session_id: str = Form(""),
    baseline_model: str = Form("sonnet"),
    router_strategy: str = Form("balanced"),
):
    """Chat with both agents and stream responses via SSE.

    Each agent has its own session_id for independent conversation memory.

    Events:
    - session_init: returns both session IDs
    - baseline_complete: baseline turn metrics + response
    - router_complete: router turn metrics + response (includes routing decision)
    - baseline_error / router_error: if an agent fails
    - done: stream end
    """
    if not baseline_session_id:
        baseline_session_id = str(uuid.uuid4())
    if not router_session_id:
        router_session_id = str(uuid.uuid4())

    baseline_session = _get_or_create_session(baseline_session_id, "baseline", baseline_model, None)
    router_session = _get_or_create_session(router_session_id, "router", None, router_strategy)

    async def event_stream() -> AsyncGenerator[str, None]:
        loop = asyncio.get_event_loop()
        baseline_q: queue.Queue = queue.Queue()
        router_q: queue.Queue = queue.Queue()

        # Send session IDs
        yield f"event: session_init\ndata: {json.dumps({'baseline_session_id': baseline_session_id, 'router_session_id': router_session_id})}\n\n"

        def run_baseline():
            """Run baseline agent turn."""
            agent = baseline_session["agent"]
            t_start = time.perf_counter()
            try:
                response = agent(message)
                latency_ms = (time.perf_counter() - t_start) * 1000
                response_text = str(response) if response else ""
                baseline_q.put(("done", {
                    "response_text": response_text,
                    "model_used": display_model_name(baseline_session["model_id"]),
                    "latency_ms": round(latency_ms, 1),
                    "strategy_used": "direct (fixed model)",
                }))
            except Exception as e:
                latency_ms = (time.perf_counter() - t_start) * 1000
                baseline_q.put(("error", {
                    "error": str(e)[:200],
                    "latency_ms": round(latency_ms, 1),
                }))

        def run_router():
            """Run smart router agent turn."""
            agent = router_session["agent"]
            smart_model = router_session["smart_model"]
            t_start = time.perf_counter()
            try:
                response = agent(message)
                latency_ms = (time.perf_counter() - t_start) * 1000
                response_text = str(response) if response else ""

                # Get routing decision
                decision = smart_model.last_routing_decision
                result = {
                    "response_text": response_text,
                    "latency_ms": round(latency_ms, 1),
                }
                if decision:
                    result.update({
                        "model_used": display_model_name(decision.selected_model),
                        "model_id_full": decision.cris_profile or decision.selected_model,
                        "cost": round(decision.actual_cost or 0, 6),
                        "input_tokens": decision.input_tokens or 0,
                        "output_tokens": decision.output_tokens or 0,
                        "complexity_detected": decision.complexity_detected,
                        "strategy_used": decision.strategy_used,
                        "fallback_used": decision.fallback_used,
                    })
                else:
                    result["model_used"] = "unknown"
                    result["strategy_used"] = router_strategy

                router_q.put(("done", result))
            except Exception as e:
                latency_ms = (time.perf_counter() - t_start) * 1000
                router_q.put(("error", {
                    "error": str(e)[:200],
                    "latency_ms": round(latency_ms, 1),
                }))

        # Run both agents in parallel
        loop.run_in_executor(executor, run_baseline)
        loop.run_in_executor(executor, run_router)

        baseline_done = False
        router_done = False

        while not (baseline_done and router_done):
            await asyncio.sleep(0.1)

            while not baseline_q.empty():
                msg_type, data = baseline_q.get_nowait()
                if msg_type == "done":
                    baseline_done = True
                    yield f"event: baseline_complete\ndata: {json.dumps(data, default=str)}\n\n"
                elif msg_type == "error":
                    baseline_done = True
                    yield f"event: baseline_error\ndata: {json.dumps(data)}\n\n"

            while not router_q.empty():
                msg_type, data = router_q.get_nowait()
                if msg_type == "done":
                    router_done = True
                    yield f"event: router_complete\ndata: {json.dumps(data, default=str)}\n\n"
                elif msg_type == "error":
                    router_done = True
                    yield f"event: router_error\ndata: {json.dumps(data)}\n\n"

        yield f"event: done\ndata: {{}}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"})


@router.post("/strands-reset")
async def strands_reset(
    baseline_session_id: str = Form(""),
    router_session_id: str = Form(""),
):
    """Reset sessions (clear conversation history for both agents)."""
    if baseline_session_id in _sessions:
        del _sessions[baseline_session_id]
    if router_session_id in _sessions:
        del _sessions[router_session_id]
    return {"status": "ok"}


@router.get("/diagrams/{filename}")
async def get_diagram(filename: str):
    """Serve a generated diagram file."""
    path = _DIAGRAM_DIR / filename
    if not path.exists():
        return {"error": "Diagram not found"}
    return FileResponse(path, media_type="image/png")
