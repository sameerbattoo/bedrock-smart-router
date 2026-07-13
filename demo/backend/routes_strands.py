# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

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
import os
import re
import time
import uuid
import queue
import threading
from pathlib import Path
from typing import Any, AsyncGenerator

from fastapi import APIRouter, Form
from fastapi.responses import StreamingResponse, FileResponse
from strands import Agent
from strands.models import BedrockModel
from strands.agent.conversation_manager.sliding_window_conversation_manager import SlidingWindowConversationManager
from mcp import stdio_client, StdioServerParameters
from strands.tools.mcp import MCPClient

from shared import (
    REGION, BASELINE_MODELS, ROUTER_STRATEGIES,
    router as smart_router, executor, display_model_name, judge_response,
    compute_cost,
)
from bedrock_smart_router.strands_model import SmartRouterModel

router = APIRouter()

# ── Constants ───────────────────────────────────────────────────────
MAX_CONVERSATION_TURNS = 5
DIAGRAM_DIR = Path("/tmp/generated-diagrams")
DIAGRAM_DIR.mkdir(exist_ok=True)
# Lock for diagram file renaming (prevents race between baseline and router)
_diagram_rename_lock = threading.Lock()
_IMG_PATH_RE = re.compile(r'!\[([^\]]*)\]\(([^)]+\.png)\)')

# Strategy name → preset name mapping
STRATEGY_TO_PRESET = {
    "balanced": None,
    "cost-optimized": "economy",
    "latency-optimized": "speed",
    "quality-optimized": "quality",
}

# ── System Prompt ───────────────────────────────────────────────────

AWS_SYSTEM_PROMPT = """You are an AWS Technical Assistant — a senior solutions architect with deep expertise across all AWS services.

Your capabilities:
- Search and reference official AWS documentation for accurate, up-to-date information
- Generate architecture diagrams using AWS service icons
- Provide best practices for cost optimization, security, performance, and reliability
- Help design well-architected solutions following the AWS Well-Architected Framework
- Explain complex AWS concepts clearly with examples

Guidelines:
- ALWAYS use the search_documentation or read_documentation tools when asked about ANY AWS service, feature, or concept — even if you think you know the answer. AWS launches new services and features constantly, and your training data may be outdated.
- NEVER say "I'm not familiar with" an AWS service or feature. Instead, search for it first using your tools. If the search returns no results, then explain that you couldn't find documentation for it.
- Always cite AWS documentation when providing technical details
- When the user asks for a diagram, use the generate_diagram tool and include the generated image path in your response using markdown: ![description](path)
- Provide cost estimates when relevant
- Suggest alternatives and trade-offs
- Keep responses concise but thorough (under 800 words unless asked for more)
- Use markdown formatting with code blocks for CLI commands and configurations
"""


# ══════════════════════════════════════════════════════════════════════
# MCP Client Factories
# ══════════════════════════════════════════════════════════════════════

def _create_aws_docs_client():
    return MCPClient(lambda: stdio_client(
        StdioServerParameters(
            command="uvx",
            args=["awslabs.aws-documentation-mcp-server@latest"],
            env={"FASTMCP_LOG_LEVEL": "ERROR"},
        )
    ), startup_timeout=120)


def _create_aws_diagram_client():
    return MCPClient(lambda: stdio_client(
        StdioServerParameters(
            command="uvx",
            args=["awslabs.aws-diagram-mcp-server@1.0.23"],
            env={
                "FASTMCP_LOG_LEVEL": "ERROR",
                "OUTPUT_DIR": str(DIAGRAM_DIR),
            },
            cwd=str(DIAGRAM_DIR),
        )
    ), startup_timeout=120)


# ══════════════════════════════════════════════════════════════════════
# Diagram Path Rewriting
# ══════════════════════════════════════════════════════════════════════

def _find_diagram_file(filename: str) -> str | None:
    """Try to find a diagram file with various name transformations."""
    if (DIAGRAM_DIR / filename).exists():
        return filename
    cleaned = filename.lower().replace(' ', '_').replace('-', '_').replace('+', '').replace('__', '_').strip('_')
    if (DIAGRAM_DIR / cleaned).exists():
        return cleaned
    if not filename.endswith('.png'):
        if (DIAGRAM_DIR / f"{cleaned}.png").exists():
            return f"{cleaned}.png"
    # Try with bl_ or rt_ prefix
    for prefix in ('bl_', 'rt_'):
        if (DIAGRAM_DIR / f"{prefix}{cleaned}").exists():
            return f"{prefix}{cleaned}"
    return None


def _replace_img_match(match: re.Match) -> str:
    """Regex replacement for markdown image references."""
    alt = match.group(1)
    path = match.group(2)
    filename = os.path.basename(path)
    found = _find_diagram_file(filename)
    if found:
        return f'![{alt}](/api/diagrams/{found})'
    # Try using the alt text as filename hint (various transformations)
    if alt:
        alt_as_file = alt.lower().replace(' ', '_').replace('-', '_').replace('+', '').replace('__', '_').strip('_')
        if not alt_as_file.endswith('.png'):
            alt_as_file += '.png'
        if (DIAGRAM_DIR / alt_as_file).exists():
            return f'![{alt}](/api/diagrams/{alt_as_file})'
        # Try with bl_ or rt_ prefix
        for prefix in ('bl_', 'rt_'):
            if (DIAGRAM_DIR / f"{prefix}{alt_as_file}").exists():
                return f'![{alt}](/api/diagrams/{prefix}{alt_as_file})'
    # Last resort: find the most recently created png with matching prefix
    # (handles cases where filename doesn't match alt text at all)
    return match.group(0)


def _replace_plain_path_match(match: re.Match) -> str:
    """Regex replacement for plain /tmp/generated-diagrams/ paths."""
    filename = match.group(2)
    if (DIAGRAM_DIR / filename).exists():
        return f'![diagram](/api/diagrams/{filename})'
    return match.group(1)


_PLAIN_PATH_RE = re.compile(r'(/tmp/generated-diagrams/([^\s"\'<>]+\.png))')


def _rewrite_diagram_paths(text: str) -> str:
    """Rewrite local diagram file paths to the /api/diagrams/ endpoint.

    Handles: ![alt](file.png), ![alt](/tmp/generated-diagrams/file.png),
    and plain text paths like /tmp/generated-diagrams/file.png.
    """
    result = _IMG_PATH_RE.sub(_replace_img_match, text)
    return _PLAIN_PATH_RE.sub(_replace_plain_path_match, result)


def _rename_new_diagrams(pre_files: set[str], prefix: str, after_time: float = 0) -> dict[str, str]:
    """Rename newly created diagram files with a prefix. Returns {old: new} map.
    
    Args:
        pre_files: Set of filenames that existed before the agent ran.
        prefix: Prefix to add (e.g., "bl_" or "rt_").
        after_time: Only rename files created after this timestamp (time.time()).
                    Prevents one agent from claiming another agent's files.
    """
    if not DIAGRAM_DIR.exists():
        return {}
    post_files = set(f.name for f in DIAGRAM_DIR.glob("*.png"))
    rename_map = {}
    for new_file in (post_files - pre_files):
        if new_file.startswith("bl_") or new_file.startswith("rt_"):
            continue
        src = DIAGRAM_DIR / new_file
        if not src.exists():
            continue
        # Only claim files created after our agent started
        if after_time > 0 and src.stat().st_mtime < after_time:
            continue
        dst = DIAGRAM_DIR / f"{prefix}{new_file}"
        if not dst.exists():
            src.rename(dst)
            rename_map[new_file] = f"{prefix}{new_file}"
    return rename_map


def _snapshot_diagrams() -> set[str]:
    """Snapshot current diagram filenames."""
    if DIAGRAM_DIR.exists():
        return set(f.name for f in DIAGRAM_DIR.glob("*.png"))
    return set()


# ══════════════════════════════════════════════════════════════════════
# Session Management
# ══════════════════════════════════════════════════════════════════════

_sessions: dict[str, dict] = {}
_session_locks: dict[str, threading.Lock] = {}
_mcp_ready = False


def _get_or_create_session(session_id: str, agent_type: str,
                           baseline_model: str | None,
                           router_strategy: str | None) -> dict:
    """Get or create a single agent session."""
    if session_id in _sessions:
        return _sessions[session_id]

    global _mcp_ready
    tool_list = []

    # Try to load MCP tools — gracefully degrade if they fail
    docs_client = None
    diagram_client = None
    try:
        docs_client = _create_aws_docs_client()
        tool_list.append(docs_client)
    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning("AWS docs MCP client failed to init: %s", exc)

    try:
        diagram_client = _create_aws_diagram_client()
        tool_list.append(diagram_client)
    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning("AWS diagram MCP client failed to init: %s", exc)

    _mcp_ready = len(tool_list) > 0

    if agent_type == "baseline":
        bl_config = BASELINE_MODELS.get(baseline_model or "sonnet", BASELINE_MODELS["sonnet"])
        model = BedrockModel(model_id=bl_config["model_id"], region_name=REGION)
        agent = Agent(
            model=model, tools=tool_list, system_prompt=AWS_SYSTEM_PROMPT,
            conversation_manager=SlidingWindowConversationManager(window_size=MAX_CONVERSATION_TURNS * 2),
        )
        session = {"id": session_id, "type": "baseline", "agent": agent,
                   "model_id": bl_config["model_id"], "smart_model": None,
                   "docs_client": docs_client, "diagram_client": diagram_client,
                   "tool_list": tool_list}
    else:
        init_preset = STRATEGY_TO_PRESET.get(router_strategy)
        smart_model = SmartRouterModel(
            router=smart_router, routing_preset=init_preset, explain=True,
            exclude_models=[".*thinking.*", ".*kimi.*thinking.*", "amazon.nova-lite.*", "amazon.nova-micro.*"],
        )
        agent = Agent(
            model=smart_model, tools=tool_list, system_prompt=AWS_SYSTEM_PROMPT,
            conversation_manager=SlidingWindowConversationManager(window_size=MAX_CONVERSATION_TURNS * 2),
        )
        session = {"id": session_id, "type": "router", "agent": agent,
                   "model_id": None, "smart_model": smart_model,
                   "docs_client": docs_client, "diagram_client": diagram_client,
                   "tool_list": tool_list}

    _sessions[session_id] = session
    _session_locks[session_id] = threading.Lock()
    return session


# ══════════════════════════════════════════════════════════════════════
# Agent Execution (run in ThreadPoolExecutor)
# ══════════════════════════════════════════════════════════════════════

def _run_baseline_agent(session_id: str, baseline_model: str, message: str,
                       result_queue: queue.Queue) -> None:
    """Execute baseline agent turn. Puts result/error/progress into queue."""
    session = _get_or_create_session(session_id, "baseline", baseline_model, None)
    lock = _session_locks.get(session_id)
    if lock:
        lock.acquire()
    try:
        agent = session["agent"]
        t_start = time.perf_counter()
        ttft_ms = [None]
        tools_used = []
        tool_interactions = []  # Capture tool name + input for judge context

        def _callback(**kwargs):
            if "current_tool_use" in kwargs and kwargs["current_tool_use"].get("name"):
                tool_name = kwargs['current_tool_use']['name']
                tool_input = kwargs['current_tool_use'].get('input', {})
                if tool_name not in tools_used:
                    tools_used.append(tool_name)
                # Capture a summary of what the tool was called with
                input_summary = ""
                if isinstance(tool_input, dict):
                    # Common MCP tool input patterns
                    input_summary = tool_input.get('query', '') or tool_input.get('search_phrase', '') or tool_input.get('url', '') or tool_input.get('topic', '') or str(tool_input)[:200]
                elif isinstance(tool_input, str):
                    input_summary = tool_input[:200]
                tool_interactions.append(f"{tool_name}({input_summary})")
                result_queue.put(("progress", f"🔧 Using tool: {tool_name}"))
            elif "data" in kwargs:
                if ttft_ms[0] is None:
                    ttft_ms[0] = (time.perf_counter() - t_start) * 1000
                result_queue.put(("token", kwargs["data"]))

        agent.callback_handler = _callback
        pre_diagrams = _snapshot_diagrams()
        t_agent_start = time.time()
        response = agent(message)
        agent.callback_handler = None
        wall_clock_ms = (time.perf_counter() - t_start) * 1000

        # Rename diagrams with baseline prefix (only files created during this agent's run)
        response_text = str(response) if response else ""
        with _diagram_rename_lock:
            rename_map = _rename_new_diagrams(pre_diagrams, "bl_", after_time=t_agent_start)
        for old_name, new_name in rename_map.items():
            response_text = response_text.replace(old_name, new_name)
        response_text = _rewrite_diagram_paths(response_text)

        # Extract metrics (ignore cache tokens for fair cost comparison)
        input_tokens = output_tokens = 0
        latency_ms = wall_clock_ms
        if hasattr(response, 'metrics') and response.metrics:
            usage = getattr(response.metrics, 'accumulated_usage', {}) or {}
            metrics_data = getattr(response.metrics, 'accumulated_metrics', {}) or {}
            input_tokens = usage.get('inputTokens', 0)
            output_tokens = usage.get('outputTokens', 0)
            if metrics_data.get('latencyMs'):
                latency_ms = metrics_data['latencyMs']

        cost = compute_cost(session["model_id"], input_tokens, output_tokens)

        result_queue.put(("done", {
            "response_text": response_text,
            "model_used": display_model_name(session["model_id"]),
            "latency_ms": round(latency_ms, 1),
            "ttft_ms": round(ttft_ms[0], 1) if ttft_ms[0] else round(latency_ms, 1),
            "input_tokens": input_tokens, "output_tokens": output_tokens,
            "cache_read_tokens": 0, "cache_write_tokens": 0,
            "cost": round(cost, 6), "strategy_used": "direct (fixed model)",
            "_tools_used": tools_used,
            "_tool_interactions": tool_interactions,
        }))
    except Exception as e:
        latency_ms = (time.perf_counter() - t_start) * 1000 if 't_start' in locals() else 0
        result_queue.put(("error", {"error": str(e)[:200], "latency_ms": round(latency_ms, 1)}))
    finally:
        if lock:
            lock.release()


def _run_router_agent(session_id: str, router_strategy: str, preferred_model: str,
                     message: str, result_queue: queue.Queue, classifier: str = "heuristic") -> None:
    """Execute smart router agent turn. Puts result/error/progress into queue."""
    session = _get_or_create_session(session_id, "router", None, router_strategy)
    lock = _session_locks.get(session_id)
    if lock:
        lock.acquire()
    try:
        agent = session["agent"]
        smart_model = session["smart_model"]

        # Set classifier on the SmartRouterModel config (per-request override)
        smart_model.update_config(classifier=classifier)

        # Update strategy/preferred model if changed mid-conversation
        current_preset = STRATEGY_TO_PRESET.get(router_strategy)
        if smart_model.config.get("routing_preset") != current_preset:
            smart_model.update_config(routing_preset=current_preset)
        current_preferred = preferred_model if preferred_model else None
        if smart_model.config.get("preferred_model") != current_preferred:
            smart_model.update_config(preferred_model=current_preferred)

        t_start = time.perf_counter()
        ttft_ms = [None]
        tools_used = []
        tool_interactions = []  # Capture tool name + input for judge context

        def _callback(**kwargs):
            if "current_tool_use" in kwargs and kwargs["current_tool_use"].get("name"):
                tool_name = kwargs['current_tool_use']['name']
                tool_input = kwargs['current_tool_use'].get('input', {})
                if tool_name not in tools_used:
                    tools_used.append(tool_name)
                # Capture a summary of what the tool was called with
                input_summary = ""
                if isinstance(tool_input, dict):
                    input_summary = tool_input.get('query', '') or tool_input.get('search_phrase', '') or tool_input.get('url', '') or tool_input.get('topic', '') or str(tool_input)[:200]
                elif isinstance(tool_input, str):
                    input_summary = tool_input[:200]
                tool_interactions.append(f"{tool_name}({input_summary})")
                result_queue.put(("progress", f"🔧 Using tool: {tool_name}"))
            elif "data" in kwargs:
                if ttft_ms[0] is None:
                    ttft_ms[0] = (time.perf_counter() - t_start) * 1000
                result_queue.put(("token", kwargs["data"]))

        agent.callback_handler = _callback
        pre_diagrams = _snapshot_diagrams()
        t_agent_start = time.time()
        response = agent(message)
        agent.callback_handler = None
        wall_clock_ms = (time.perf_counter() - t_start) * 1000

        # Rename diagrams with router prefix (only files created during this agent's run)
        response_text = str(response) if response else ""
        with _diagram_rename_lock:
            rename_map = _rename_new_diagrams(pre_diagrams, "rt_", after_time=t_agent_start)
        for old_name, new_name in rename_map.items():
            response_text = response_text.replace(old_name, new_name)
        response_text = _rewrite_diagram_paths(response_text)

        # Extract metrics
        strands_input = strands_output = 0
        strands_latency_ms = 0
        if hasattr(response, 'metrics') and response.metrics:
            usage = getattr(response.metrics, 'accumulated_usage', {}) or {}
            metrics_data = getattr(response.metrics, 'accumulated_metrics', {}) or {}
            strands_input = usage.get('inputTokens', 0)
            strands_output = usage.get('outputTokens', 0)
            strands_latency_ms = metrics_data.get('latencyMs', 0)

        # Build result from routing decision
        decision = smart_model.last_routing_decision
        result = {
            "response_text": response_text,
            "latency_ms": round(strands_latency_ms or wall_clock_ms, 1),
            "ttft_ms": round(ttft_ms[0], 1) if ttft_ms[0] else round(strands_latency_ms or wall_clock_ms, 1),
        }
        if decision:
            if decision.fallback_used:
                import logging
                logging.getLogger(__name__).warning(
                    "Fallback triggered: primary was %s, fell back to %s",
                    decision.explanation.get("reason", "unknown") if decision.explanation else "unknown",
                    decision.selected_model,
                )
            # Recalculate cost WITHOUT cache tokens for fair comparison
            # (Bedrock's native prompt caching makes the router appear cheaper
            #  which confuses the demo — we want apples-to-apples cost comparison)
            fair_cost = compute_cost(
                decision.selected_model, strands_input, strands_output
            )
            result.update({
                "model_used": display_model_name(decision.selected_model),
                "model_id_full": decision.cris_profile or decision.selected_model,
                "cost": round(fair_cost, 6),
                "input_tokens": strands_input, "output_tokens": strands_output,
                "complexity_detected": decision.complexity_detected,
                "strategy_used": decision.strategy_used,
                "fallback_used": decision.fallback_used,
                "routing_overhead_ms": decision.routing_decision_ms,
                "explanation": decision.explanation,
                "_tools_used": tools_used,
                "_tool_interactions": tool_interactions,
            })
        else:
            result["model_used"] = "unknown"
            result["strategy_used"] = router_strategy
            result["_tools_used"] = tools_used
            result["_tool_interactions"] = tool_interactions

        result_queue.put(("done", result))
    except Exception as e:
        latency_ms = (time.perf_counter() - t_start) * 1000 if 't_start' in locals() else 0
        result_queue.put(("error", {"error": str(e)[:200], "latency_ms": round(latency_ms, 1)}))
    finally:
        if lock:
            lock.release()


def _find_diagram_bytes(response_text: str) -> tuple[bytes | None, str | None]:
    """Find a generated diagram PNG referenced in the response text."""
    matches = _IMG_PATH_RE.findall(response_text or "")
    for _, path in matches:
        filename = os.path.basename(path)
        fpath = DIAGRAM_DIR / filename
        if fpath.exists():
            return fpath.read_bytes(), "image/png"
        cleaned = filename.replace(' ', '_')
        fpath = DIAGRAM_DIR / cleaned
        if fpath.exists():
            return fpath.read_bytes(), "image/png"
    return None, None


# ══════════════════════════════════════════════════════════════════════
# API Endpoints
# ══════════════════════════════════════════════════════════════════════

@router.get("/strands-status")
async def strands_status():
    """Check MCP server readiness status."""
    if _mcp_ready:
        return {"docs": "ready", "diagram": "ready"}
    return {"docs": "starting", "diagram": "starting"}


@router.get("/strands-system-prompt")
async def strands_system_prompt():
    """Return the system prompt used by both agents (single source of truth)."""
    return {"system_prompt": AWS_SYSTEM_PROMPT}


@router.post("/strands-chat")
async def strands_chat(
    message: str = Form(...),
    baseline_session_id: str = Form(""),
    router_session_id: str = Form(""),
    baseline_model: str = Form("sonnet"),
    router_strategy: str = Form("balanced"),
    classifier: str = Form("heuristic"),
    preferred_model: str = Form(""),
    send_target: str = Form("both"),
    skip_judge: bool = Form(False),
):
    """Chat with both agents and stream responses via SSE."""
    if not baseline_session_id:
        baseline_session_id = str(uuid.uuid4())
    if not router_session_id:
        router_session_id = str(uuid.uuid4())

    async def event_stream() -> AsyncGenerator[str, None]:
        loop = asyncio.get_event_loop()
        baseline_q: queue.Queue = queue.Queue()
        router_q: queue.Queue = queue.Queue()

        yield f"event: session_init\ndata: {json.dumps({'baseline_session_id': baseline_session_id, 'router_session_id': router_session_id})}\n\n"

        # Dispatch agents based on send_target
        if send_target in ("both", "baseline"):
            loop.run_in_executor(executor, _run_baseline_agent, baseline_session_id, baseline_model, message, baseline_q)
        else:
            baseline_q.put(("skip", None))

        if send_target in ("both", "router"):
            loop.run_in_executor(executor, _run_router_agent, router_session_id, router_strategy, preferred_model, message, router_q, classifier)
        else:
            router_q.put(("skip", None))

        # Stream events from both queues
        baseline_done = router_done = False
        baseline_result_data = router_result_data = None

        while not (baseline_done and router_done):
            await asyncio.sleep(0.05)

            while not baseline_q.empty():
                msg_type, data = baseline_q.get_nowait()
                if msg_type == "skip":
                    baseline_done = True
                elif msg_type == "done":
                    baseline_done = True
                    baseline_result_data = data
                    yield f"event: baseline_complete\ndata: {json.dumps(data, default=str)}\n\n"
                elif msg_type == "error":
                    baseline_done = True
                    yield f"event: baseline_error\ndata: {json.dumps(data)}\n\n"
                elif msg_type == "progress":
                    yield f"event: baseline_progress\ndata: {json.dumps({'message': data})}\n\n"
                elif msg_type == "token":
                    yield f"event: baseline_token\ndata: {json.dumps({'text': data})}\n\n"

            while not router_q.empty():
                msg_type, data = router_q.get_nowait()
                if msg_type == "skip":
                    router_done = True
                elif msg_type == "done":
                    router_done = True
                    router_result_data = data
                    yield f"event: router_complete\ndata: {json.dumps(data, default=str)}\n\n"
                elif msg_type == "error":
                    router_done = True
                    yield f"event: router_error\ndata: {json.dumps(data)}\n\n"
                elif msg_type == "progress":
                    yield f"event: router_progress\ndata: {json.dumps({'message': data})}\n\n"
                elif msg_type == "token":
                    yield f"event: router_token\ndata: {json.dumps({'text': data})}\n\n"

        yield f"event: done\ndata: {{}}\n\n"

        # LLM Judge scoring (non-blocking, arrives after done)
        if not skip_judge and baseline_result_data and router_result_data:
            # Build context that includes tool usage info so the judge knows
            # responses are grounded in real MCP tool data (AWS docs, diagrams)
            tools_context_parts = []
            bl_tools = baseline_result_data.get("_tools_used") or []
            rt_tools = router_result_data.get("_tools_used") or []
            bl_interactions = baseline_result_data.get("_tool_interactions") or []
            rt_interactions = router_result_data.get("_tool_interactions") or []
            # Also detect tool usage from progress messages embedded in response
            bl_text = baseline_result_data.get("response_text", "")
            rt_text = router_result_data.get("response_text", "")

            if bl_interactions:
                tools_context_parts.append(f"Baseline agent tool calls: {'; '.join(bl_interactions)}")
            elif bl_tools:
                tools_context_parts.append(f"Baseline agent used MCP tools: {', '.join(bl_tools)}")
            if rt_interactions:
                tools_context_parts.append(f"Router agent tool calls: {'; '.join(rt_interactions)}")
            elif rt_tools:
                tools_context_parts.append(f"Router agent used MCP tools: {', '.join(rt_tools)}")

            tools_note = ""
            if tools_context_parts:
                tools_note = "\n\nTool usage context:\n" + "\n".join(tools_context_parts) + "\n\nIMPORTANT: Both agents have access to live AWS documentation via MCP tools (search_documentation, read_documentation). When an agent uses these tools, its response is grounded in official, up-to-date AWS documentation. Do NOT penalize responses for containing information about new or unfamiliar AWS services if the agent retrieved that information from official docs via tools."

            conv_context = f"User question: {message}{tools_note}"
            bl_img, bl_type = _find_diagram_bytes(bl_text)
            rt_img, rt_type = _find_diagram_bytes(rt_text)

            judge_bl = loop.run_in_executor(executor, judge_response, conv_context, bl_text, bl_img, bl_type)
            judge_rt = loop.run_in_executor(executor, judge_response, conv_context, rt_text, rt_img, rt_type)
            bj, rj = await asyncio.gather(judge_bl, judge_rt)
            yield f"event: judge_scores\ndata: {json.dumps({'baseline_score': bj['score'], 'baseline_reasoning': bj['reasoning'], 'router_score': rj['score'], 'router_reasoning': rj['reasoning']})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"})


@router.post("/strands-reset")
async def strands_reset(
    baseline_session_id: str = Form(""),
    router_session_id: str = Form(""),
):
    """Reset conversation history but keep MCP clients alive for fast restart."""
    for sid in (baseline_session_id, router_session_id):
        session = _sessions.get(sid)
        if session and session.get("agent"):
            agent = session["agent"]
            agent.messages.clear()
    return {"status": "ok"}


@router.get("/diagrams/{filename}")
async def get_diagram(filename: str):
    """Serve a generated diagram file (path traversal protected)."""
    safe_name = os.path.basename(filename)
    path = DIAGRAM_DIR / safe_name
    if not path.exists() or not path.is_file():
        return {"error": "Diagram not found"}
    return FileResponse(path, media_type="image/png")
