"""Use-case 8: Safe Model Rollouts — A/B Testing, Canary, and Shadow Mode.

Demonstrates the full safe rollout lifecycle:
- A/B Testing: split traffic between two models, sticky user assignment
- Canary: roll out a new model at X% with auto-rollback on errors
- Shadow: mirror traffic to a new model without affecting users

Each mode loads from a JSON config file (production pattern).
"""
from __future__ import annotations

import asyncio
import json
import random
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Form
from fastapi.responses import StreamingResponse

from bedrock_smart_router import BedrockRouter, RoutingConfig
from shared import display_model_name, compute_cost

router = APIRouter()

# ── Load configs from JSON ──────────────────────────────────────────
CONFIG_DIR = Path(__file__).parent / "rollout_configs"


def _load_config(filename: str) -> dict:
    """Load a JSONC file (JSON with comments), stripping // comments."""
    raw = (CONFIG_DIR / filename).read_text()
    # Strip single-line comments (// ...) but not inside strings
    import re
    stripped = re.sub(r'(?<!:)//.*', '', raw)
    return json.loads(stripped)


def _strip_notes(obj):
    """Remove _note/_*_note keys for router creation (keep for display)."""
    if isinstance(obj, dict):
        return {k: _strip_notes(v) for k, v in obj.items() if not k.startswith('_')}
    if isinstance(obj, list):
        return [_strip_notes(i) for i in obj]
    return obj


AB_CONFIG_RAW = (CONFIG_DIR / "ab_test.jsonc").read_text()
CANARY_CONFIG_RAW = (CONFIG_DIR / "canary.jsonc").read_text()
SHADOW_CONFIG_RAW = (CONFIG_DIR / "shadow.jsonc").read_text()

AB_CONFIG = _load_config("ab_test.jsonc")
AB_CONFIG["cache"] = {"enabled": False}  # Disable cache for A/B (each user needs own assignment)
CANARY_CONFIG = _load_config("canary.jsonc")
CANARY_CONFIG["cache"] = {"enabled": False}
SHADOW_CONFIG = _load_config("shadow.jsonc")
SHADOW_CONFIG["cache"] = {"enabled": False}

# ── Create routers from configs ─────────────────────────────────────
_ab_router = BedrockRouter.create(AB_CONFIG)
_canary_router = BedrockRouter.create(CANARY_CONFIG)
_shadow_router = BedrockRouter.create(SHADOW_CONFIG)

# ── Simulated Users ─────────────────────────────────────────────────
SIM_USERS = [f"user-{i:03d}" for i in range(100)]

# ── Prompts (same categories as Usage Tracking) ────────────────────
SIM_PROMPTS = {
    "simple": [
        "What is Amazon S3?",
        "Define serverless computing.",
        "What is a VPC?",
        "Explain IAM roles.",
        "What is DynamoDB?",
    ],
    "moderate": [
        "Compare REST and GraphQL APIs. List the pros and cons of each approach with specific use cases where one is better than the other.",
        "Explain the CAP theorem and its implications for distributed databases. How does DynamoDB handle this trade-off? Provide examples.",
        "How does AWS Lambda cold start work? What are the best practices to mitigate cold starts? Include code examples for provisioned concurrency.",
        "Describe the differences between SQS and SNS. When would you use each? Design a fanout pattern using both services together.",
        "Explain blue-green deployment strategy on AWS. Walk through the steps to implement it with CodeDeploy and ALB target groups.",
    ],
    "complex": [
        "Design a real-time fraud detection system that processes 1 million transactions per second with sub-100ms latency. Analyze step by step the data pipeline architecture, ML model serving strategy, feature store design, and alerting system. Compare at least 3 approaches.",
        "Architect a multi-region active-active database with conflict resolution. Step by step, explain the trade-offs between eventual consistency and strong consistency. Design the replication topology and failover strategy.",
        "Design a zero-trust security architecture for a multi-account AWS organization. Analyze the requirements for identity federation, network segmentation, data encryption at rest and in transit, secrets management, and automated incident response.",
        "Build a serverless event-driven architecture for an e-commerce platform with CQRS and event sourcing. Step by step, design the service boundaries, event schema, saga pattern for distributed transactions, and dead letter queue handling.",
        "Design a comprehensive ML pipeline with feature store, model registry, A/B testing, and canary deployments. Analyze step by step how to handle data drift detection, model retraining triggers, and rollback strategies.",
    ],
}


# ── Endpoints ───────────────────────────────────────────────────────

@router.get("/rollout-configs")
def get_configs():
    """Return the 3 config files (raw JSONC with comments) for display in the UI."""
    return {
        "ab_test": AB_CONFIG_RAW,
        "canary": CANARY_CONFIG_RAW,
        "shadow": SHADOW_CONFIG_RAW,
    }


@router.get("/rollout-stats")
def get_stats(mode: str = "ab_test"):
    """Return current stats for the selected mode."""
    if mode == "ab_test":
        return {"stats": _ab_router.ab_test.stats}
    elif mode == "canary":
        return {"stats": _canary_router.canary.stats}
    elif mode == "shadow":
        return {"stats": _shadow_router.shadow.stats}
    return {"stats": {}}


@router.post("/rollout-simulate")
async def simulate_rollout(
    mode: str = Form("ab_test"),
    num_users: int = Form(20),
    requests_per_user: int = Form(3),
    complexity: str = Form("mixed"),
    speed: str = Form("fast"),
    config_overrides: str = Form("{}"),
):
    """Run a rollout simulation. Streams results via SSE."""
    delay_map = {"slow": 1.5, "normal": 0.8, "fast": 0.3}
    delay = delay_map.get(speed, 0.5)

    # Apply config overrides (from UI sliders)
    overrides = json.loads(config_overrides) if config_overrides != "{}" else {}

    # Select or create router based on mode + overrides
    if mode == "ab_test":
        if overrides:
            config = {**AB_CONFIG, **overrides}
            sim_router = BedrockRouter.create(config)
        else:
            sim_router = _ab_router
    elif mode == "canary":
        if overrides:
            config = {**CANARY_CONFIG, **overrides}
            sim_router = BedrockRouter.create(config)
        else:
            sim_router = _canary_router
    elif mode == "shadow":
        if overrides:
            config = {**SHADOW_CONFIG, **overrides}
            sim_router = BedrockRouter.create(config)
        else:
            sim_router = _shadow_router
    else:
        sim_router = _ab_router

    # Pick simulated users
    users = random.sample(SIM_USERS, min(num_users, len(SIM_USERS)))

    async def event_stream():
        yield f"event: session_init\ndata: {json.dumps({'mode': mode, 'num_users': len(users)})}\n\n"

        variant_counts: dict[str, int] = {}
        variant_costs: dict[str, list] = {}
        variant_latencies: dict[str, list] = {}
        user_assignments: dict[str, str] = {}
        loop = asyncio.get_event_loop()

        for req_num in range(requests_per_user):
            # Fire all users in parallel
            futures = {}
            for user_id in users:
                cx = random.choice(["simple", "moderate", "complex"]) if complexity == "mixed" else complexity
                prompt = random.choice(SIM_PROMPTS[cx])
                futures[user_id] = {
                    "future": loop.run_in_executor(
                        None, _run_rollout_request, sim_router, prompt, user_id, mode
                    ),
                    "prompt": prompt,
                }

            # Collect results
            for user_id, ctx in futures.items():
                try:
                    result = await ctx["future"]

                    variant = result.get("variant", "primary")
                    variant_counts[variant] = variant_counts.get(variant, 0) + 1
                    variant_costs.setdefault(variant, []).append(result["cost"])
                    variant_latencies.setdefault(variant, []).append(result["latency_ms"])
                    user_assignments[user_id] = variant

                    event_data = {
                        "user_id": user_id,
                        "request_num": req_num + 1,
                        "prompt": ctx["prompt"][:80],
                        "response_text": result.get("response_text", ""),
                        "model": result["display_model"],
                        "variant": variant,
                        "is_canary": result.get("is_canary", False),
                        "is_shadow": result.get("is_shadow", False),
                        "cost": round(result["cost"], 6),
                        "latency_ms": round(result["latency_ms"], 0),
                        "complexity": result.get("complexity", "simple"),
                    }
                    yield f"event: request_complete\ndata: {json.dumps(event_data)}\n\n"

                except Exception as e:
                    yield f"event: request_error\ndata: {json.dumps({'user_id': user_id, 'error': str(e)[:100]})}\n\n"

            # Send stats update after each round
            total = sum(variant_counts.values())
            stats_data = {
                "total_requests": total,
                "variant_counts": variant_counts,
                "variant_pcts": {k: round(v / total * 100, 1) for k, v in variant_counts.items()} if total > 0 else {},
                "variant_avg_cost": {k: round(sum(v) / len(v), 6) for k, v in variant_costs.items()},
                "variant_avg_latency": {k: round(sum(v) / len(v), 0) for k, v in variant_latencies.items()},
            }

            # Mode-specific stats
            if mode == "canary":
                stats_data["canary_stats"] = sim_router.canary.stats
            elif mode == "ab_test":
                stats_data["ab_stats"] = sim_router.ab_test.stats
                # Sticky verification: show first 5 users and their consistent assignment
                sticky_sample = dict(user_assignments)
                stats_data["sticky_sample"] = sticky_sample
            elif mode == "shadow":
                stats_data["shadow_stats"] = sim_router.shadow.stats
                # Include recent shadow results for the log
                shadow_results = sim_router.shadow.results  # All results
                stats_data["shadow_log"] = [
                    {"model": r.shadow_model, "latency_ms": round(r.latency_ms, 0), "success": r.success, "error": r.error, "prompt": r.prompt, "response_text": r.response_text, "cost": round(r.cost, 6)}
                    for r in shadow_results
                ]

            yield f"event: stats_update\ndata: {json.dumps(stats_data)}\n\n"

            if req_num < requests_per_user - 1:
                await asyncio.sleep(delay)

        # Final summary
        yield f"event: simulation_complete\ndata: {json.dumps({'total': sum(variant_counts.values()), 'variants': variant_counts})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.post("/rollout-reset")
def reset_rollout(mode: str = Form("ab_test")):
    """Reset a rollout router (re-create from config)."""
    global _ab_router, _canary_router, _shadow_router
    if mode == "ab_test":
        _ab_router = BedrockRouter.create(AB_CONFIG)
    elif mode == "canary":
        _canary_router = BedrockRouter.create(CANARY_CONFIG)
    elif mode == "shadow":
        _shadow_router = BedrockRouter.create(SHADOW_CONFIG)
    return {"status": "ok"}


# ── Internal ────────────────────────────────────────────────────────

def _run_rollout_request(sim_router: BedrockRouter, prompt: str,
                         user_id: str, mode: str) -> dict:
    """Execute a single request and return result with variant info."""
    t_start = time.perf_counter()
    response = sim_router.converse(
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        routing=RoutingConfig(metadata={"user_id": user_id}),
        inferenceConfig={"maxTokens": 100},
    )
    latency_ms = (time.perf_counter() - t_start) * 1000

    d = response["routing_decision"]
    usage = response.get("usage", {})
    cost = d.actual_cost if d.actual_cost and d.actual_cost > 0 else compute_cost(
        d.selected_model,
        usage.get("inputTokens", 0),
        usage.get("outputTokens", 0),
    )

    # Determine variant assignment
    variant = "primary"
    is_canary = False
    is_shadow = False
    metadata = d.metadata or {}

    if mode == "ab_test":
        variant = metadata.get("ab_variant", "unknown")
    elif mode == "canary":
        is_canary = metadata.get("is_canary", False)
        variant = "canary" if is_canary else "baseline"
    elif mode == "shadow":
        is_shadow = True
        variant = "primary"

    return {
        "display_model": display_model_name(d.selected_model),
        "model_id": d.selected_model,
        "variant": variant,
        "is_canary": is_canary,
        "is_shadow": is_shadow,
        "cost": cost,
        "latency_ms": latency_ms,
        "complexity": d.complexity_detected,
        "response_text": _extract_text(response),
    }


def _extract_text(response: dict) -> str:
    """Extract text from Bedrock Converse response."""
    try:
        for block in response.get("output", {}).get("message", {}).get("content", []):
            if isinstance(block, dict) and "text" in block:
                return block["text"][:150]
    except Exception:
        pass
    return ""
