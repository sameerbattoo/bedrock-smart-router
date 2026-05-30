"""Use-case 7: Usage & Cost Tracking — per-user budget enforcement.

Demonstrates the core library's integrated budget enforcement:
- Router configured with BudgetRules per tier
- SQLiteBudgetStore for persistence
- Automatic reject (free tier) and downgrade (pro/enterprise) on exceed
- No manual budget checks — the router handles everything
"""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Any

from fastapi import APIRouter, Form
from fastapi.responses import StreamingResponse

from bedrock_smart_router import BedrockRouter, RoutingConfig
from bedrock_smart_router.budget_strategy import BudgetExceededError, BudgetTracker
from shared import display_model_name, compute_cost

router = APIRouter()

# ── Router with Budget Enforcement ──────────────────────────────────
# This is the key demo: budget rules in the router config.
# The router automatically checks + records spend on every converse() call.

_smart_router = BedrockRouter.create({
    "strategy": "quality-optimized",
    "budget": {
        "tracker_backend": "sqlite",
        "sqlite_path": "/tmp/bsr_budget.db",
        "scope_key": "user_id",
        "rule_key": "tier",
        "sync_interval_seconds": 2,
        "rules": {
            # Per-user rules (keyed by tier)
            "free": {"max_hourly_spend": 0.005, "on_exceeded": "reject"},
            "pro": {"max_hourly_spend": 0.015, "on_exceeded": "downgrade"},
            "enterprise": {"max_hourly_spend": 0.05, "on_exceeded": "downgrade"},
            # Per-tenant rules (keyed by tenant name)
            "acme-corp": {"max_hourly_spend": 0.03, "on_exceeded": "downgrade"},
            "globex-inc": {"max_hourly_spend": 0.008, "on_exceeded": "reject"},
            "initech": {"max_hourly_spend": 0.05, "on_exceeded": "downgrade"},
            "umbrella-co": {"max_hourly_spend": 0.06, "on_exceeded": "downgrade"},
        },
    },
})

# ── User Personas ───────────────────────────────────────────────────

BUDGET_RULES = _smart_router._budget_rules
USERS = [
    {"id": "alice", "name": "Alice Chen", "team": "Engineering", "tier": "pro", "color": "purple"},
    {"id": "bob", "name": "Bob Martinez", "team": "Marketing", "tier": "free", "color": "blue"},
    {"id": "charlie", "name": "Charlie Kim", "team": "Data Science", "tier": "enterprise", "color": "green"},
    {"id": "diana", "name": "Diana Patel", "team": "Executive", "tier": "enterprise", "color": "orange"},
]

for u in USERS:
    rule = BUDGET_RULES.get(u["tier"])
    u["budget"] = rule.max_hourly_spend if rule else 0
    u["on_exceeded"] = rule.on_exceeded if rule else "downgrade"

# ── Tenant (Department) Personas ────────────────────────────────────

TENANTS = [
    {"id": "acme-corp", "name": "Acme Corp", "members": 12, "color": "purple"},
    {"id": "globex-inc", "name": "Globex Inc", "members": 8, "color": "blue"},
    {"id": "initech", "name": "Initech", "members": 6, "color": "green"},
    {"id": "umbrella-co", "name": "Umbrella Co", "members": 4, "color": "orange"},
]

for t in TENANTS:
    rule = BUDGET_RULES.get(t["id"])
    t["budget"] = rule.max_hourly_spend if rule else 0
    t["on_exceeded"] = rule.on_exceeded if rule else "downgrade"
    t["team"] = f"{t['members']} members"
    t["tier"] = t["id"]

# ── Simulation Prompts ──────────────────────────────────────────────

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

@router.get("/usage-users")
def get_users():
    """Return available user personas and tenant personas."""
    return {"users": USERS, "tenants": TENANTS}


@router.get("/usage-dashboard")
def get_dashboard(scope: str = "user"):
    """Return current spend from the persistent store (wall-clock accurate)."""
    from bedrock_smart_router.budget_store import SQLiteBudgetStore
    store = _smart_router._budget_tracker._store if _smart_router._budget_tracker else None

    entities = TENANTS if scope == "tenant" else USERS
    entities_data = []
    for entity in entities:
        eid = entity["id"]
        rule = BUDGET_RULES.get(entity.get("tier", eid))
        budget = rule.max_hourly_spend if rule else 0
        # Query the persistent store with wall-clock time (accurate after restart)
        spend = store.get_spend(eid, 3600) if store else 0
        remaining = max(0, budget - spend)
        pct = (spend / budget * 100) if budget > 0 else 0
        status = "over" if pct >= 100 else "warning" if pct >= 80 else "ok"
        entities_data.append({
            "user_id": eid, "name": entity["name"],
            "team": entity.get("team", ""),
            "tier": entity.get("tier", eid),
            "total_cost": round(spend, 6),
            "budget": budget, "remaining": round(remaining, 6),
            "pct_used": round(pct, 1), "status": status,
            "on_exceeded": entity.get("on_exceeded", "downgrade"),
            "color": entity["color"],
        })
    return {"entities": entities_data}


@router.post("/usage-simulate")
async def simulate_usage(
    selected_users: str = Form("alice,bob,charlie"),
    scope: str = Form("user"),
    strategy: str = Form("quality-optimized"),
    classifier: str = Form("heuristic"),
    complexity: str = Form("mixed"),
    requests_per_user: int = Form(20),
    speed: str = Form("normal"),
):
    """Run simulation. The router handles budget enforcement automatically."""
    import random

    entity_ids = [u.strip() for u in selected_users.split(",") if u.strip()]

    if scope == "tenant":
        entities = [t for t in TENANTS if t["id"] in entity_ids]
        if not entities:
            entities = TENANTS[:2]
    else:
        entities = [u for u in USERS if u["id"] in entity_ids]
        if not entities:
            entities = USERS[:2]

    session_id = str(uuid.uuid4())[:8]
    delay_map = {"slow": 2.0, "normal": 1.0, "fast": 0.3}
    delay = delay_map.get(speed, 1.0)

    async def event_stream():
        yield f"event: session_init\ndata: {json.dumps({'session_id': session_id})}\n\n"

        entity_was_rejected: dict[str, bool] = {e["id"]: False for e in entities}
        entity_was_downgraded: dict[str, bool] = {e["id"]: False for e in entities}
        loop = asyncio.get_event_loop()

        for req_num in range(requests_per_user):
            for entity in entities:
                eid = entity["id"]
                rule = BUDGET_RULES.get(entity.get("tier", eid))
                budget = rule.max_hourly_spend if rule else 0

                # Pick a prompt
                cx = random.choice(["simple", "moderate", "complex"]) if complexity == "mixed" else complexity
                prompt = random.choice(SIM_PROMPTS[cx])

                # Build metadata — for tenant mode, use tenant ID as user_id
                # so the router's scope_key ("user_id") tracks per-tenant
                if scope == "tenant":
                    metadata = {"user_id": eid, "tier": eid}
                else:
                    metadata = {"user_id": eid, "team": entity.get("team", ""), "tier": entity.get("tier", "pro")}

                try:
                    result = await loop.run_in_executor(
                        None, _run_request, prompt, metadata, strategy, classifier
                    )

                    # Detect downgrade (router switched to cost-optimized)
                    downgraded = result.get("strategy_used") == "cost-optimized" and strategy != "cost-optimized"
                    if downgraded and not entity_was_downgraded[eid]:
                        entity_was_downgraded[eid] = True
                        yield f"event: budget_exceeded\ndata: {json.dumps({'user_id': eid, 'name': entity['name'], 'budget': budget, 'action': 'downgrade'})}\n\n"

                    spend = _smart_router._budget_tracker.get_spend(eid, 3600)
                    event_data = {
                        "user_id": eid, "name": entity["name"],
                        "request_num": req_num + 1, "total_requests": requests_per_user,
                        "prompt": prompt[:80], "response_text": result["response_text"],
                        "model": result["display_model"], "complexity": result["complexity"],
                        "cost": round(result["cost"], 6),
                        "cumulative_cost": round(spend, 6), "budget": budget,
                        "remaining": round(max(0, budget - spend), 6),
                        "pct_used": round(spend / budget * 100, 1) if budget > 0 else 0,
                        "latency_ms": round(result["latency_ms"], 0),
                        "downgraded": downgraded, "rejected": False,
                        "status": "over" if spend >= budget else "warning" if spend >= budget * 0.8 else "ok",
                    }
                    yield f"event: request_complete\ndata: {json.dumps(event_data)}\n\n"

                except BudgetExceededError as e:
                    if not entity_was_rejected[eid]:
                        entity_was_rejected[eid] = True
                        yield f"event: budget_exceeded\ndata: {json.dumps({'user_id': eid, 'name': entity['name'], 'budget': budget, 'action': 'reject', 'reason': str(e)})}\n\n"

                    spend = _smart_router._budget_tracker.get_spend(eid, 3600)
                    event_data = {
                        "user_id": eid, "name": entity["name"],
                        "request_num": req_num + 1, "total_requests": requests_per_user,
                        "prompt": prompt[:80],
                        "response_text": "[REJECTED] Budget exceeded — no model called, $0 cost",
                        "model": "—", "complexity": "—", "cost": 0,
                        "cumulative_cost": round(spend, 6), "budget": budget,
                        "remaining": 0, "pct_used": 100.0, "latency_ms": 0,
                        "downgraded": False, "rejected": True, "status": "over",
                    }
                    yield f"event: request_complete\ndata: {json.dumps(event_data)}\n\n"

                except Exception as e:
                    yield f"event: request_error\ndata: {json.dumps({'user_id': eid, 'error': str(e)[:100]})}\n\n"

            if req_num < requests_per_user - 1:
                await asyncio.sleep(delay)

        yield f"event: simulation_complete\ndata: {json.dumps({'total_requests': requests_per_user * len(entities)})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.post("/usage-reset")
def reset_usage():
    """Reset budget tracking data."""
    tracker = _smart_router._budget_tracker
    if tracker:
        # Flush any pending writes before cleanup
        if hasattr(tracker, '_flush_pending'):
            tracker._flush_pending()
        if tracker._store:
            tracker._store.cleanup(older_than_seconds=0)
        # Clear in-memory spend (reuse existing tracker, don't orphan sync thread)
        with tracker._lock:
            tracker._spend.clear()
    return {"status": "ok"}


# ── Internal ────────────────────────────────────────────────────────

def _run_request(prompt: str, metadata: dict, strategy: str, classifier: str) -> dict:
    """Execute a single request through the router. Budget enforcement is automatic."""
    _smart_router._cache.invalidate()

    response = _smart_router.converse(
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        routing=RoutingConfig(
            strategy=strategy,
            metadata=metadata,
            classifier=classifier,
        ),
        inferenceConfig={"maxTokens": 100},
    )

    d = response["routing_decision"]
    usage = response.get("usage", {})
    cost = d.actual_cost if d.actual_cost and d.actual_cost > 0 else compute_cost(
        d.selected_model,
        usage.get("inputTokens", 0),
        usage.get("outputTokens", 0),
    )

    return {
        "display_model": display_model_name(d.selected_model),
        "complexity": d.complexity_detected,
        "strategy_used": d.strategy_used,
        "cost": cost,
        "latency_ms": d.latency_ms or 0,
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
