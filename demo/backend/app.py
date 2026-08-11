# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""FastAPI backend for the Smart Router demo.

Modular structure:
- shared.py: Configuration, clients, helpers
- routes_compare.py: Use-case 1 (Baseline vs Smart Router)
- routes_throttle.py: Use-case 2 (Throttle Handling)
"""
from __future__ import annotations

import os
# Must be set before any strands_tools imports
os.environ["BYPASS_TOOL_CONSENT"] = "true"
# Force matplotlib to use non-interactive backend (no GUI popups)
os.environ["MPLBACKEND"] = "Agg"

# Enable DEBUG logging for semantic cache to see scores
import logging
logging.getLogger("bedrock_smart_router.semantic_cache").setLevel(logging.DEBUG)
logging.getLogger("bedrock_smart_router.intent_extractor").setLevel(logging.DEBUG)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from shared import (
    REGION, BASELINE_MODEL, BASELINE_MODELS, ROUTER_STRATEGIES,
    TEMPLATES, router, get_temp_file, display_model_name, JUDGE_MODEL,
)
from routes_compare import router as compare_router
from routes_throttle import router as throttle_router
from routes_strands import router as strands_router
from routes_multi_tenant import router as multi_tenant_router
from routes_text2sql import router as text2sql_router
from routes_guardrails import router as guardrails_router
from routes_usage import router as usage_router
from routes_rollout import router as rollout_router

app = FastAPI(title="Bedrock Smart Router Demo")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount use-case routers
app.include_router(compare_router, prefix="/api")
app.include_router(throttle_router, prefix="/api")
app.include_router(strands_router, prefix="/api")
app.include_router(multi_tenant_router, prefix="/api")
app.include_router(text2sql_router, prefix="/api")
app.include_router(guardrails_router, prefix="/api")
app.include_router(usage_router, prefix="/api")
app.include_router(rollout_router, prefix="/api")

# Initialize Text2SQL database on startup
import threading

def _init_text2sql_db():
    from text2sql.db import init_database
    init_database()
threading.Thread(target=_init_text2sql_db, daemon=True).start()


# ── Common Endpoints ────────────────────────────────────────────────

@app.get("/api/templates")
def get_templates():
    return TEMPLATES


@app.get("/api/options")
def get_options():
    """Return available baseline models, router strategies, and preferred models for the configured region."""
    all_models = router._registry.all_models
    seen = {}
    for m in all_models:
        if m.display_name not in seen or not m.model_id.startswith("global."):
            seen[m.display_name] = m

    preferred = [{"id": "", "label": "Auto (router decides)"}]
    for m in seen.values():
        # Only show models reachable via Converse or Chat Completions API
        # (Responses-only models like GPT-5.x can't be used as preferred targets)
        if any(api in m.api_support for api in ("converse", "chat_completions")):
            preferred.append({"id": m.model_id, "label": m.display_name})

    # Models available in the configured region (with their resolved profile IDs)
    from shared import REGION
    region_models = []
    for m in all_models:
        if m.model_id.startswith("global."):
            continue  # Skip global duplicates
        # Check if available in our region
        for r in m.regions:
            if r.get("name") == REGION:
                # Resolve the profile ID that would be used
                profile_id = m.model_id  # default: direct
                if "global" in r.get("cris_profiles", []):
                    profile_id = f"global.{m.model_id}"
                elif r.get("cris_profiles"):
                    profile_id = f"{r['cris_profiles'][0]}.{m.model_id}"
                region_models.append({
                    "id": m.model_id,
                    "label": m.display_name,
                    "profile_id": profile_id,
                    "tier": m.tier.value,
                })
                break

    return {
        "baseline_models": [{"id": k, "label": v["label"]} for k, v in BASELINE_MODELS.items()],
        "router_strategies": ROUTER_STRATEGIES,
        "preferred_models": preferred,
        "region_models": region_models,
        "region": REGION,
        "judge_model": display_model_name(JUDGE_MODEL),
    }


@app.get("/api/health")
def health():
    return {"status": "ok", "region": REGION, "baseline_model": BASELINE_MODEL}


@app.get("/api/check-file")
def check_file(file_id: str = ""):
    """Check if a temp file still exists (not expired)."""
    if not file_id:
        return {"exists": False}
    fb, _, _ = get_temp_file(file_id)
    return {"exists": fb is not None}
