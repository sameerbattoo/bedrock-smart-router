"""FastAPI backend for the Smart Router demo.

Modular structure:
- shared.py: Configuration, clients, helpers
- routes_compare.py: Use-case 1 (Baseline vs Smart Router)
- routes_throttle.py: Use-case 2 (Throttle Handling)
"""
from __future__ import annotations

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
