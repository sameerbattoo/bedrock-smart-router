"""Model Catalog — JSON-driven registry with overlays and custom models.

Demonstrates:
  - Listing models by family, tier, capability
  - Loading a custom model catalog overlay
  - Registering distilled models
  - Refreshing pricing from AWS APIs
"""

from bedrock_smart_router import BedrockRouter

router = BedrockRouter.create()


# ── Example 1: List models by family ─────────────────────────────────

print("Anthropic models:")
for m in router.registry.list_models(family="anthropic"):
    print(f"  {m.display_name} ({m.tier.value}) — ${m.pricing.input_per_1k}/1K in")

print("\nAmazon Nova models:")
for m in router.registry.list_models(family="amazon"):
    print(f"  {m.display_name} ({m.tier.value}) — ${m.pricing.input_per_1k}/1K in")


# ── Example 2: Filter by capability ──────────────────────────────────

print("\nModels with vision:")
for m in router.registry.eligible_models(requires_vision=True):
    print(f"  {m.display_name}")

print("\nModels with tool use + 200K+ context:")
for m in router.registry.eligible_models(requires_tool_use=True, min_context=200_000):
    print(f"  {m.display_name} ({m.max_input_tokens:,} tokens)")


# ── Example 3: Load a custom overlay ─────────────────────────────────
# Add or override models without replacing the entire catalog.

import json, tempfile, os

overlay = {
    "models": [{
        "model_id": "my-fine-tuned-nova",
        "family": "amazon",
        "tier": "mid",
        "display_name": "My Fine-Tuned Nova Pro",
        "max_input_tokens": 300000,
        "max_output_tokens": 5000,
        "pricing": {"input_per_1k": 0.001, "output_per_1k": 0.004},
        "capabilities": {"tool_use": True, "vision": True, "streaming": True},
        "supported_inference_tiers": ["standard"],
    }]
}

with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
    json.dump(overlay, f)
    overlay_path = f.name

count = router.registry.load_overlay(overlay_path)
print(f"\nLoaded {count} models from overlay")
print(f"Custom model: {router.registry.get('my-fine-tuned-nova').display_name}")
os.unlink(overlay_path)


# ── Example 4: Register a distilled model ────────────────────────────

from bedrock_smart_router.distilled_models import DistilledModelManager

mgr = DistilledModelManager(router.registry)
distilled = mgr.register_distilled(
    model_id="my-distilled-sonnet",
    teacher_model_id="us.anthropic.claude-sonnet-4-6",
    quality_delta=-0.02,       # 2% quality loss
    cost_multiplier=0.25,      # 75% cheaper
    speed_multiplier=5.0,      # 5x faster
)
print(f"\nDistilled: {distilled.display_name}")
print(f"  Tier: {distilled.tier.value} (one below teacher)")
print(f"  Price: ${distilled.pricing.input_per_1k}/1K (vs ${0.003}/1K teacher)")
