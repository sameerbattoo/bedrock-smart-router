"""Error Handling — graceful no-models-match with actionable feedback.

Demonstrates:
  - Catching NoModelsMatchError
  - Inspecting per-model rejection reasons
  - Using suggestions to fix the issue
  - Structured error for API responses
"""

from bedrock_smart_router import BedrockRouter, RoutingConfig, NoModelsMatchError

router = BedrockRouter.create()


# ── Example 1: Impossible family filter ──────────────────────────────

try:
    router.converse(
        messages=[{"role": "user", "content": [{"text": "Hello"}]}],
        routing=RoutingConfig(preferred_family="nonexistent-family"),
    )
except NoModelsMatchError as e:
    print("Caught NoModelsMatchError:")
    print(f"  Constraints: {e.constraints}")
    print(f"  Suggestions: {e.suggestions}")
    print(f"  Rejections ({len(e.rejections)} models):")
    for r in e.rejections[:3]:
        print(f"    {r.display_name}: {', '.join(r.reasons)}")


# ── Example 2: Budget too tight ──────────────────────────────────────

try:
    router.converse(
        messages=[{"role": "user", "content": [{"text": "Hello"}]}],
        routing=RoutingConfig(
            preset="economy",
            max_cost_per_request=0.0000001,  # Impossibly low
            preferred_family="anthropic",
        ),
    )
except NoModelsMatchError as e:
    print(f"\nBudget too tight:")
    print(f"  Suggestions: {e.suggestions}")


# ── Example 3: Structured error for API responses ────────────────────

try:
    router.converse(
        messages=[{"role": "user", "content": [{"text": "Hello"}]}],
        routing=RoutingConfig(preferred_family="nonexistent"),
    )
except NoModelsMatchError as e:
    # Convert to dict for JSON API response
    error_dict = e.to_dict()
    print(f"\nStructured error:")
    print(f"  error: {error_dict['error']}")
    print(f"  rejections: {len(error_dict['rejections'])} models")
    print(f"  suggestions: {error_dict['suggestions']}")
    # Return as HTTP 422: json.dumps(error_dict)
