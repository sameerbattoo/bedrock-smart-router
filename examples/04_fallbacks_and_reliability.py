"""Fallbacks and Reliability — circuit breakers, retries, fallback chains.

Demonstrates:
  - Viewing the fallback chain for a routing decision
  - Configuring circuit breaker thresholds
  - Configuring retry behavior
  - Handling fallback scenarios
"""

from bedrock_smart_router import BedrockRouter, RoutingConfig
from bedrock_smart_router.config import RouterConfig

# ── Example 1: Inspect the fallback chain ────────────────────────────

router = BedrockRouter.create()

response = router.converse(
    messages=[{"role": "user", "content": [{"text": "Hello"}]}],
)
d = response["routing_decision"]
print(f"Primary model: {d.selected_model}")
print(f"Fallback chain: {d.fallback_chain}")
print(f"Fallback used: {d.fallback_used}")
if d.fallback_model:
    print(f"Fell back to: {d.fallback_model}")


# ── Example 2: Aggressive circuit breaker config ─────────────────────
# Open the circuit after just 3 failures in 30 seconds.

router = BedrockRouter.create({
    "circuit_breaker": {
        "failure_threshold": 3,
        "window_seconds": 30,
        "cooldown_seconds": 15,
        "throttle_cooldown_seconds": 5,
    },
    "fallback": {
        "max_depth": 5,
        "default_safe_model": "us.amazon.nova-lite-v1:0",
    },
    "retry": {
        "max_retries": 2,
        "backoff_base_seconds": 0.3,
    },
})

response = router.converse(
    messages=[{"role": "user", "content": [{"text": "Test reliability"}]}],
)
print(f"\nReliable config → {response['routing_decision'].selected_model}")
print(f"Circuit breaker skipped: {response['routing_decision'].circuit_breaker_skipped}")


# ── Example 3: Check circuit breaker states ──────────────────────────

from bedrock_smart_router.circuit_breaker import CircuitBreakerRegistry

# After some usage, inspect which models have open circuits
# (In a real app, this would be after actual failures)
states = router._circuit_breakers.get_all_states()
for model_id, state in states.items():
    print(f"  {model_id}: {state.value}")
