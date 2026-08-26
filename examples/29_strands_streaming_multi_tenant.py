# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strands Streaming + Multi-Tenant Routing with Smart Router.

Adapted from the official Strands Agents SDK streaming sample:
https://github.com/strands-agents/samples/blob/main/python/01-learn/04-streaming/advanced_processing_agent_response.ipynb

The original sample demonstrates async streaming and callback handlers
with a fixed Claude Sonnet 4.5 model. This adapted version:

  1. Replaces BedrockModel with SmartRouterModel
  2. Shows multi-tenant routing — premium vs freemium customers get
     different models via routing presets and update_config()
  3. Uses callback handlers for real-time streaming output
  4. Tracks per-tenant cost and model selection

Requirements:
    pip install bedrock-smart-router[strands]
"""

from strands import Agent, tool
from bedrock_smart_router.strands_model import SmartRouterModel


# ═══════════════════════════════════════════════════════════════════
# Custom Tools — weather forecast and calculator (from AWS sample)
# ═══════════════════════════════════════════════════════════════════

@tool
def weather_forecast(city: str, days: int = 3) -> str:
    """Get the weather forecast for a city.

    Args:
        city: The city name.
        days: Number of forecast days (default 3).

    Returns:
        Weather forecast summary.
    """
    forecasts = {
        "NYC": "Partly cloudy, 18-24°C, 20% chance of rain",
        "Seattle": "Overcast, 12-17°C, 60% chance of rain",
        "London": "Foggy mornings, 10-15°C, 40% chance of rain",
        "Tokyo": "Sunny, 22-28°C, 5% chance of rain",
    }
    forecast = forecasts.get(city, f"Mild conditions, 15-22°C for {city}")
    return f"Weather forecast for {city} ({days} days): {forecast}"


@tool
def calculate(expression: str) -> str:
    """Evaluate a math expression.

    Args:
        expression: The math expression to evaluate.

    Returns:
        The result as a string.
    """
    return str(eval(expression))  # noqa: S307


# ═══════════════════════════════════════════════════════════════════
# Callback Handler — real-time streaming output (from AWS sample)
# ═══════════════════════════════════════════════════════════════════

def streaming_callback(**kwargs):
    """Custom callback that shows streaming text and tool usage."""
    if "data" in kwargs:
        print(kwargs["data"], end="", flush=True)
    elif "current_tool_use" in kwargs and kwargs["current_tool_use"].get("name"):
        print(f"\n  🔧 Tool: {kwargs['current_tool_use']['name']}", flush=True)


# ═══════════════════════════════════════════════════════════════════
# Multi-Tenant Setup
# ═══════════════════════════════════════════════════════════════════
#
# Two tenants share the same agent code but get different routing:
#
# Premium tenant → quality preset → best models (Claude Opus, Sonnet)
# Freemium tenant → economy preset → cheapest models (Nova Micro, Llama)
#
# The SmartRouterModel's update_config() switches routing per tenant
# without recreating the agent.

# max_tokens must be generous enough for the longest response in the demo
# (the premium "Compare VPC peering vs Transit Gateway" query). Strands raises
# MaxTokensReachedException if a response is truncated at the limit, so a tight
# budget (e.g. 512) would abort the agent mid-answer.
model = SmartRouterModel(
    router_config={"region": "us-west-2"},
    max_tokens=2048,
)

agent = Agent(
    model=model,
    tools=[weather_forecast, calculate],
    callback_handler=streaming_callback,
    system_prompt="You are a helpful assistant. Use tools when needed. Keep responses concise.",
)

# Track costs per tenant
tenant_costs: dict[str, float] = {"premium": 0.0, "freemium": 0.0}


def serve_tenant(tenant_id: str, query: str, preset: str) -> None:
    """Handle a request for a specific tenant with the appropriate routing."""
    # Switch routing config for this tenant
    model.update_config(
        routing_preset=preset,
        tags=[f"{tenant_id}-tier"],
        metadata={"tenant": tenant_id},
    )

    print(f"\n{'─' * 60}")
    print(f"  Tenant: {tenant_id} | Preset: {preset}")
    print(f"  Query: {query}")
    print(f"{'─' * 60}")

    # Call the agent — streaming output via callback handler
    response = agent(query)

    # Show routing decision
    d = model.last_routing_decision
    tenant_costs[tenant_id] += d.actual_cost or 0
    print(f"\n  → Model: {d.selected_model}")
    print(f"  → Strategy: {d.strategy_used}")
    print(f"  → Cost: ${d.actual_cost:.6f}")
    print(f"  → Complexity: {d.complexity_detected}")
    print(f"  → Tenant cumulative cost: ${tenant_costs[tenant_id]:.6f}")


# ═══════════════════════════════════════════════════════════════════
# Demo: Premium vs Freemium Tenant Requests
# ═══════════════════════════════════════════════════════════════════

print("=" * 60)
print("Multi-Tenant Streaming Demo")
print("Premium (quality) vs Freemium (economy)")
print("=" * 60)

# --- Simple question: both tenants ---
serve_tenant("freemium", "What is 42 * 17?", preset="economy")
serve_tenant("premium", "What is 42 * 17?", preset="quality")

# --- Weather query with tool use ---
serve_tenant("freemium", "What's the weather in Seattle?", preset="economy")
serve_tenant("premium", "What's the weather in NYC and Tokyo?", preset="quality")

# --- Complex question: routing difference is most visible here ---
serve_tenant(
    "freemium",
    "Explain VPC peering in one sentence.",
    preset="economy",
)
serve_tenant(
    "premium",
    "Compare VPC peering vs Transit Gateway. When should I use each?",
    preset="quality",
)

# ═══════════════════════════════════════════════════════════════════
# Summary: Per-Tenant Cost Comparison
# ═══════════════════════════════════════════════════════════════════

print(f"\n{'=' * 60}")
print("Per-Tenant Cost Summary")
print(f"{'=' * 60}")
print(f"  Freemium (economy):  ${tenant_costs['freemium']:.6f}")
print(f"  Premium  (quality):  ${tenant_costs['premium']:.6f}")
if tenant_costs["freemium"] > 0:
    ratio = tenant_costs["premium"] / tenant_costs["freemium"]
    print(f"  Premium/Freemium ratio: {ratio:.1f}x")
print(f"{'=' * 60}")
