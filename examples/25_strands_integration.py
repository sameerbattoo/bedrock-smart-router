# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strands Agents Integration — use the smart router as a Strands Model provider.

Demonstrates:
  1. Basic agent with smart routing
  2. Routing presets (economy, speed, quality)
  3. Accessing routing decisions after each call
  4. Tool use with smart routing
  5. Multi-turn conversation
  6. Non-streaming mode
  7. Passing an existing router instance
  8. Per-request routing overrides via update_config

Requirements:
    pip install strands-agents
    # or
    pip install bedrock-smart-router[strands]
"""

from strands import Agent, tool
from bedrock_smart_router import RoutingConfig
from bedrock_smart_router.strands_model import SmartRouterModel


# ── Example 1: Basic agent with smart routing ────────────────────────
# The SmartRouterModel replaces Strands' default BedrockModel.
# All routing intelligence is automatic — complexity analysis, model
# selection, fallbacks, circuit breakers, etc.

model = SmartRouterModel(router_config={"region": "us-west-2"})
agent = Agent(model=model)

response = agent("What is Amazon S3?")
print(response)

# Every call produces a routing decision you can inspect.
d = model.last_routing_decision
print(f"\nRouted to:   {d.selected_model}")
print(f"Strategy:    {d.strategy_used}")
print(f"Complexity:  {d.complexity_detected}")
print(f"Cost:        ${d.actual_cost:.6f}")
print(f"Latency:     {d.latency_ms:.0f}ms")


# ── Example 2: Routing presets ───────────────────────────────────────
# Use presets to control the cost/quality/speed trade-off.

economy_model = SmartRouterModel(
    router_config={"region": "us-west-2"},
    routing_preset="economy",
)
economy_agent = Agent(model=economy_model)
response = economy_agent("Summarise S3 in one sentence.")
print(f"\nEconomy → {economy_model.last_routing_decision.selected_model}")
print(f"Cost: ${economy_model.last_routing_decision.actual_cost:.6f}")

quality_model = SmartRouterModel(
    router_config={"region": "us-west-2"},
    routing_preset="quality",
)
quality_agent = Agent(model=quality_model)
response = quality_agent("Compare eventual consistency vs strong consistency in distributed systems.")
print(f"\nQuality → {quality_model.last_routing_decision.selected_model}")
print(f"Cost: ${quality_model.last_routing_decision.actual_cost:.6f}")


# ── Example 3: Tool use ─────────────────────────────────────────────
# Strands handles the agent loop (call model → execute tool → feed
# result back).  The router picks the best model that supports tool_use.

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"The weather in {city} is 22°C and sunny."

@tool
def calculate(expression: str) -> str:
    """Evaluate a math expression."""
    return str(eval(expression))  # noqa: S307

model = SmartRouterModel(router_config={"region": "us-west-2"})
agent = Agent(
    model=model,
    tools=[get_weather, calculate],
    system_prompt="You are a helpful assistant. Use tools when needed.",
)

response = agent("What's the weather in Seattle and what is 42 * 17?")
print(f"\n{response}")
print(f"Model used: {model.last_routing_decision.selected_model}")


# ── Example 4: Multi-turn conversation ───────────────────────────────
# Each turn is independently routed — a simple follow-up may go to a
# cheaper model while a complex question routes to a heavier one.

model = SmartRouterModel(router_config={"region": "us-west-2"}, max_tokens=1024)
agent = Agent(model=model)

response = agent("Hi, what's your name?")
print(f"\nTurn 1 → {model.last_routing_decision.selected_model} "
      f"({model.last_routing_decision.complexity_detected})")

response = agent("Explain the CAP theorem in 3 sentences.")
print(f"Turn 2 → {model.last_routing_decision.selected_model} "
      f"({model.last_routing_decision.complexity_detected})")


# ── Example 5: Non-streaming mode ────────────────────────────────────
# Some models don't support streaming tool use.  Set streaming=False
# and the adapter uses router.converse() instead.

model = SmartRouterModel(
    router_config={"region": "us-west-2"},
    streaming=False,
)
agent = Agent(model=model)
response = agent("What is Lambda?")
print(f"\nNon-streaming → {model.last_routing_decision.selected_model}")


# ── Example 6: Bring your own router ────────────────────────────────
# If you already have a configured BedrockRouter, pass it directly.

from bedrock_smart_router import BedrockRouter

router = BedrockRouter.create({
    "region": "us-west-2",
    "strategy": "cost-optimized",
    "cache": {"enabled": True, "ttl": 300},
})

model = SmartRouterModel(router=router)
agent = Agent(model=model)
response = agent("What is DynamoDB?")
print(f"\nBYO router → {model.last_routing_decision.selected_model}")


# ── Example 7: Runtime config changes ────────────────────────────────
# Switch routing behaviour mid-conversation.

model = SmartRouterModel(router_config={"region": "us-west-2"})
agent = Agent(model=model)

response = agent("Hello!")
print(f"\nDefault → {model.last_routing_decision.selected_model}")

model.update_config(routing_preset="economy", max_cost_per_request=0.001)
response = agent("What is EC2?")
print(f"Economy  → {model.last_routing_decision.selected_model}")

model.update_config(routing_preset="quality", max_cost_per_request=None)
response = agent("Design a microservices architecture for a banking platform.")
print(f"Quality  → {model.last_routing_decision.selected_model}")
