# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strands First Agent — the 1st official Strands Agents SDK sample, adapted to use our smart router.

Based on the official AWS Strands Agents "First Agent" quickstart sample:
https://github.com/strands-agents/samples/blob/main/python/01-learn/01-first-agent/01-first-agent.ipynb

The original sample uses a fixed Bedrock model (Claude Sonnet 4.5) for every
call. This adapted version replaces it with SmartRouterModel so that every
agent call is intelligently routed across Bedrock models based on cost,
latency, quality, and task complexity.

Demonstrates:
  1. Simple agent (original → smart-routed)
  2. Agent with tools (calculator + custom weather tool)
  3. Changing model provider (BedrockModel → SmartRouterModel)
  4. RecipeBot — a task-specific agent with web search

Requirements:
    pip install bedrock-smart-router[strands]
    pip install strands-agents-tools   # for calculator tool
    pip install ddgs                   # for RecipeBot web search
"""

from strands import Agent, tool
from bedrock_smart_router.strands_model import SmartRouterModel


# ═══════════════════════════════════════════════════════════════════════
# 1. SIMPLE AGENT
# ═══════════════════════════════════════════════════════════════════════
#
# ORIGINAL (from AWS sample — fixed model, no routing):
#
#   agent = Agent(
#       model="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
#       system_prompt="You are a helpful assistant that provides concise responses."
#   )
#   response = agent("Hello! Tell me a joke.")
#
# SMART-ROUTED (router picks the optimal model per request):

print("=" * 60)
print("1. Simple Agent — smart-routed")
print("=" * 60)

model = SmartRouterModel(
    router_config={"region": "us-west-2"},
    system_prompt="You are a helpful assistant that provides concise responses.",
)
agent = Agent(
    model=model,
    system_prompt="You are a helpful assistant that provides concise responses.",
)

response = agent("Hello! Tell me a joke.")
print(response)

d = model.last_routing_decision
print(f"\n  → Model:      {d.selected_model}")
print(f"  → Complexity: {d.complexity_detected}")
print(f"  → Cost:       ${d.actual_cost:.6f}")
print(f"  → Strategy:   {d.strategy_used}")


# ═══════════════════════════════════════════════════════════════════════
# 2. AGENT WITH TOOLS
# ═══════════════════════════════════════════════════════════════════════
#
# ORIGINAL (from AWS sample — fixed model):
#
#   from strands_tools import calculator
#
#   @tool
#   def weather():
#       """Get weather"""
#       return "sunny"
#
#   agent = Agent(
#       model="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
#       tools=[calculator, weather],
#       system_prompt="You're a helpful assistant. You can do simple math
#                      calculation, and tell the weather."
#   )
#   response = agent("What is the weather today?")
#
# SMART-ROUTED (router picks a tool-capable model automatically):

print("\n" + "=" * 60)
print("2. Agent with Tools — smart-routed")
print("=" * 60)

@tool
def weather() -> str:
    """Get the current weather."""
    return "sunny, 24°C"

# Note: we skip strands_tools.calculator here to avoid the extra dependency.
# In your code, just add it to the tools list: tools=[calculator, weather]
@tool
def calculate(expression: str) -> str:
    """Evaluate a math expression and return the result."""
    return str(eval(expression))  # noqa: S307

model = SmartRouterModel(router_config={"region": "us-west-2"})
agent = Agent(
    model=model,
    tools=[calculate, weather],
    system_prompt="You're a helpful assistant. You can do simple math calculation, and tell the weather.",
)

response = agent("What is the weather today?")
print(response)
d = model.last_routing_decision
print(f"\n  → Model: {d.selected_model} (tool use)")

response = agent("What is 123 * 456?")
print(response)
d = model.last_routing_decision
print(f"\n  → Model: {d.selected_model} (tool use)")


# ═══════════════════════════════════════════════════════════════════════
# 3. MODEL PROVIDER — BedrockModel vs SmartRouterModel
# ═══════════════════════════════════════════════════════════════════════
#
# ORIGINAL (from AWS sample — explicit BedrockModel with fixed config):
#
#   import boto3
#   from strands.models import BedrockModel
#
#   bedrock_model = BedrockModel(
#       model_id="us.anthropic.claude-haiku-4-5-20251001-v1:0",
#       region_name='us-west-2',
#       temperature=0.3,
#   )
#   agent = Agent(model=bedrock_model)
#
# SMART-ROUTED (router picks the model, you control the strategy):

print("\n" + "=" * 60)
print("3. Model Provider — SmartRouterModel replaces BedrockModel")
print("=" * 60)

# Economy preset — cheapest model for simple tasks
economy_model = SmartRouterModel(
    router_config={"region": "us-west-2"},
    routing_preset="economy",
    temperature=0.3,
)
agent = Agent(model=economy_model)
response = agent("What is Amazon S3?")
d = economy_model.last_routing_decision
print(f"  Economy  → {d.selected_model}, cost: ${d.actual_cost:.6f}")

# Quality preset — best model for complex tasks
quality_model = SmartRouterModel(
    router_config={"region": "us-west-2"},
    routing_preset="quality",
    temperature=0.3,
)
agent = Agent(model=quality_model)
response = agent("Design a serverless event-driven architecture for an e-commerce platform.")
d = quality_model.last_routing_decision
print(f"  Quality  → {d.selected_model}, cost: ${d.actual_cost:.6f}")

# Balanced (default) — weighted composite of cost, latency, quality
balanced_model = SmartRouterModel(
    router_config={"region": "us-west-2"},
    temperature=0.3,
)
agent = Agent(model=balanced_model)
response = agent("Explain VPC peering in 2 sentences.")
d = balanced_model.last_routing_decision
print(f"  Balanced → {d.selected_model}, cost: ${d.actual_cost:.6f}")


# ═══════════════════════════════════════════════════════════════════════
# 4. RECIPEBOT — Task-Specific Agent with Web Search
# ═══════════════════════════════════════════════════════════════════════
#
# ORIGINAL (from AWS sample — fixed Claude Sonnet 4.5):
#
#   from ddgs import DDGS
#
#   @tool
#   def websearch(keywords: str, ...) -> str:
#       ...
#
#   recipe_agent = Agent(
#       model="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
#       system_prompt="You are RecipeBot, a helpful cooking assistant...",
#       tools=[websearch],
#   )
#   response = recipe_agent("Suggest a recipe with chicken and broccoli.")
#
# SMART-ROUTED (router picks the best model — a simple recipe lookup
# goes to a cheap model, a complex cooking science question goes to a
# heavier model):

print("\n" + "=" * 60)
print("4. RecipeBot — smart-routed task-specific agent")
print("=" * 60)

try:
    from ddgs import DDGS
    from ddgs.exceptions import RatelimitException, DDGSException

    @tool
    def websearch(keywords: str, region: str = "us-en", max_results: int = 3) -> str:
        """Search the web to get updated information.

        Args:
            keywords: The search query keywords.
            region: The search region (e.g. us-en, uk-en).
            max_results: Maximum number of results to return.

        Returns:
            Search results as a string.
        """
        try:
            results = DDGS().text(keywords, region=region, max_results=max_results)
            return str(results) if results else "No results found."
        except RatelimitException:
            return "Rate limited — please try again shortly."
        except DDGSException as e:
            return f"Search error: {e}"
        except Exception as e:
            return f"Error: {e}"

    # SmartRouterModel with balanced strategy — the router will pick
    # a cheap model for simple questions and a heavier model for
    # complex cooking science questions.
    model = SmartRouterModel(
        router_config={"region": "us-west-2"},
        max_tokens=1024,
    )

    recipe_agent = Agent(
        model=model,
        system_prompt="""You are RecipeBot, a helpful cooking assistant.
        Help users find recipes based on ingredients and answer cooking questions.
        Use the websearch tool to find recipes when users mention ingredients or to
        look up cooking information. Keep responses concise.""",
        tools=[websearch],
    )

    # Simple recipe lookup
    response = recipe_agent("Suggest a quick recipe with chicken and broccoli.")
    print(response)
    d = model.last_routing_decision
    print(f"\n  → Model: {d.selected_model}, complexity: {d.complexity_detected}, cost: ${d.actual_cost:.6f}")

except ImportError:
    print("  Skipped — install 'ddgs' package for RecipeBot: pip install ddgs")


# ═══════════════════════════════════════════════════════════════════════
# SUMMARY: What changed from the original AWS sample
# ═══════════════════════════════════════════════════════════════════════
#
# 1. Replace:  model="us.anthropic.claude-sonnet-4-5-20250929-v1:0"
#    With:     model=SmartRouterModel(router_config={"region": "us-west-2"})
#
# 2. That's it. Everything else stays the same — tools, system prompts,
#    agent invocation, response handling. The router handles:
#    - Model selection based on request complexity
#    - Fallbacks if a model fails or is throttled
#    - Circuit breakers to skip failing models
#    - Cost tracking on every call
#    - CRIS profile and inference tier selection
#
# 3. Optional: add routing_preset="economy" or "quality" to control
#    the cost/quality trade-off. Add max_cost_per_request for cost caps.
#
# 4. Access routing decisions via model.last_routing_decision after
#    each call for observability.
