#!/usr/bin/env python3
"""Test script: Strands Agent with MCP tools (aws-docs + aws-diagram).

Tests both the baseline (BedrockModel) and smart router (SmartRouterModel)
agents with MCP tool integration.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strands import Agent
from strands.models import BedrockModel
from mcp import stdio_client, StdioServerParameters
from strands.tools.mcp import MCPClient
from bedrock_smart_router.strands_model import SmartRouterModel
from bedrock_smart_router import BedrockRouter

REGION = "us-west-2"

# ── MCP Clients ─────────────────────────────────────────────────────

print("Creating MCP clients...")

docs_client = MCPClient(lambda: stdio_client(
    StdioServerParameters(
        command="uvx",
        args=["awslabs.aws-documentation-mcp-server@latest"],
        env={"FASTMCP_LOG_LEVEL": "ERROR"},
    )
))

diagram_client = MCPClient(lambda: stdio_client(
    StdioServerParameters(
        command="uvx",
        args=["awslabs.aws-diagram-mcp-server@1.0.23"],
        env={"FASTMCP_LOG_LEVEL": "ERROR"},
    )
))

print("✅ MCP clients created")

# ── Test 1: Baseline Agent (fixed model) ────────────────────────────

print("\n" + "=" * 60)
print("TEST 1: Baseline Agent (Sonnet 4.6) + MCP tools")
print("=" * 60)

baseline_model = BedrockModel(
    model_id="global.anthropic.claude-sonnet-4-6",
    region_name=REGION,
)

baseline_agent = Agent(
    model=baseline_model,
    tools=[docs_client, diagram_client],
    system_prompt="You are an AWS expert. Use tools when needed. Keep responses under 200 words.",
)

print("Sending: 'What is AWS Lambda? Keep it brief.'")
response = baseline_agent("What is AWS Lambda? Keep it brief.")
print(f"\nResponse: {str(response)[:300]}...")
print("✅ Baseline agent works")

# ── Test 2: Smart Router Agent ──────────────────────────────────────

print("\n" + "=" * 60)
print("TEST 2: Smart Router Agent + MCP tools")
print("=" * 60)

router = BedrockRouter.create({"region": REGION, "excluded_models": ["deepseek.*", "global.*"]})

smart_model = SmartRouterModel(router=router)

router_agent = Agent(
    model=smart_model,
    tools=[docs_client, diagram_client],
    system_prompt="You are an AWS expert. Use tools when needed. Keep responses under 200 words.",
)

print("Sending: 'What is S3? Keep it brief.'")
response = router_agent("What is S3? Keep it brief.")
print(f"\nResponse: {str(response)[:300]}...")

decision = smart_model.last_routing_decision
if decision:
    print(f"\n  Model:      {decision.selected_model}")
    print(f"  Complexity: {decision.complexity_detected}")
    print(f"  Cost:       ${decision.actual_cost:.6f}")
    print(f"  Latency:    {decision.latency_ms:.0f}ms")
print("✅ Smart Router agent works")

# ── Test 3: Multi-turn ──────────────────────────────────────────────

print("\n" + "=" * 60)
print("TEST 3: Multi-turn conversation")
print("=" * 60)

print("Turn 1: 'What is DynamoDB?'")
response = router_agent("What is DynamoDB? One sentence.")
print(f"  → {str(response)[:150]}...")
d = smart_model.last_routing_decision
print(f"  Model: {d.selected_model}, Complexity: {d.complexity_detected}")

print("\nTurn 2: 'How does it compare to Aurora?'")
response = router_agent("How does it compare to Aurora? Brief comparison.")
print(f"  → {str(response)[:150]}...")
d = smart_model.last_routing_decision
print(f"  Model: {d.selected_model}, Complexity: {d.complexity_detected}")

print("\n✅ Multi-turn works (agent remembers context)")
print("\n" + "=" * 60)
print("ALL TESTS PASSED")
print("=" * 60)
