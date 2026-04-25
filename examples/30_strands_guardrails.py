"""Strands Agent with Bedrock Guardrails via Smart Router.

Adapted from the official Strands Agents SDK guardrails sample:
https://github.com/strands-agents/samples/blob/main/python/01-learn/05-guardrails/bedrock_guardrails_sample.ipynb

The original sample configures guardrails on the BedrockModel directly.
With the Smart Router, guardrails are configured at the ROUTER level:

  - Set once in the router config (guardrails.pre_route / post_route)
  - Input screened via ApplyGuardrail API BEFORE model selection
  - Works regardless of which model the router picks
  - Blocked requests never reach Bedrock inference (saves cost)

Demonstrates:
  1. Create a Bedrock Guardrail (blocks financial advice)
  2. Safe question → tool use works, guardrail_checked=True
  3. Blocked question → rejected pre-route, no inference cost
  4. Multi-turn conversation with guardrail intervention mid-chat
  5. Clean up the guardrail

Requirements:
    pip install bedrock-smart-router[strands]

Note: This example creates and deletes a Bedrock Guardrail. You need
      IAM permissions for bedrock:CreateGuardrail and bedrock:DeleteGuardrail.
"""

import boto3
from strands import Agent, tool
from bedrock_smart_router.strands_model import SmartRouterModel
from bedrock_smart_router import BedrockRouter


# ═══════════════════════════════════════════════════════════════════
# Step 1: Create a Bedrock Guardrail (same as AWS sample)
# ═══════════════════════════════════════════════════════════════════

print("=" * 60)
print("Creating Bedrock Guardrail...")
print("=" * 60)

bedrock_client = boto3.client("bedrock", region_name="us-west-2")

response = bedrock_client.create_guardrail(
    name="smart-router-demo-guardrail",
    description="Prevents financial advice and filters inappropriate content.",
    topicPolicyConfig={
        "topicsConfig": [
            {
                "name": "Fiduciary Advice",
                "definition": "Providing personalized advice on managing financial "
                              "assets, investments, or trusts.",
                "examples": [
                    "What stocks should I invest in for my retirement?",
                    "How should I allocate my 401(k) investments?",
                    "Should I hire a financial advisor?",
                ],
                "type": "DENY",
            }
        ]
    },
    contentPolicyConfig={
        "filtersConfig": [
            {"type": "SEXUAL", "inputStrength": "HIGH", "outputStrength": "HIGH"},
            {"type": "VIOLENCE", "inputStrength": "HIGH", "outputStrength": "HIGH"},
            {"type": "HATE", "inputStrength": "HIGH", "outputStrength": "HIGH"},
            {"type": "INSULTS", "inputStrength": "HIGH", "outputStrength": "HIGH"},
            {"type": "PROMPT_ATTACK", "inputStrength": "HIGH", "outputStrength": "NONE"},
        ]
    },
    wordPolicyConfig={
        "wordsConfig": [
            {"text": "fiduciary advice"},
            {"text": "investment recommendations"},
            {"text": "stock picks"},
            {"text": "portfolio allocation advice"},
        ],
        "managedWordListsConfig": [{"type": "PROFANITY"}],
    },
    blockedInputMessaging="I can't help with financial advice. Please ask about something else.",
    blockedOutputsMessaging="I can't provide that type of advice. Let me help with something else.",
)

guardrail_id = response["guardrailId"]
guardrail_version = "DRAFT"
print(f"  Guardrail ID: {guardrail_id}")
print(f"  Version: {guardrail_version}")


# ═══════════════════════════════════════════════════════════════════
# Custom tools for the customer support agent
# ═══════════════════════════════════════════════════════════════════

CUSTOMERS = {
    "CUST100": {"name": "Alice Johnson", "email": "alice@example.com", "tier": "premium"},
    "CUST101": {"name": "Bob Smith", "email": "bob@example.com", "tier": "standard"},
}


@tool
def get_customer(customer_id: str) -> str:
    """Look up a customer by their ID.

    Args:
        customer_id: The customer ID (e.g. CUST100).

    Returns:
        Customer profile as JSON string.
    """
    customer = CUSTOMERS.get(customer_id)
    if customer:
        return f"Customer {customer_id}: {customer['name']}, {customer['email']}, tier: {customer['tier']}"
    return f"Customer {customer_id} not found."


@tool
def get_account_balance(customer_id: str) -> str:
    """Get the account balance for a customer.

    Args:
        customer_id: The customer ID.

    Returns:
        Account balance information.
    """
    balances = {"CUST100": "$12,450.00", "CUST101": "$3,200.00"}
    balance = balances.get(customer_id, "Unknown")
    return f"Account balance for {customer_id}: {balance}"


# ═══════════════════════════════════════════════════════════════════
# Step 2: Create Strands Agent with Router-Level Guardrails
# ═══════════════════════════════════════════════════════════════════
# The guardrail is configured in the ROUTER, not the model.
# Input is screened via ApplyGuardrail API BEFORE model selection.
# Output is screened AFTER the model responds.
# This works regardless of which model the router picks.

print(f"\n{'=' * 60}")
print("Strands Agent with Router-Level Guardrails")
print("=" * 60)

router = BedrockRouter.create({
    "region": "us-west-2",
    "guardrails": {
        "pre_route": {
            "guardrail_id": guardrail_id,
            "guardrail_version": guardrail_version,
            "action_on_block": "reject",  # Reject blocked input
        },
    },
})

# Wrap the router as a Strands model
model1 = SmartRouterModel(router=router, max_tokens=256)
agent1 = Agent(
    model=model1,
    tools=[get_customer, get_account_balance],
    system_prompt="You are a helpful customer support assistant for a retail company.",
)

# Test 1a: Safe question — should work normally
print("\n  Test 1a: Safe question")
print("  Query: 'What is the account balance for CUST100?'")
try:
    response = agent1("What is the account balance for CUST100?")
    d = model1.last_routing_decision
    print(f"  → Model: {d.selected_model}")
    print(f"  → Guardrail checked: {d.guardrail_checked}")
except Exception as e:
    print(f"  → Error: {e}")

# Test 1b: Blocked question — guardrail should intervene
print("\n  Test 1b: Financial advice question (should be blocked)")
print("  Query: 'What stocks should I invest in for my retirement?'")
try:
    response = agent1("What stocks should I invest in for my retirement?")
    print(f"  → Response: {response}")
except Exception as e:
    error_msg = str(e)
    if "guardrail" in error_msg.lower() or "blocked" in error_msg.lower():
        print(f"  → ⚠️ GUARDRAIL BLOCKED (pre-route): Input rejected before model selection ✅")
    else:
        print(f"  → Error: {error_msg[:100]}")


# ═══════════════════════════════════════════════════════════════════
# Step 3: Multi-turn conversation with guardrails
# ═══════════════════════════════════════════════════════════════════

print(f"\n{'=' * 60}")
print("Multi-turn conversation with guardrails")
print("=" * 60)

# Create a fresh agent for multi-turn
model3 = SmartRouterModel(router=router, max_tokens=256)
agent3 = Agent(
    model=model3,
    tools=[get_customer, get_account_balance],
    system_prompt="You are a helpful customer support assistant for a retail company.",
)

# Turn 1: Safe question
print("\n  Turn 1: 'Look up customer CUST100'")
try:
    agent3("Look up customer CUST100")
    d = model3.last_routing_decision
    print(f"  → Model: {d.selected_model}, Cost: ${d.actual_cost:.6f}")
except Exception as e:
    print(f"  → Error: {e}")

# Turn 2: Follow-up safe question
print("\n  Turn 2: 'What is their account balance?'")
try:
    agent3("What is their account balance?")
    d = model3.last_routing_decision
    print(f"  → Model: {d.selected_model}, Cost: ${d.actual_cost:.6f}")
except Exception as e:
    print(f"  → Error: {e}")

# Turn 3: Blocked question mid-conversation
print("\n  Turn 3: 'How should they invest their balance?' (should be blocked)")
try:
    agent3("How should they invest their balance for maximum returns?")
    print(f"  → Response went through (unexpected)")
except Exception as e:
    error_msg = str(e)
    if "guardrail" in error_msg.lower() or "blocked" in error_msg.lower():
        print(f"  → ⚠️ GUARDRAIL BLOCKED: Financial advice rejected mid-conversation ✅")
    else:
        print(f"  → Error: {error_msg[:100]}")

# Turn 4: Safe question after blocked one — should work
print("\n  Turn 4: 'Look up customer CUST101' (should work)")
try:
    agent3("Look up customer CUST101")
    d = model3.last_routing_decision
    print(f"  → Model: {d.selected_model}, Cost: ${d.actual_cost:.6f} ✅")
except Exception as e:
    print(f"  → Error: {str(e)[:80]}")


# ═══════════════════════════════════════════════════════════════════
# Clean up: Delete the guardrail
# ═══════════════════════════════════════════════════════════════════

print(f"\n{'=' * 60}")
print("Cleaning up...")
print("=" * 60)

try:
    bedrock_client.delete_guardrail(guardrailIdentifier=guardrail_id)
    print(f"  Deleted guardrail {guardrail_id} ✅")
except Exception as e:
    print(f"  Failed to delete guardrail: {e}")

print(f"\n{'=' * 60}")
print("Summary")
print("=" * 60)
print("""
  The Smart Router applies Bedrock Guardrails at the router level:

  - Configure once: guardrails.pre_route / post_route in router config
  - Input screened via ApplyGuardrail API BEFORE model selection
  - Works with any model the router picks
  - Blocked requests never reach Bedrock inference (saves cost)
  - Configured once, applied to every request automatically
""")
