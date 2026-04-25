"""Strands Agent with Bedrock Guardrails via Smart Router.

Adapted from the official Strands Agents SDK guardrails sample:
https://github.com/strands-agents/samples/blob/main/python/01-learn/05-guardrails/bedrock_guardrails_sample.ipynb

The original sample configures guardrails on the BedrockModel directly.
With the Smart Router, you have TWO ways to use guardrails:

  Method 1: Router-level guardrails (pre-route + post-route)
    - Configured once in the router config
    - Applied via the ApplyGuardrail API BEFORE and AFTER model selection
    - Works regardless of which model the router picks
    - Input is screened before any Bedrock inference call

  Method 2: Model-level guardrails (Bedrock native, via kwargs)
    - Passed as guardrailConfig in the Bedrock Converse API call
    - Applied by Bedrock during inference
    - Same as the original AWS sample, but with smart routing

This example demonstrates both methods.

Requirements:
    pip install bedrock-smart-router[strands]

Note: This example creates and deletes a Bedrock Guardrail. You need
      IAM permissions for bedrock:CreateGuardrail and bedrock:DeleteGuardrail.
"""

import boto3
from strands import Agent, tool
from bedrock_smart_router import BedrockRouter
from bedrock_smart_router.strands_model import SmartRouterModel


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
# Method 1: Router-Level Guardrails (Pre-Route + Post-Route)
# ═══════════════════════════════════════════════════════════════════
# The guardrail is configured in the ROUTER, not the model.
# Input is screened via ApplyGuardrail API BEFORE model selection.
# Output is screened AFTER the model responds.
# This works regardless of which model the router picks.

print(f"\n{'=' * 60}")
print("Method 1: Router-Level Guardrails")
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
# Method 2: Model-Level Guardrails (via kwargs passthrough)
# ═══════════════════════════════════════════════════════════════════
# The guardrail is passed as guardrailConfig in the Bedrock Converse
# API call. The router forwards it via **kwargs. This works with the
# router's converse() / converse_stream() directly (not via Strands).

print(f"\n{'=' * 60}")
print("Method 2: Model-Level Guardrails (Bedrock native via kwargs)")
print("=" * 60)

# Create a router WITHOUT pre-route guardrails
router2 = BedrockRouter.create({"region": "us-west-2"})

guardrail_config = {
    "guardrailIdentifier": guardrail_id,
    "guardrailVersion": guardrail_version,
    "trace": "enabled",
}

# Test 2a: Safe question — guardrail passed per-request via kwargs
print("\n  Test 2a: Safe question (guardrailConfig in kwargs)")
response = router2.converse(
    messages=[{"role": "user", "content": [{"text": "What is Amazon S3?"}]}],
    guardrailConfig=guardrail_config,
)
d = response["routing_decision"]
print(f"  → Model: {d.selected_model}, Cost: ${d.actual_cost:.6f}")
print(f"  → Guardrail trace: {'present' if d.guardrail_trace else 'none'}")

# Test 2b: Blocked question — Bedrock applies guardrail during inference
print("\n  Test 2b: Financial advice (should be blocked by Bedrock)")
try:
    response = router2.converse(
        messages=[{"role": "user", "content": [{"text": "What stocks should I invest in?"}]}],
        guardrailConfig=guardrail_config,
    )
    stop = response.get("stopReason", "")
    if stop == "guardrail_intervened":
        print(f"  → ⚠️ GUARDRAIL INTERVENED during inference ✅")
    else:
        d = response["routing_decision"]
        print(f"  → Model: {d.selected_model}, stop: {d.stop_reason}")
except Exception as e:
    print(f"  → Blocked: {str(e)[:80]}")

print("\n  Note: Method 1 (router-level) is preferred for Strands agents")
print("  because it screens input BEFORE inference and costs nothing.")
print("  Method 2 is useful when calling router.converse() directly.")


# ═══════════════════════════════════════════════════════════════════
# Method 1 continued: Multi-turn with guardrails
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
  The Smart Router supports Bedrock Guardrails at two levels:

  1. Router-level (recommended for Strands agents):
     - Configure once in router config: guardrails.pre_route / post_route
     - Input screened via ApplyGuardrail API BEFORE model selection
     - Works with any model the router picks
     - Blocked requests never reach Bedrock inference (saves cost)

  2. Model-level (via Bedrock Converse API kwargs):
     - Pass guardrailConfig per request
     - Applied by Bedrock during inference
     - Same as using BedrockModel directly

  Router-level is preferred because:
  - Blocked input is caught before any inference cost
  - Works regardless of which model is selected
  - Configured once, not per-request
""")
