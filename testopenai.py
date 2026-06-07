"""
testopenai.py - Chat Completions using OpenAI SDK pointed at Bedrock Mantle.

Based on: https://gist.github.com/pszemraj/c643cfe422d3769fd13b97729cf517c5
Modernized for the current OpenAI Python SDK (v1+) and Amazon Bedrock Mantle endpoint.

Step 1: Test with OpenAI SDK → Bedrock Mantle directly
Step 2: Replace with our BedrockRouter (drop-in replacement)

Usage:
    # Step 1: Direct Mantle call via OpenAI SDK
    python testopenai.py

    # Step 2: Via Smart Router (uncomment the router section below)
    python testopenai.py --use-router
"""

import os
import sys
import time

from openai import OpenAI


# ═══════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════

REGION = "us-west-2"
MODEL = "openai.gpt-oss-120b"  # Available on Bedrock Mantle

# Bedrock Mantle endpoint (OpenAI-compatible)
BASE_URL = f"https://bedrock-mantle.{REGION}.api.aws/v1"

# For Bedrock Mantle with SigV4, we need to generate a short-term API key
# or use a helper. For this test, we'll use our MantleClient which handles SigV4.
# But first, let's see if we can use the OpenAI SDK directly with SigV4...


# ═══════════════════════════════════════════════════════════════════
# Helper: Get a signed request for Mantle (since OpenAI SDK needs a key)
# ═══════════════════════════════════════════════════════════════════

def get_mantle_api_key():
    """
    For the OpenAI SDK to work with Bedrock Mantle, you need either:
    1. A Bedrock API key (brk_xxx) - set as BEDROCK_API_KEY env var
    2. Or use our MantleClient which handles SigV4 signing
    
    Returns the API key if available, None otherwise.
    """
    return os.environ.get("BEDROCK_API_KEY") or os.environ.get("AWS_BEARER_TOKEN_BEDROCK")


# ═══════════════════════════════════════════════════════════════════
# Step 1: Direct call via OpenAI SDK → Bedrock Mantle
# ═══════════════════════════════════════════════════════════════════

def test_openai_sdk_direct():
    """Test using the OpenAI SDK pointed at Bedrock Mantle."""
    print("=" * 60)
    print("STEP 1: OpenAI SDK → Bedrock Mantle (direct)")
    print("=" * 60)

    api_key = get_mantle_api_key()
    if not api_key:
        print("\n  ⚠️  No BEDROCK_API_KEY found.")
        print("  Falling back to MantleClient (SigV4) for this test.\n")
        return test_mantle_client_direct()

    client = OpenAI(
        base_url=BASE_URL,
        api_key=api_key,
    )

    # Basic completion
    print(f"\n  Model: {MODEL}")
    print(f"  Endpoint: {BASE_URL}")
    print()

    t0 = time.time()
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Be concise."},
            {"role": "user", "content": "What is Amazon Bedrock? Answer in 2 sentences."},
        ],
        max_tokens=100,
        temperature=0.5,
    )
    elapsed = time.time() - t0

    print(f"  Response ({elapsed:.2f}s):")
    print(f"  Model: {response.model}")
    print(f"  Content: {response.choices[0].message.content}")
    print(f"  Finish: {response.choices[0].finish_reason}")
    print(f"  Usage: {response.usage.prompt_tokens} in / {response.usage.completion_tokens} out")
    print()
    return True


def test_mantle_client_direct():
    """Fallback: Use our MantleClient (SigV4, no API key needed)."""
    from bedrock_smart_router.mantle_client import MantleClient

    client = MantleClient(region=REGION)

    print(f"  Model: {MODEL}")
    print(f"  Auth: SigV4 (AWS credentials)")
    print()

    t0 = time.time()
    response = client.chat_completions(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Be concise."},
            {"role": "user", "content": "What is Amazon Bedrock? Answer in 2 sentences."},
        ],
        max_tokens=100,
        temperature=0.5,
    )
    elapsed = time.time() - t0

    content = response["choices"][0]["message"]["content"]
    usage = response.get("usage", {})
    print(f"  Response ({elapsed:.2f}s):")
    print(f"  Model: {response.get('model', MODEL)}")
    print(f"  Content: {content}")
    print(f"  Usage: {usage.get('prompt_tokens', '?')} in / {usage.get('completion_tokens', '?')} out")
    print()
    return True


# ═══════════════════════════════════════════════════════════════════
# Step 2: Drop-in replacement using BedrockRouter
# ═══════════════════════════════════════════════════════════════════

def test_smart_router():
    """Test using BedrockRouter as a drop-in for the OpenAI SDK."""
    print("=" * 60)
    print("STEP 2: BedrockRouter (drop-in replacement)")
    print("=" * 60)

    from bedrock_smart_router import BedrockRouter

    # Create router — same interface as OpenAI client
    router = BedrockRouter.create({"region": REGION})

    print(f"\n  Using: router.chat.completions.create()")
    print(f"  Same parameters as OpenAI SDK — zero code changes needed")
    print()

    # Same call — identical to OpenAI SDK interface
    t0 = time.time()
    response = router.chat.completions.create(
        model=MODEL,  # Can specify model (like OpenAI) or omit for auto-routing
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Be concise."},
            {"role": "user", "content": "What is Amazon Bedrock? Answer in 2 sentences."},
        ],
        max_tokens=100,
        temperature=0.5,
    )
    elapsed = time.time() - t0

    print(f"  Response ({elapsed:.2f}s):")
    print(f"  Model: {response['model']}")
    print(f"  Content: {response['choices'][0]['message']['content']}")
    print(f"  Finish: {response['choices'][0]['finish_reason']}")
    print(f"  Usage: {response['usage']['prompt_tokens']} in / {response['usage']['completion_tokens']} out")

    # Bonus: routing decision (extra info not available with raw OpenAI SDK)
    rd = response.get("routing_decision")
    if rd:
        print(f"\n  🔀 Routing Decision:")
        print(f"     Strategy: {rd.strategy_used}")
        print(f"     Complexity: {rd.complexity_detected}")
        print(f"     Cost: ${rd.actual_cost:.6f}")
    print()

    # Test 2: Auto-routing (omit model — router picks the best one)
    print("  --- Auto-routing (no model specified) ---")
    t0 = time.time()
    response2 = router.chat.completions.create(
        messages=[
            {"role": "user", "content": "What is 2+2?"},
        ],
        max_tokens=20,
    )
    elapsed2 = time.time() - t0
    print(f"  Model selected: {response2['model']} ({elapsed2:.2f}s)")
    print(f"  Content: {response2['choices'][0]['message']['content']}")
    rd2 = response2.get("routing_decision")
    if rd2:
        print(f"  Complexity: {rd2.complexity_detected}, Cost: ${rd2.actual_cost:.6f}")
    print()

    # Test 3: Tool use
    print("  --- Tool use ---")
    response3 = router.chat.completions.create(
        messages=[{"role": "user", "content": "What's the weather in Paris?"}],
        tools=[{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }],
        max_tokens=100,
    )
    choice = response3["choices"][0]
    if choice["message"].get("tool_calls"):
        tc = choice["message"]["tool_calls"][0]
        print(f"  Tool call: {tc['function']['name']}({tc['function']['arguments']})")
    else:
        print(f"  Response: {choice['message']['content'][:80]}")
    print(f"  Finish reason: {choice['finish_reason']}")
    print()

    return True


# ═══════════════════════════════════════════════════════════════════
# Step 3: BedrockRouter with API key (no SigV4 needed)
# ═══════════════════════════════════════════════════════════════════

def test_smart_router_with_api_key():
    """Test using BedrockRouter configured with a Bedrock API key."""
    print("=" * 60)
    print("STEP 3: BedrockRouter with Bedrock API Key")
    print("=" * 60)

    api_key = get_mantle_api_key()
    if not api_key:
        print("\n  ⚠️  No BEDROCK_API_KEY found. Skipping this test.")
        print("  Set BEDROCK_API_KEY env var to test API key auth.\n")
        return False

    from bedrock_smart_router import BedrockRouter

    # Create router with API key — works for both bedrock-runtime and mantle
    router = BedrockRouter.create({
        "region": REGION,
        "api_key": api_key,  # Single key for both endpoints
    })

    print(f"\n  Using: router with api_key configured")
    print(f"  Auth: Bearer token (Bedrock API key)")
    print(f"  Both bedrock-runtime and bedrock-mantle use the same key")
    print()

    # Test: Chat Completions with API key auth
    t0 = time.time()
    response = router.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Be concise."},
            {"role": "user", "content": "What is Amazon S3? One sentence."},
        ],
        max_tokens=60,
        temperature=0.5,
    )
    elapsed = time.time() - t0

    print(f"  Response ({elapsed:.2f}s):")
    print(f"  Model: {response['model']}")
    print(f"  Content: {response['choices'][0]['message']['content']}")
    print(f"  Usage: {response['usage']['prompt_tokens']} in / {response['usage']['completion_tokens']} out")

    rd = response.get("routing_decision")
    if rd:
        print(f"\n  🔀 Routing Decision:")
        print(f"     Strategy: {rd.strategy_used}")
        print(f"     Complexity: {rd.complexity_detected}")
        print(f"     Cost: ${rd.actual_cost:.6f}")
    print()

    # Also test Converse API with same API key
    print("  --- Converse API (same api_key) ---")
    t0 = time.time()
    converse_resp = router.converse(
        messages=[{"role": "user", "content": [{"text": "What is DynamoDB? One sentence."}]}],
        inferenceConfig={"maxTokens": 60},
    )
    elapsed = time.time() - t0
    text = converse_resp["output"]["message"]["content"][0]["text"]
    print(f"  Response ({elapsed:.2f}s): {text[:100]}")
    print(f"  Model: {converse_resp['routing_decision'].selected_model}")
    print()

    return True


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    use_router = "--use-router" in sys.argv or "--router" in sys.argv

    if use_router:
        test_smart_router()
    else:
        # Run all steps
        test_openai_sdk_direct()
        print()
        test_smart_router()
        print()
        test_smart_router_with_api_key()
        test_smart_router()
