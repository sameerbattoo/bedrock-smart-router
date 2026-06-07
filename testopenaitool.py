import json
import os
from openai import OpenAI
import time

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


# 1. Define the actual Python function you want the model to be able to trigger
def get_weather(location: str, unit: str = "celsius") -> str:
    """Get the current weather for a given location."""
    # In a real app, you would make an API request to a weather service here
    if "tokyo" in location.lower():
        return json.dumps({"location": "Tokyo", "temperature": "10", "unit": unit, "condition": "sunny"})
    elif "san francisco" in location.lower():
        return json.dumps({"location": "San Francisco", "temperature": "15", "unit": unit, "condition": "windy"})
    else:
        return json.dumps({"location": location, "temperature": "unknown"})

def run_conversation():
    """Test using BedrockRouter configured with a Bedrock API key."""
    print("=" * 60)
    print("STEP : BedrockRouter with Bedrock API Key")
    print("=" * 60)

    api_key = get_mantle_api_key()
    if not api_key:
        print("\n  ⚠️  No BEDROCK_API_KEY found. Skipping this test.")
        print("  Set BEDROCK_API_KEY env var to test API key auth.\n")
        return False

    from bedrock_smart_router import BedrockRouter

    # Create router with API key — works for both bedrock-runtime and mantle
    client = BedrockRouter.create({
        "region": REGION,
        "api_key": api_key,  # Single key for both endpoints
    })

    # 2. Define the tool specification matching the function above
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a specific city location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {
                            "type": "string", 
                            "enum": ["celsius", "fahrenheit"]
                        },
                    },
                    "required": ["location"],
                },
            },
        }
    ]

    # Initialize the conversation history
    messages = [{"role": "user", "content": "What is the weather like in Tokyo right now?"}]

    # Step 1: Send the conversation and available tools to the model
    print("Sending initial request to model...")
    t0 = time.time()
    response = client.chat.completions.create(
        # model=MODEL,  # or gpt-4o-mini
        messages=messages,
        tools=tools,
        tool_choice="auto",
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

    assistant_message = response.choices[0].message
    messages.append(assistant_message)

    # Step 2: Check if the model wants to call a function
    if assistant_message.tool_calls:
        print(f"Model selected tool: {assistant_message.tool_calls[0].function.name}")
        
        # Mapping available functions
        available_functions = {
            "get_weather": get_weather,
        }
        
        # Step 3: Execute the local function call
        for tool_call in assistant_message.tool_calls:
            function_name = tool_call.function.name
            function_to_call = available_functions[function_name]
            function_args = json.loads(tool_call.function.arguments)
            
            # Execute the function with arguments supplied by the model
            function_response = function_to_call(
                location=function_args.get("location"),
                unit=function_args.get("unit", "celsius"),
            )
            print(f"Executed local function output: {function_response}")

            # Step 4: Append the tool output back into the message history
            messages.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": function_response,
                }
            )
        
        # Step 5: Send the complete updated conversation history back to the model
        print("Sending tool outputs back to model...")
        t0 = time.time()
        second_response = client.chat.completions.create(
            # model=MODEL,
            messages=messages,
        )
        elapsed = time.time() - t0
        print(f"  Response ({elapsed:.2f}s)")
        print(f"  Model: {second_response['routing_decision'].selected_model}")
        print()
        
        final_output = second_response.choices[0].message.content
        print(f"\nFinal AI Answer:\n{final_output}")
    else:
        print(f"\nFinal AI Answer (No tool needed):\n{assistant_message.content}")

if __name__ == "__main__":
    run_conversation()
