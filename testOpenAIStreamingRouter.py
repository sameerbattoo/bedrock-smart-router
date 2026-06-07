"""Test: BedrockRouter as drop-in replacement for OpenAI SDK (streaming).

This uses the synchronous OpenAI SDK pattern:
    from openai import OpenAI
    client = OpenAI()
    stream = client.chat.completions.create(stream=True, ...)
    for chunk in stream:
        print(chunk.choices[0].delta.content, end="")

We swap in BedrockRouter and verify it works identically.
"""

import json
import logging
from typing import Literal
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
# Drop-in replacement: BedrockRouter instead of OpenAI()
# ═══════════════════════════════════════════════════════════════
from bedrock_smart_router import BedrockRouter

REGION = "us-west-2"
client = BedrockRouter.create({"region": REGION})

# =====================================================================
# 1. SCHEMAS & MOCK BACKEND
# =====================================================================
class QueryDatabaseArgs(BaseModel):
    customer_id: str = Field(..., description="6-digit alphanumeric ID, e.g., 'CU1234'.")
    metric: Literal["balance", "tier", "risk_score"] = Field(...)

def query_customer_database(customer_id: str, metric: str) -> str:
    database = {
        "CU1234": {"balance": "$12,450.00", "tier": "Platinum", "risk_score": "low"},
        "CU5678": {"balance": "$150.25", "tier": "Standard", "risk_score": "high"}
    }
    customer = database.get(customer_id.upper())
    if not customer:
        return json.dumps({"error": f"Customer ID {customer_id} not found."})
    return json.dumps({customer_id: {metric: customer.get(metric)}})

TOOL_MAPPING = {"query_customer_database": query_customer_database}

tools = [{
    "type": "function",
    "function": {
        "name": "query_customer_database",
        "description": "Extract critical real-time infrastructure data metrics.",
        "parameters": QueryDatabaseArgs.model_json_schema(),
        "strict": True,
    },
}]

# =====================================================================
# 2. STREAMING AGENT (Synchronous — matches OpenAI sync SDK pattern)
# =====================================================================
class StreamingAgent:
    def __init__(self, system_prompt: str, model: str | None = None):
        self.model = model
        self.conversation_history = [{"role": "system", "content": system_prompt}]

    def interact(self, user_input: str):
        """Synchronous streaming agent loop with tool support."""
        self.conversation_history.append({"role": "user", "content": user_input})
        
        max_iterations = 5
        iteration = 0
        has_tool_results_pending = False
        
        while iteration < max_iterations:
            iteration += 1
            
            create_kwargs = {
                "messages": self.conversation_history,
                "tools": tools,
                "stream": True,
            }
            if self.model:
                create_kwargs["model"] = self.model
            if not has_tool_results_pending:
                create_kwargs["tool_choice"] = "auto"
            
            # Synchronous streaming — same as: openai.OpenAI().chat.completions.create(stream=True)
            stream = client.chat.completions.create(**create_kwargs)

            tool_calls_buffer = {}
            final_text_content = ""
            routing_decision = None
            
            for chunk in stream:
                # Capture routing decision (SmartRouter extra)
                if hasattr(chunk, 'routing_decision') and chunk.routing_decision:
                    routing_decision = chunk.routing_decision
                    continue
                
                if not chunk.choices:
                    continue
                
                delta = chunk.choices[0].delta
                if not delta:
                    continue
                
                # Text fragments
                if delta.content:
                    final_text_content += delta.content
                    yield delta.content

                # Tool call fragments
                elif hasattr(delta, 'tool_calls') and delta.tool_calls:
                    for tc_chunk in delta.tool_calls:
                        idx = tc_chunk.index
                        if idx not in tool_calls_buffer:
                            tool_calls_buffer[idx] = {"id": "", "name": "", "arguments": ""}
                        if tc_chunk.id:
                            tool_calls_buffer[idx]["id"] = tc_chunk.id
                        if tc_chunk.function and tc_chunk.function.name:
                            if not tool_calls_buffer[idx]["name"]:
                                tool_calls_buffer[idx]["name"] = tc_chunk.function.name
                        if tc_chunk.function and tc_chunk.function.arguments:
                            tool_calls_buffer[idx]["arguments"] += tc_chunk.function.arguments

            # Print routing info
            if routing_decision:
                yield f"\n  🔀 [{routing_decision.selected_model} | {routing_decision.strategy_used} | {routing_decision.complexity_detected} | {routing_decision.api_backend}]\n"

            # Save assistant message to history
            assistant_message = {"role": "assistant", "content": final_text_content or None}
            if tool_calls_buffer:
                assistant_message["tool_calls"] = [
                    {"id": item["id"], "type": "function", "function": {"name": item["name"], "arguments": item["arguments"]}}
                    for _, item in sorted(tool_calls_buffer.items())
                ]

            self.conversation_history.append(assistant_message)

            if not tool_calls_buffer:
                break

            # Execute tools
            for item in tool_calls_buffer.values():
                yield f"\n  ⚙️  Executing: {item['name']}({item['arguments']})\n"
                try:
                    args = json.loads(item["arguments"])
                    result = TOOL_MAPPING[item["name"]](**args)
                except Exception as e:
                    result = json.dumps({"error": str(e)})
                
                self.conversation_history.append({
                    "tool_call_id": item["id"],
                    "role": "tool",
                    "name": item["name"],
                    "content": result,
                })
            
            has_tool_results_pending = True
            logger.info("Tool executions complete. Re-evaluating...")


# =====================================================================
# 3. RUN
# =====================================================================
if __name__ == "__main__":
    system_instruction = "You are a concise financial risk data auditing system."
    agent = StreamingAgent(system_prompt=system_instruction)

    turns = [
        "Write a detailed 500-word essay explaining the evolution of cloud computing from mainframes to serverless.",
        "Now check the risk score for customer CU1234.",
    ]

    for turn_idx, user_query in enumerate(turns, 1):
        print(f"\n{'='*60}")
        print(f"--- Turn {turn_idx} ---")
        print(f"User: {user_query}")
        print(f"{'='*60}")
        print("Assistant: ", end="", flush=True)
        
        for token in agent.interact(user_query):
            print(token, end="", flush=True)
        print()
