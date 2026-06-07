import asyncio
import json
import logging
import os
from typing import Literal
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Drop-in replacement: BedrockRouter instead of AsyncOpenAI
from bedrock_smart_router import BedrockRouter

REGION = "us-west-2"
client = BedrockRouter.create({"region": REGION})

# =====================================================================
# 1. SCHEMAS & MOCK BACKEND
# =====================================================================
class QueryDatabaseArgs(BaseModel):
    customer_id: str = Field(..., description="6-digit alphanumeric ID, e.g., 'CU1234'.")
    metric: Literal["balance", "tier", "risk_score"] = Field(..., description="The account data point.")

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
# 2. STREAMING AGENT ENGINE
# =====================================================================
class StreamingAgent:
    def __init__(self, system_prompt: str, model: str | None = None):
        self.model = model
        self.conversation_history = [{"role": "system", "content": system_prompt}]

    async def interact(self, user_input: str):
        """
        Accepts user input, handles the async stream generation, 
        yields text chunks immediately to the UI, and handles tool execution.
        """
        self.conversation_history.append({"role": "user", "content": user_input})
        
        max_iterations = 5  # Safety limit to prevent infinite tool loops
        iteration = 0
        has_tool_results_pending = False
        
        while iteration < max_iterations:
            iteration += 1
            # Request an asynchronous stream from OpenAI
            # After tool results, don't force tool_choice to avoid infinite loops
            create_kwargs = {
                "messages": self.conversation_history,
                "tools": tools,
                "stream": True,
            }
            if self.model:
                create_kwargs["model"] = self.model
            if not has_tool_results_pending:
                create_kwargs["tool_choice"] = "auto"
            
            stream = await client.chat.completions.create(**create_kwargs)

            # Local state variables to stitch tool calls together from chunks
            tool_calls_buffer = {}
            final_text_content = ""
            
            async for chunk in stream:
                if not chunk.choices:
                    continue
                
                delta = chunk.choices[0].delta
                if not delta:
                    continue
                
                # Case A: Handling incoming stream text fragments
                if delta.content:
                    final_text_content += delta.content
                    yield delta.content  # Immediately stream text to the user interface

                # Case B: Handling incoming tool calls fragments
                elif hasattr(delta, 'tool_calls') and delta.tool_calls:
                    for tc_chunk in delta.tool_calls:
                        idx = tc_chunk.index
                        
                        # Initialize the data slot if this index hasn't been seen yet
                        if idx not in tool_calls_buffer:
                            tool_calls_buffer[idx] = {
                                "id": tc_chunk.id or "",
                                "name": "",
                                "arguments": ""
                            }
                        
                        # Accumulate function name and argument string tokens as they arrive
                        if tc_chunk.id:
                            tool_calls_buffer[idx]["id"] = tc_chunk.id
                        if tc_chunk.function and tc_chunk.function.name:
                            # Set name (first chunk has full name, don't keep appending)
                            if not tool_calls_buffer[idx]["name"]:
                                tool_calls_buffer[idx]["name"] = tc_chunk.function.name
                        if tc_chunk.function and tc_chunk.function.arguments:
                            tool_calls_buffer[idx]["arguments"] += tc_chunk.function.arguments

            # Create a structured message format to store back into history logs
            assistant_message = {"role": "assistant"}
            if final_text_content:
                assistant_message["content"] = final_text_content
            else:
                assistant_message["content"] = None
            
            # Map buffer back to a standard format OpenAI expects for context verification
            if tool_calls_buffer:
                assistant_message["tool_calls"] = [
                    {
                        "id": item["id"],
                        "type": "function",
                        "function": {"name": item["name"], "arguments": item["arguments"]}
                    }
                    for idx, item in sorted(tool_calls_buffer.items())
                ]

            # Save the complete reconstructed message to conversation history
            self.conversation_history.append(assistant_message)

            # Break out of the loop if the model didn't ask to run tools
            if not tool_calls_buffer:
                break

            # Execute tool queries concurrently using asyncio.to_thread
            tool_outputs = []
            tasks = []

            async def run_tool_async(call_id, name, args_str):
                try:
                    args = json.loads(args_str)
                    executable = TOOL_MAPPING[name]
                    # Run the CPU-bound/blocking mock method in a separate background thread
                    result = await asyncio.to_thread(executable, **args)
                except Exception as e:
                    result = json.dumps({"error": f"Execution failed: {str(e)}"})
                
                return {
                    "tool_call_id": call_id,
                    "role": "tool",
                    "name": name,
                    "content": result
                }

            # Add all requested tool tasks into the execution loop queue
            for item in tool_calls_buffer.values():
                yield f"\n*[System Executing: {item['name']} with {item['arguments']}...]*\n"
                tasks.append(run_tool_async(item["id"], item["name"], item["arguments"]))

            # Execute parallel operations simultaneously
            tool_outputs = await asyncio.gather(*tasks)
            self.conversation_history.extend(tool_outputs)
            has_tool_results_pending = True
            
            logger.info("Parallel tool executions finalized. Loop re-evaluating...")

# =====================================================================
# 3. INTERACTIVE COROUTINE RUNNER
# =====================================================================
async def main():
    system_instruction = "You are a concise financial risk data auditing system."
    agent = StreamingAgent(system_prompt=system_instruction)

    # Multi-turn script simulator
    turns = [
        "Write a detailed 500-word essay explaining the evolution of cloud computing from mainframes to serverless.",
    ]

    for turn_idx, user_query in enumerate(turns, 1):
        print(f"\n--- Turn {turn_idx} ---")
        print(f"User: {user_query}")
        print("Assistant: ", end="", flush=True)
        
        # Consume the stream generator yielding tokens directly to standard output
        async for token in agent.interact(user_query):
            print(token, end="", flush=True)
        print()

if __name__ == "__main__":
    asyncio.run(main())
