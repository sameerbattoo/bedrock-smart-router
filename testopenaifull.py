import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Literal, Optional
from openai import OpenAI
from pydantic import BaseModel, Field

# Configure logging for production visibility
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Initialize OpenAI client — using BedrockRouter (drop-in replacement)
import os
REGION = "us-west-2"

from bedrock_smart_router import BedrockRouter
client = BedrockRouter.create({"region": REGION})

# =====================================================================
# 1. STRUCTURED SCHEMAS (Pydantic Models)
# Using strict Pydantic structures ensures the model cannot hallucinate 
# invalid arguments or miss required fields.
# =====================================================================

class QueryDatabaseArgs(BaseModel):
    customer_id: str = Field(..., description="The unique 6-digit alphanumeric customer ID, e.g., 'CU7891'.")
    metric: Literal["balance", "tier", "risk_score"] = Field(..., description="The specific account data point requested.")

class SendAlertArgs(BaseModel):
    recipient_email: str = Field(..., description="Validated target destination email address.")
    priority: Literal["low", "medium", "high"] = Field(..., description="Urgency categorization level of the notification.")
    message: str = Field(..., description="The brief alert body text describing what action is needed.")

# =====================================================================
# 2. ACTUAL SYSTEM TOOL CODES
# Mock backend services mimicking production databases/APIs.
# =====================================================================

def query_customer_database(customer_id: str, metric: str) -> str:
    """Mock CRM/Database query service."""
    logger.info(f"Executing database search for {customer_id} -> metric: {metric}")
    database = {
        "CU1234": {"balance": "$12,450.00", "tier": "Platinum", "risk_score": "low"},
        "CU5678": {"balance": "$150.25", "tier": "Standard", "risk_score": "high"}
    }
    customer = database.get(customer_id.upper())
    if not customer:
        return json.dumps({"error": f"Customer ID {customer_id} not found in the records."})
    return json.dumps({customer_id: {metric: customer.get(metric)}})

def send_security_alert(recipient_email: str, priority: str, message: str) -> str:
    """Mock Notification/SES service."""
    logger.info(f"Dispatching emergency notification to {recipient_email} [{priority.upper()}]")
    return json.dumps({"status": "success", "dispatched_to": recipient_email, "priority": priority})

# Map string keys to executable routines
TOOL_MAPPING = {
    "query_customer_database": query_customer_database,
    "send_security_alert": send_security_alert
}

# =====================================================================
# 3. OPENAI TOOL SPECIFICATIONS
# Incorporating 'strict: True' to activate hard API-level JSON validation.
# =====================================================================

tools = [
    {
        "type": "function",
        "function": {
            "name": "query_customer_database",
            "description": "Extract critical real-time infrastructure data metrics for a specific customer profile.",
            "parameters": QueryDatabaseArgs.model_json_schema(),
            "strict": True,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_security_alert",
            "description": "Trigger automated high-priority email alerts when anomalies or compliance hazards are verified.",
            "parameters": SendAlertArgs.model_json_schema(),
            "strict": True,
        },
    }
]

# =====================================================================
# 4. AGENT ENGINE (The Engine Framework)
# =====================================================================

class ProductionAgent:
    def __init__(self, system_prompt: str, model: str | None = None):
        self.model = model  # None = let the smart router decide
        self.conversation_history = [{"role": "system", "content": system_prompt}]

    def interact(self, user_input: str) -> str:
        """Appends new input, runs the multi-turn generation, handles parallel calls, and yields a response."""
        self.conversation_history.append({"role": "user", "content": user_input})
        
        # Execute loops internally until model is finished requesting tools (handles recursive chaining)
        while True:
            response = client.chat.completions.create(
                model=self.model,
                messages=self.conversation_history,
                tools=tools,
                tool_choice="auto"
            )
            
            # Print routing decision
            rd = response.routing_decision
            if rd:
                print(f"  🔀 Routed to: {rd.selected_model} | strategy={rd.strategy_used} | complexity={rd.complexity_detected} | backend={rd.api_backend}")
            
            response_message = response.choices[0].message
            tool_calls = response_message.tool_calls
            
            # Scenario A: The model did not call tools, it returned its final message answer
            if not tool_calls:
                self.conversation_history.append(response_message)
                return response_message.content

            # Scenario B: The model called tools. State preservation requires pushing assistant intent first.
            self.conversation_history.append(response_message)
            
            # Execute tools concurrently utilizing a thread pool for maximum performance speed
            tool_outputs = []
            with ThreadPoolExecutor() as executor:
                futures = {}
                for tool_call in tool_calls:
                    func_name = tool_call.function.name
                    func_args = json.loads(tool_call.function.arguments)
                    executable = TOOL_MAPPING[func_name]
                    
                    # Submit threads
                    futures[executor.submit(executable, **func_args)] = tool_call

                for future in as_completed(futures):
                    matched_call = futures[future]
                    try:
                        result_content = future.result()
                    except Exception as e:
                        result_content = json.dumps({"error": f"Execution failed: {str(e)}"})
                    
                    tool_outputs.append({
                        "tool_call_id": matched_call.id,
                        "role": "tool",
                        "name": matched_call.function.name,
                        "content": result_content
                    })

            # Append complete parallel tool outputs block chronologically back to context array
            self.conversation_history.extend(tool_outputs)
            logger.info("Parallel tool executions integrated into memory. Re-evaluating status...")

# =====================================================================
# 5. EXECUTION PIPELINE (Multi-Turn Scenario Simulation)
# =====================================================================

if __name__ == "__main__":
    system_instruction = (
        "You are an automated SecOps Compliance Intelligence Assistant. Your objective is to audit accounts, "
        "analyze potential operational risk scores, and coordinate safety alerts immediately when flags arise. "
        "Be concise, clear, and act systematically step-by-step."
    )
    
    # Initialize stateful operational session
    agent = ProductionAgent(system_prompt=system_instruction)

    print("--- TURN 1: Complex Multi-Tool Trigger Request ---")
    prompt_1 = "Check the risk score and account balance for customer CU1234. Let me know what you find."
    print(f"User: {prompt_1}\n")
    answer_1 = agent.interact(prompt_1)
    print(f"Assistant: {answer_1}\n")

    print("--- TURN 2: Evaluation with Conditional Concluding Action ---")
    prompt_2 = (
        "Interesting. Now check the details for customer CU5678 instead. If their risk score is high, "
        "immediately email a high-priority alert to compliance@company.com letting them know the account "
        "is flagged due to low balances and systemic risks."
    )
    print(f"User: {prompt_2}\n")
    answer_2 = agent.interact(prompt_2)
    print(f"Assistant: {answer_2}\n")
    
    print("--- TURN 3: Context Retention Validation Check ---")
    prompt_3 = "Summarize what actions you just completed regarding both clients for my daily audit log."
    print(f"User: {prompt_3}\n")
    answer_3 = agent.interact(prompt_3)
    print(f"Assistant: {answer_3}")

    # =====================================================================
    # 6. STREAMING TEST
    # =====================================================================
    print("\n\n--- STREAMING TEST: chat.completions.create(stream=True) ---")
    print("User: Write a haiku about cloud computing.\n")
    print("Assistant (streaming): ", end="", flush=True)
    
    stream = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Write a haiku about cloud computing."},
        ],
        stream=True,
    )
    
    routing_decision = None
    for chunk in stream:
        if hasattr(chunk, "routing_decision") and chunk.routing_decision:
            routing_decision = chunk.routing_decision
            continue
        if chunk.choices:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                print(delta.content, end="", flush=True)
    
    print()
    if routing_decision:
        print(f"  🔀 Routed to: {routing_decision.selected_model} | strategy={routing_decision.strategy_used} | complexity={routing_decision.complexity_detected} | backend={routing_decision.api_backend}")
