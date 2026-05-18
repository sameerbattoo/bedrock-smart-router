"""Orchestrator Agent — routes user queries to SQL and Chart agents.

Single Strands agent with tools that delegate to sub-agents.
Uses SmartRouterModel with semantic cache for intelligent routing.
"""
import json
import logging
import threading
import time
from typing import Any, Callable

from strands import Agent, tool
from strands.types.content import SystemContentBlock
from strands.agent.conversation_manager.sliding_window_conversation_manager import SlidingWindowConversationManager
from strands_tools import current_time

from bedrock_smart_router.strands_model import SmartRouterModel
from text2sql.sql_agent import SQLAgent
from text2sql.chart_agent import ChartAgent
from text2sql.cache import FilesystemSemanticCache
from text2sql.db import get_table_list

logger = logging.getLogger(__name__)

# Session registry: maps agent id → session (since @tool runs on different thread)
_session_by_agent: dict[int, "Text2SQLSession"] = {}


def _get_active_session() -> "Text2SQLSession | None":
    """Find the session by checking all registered sessions.
    
    Since we can't reliably use thread ID (Strands may run tools on different threads),
    we use a simple approach: for single-user demo, just return the most recently set session.
    """
    return getattr(_get_active_session, "_current", None)


def set_active_session(session: "Text2SQLSession | None") -> None:
    """Set the active session for @tool functions."""
    _get_active_session._current = session


@tool
def query_database(user_query: str) -> str:
    """Query the e-commerce database using natural language.

    Generates SQL from the question, executes it, and returns results.
    If the user asks for a chart/visual, also generates one.

    Args:
        user_query: Natural language question about the data.
    """
    session = _get_active_session()
    if session is None:
        return json.dumps({"error": "Session not initialized."})

    def _status(msg: str):
        if session._status_callback:
            session._status_callback(msg)

    # Get conversation messages for multi-turn intent resolution
    conversation_messages = None
    if session.orchestrator and hasattr(session.orchestrator, 'messages'):
        messages = session.orchestrator.messages
        if messages and len(messages) >= 2:
            conversation_messages = messages

    # Check semantic cache (with auto_extract + variable hashing + multi-turn)
    _status("🔍 Checking semantic cache...")
    cached = session.cache.get(
        query_text=user_query,
        messages=conversation_messages,
    )
    if cached is not None:
        session.metrics["cache_hits"] += 1
        cached["cache_hit"] = True
        _status("⚡ Cache hit — returning cached result")
        return json.dumps(cached, default=str)

    _status("🧠 Cache miss — calling LLM to generate SQL...")

    # Generate and execute SQL
    result = session.sql_agent.generate_and_execute(user_query, status_callback=_status)
    logger.info("SQL agent result: error=%s, row_count=%s", result.get("error"), result.get("row_count"))

    # If SQL generation/execution failed, return error immediately
    if result.get("error"):
        _status("❌ SQL generation failed")
        return json.dumps({"error": result["error"], "cache_hit": False}, default=str)

    _status(f"✅ Query returned {result.get('row_count', 0)} rows — generating chart...")

    # Always attempt chart generation when there's numeric data
    if result.get("results") and result.get("row_count", 0) > 1:
        chart_result = session.chart_agent.generate_chart(
            user_query, result["results"], result.get("columns", [])
        )
        if chart_result.get("success"):
            result["chart_filename"] = chart_result["filename"]
            _status("📊 Chart generated")
        else:
            _status("📊 Chart skipped (not suitable for visualization)")

    _status("✍️ Formatting response...")
    # Store the tool result on the session so the route can cache it with the original message
    session._last_tool_result = result
    result["cache_hit"] = False
    return json.dumps(result, default=str)


@tool
def get_sample_questions() -> str:
    """Return sample questions the user can ask about the e-commerce data."""
    return json.dumps([
        "What are the top 5 products by total revenue?",
        "Show monthly order trends for 2024",
        "Which customers have the most orders?",
        "What is the average order value by category?",
        "List products with low stock (below 60 units)",
        "Show me a chart of sales by category",
    ])


# Global shared semantic cache (singleton across all sessions)
_shared_cache: "FilesystemSemanticCache | None" = None


def _get_shared_cache(region: str = "us-west-2") -> "FilesystemSemanticCache":
    global _shared_cache
    if _shared_cache is None:
        _shared_cache = FilesystemSemanticCache(region=region)
    return _shared_cache


class Text2SQLSession:
    """Manages a single user session with all agents and cache."""

    def __init__(self, router_model: SmartRouterModel, region: str = "us-west-2"):
        self._status_callback: Any = None  # Set per-request for streaming status updates
        self.metrics: dict[str, Any] = {
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
            "cache_hits": 0,
            "steps": 0,
        }

        # Each agent gets its own SmartRouterModel instance (shares underlying router)
        # to avoid last_routing_decision conflicts between concurrent calls
        underlying_router = router_model.router
        sql_model = SmartRouterModel(router=underlying_router, routing_preset="balanced", explain=True)
        # Chart agent uses a preferred model to avoid reasoning content issues with small models
        chart_model = SmartRouterModel(
            router=underlying_router,
            routing_preset="balanced",
            explain=True,
            preferred_model="anthropic.claude-haiku-4-5-20251001-v1:0",
        )
        orch_model = SmartRouterModel(router=underlying_router, routing_preset="balanced", explain=True)

        self._orch_model = orch_model  # Keep reference for metrics extraction
        self._sql_model = sql_model
        self._chart_model = chart_model

        # Shared cache across all sessions (FAISS in-memory persists across requests)
        self.cache = _get_shared_cache(region)
        self.sql_agent = SQLAgent(router_model=sql_model, token_callback=self._token_cb)
        self.chart_agent = ChartAgent(router_model=chart_model, token_callback=self._token_cb)

        # Build orchestrator
        tables = get_table_list()
        system_prompt = f"""# E-Commerce Data Assistant

<role>
You are an intelligent e-commerce data assistant that helps users query product,
order, customer, and shipment data from a SQLite database.
You analyse each question and call the right tool.
</role>

<available_tables>
{', '.join(tables)}
</available_tables>

<available_tools>
1. **query_database** — Handles queries about orders, products, customers, shipments,
   inventory, sales, and other transactional data. Generates SQL, executes it, and
   returns formatted results. If the user asks for a chart/visual, the tool creates one.
2. **get_sample_questions** — Lists sample questions users can ask.
3. **current_time** — Returns the current date and time. Use when the user asks about
   "today", "this month", "recent", or needs time-relative context.
</available_tools>

<tool_instructions>
## When using **query_database**:
- Use this HTML table format for results:
  ```html
  <div style="overflow-x: auto; margin: 20px 0;">
  <table>
    <thead>
      <tr>
        <th>Column 1</th>
        <th>Column 2</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>Value 1</td>
        <td>Value 2</td>
      </tr>
    </tbody>
  </table>
  </div>
  ```
- Show maximum 20 rows and include the total row count above the table.
- Include the generated SQL statement within `<details>` tags at the end:
  ```html
  <details>
  <summary>SQL Query</summary>
  <pre><code>SELECT ... FROM ...</code></pre>
  </details>
  ```
</tool_instructions>

<chart_display_rules>
CRITICAL: Always check tool responses for chart data.

**Chart Detection and Display Protocol:**
1. The query_database tool automatically generates charts when data has numeric columns.
   After calling this tool, inspect the response for a `chart_filename` field.
2. If `chart_filename` IS present in the tool response, display it:
   ```html
   <img src="/api/text2sql/charts/{{{{chart_filename}}}}" alt="Data Visualization"
        style="max-width: 100%; height: auto; border: 1px solid #ddd;
               border-radius: 4px; padding: 5px;" />
   ```
3. The chart should appear AFTER the data table and BEFORE the insights section.
4. If no `chart_filename` field exists, the chart generation failed or data wasn't suitable — simply skip the chart section. Do NOT mention the failure.
</chart_display_rules>

<response_format>
When responding to a user query, follow this structure:

**Data Table**
- Present tabular data using the HTML table format above.
- Show row count above the table.

**Chart (only when chart_filename is present in tool response)**
- Display the chart image using the format from chart_display_rules.

**Key Insights**
- Provide 2-3 brief, actionable insights about the data.

**Technical Details**
- Show SQL in collapsible `<details>` section at the end.

**Follow-up Suggestions**
- End with 2-3 relevant follow-up questions as bullet points.
</response_format>
"""
        self.orchestrator = Agent(
            model=orch_model,
            system_prompt=[
                SystemContentBlock(text=system_prompt),
                SystemContentBlock(cachePoint={"type": "default"}),
            ],
            tools=[query_database, get_sample_questions, current_time],
            conversation_manager=SlidingWindowConversationManager(window_size=6),
        )

    def reset_metrics(self) -> None:
        self.metrics = {
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
            "cache_hits": 0,
            "steps": 0,
        }

    def update_strategy(self, strategy: str) -> None:
        """Update routing strategy on all agent models.
        
        Accepts strategy names (balanced, cost-optimized, quality-optimized, latency-optimized)
        and maps them to the correct routing config.
        """
        # Map frontend strategy names to preset names
        strategy_to_preset = {
            "balanced": "balanced",
            "cost-optimized": "economy",
            "quality-optimized": "quality",
            "latency-optimized": "speed",
        }
        preset = strategy_to_preset.get(strategy, "balanced")
        self._orch_model.update_config(routing_preset=preset)
        self._sql_model.update_config(routing_preset=preset)
        # Chart model keeps its preferred_model to avoid reasoning content issues

    def get_metrics(self) -> dict[str, Any]:
        cache_stats = self.cache.stats() if hasattr(self.cache, 'stats') else {}
        return {**self.metrics, "cache_stats": cache_stats}

    def _token_cb(
        self, input_tokens: int, output_tokens: int, step: str,
        cache_read: int = 0, cache_write: int = 0,
    ) -> None:
        self.metrics["total_input_tokens"] += input_tokens
        self.metrics["total_output_tokens"] += output_tokens
        self.metrics["cache_read_tokens"] += cache_read
        self.metrics["cache_write_tokens"] += cache_write
        self.metrics["steps"] += 1

