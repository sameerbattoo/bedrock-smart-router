"""Chart Agent — generates visualizations using strands python_repl tool.

Uses SmartRouterModel for code generation and python_repl for execution.
Charts are saved locally and served via FastAPI.
"""
import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Callable

from strands import Agent
from strands.agent.conversation_manager.sliding_window_conversation_manager import SlidingWindowConversationManager
from strands_tools import python_repl, file_write

from bedrock_smart_router.strands_model import SmartRouterModel

logger = logging.getLogger(__name__)

CHART_DIR = Path("/tmp/text2sql_charts")
CHART_DIR.mkdir(exist_ok=True)


class ChartAgent:
    """Generates chart images from query results using python_repl."""

    def __init__(self, router_model: SmartRouterModel, token_callback: Callable | None = None):
        self._router_model = router_model
        self._token_callback = token_callback

    def generate_chart(
        self, user_query: str, rows: list[dict], columns: list[str],
    ) -> dict[str, Any]:
        """Generate a chart from query results. Returns {success, filename}."""
        if not rows or len(rows) < 2:
            return {"success": False, "filename": None, "message": "Not enough data for chart."}

        start = time.perf_counter()
        timestamp = int(time.time())
        filename = f"chart_{timestamp}.png"
        filepath = CHART_DIR / filename

        data_sample = json.dumps(rows[:50], default=str)
        prompt = (
            f"Generate Python matplotlib code to visualize this data and save it to '{filepath}'.\n"
            f"IMPORTANT: Start code with: import matplotlib; matplotlib.use('Agg')\n"
            f"User question: {user_query}\n"
            f"Columns: {columns}\n"
            f"Data ({len(rows)} rows, showing first 50):\n{data_sample}\n\n"
            f"Requirements:\n"
            f"- FIRST line: import matplotlib; matplotlib.use('Agg')\n"
            f"- Use dark theme: facecolor='#1a2332', text color '#ecf0f1'\n"
            f"- Colors: ['#5DADE2', '#EC7063', '#58D68D', '#F39C12', '#AF7AC5', '#48C9B0']\n"
            f"- Save with dpi=100, bbox_inches='tight'\n"
            f"- Embed the full data as a Python literal in the code\n"
            f"- Do NOT use plt.show()\n"
            f"- Save to EXACTLY: {filepath}"
        )

        try:
            # Fresh agent per chart to avoid conversation pollution
            # Use a no-op callback to prevent chart agent output from leaking
            # into the orchestrator's streaming response
            agent = Agent(
                model=self._router_model,
                system_prompt=self._build_system_prompt(),
                tools=[python_repl, file_write],
                conversation_manager=SlidingWindowConversationManager(window_size=4),
                callback_handler=None,
            )
            agent(prompt)

            # Report tokens
            decision = self._router_model.last_routing_decision
            if decision and self._token_callback:
                self._token_callback(
                    decision.input_tokens or 0,
                    decision.output_tokens or 0,
                    "chart_generation",
                    getattr(decision, "prompt_cache_read_tokens", 0),
                    getattr(decision, "prompt_cache_write_tokens", 0),
                )

            if filepath.exists():
                elapsed = time.perf_counter() - start
                logger.info("Chart generated in %.1fs: %s", elapsed, filename)
                return {"success": True, "filename": filename, "message": "Chart generated."}
            else:
                return {"success": False, "filename": None, "message": "Chart file not created."}

        except Exception as exc:
            logger.error("Chart generation failed: %s", exc)
            return {"success": False, "filename": None, "message": str(exc)[:200]}

    @staticmethod
    def _build_system_prompt() -> str:
        return """You are a chart visualization expert. When asked to create a chart:

1. Write Python code using matplotlib and pandas
2. Use the python_repl tool to execute the code
3. Save the chart to the EXACT file path specified in the prompt
4. Use dark theme: facecolor='#1a2332', text color '#ecf0f1'
5. Use these colors: ['#5DADE2', '#EC7063', '#58D68D', '#F39C12', '#AF7AC5', '#48C9B0']
6. Include proper labels, title, and legend
7. Use plt.tight_layout() before saving

CRITICAL RULES:
- ALWAYS start your code with these two lines FIRST before any other imports:
  import matplotlib
  matplotlib.use('Agg')
- Then import matplotlib.pyplot as plt
- Save to the exact path given. Do NOT use plt.show().
- Always embed the full data as a Python literal. Do NOT use json.loads().
- After the chart is saved successfully, respond with ONLY: "Chart saved successfully to <path>"
- Do NOT describe the chart, explain its features, or provide commentary.
- Keep your response minimal — just generate the code, execute it, confirm success.
"""
