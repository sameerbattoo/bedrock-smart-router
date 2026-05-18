"""Test script: Strands agent + python_repl tool to generate a matplotlib chart.

Run: BYPASS_TOOL_CONSENT=true python scripts/test_chart_repl.py
"""
import os
os.environ["BYPASS_TOOL_CONSENT"] = "true"
os.environ["MPLBACKEND"] = "Agg"

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pathlib import Path
from strands import Agent
from strands.models.bedrock import BedrockModel
from strands_tools import python_repl

CHART_PATH = Path("/tmp/test_chart_repl.png")
CHART_PATH.unlink(missing_ok=True)

# Use a simple Bedrock model directly (no smart router needed for this test)
model = BedrockModel(
    model_id="us.anthropic.claude-haiku-4-5-20251001-v1:0",
    region_name="us-west-2",
)

agent = Agent(
    model=model,
    system_prompt="You are a chart expert. Generate and execute Python code using the python_repl tool. Do NOT use plt.show(). Save charts to the exact path given.",
    tools=[python_repl],
)

data = [
    {"month": "2025-01", "orders": 61},
    {"month": "2025-02", "orders": 46},
    {"month": "2025-03", "orders": 49},
    {"month": "2025-04", "orders": 59},
    {"month": "2025-05", "orders": 58},
    {"month": "2025-06", "orders": 72},
    {"month": "2025-07", "orders": 86},
    {"month": "2025-08", "orders": 67},
    {"month": "2025-09", "orders": 72},
    {"month": "2025-10", "orders": 87},
    {"month": "2025-11", "orders": 78},
    {"month": "2025-12", "orders": 24},
]

prompt = f"""Generate a matplotlib bar chart for this data and save it to '{CHART_PATH}'.

Data: {data}

Requirements:
- FIRST two lines of code MUST be: import matplotlib; matplotlib.use('Agg')
- Dark theme: facecolor='#1a2332', text color '#ecf0f1'
- Bar color: '#5DADE2'
- Title: 'Monthly Order Trends 2025'
- X-axis: month, Y-axis: order count
- Save with dpi=100, bbox_inches='tight'
- Do NOT use plt.show()
"""

print("=" * 60)
print("Testing Strands Agent + python_repl chart generation")
print("=" * 60)
print(f"Target file: {CHART_PATH}")
print()

result = agent(prompt)

print()
print("=" * 60)
if CHART_PATH.exists():
    size = CHART_PATH.stat().st_size
    print(f"✅ SUCCESS: Chart generated at {CHART_PATH} ({size:,} bytes)")
else:
    print(f"❌ FAIL: Chart file not found at {CHART_PATH}")
print("=" * 60)
