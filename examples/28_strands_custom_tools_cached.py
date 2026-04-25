"""Strands Custom Tools with Smart Router + Semantic Cache.

Adapted from the 2nd official Strands Agents SDK sample (Custom Tools):
https://github.com/strands-agents/samples/blob/main/python/01-learn/02-tools-and-mcp/02-custom-tools/custom-tools-with-strands-agents.ipynb

The original sample builds a personal appointment assistant with custom
tools (create, list, update appointments) backed by SQLite, using a
fixed Claude Sonnet 4.5 model.

This adapted version:
  1. Replaces BedrockModel with SmartRouterModel for intelligent routing
  2. Adds auto-extracting semantic cache so repeated questions are free
  3. Shows multi-turn caching with native Strands agent.messages history
  4. Demonstrates streaming output with routing decisions

Requirements:
    pip install bedrock-smart-router[strands]
"""

import json
import os
import sqlite3
import uuid
from datetime import datetime

from strands import Agent, tool
from bedrock_smart_router.strands_model import SmartRouterModel
from bedrock_smart_router.semantic_cache import SemanticCache, SemanticCacheConfig


# ═══════════════════════════════════════════════════════════════════
# Custom Tools — same as the AWS sample, simplified for this example
# ═══════════════════════════════════════════════════════════════════

DB_PATH = "/tmp/appointments_demo.db"


def _get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS appointments (
            id TEXT PRIMARY KEY,
            date TEXT,
            location TEXT,
            title TEXT,
            description TEXT
        )
    """)
    return conn


@tool
def create_appointment(date: str, location: str, title: str, description: str) -> str:
    """Create a new personal appointment in the database.

    Args:
        date: Date and time (format: YYYY-MM-DD HH:MM).
        location: Location of the appointment.
        title: Title of the appointment.
        description: Description of the appointment.

    Returns:
        Confirmation with the appointment ID.
    """
    try:
        datetime.strptime(date, "%Y-%m-%d %H:%M")
    except ValueError:
        return "Error: Date must be in format 'YYYY-MM-DD HH:MM'"

    appointment_id = str(uuid.uuid4())[:8]
    conn = _get_db()
    conn.execute(
        "INSERT INTO appointments (id, date, location, title, description) VALUES (?, ?, ?, ?, ?)",
        (appointment_id, date, location, title, description),
    )
    conn.commit()
    conn.close()
    return f"Appointment '{title}' created with ID {appointment_id}"


@tool
def list_appointments() -> str:
    """List all available appointments from the database.

    Returns:
        JSON list of all appointments, or a message if none exist.
    """
    if not os.path.exists(DB_PATH):
        return "No appointments available."
    conn = _get_db()
    rows = conn.execute("SELECT * FROM appointments ORDER BY date").fetchall()
    conn.close()
    if not rows:
        return "No appointments available."
    appointments = [dict(row) for row in rows]
    return json.dumps(appointments, indent=2)


@tool
def delete_appointment(appointment_id: str) -> str:
    """Delete an appointment by its ID.

    Args:
        appointment_id: The ID of the appointment to delete.

    Returns:
        Confirmation message.
    """
    conn = _get_db()
    cursor = conn.execute("DELETE FROM appointments WHERE id = ?", (appointment_id,))
    conn.commit()
    conn.close()
    if cursor.rowcount == 0:
        return f"Appointment {appointment_id} not found."
    return f"Appointment {appointment_id} deleted."


# ═══════════════════════════════════════════════════════════════════
# Setup: Smart Router Model + Semantic Cache
# ═══════════════════════════════════════════════════════════════════

# Clean up any previous demo database
if os.path.exists(DB_PATH):
    os.remove(DB_PATH)

# SmartRouterModel — replaces the fixed BedrockModel from the AWS sample
model = SmartRouterModel(
    router_config={"region": "us-west-2"},
    max_tokens=512,
)

# Auto-extracting semantic cache with FAISS backend and multi-turn resolution
cache = SemanticCache(
    config=SemanticCacheConfig(
        threshold=0.85,
        auto_extract=True,
        extraction_model="us.amazon.nova-micro-v1:0",
        vector_store_backend="faiss",       # FAISS for fast similarity search
        embedding_dimension=1024,           # Must match Titan v2 output
    ),
    region="us-west-2",
)

# System prompt — same as the AWS sample
system_prompt = """You are a helpful personal assistant that manages appointments.
You have tools to create, list, and delete appointments.
Always provide the appointment ID so the user can reference it later.
Keep responses concise."""

agent = Agent(
    model=model,
    system_prompt=system_prompt,
    tools=[create_appointment, list_appointments, delete_appointment],
)


# ═══════════════════════════════════════════════════════════════════
# Helper: Agent call with semantic cache check
# ═══════════════════════════════════════════════════════════════════

def ask(query: str, use_cache: bool = True) -> str:
    """Call the agent with optional semantic cache lookup.

    For read-only queries (list, show, count), the cache is checked
    first. For write operations (create, delete, update), the cache
    is skipped to ensure the action is performed.
    """
    # Check cache using the full conversation history for multi-turn
    if use_cache and len(agent.messages) >= 2:
        full_messages = list(agent.messages) + [
            {"role": "user", "content": [{"text": query}]},
        ]
        cached = cache.get(messages=full_messages)
        if cached is not None:
            print(f"  [CACHE HIT — saved a Bedrock call]")
            print(f"  {cached['text']}")
            return cached["text"]

    # Single-turn cache check
    if use_cache:
        cached = cache.get(query)
        if cached is not None:
            print(f"  [CACHE HIT — saved a Bedrock call]")
            print(f"  {cached['text']}")
            return cached["text"]

    # Cache miss — call the agent
    response = agent(query)
    text = str(response)

    # Store in cache (for read-only queries)
    if use_cache:
        cache.put(query, {"text": text})

    d = model.last_routing_decision
    print(f"  [Model: {d.selected_model}, Cost: ${d.actual_cost:.6f}]")
    return text


# ═══════════════════════════════════════════════════════════════════
# Demo: Personal Appointment Assistant
# ═══════════════════════════════════════════════════════════════════

print("=" * 60)
print("Personal Appointment Assistant")
print("Smart Router + Custom Tools + Semantic Cache")
print("=" * 60)

# --- Turn 1: Create an appointment (skip cache — write operation) ---
print("\n📅 Turn 1: Creating an appointment...")
ask("Book 'Team Standup' for 2026-05-01 09:00 in Seattle. Daily sync meeting.", use_cache=False)

# --- Turn 2: Create another appointment ---
print("\n📅 Turn 2: Creating another appointment...")
ask("Schedule 'Architecture Review' for 2026-05-01 14:00 in NYC. Review microservices design.", use_cache=False)

# --- Turn 3: List appointments (cacheable) ---
print("\n📋 Turn 3: List all appointments...")
ask("Show me all my appointments")

# --- Turn 4: Same question, different wording → CACHE HIT ---
print("\n📋 Turn 4: Same question rephrased (should hit cache)...")
ask("What appointments do I have scheduled?")

# --- Turn 5: Delete an appointment (skip cache — write operation) ---
print("\n🗑️  Turn 5: Delete an appointment...")
ask("Delete the Team Standup appointment", use_cache=False)

# --- Turn 6: List again after deletion (cache should miss — data changed) ---
# Clear the cache since data changed
cache.invalidate()
print("\n📋 Turn 6: List appointments after deletion...")
ask("What appointments do I have?")

# --- Summary ---
print(f"\n{'=' * 60}")
print(f"Cache stats: {cache.stats}")
print(f"{'=' * 60}")
