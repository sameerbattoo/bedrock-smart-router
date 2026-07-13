# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""SQL Agent — generates SQL from natural language and executes against SQLite.

Uses SmartRouterModel for LLM calls and the local SQLite database.
"""
import logging
import re
import time
from typing import Any, Callable

from strands import Agent
from strands.types.content import SystemContentBlock
from strands.agent.conversation_manager.sliding_window_conversation_manager import SlidingWindowConversationManager

from bedrock_smart_router.strands_model import SmartRouterModel
from text2sql.db import execute_query, get_schema_ddl
from text2sql.schema_optimizer import optimize_schema

logger = logging.getLogger(__name__)

MAX_DISPLAY_ROWS = 50


class SQLAgent:
    """Generates SQL from natural language, executes it, returns results."""

    def __init__(self, router_model: SmartRouterModel, token_callback: Callable | None = None):
        self._router_model = router_model
        self._token_callback = token_callback
        self._schema_ddl = get_schema_ddl()

        # Build optimized schema for the system prompt
        self._optimized_schema = self._build_optimized_schema()
        self._system_prompt = self._build_system_prompt()

        # Single agent instance per session — maintains conversation context
        self._agent = Agent(
            model=self._router_model,
            system_prompt=[
                SystemContentBlock(text=self._system_prompt),
                SystemContentBlock(cachePoint={"type": "default"}),
            ],
            conversation_manager=SlidingWindowConversationManager(window_size=4),
        )

    def _build_optimized_schema(self) -> str:
        """Parse DDL into table dict and optimize for token efficiency."""
        tables: dict[str, str] = {}
        current_table = None
        current_lines: list[str] = []

        for line in self._schema_ddl.split('\n'):
            if line.strip().startswith('CREATE TABLE'):
                if current_table and current_lines:
                    tables[current_table] = '\n'.join(current_lines)
                match = re.search(r'CREATE TABLE\s+(\w+)', line)
                current_table = match.group(1) if match else None
                current_lines = [line]
            elif current_table:
                current_lines.append(line)

        if current_table and current_lines:
            tables[current_table] = '\n'.join(current_lines)

        if tables:
            return optimize_schema(tables)
        return self._schema_ddl

    def generate_and_execute(self, user_query: str, status_callback=None) -> dict[str, Any]:
        """NL question → SQL generation → execution → results."""
        start = time.perf_counter()

        def _status(msg):
            if status_callback:
                status_callback(msg)

        # Generate SQL via LLM — fresh agent per call to avoid conversation pollution
        try:
            agent = Agent(
                model=self._router_model,
                system_prompt=[
                    SystemContentBlock(text=self._system_prompt),
                    SystemContentBlock(cachePoint={"type": "default"}),
                ],
                conversation_manager=SlidingWindowConversationManager(window_size=2),
            )
            result = agent(f"Generate a SQLite SELECT query for: {user_query}")
        except Exception as exc:
            logger.error("SQL agent LLM call failed: %s", exc)
            _status("❌ SQL generation failed")
            return {
                "sql": None, "columns": [], "results": [], "row_count": 0,
                "error": f"SQL generation failed: {str(exc)[:200]}",
                "generation_ms": round((time.perf_counter() - start) * 1000, 1),
            }

        sql = self._extract_sql(str(result))
        if not sql:
            _status("❌ Could not extract valid SQL")
            return {
                "sql": None, "columns": [], "results": [], "row_count": 0,
                "error": "Could not generate a valid SQL query.",
                "generation_ms": round((time.perf_counter() - start) * 1000, 1),
            }

        _status("⚙️ SQL generated — executing query...")

        # Report token usage
        decision = self._router_model.last_routing_decision
        gen_ms = round((time.perf_counter() - start) * 1000, 1)

        if decision and self._token_callback:
            self._token_callback(
                decision.input_tokens or 0,
                decision.output_tokens or 0,
                "sql_generation",
                getattr(decision, "prompt_cache_read_tokens", 0),
                getattr(decision, "prompt_cache_write_tokens", 0),
            )

        # Execute SQL
        try:
            exec_result = execute_query(sql)
        except Exception as exc:
            return {
                "sql": sql, "columns": [], "results": [], "row_count": 0,
                "error": str(exc),
                "generation_ms": gen_ms,
                "model_used": decision.selected_model if decision else "unknown",
            }

        # Fix #6: Always include columns in result
        return {
            "sql": sql,
            "columns": exec_result["columns"],
            "results": exec_result["rows"][:MAX_DISPLAY_ROWS],
            "row_count": exec_result["row_count"],
            "generation_ms": gen_ms,
            "model_used": decision.selected_model if decision else "unknown",
        }

    def _build_system_prompt(self) -> str:
        return f"""You are a SQLite SQL expert. Given a natural language question,
generate a single SELECT query that answers it.

SCHEMA KEY: int=integer, str=varchar, ts=timestamp, dec=decimal, num=numeric,
bool=boolean, int8=bigint, txt=text, PK=PRIMARY KEY, FK=FOREIGN KEY, REF=REFERENCES

DATABASE SCHEMA:
{self._optimized_schema}

RULES:
1. Use ONLY the tables and columns defined in the schema above. Do NOT invent or guess column names.
2. Return ONLY the SQL query. No explanation, no markdown fences.
3. Generate SELECT statements ONLY. Never use DROP, DELETE, INSERT, UPDATE, ALTER, CREATE.
4. Use SQLite syntax (e.g., strftime for dates, || for concatenation).
5. For date filtering use: strftime('%Y', column_name) or column LIKE '2025%'.
6. End with a semicolon.
7. Use appropriate JOINs based on foreign key relationships.
8. For aggregations, ensure GROUP BY includes all non-aggregated columns.
9. Use LIMIT for top-N queries.
10. CRITICAL: If a column doesn't exist in the schema, DO NOT use it. Double-check every column name against the schema before generating the query.
11. For customer names, concatenate first_name and last_name if needed (e.g., first_name || ' ' || last_name).
12. The customers table uses 'created_at' for when the customer was created, NOT 'date_first_purchase' or 'signup_date'.
"""

    @staticmethod
    def _extract_sql(text: str) -> str | None:
        """Extract SQL from LLM response."""
        match = re.search(r"```(?:sql)?\s*(SELECT.+?)```", text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip().rstrip(";") + ";"
        match = re.search(r"(SELECT\s.+?);", text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip() + ";"
        stripped = text.strip()
        if stripped.upper().startswith("SELECT"):
            return stripped.rstrip(";") + ";"
        return None
