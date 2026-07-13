# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Schema optimizer — compresses CREATE TABLE DDL for LLM token efficiency.

Converts verbose DDL like:
    CREATE TABLE tenanta.orders (
        order_id integer NOT NULL DEFAULT nextval('orders_order_id_seq'::regclass),
        customer_id integer NOT NULL,
        order_date timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
        status character varying(20) DEFAULT 'pending',
        total_amount numeric(10,2),
        PRIMARY KEY (order_id),
        FOREIGN KEY (customer_id) REFERENCES tenanta.customers(customer_id)
    );

Into compact "Schema on a String" format:
    tenanta.orders(order_id:int,customer_id:int,order_date:ts,status:str(20),total_amount:num(10,2),PK(order_id),FK(customer_id) REF tenanta.customers(customer_id))

This reduces token usage by ~60-70% while preserving all information
the LLM needs for SQL generation (column names, types, keys, relationships).
"""

import logging
import re
from typing import Dict

logger = logging.getLogger(__name__)


def optimize_schema(create_statements: Dict[str, str]) -> str:
    """Optimize all CREATE TABLE statements into a compact schema string.

    Args:
        create_statements: Mapping of table_name → CREATE TABLE DDL.

    Returns:
        Compact schema string suitable for LLM system prompts.
    """
    optimized = []
    for table, stmt in create_statements.items():
        try:
            compact = _optimize_table(stmt)
            optimized.append(f"-- {table}\n{compact}")
        except Exception as exc:
            logger.warning("Schema optimization failed for '%s': %s", table, exc)
            optimized.append(f"-- {table}\n{stmt}")
    return "\n\n".join(optimized)


def _optimize_table(create_statement: str) -> str:
    """Optimize a single CREATE TABLE statement."""
    # Step 1: Strip NOT NULL and DEFAULT constraints
    cleaned = _remove_non_essential_constraints(create_statement)
    # Step 2: Convert to schema-on-a-string format
    schema_str = _convert_to_schema_string(cleaned)
    # Step 3: Compress type names and keywords
    compressed = _compress(schema_str)
    return compressed


# ── Step 1: Remove non-essential constraints ────────────────

def _remove_non_essential_constraints(stmt: str) -> str:
    # NOT NULL
    s = re.sub(r'\s+NOT\s+NULL', '', stmt, flags=re.IGNORECASE)
    # DEFAULT with function calls: nextval('seq'::regclass)
    s = re.sub(r"\s+DEFAULT\s+[a-zA-Z_]\w*\([^)]*\)(?:::\w+)?(?=\s*[,)])", '', s, flags=re.IGNORECASE)
    # DEFAULT with quoted strings
    s = re.sub(r"\s+DEFAULT\s+'[^']*'(?=\s*[,)])", '', s, flags=re.IGNORECASE)
    s = re.sub(r'\s+DEFAULT\s+"[^"]*"(?=\s*[,)])', '', s, flags=re.IGNORECASE)
    # DEFAULT with numbers
    s = re.sub(r'\s+DEFAULT\s+\d+(?:\.\d+)?(?=\s*[,)])', '', s, flags=re.IGNORECASE)
    # DEFAULT with keywords (CURRENT_TIMESTAMP, TRUE, FALSE)
    s = re.sub(r'\s+DEFAULT\s+[A-Z_]+(?=\s*[,)])', '', s, flags=re.IGNORECASE)
    # Normalize whitespace
    s = re.sub(r'\s+', ' ', s)
    s = re.sub(r'\s*,\s*', ', ', s)
    s = re.sub(r',\s*,', ',', s)
    s = re.sub(r',\s*\)', ')', s)
    return s.strip()


# ── Step 2: Convert to schema-on-a-string ───────────────────

def _convert_to_schema_string(stmt: str) -> str:
    # Extract table name
    table_match = re.search(r'CREATE\s+TABLE\s+([^\s(]+)', stmt, re.IGNORECASE)
    if not table_match:
        return stmt
    table_name = table_match.group(1)

    # Extract content between outer parentheses
    paren_match = re.search(r'\((.*)\)', stmt, re.DOTALL)
    if not paren_match:
        return stmt
    content = paren_match.group(1)

    # Split by commas respecting nested parens
    parts = _split_respecting_parens(content)

    constraint_re = re.compile(r'^(PRIMARY\s+KEY|FOREIGN\s+KEY|UNIQUE|CHECK|CONSTRAINT)', re.IGNORECASE)

    columns = []
    constraints = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if constraint_re.match(part):
            constraints.append(part)
        else:
            formatted = _format_column(part)
            if formatted:
                columns.append(formatted)

    all_parts = columns + constraints
    return f"{table_name}({', '.join(all_parts)})"


def _split_respecting_parens(content: str) -> list:
    parts = []
    current = []
    depth = 0
    for ch in content:
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
        elif ch == ',' and depth == 0:
            parts.append(''.join(current).strip())
            current = []
            continue
        current.append(ch)
    tail = ''.join(current).strip()
    if tail:
        parts.append(tail)
    return parts


def _format_column(col_def: str) -> str:
    col_def = re.sub(r'\s+', ' ', col_def.strip())
    match = re.match(r'^(\w+)\s+(\w+(?:\([^)]+\))?)', col_def)
    if match:
        return f"{match.group(1)}: {match.group(2)}"
    parts = col_def.split(None, 1)
    if len(parts) >= 2:
        return f"{parts[0]}: {parts[1]}"
    return col_def


# ── Step 3: Compress type names and keywords ────────────────

_TYPE_MAP = [
    (re.compile(r'\binteger\b', re.I), 'int'),
    (re.compile(r'\bvarchar\b', re.I), 'str'),
    (re.compile(r'\bcharacter varying\b', re.I), 'str'),
    (re.compile(r'\bcharacter\b', re.I), 'char'),
    (re.compile(r'\btimestamp\b', re.I), 'ts'),
    (re.compile(r'\bdecimal\b', re.I), 'dec'),
    (re.compile(r'\bnumeric\b', re.I), 'num'),
    (re.compile(r'\bboolean\b', re.I), 'bool'),
    (re.compile(r'\bbigint\b', re.I), 'int8'),
    (re.compile(r'\btext\b', re.I), 'txt'),
]

_CONSTRAINT_MAP = [
    (re.compile(r'\bPRIMARY KEY\b', re.I), 'PK'),
    (re.compile(r'\bFOREIGN KEY\b', re.I), 'FK'),
    (re.compile(r'\bREFERENCES\b', re.I), 'REF'),
    (re.compile(r'\bUNIQUE\b', re.I), 'UQ'),
    (re.compile(r'\bCHECK\b', re.I), 'CK'),
]

_SPACING = [
    (re.compile(r'\s*\(\s*'), '('),
    (re.compile(r'\s*\)\s*'), ')'),
    (re.compile(r'\s*,\s*'), ','),
    (re.compile(r'\s*:\s*'), ':'),
    (re.compile(r'(FK|PK|UQ|CK)\s+\('), r'\1('),
]


def _compress(schema_str: str) -> str:
    s = schema_str
    for pattern, repl in _TYPE_MAP:
        s = pattern.sub(repl, s)
    for pattern, repl in _CONSTRAINT_MAP:
        s = pattern.sub(repl, s)
    for pattern, repl in _SPACING:
        s = pattern.sub(repl, s)
    return s
