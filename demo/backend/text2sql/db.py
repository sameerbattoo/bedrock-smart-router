"""SQLite database wrapper for the Text2SQL demo.

Creates and seeds an e-commerce database on startup.
Provides schema extraction and query execution.
"""
import logging
import sqlite3
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DB_PATH = Path("/tmp/text2sql_demo.db")
SEED_SQL = Path(__file__).parent.parent.parent / "prerequisite" / "seed_data.sql"

_local = threading.local()


def _get_conn() -> sqlite3.Connection:
    """Get a thread-local SQLite connection."""
    if not hasattr(_local, "conn") or _local.conn is None:
        _local.conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        _local.conn.row_factory = sqlite3.Row
    return _local.conn


def init_database() -> None:
    """Create and seed the database if it doesn't exist or is empty."""
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()

    # Check if tables exist
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='categories'")
    if cursor.fetchone():
        conn.close()
        logger.info("Database already seeded: %s", DB_PATH)
        return

    # Read and execute seed SQL
    if SEED_SQL.exists():
        sql = SEED_SQL.read_text()
        conn.executescript(sql)
        conn.commit()
        logger.info("Database seeded from %s", SEED_SQL)
    else:
        logger.warning("Seed file not found: %s", SEED_SQL)

    conn.close()


def execute_query(sql: str) -> dict[str, Any]:
    """Execute a SELECT query and return results."""
    # Guard: only SELECT allowed
    upper = sql.strip().upper()
    if not upper.startswith("SELECT"):
        raise ValueError("Only SELECT queries are allowed.")

    conn = _get_conn()
    try:
        cursor = conn.execute(sql)
        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        rows = [dict(row) for row in cursor.fetchall()]
        return {"columns": columns, "rows": rows, "row_count": len(rows)}
    except Exception as exc:
        raise RuntimeError(f"Query failed: {exc}") from exc


def get_schema_ddl() -> str:
    """Extract the database schema as DDL text."""
    conn = _get_conn()
    cursor = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL ORDER BY name"
    )
    tables = [row[0] for row in cursor.fetchall()]
    return "\n\n".join(tables)


def get_table_list() -> list[str]:
    """List all table names."""
    conn = _get_conn()
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    return [row[0] for row in cursor.fetchall()]
