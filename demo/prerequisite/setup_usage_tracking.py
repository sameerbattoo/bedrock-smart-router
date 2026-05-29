#!/usr/bin/env python3
"""Set up the SQLite database for the Usage & Cost Tracking demo.

Creates the usage_tracking table with indexes for per-user budget queries.
Idempotent: safe to run multiple times.
"""

import sqlite3
import sys
from pathlib import Path

DB_PATH = Path("/tmp/bsr_usage_tracking.db")


def main() -> bool:
    """Initialize the usage tracking database. Returns True on success."""
    print("Setting up Usage Tracking database...")

    try:
        conn = sqlite3.connect(str(DB_PATH))
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS usage_tracking (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                team TEXT,
                tier TEXT,
                timestamp REAL NOT NULL,
                model_id TEXT,
                display_model TEXT,
                strategy TEXT,
                complexity TEXT,
                input_tokens INTEGER DEFAULT 0,
                output_tokens INTEGER DEFAULT 0,
                cost REAL DEFAULT 0,
                latency_ms REAL DEFAULT 0,
                cache_hit INTEGER DEFAULT 0,
                budget_cap REAL DEFAULT 0,
                downgraded INTEGER DEFAULT 0
            );

            -- Primary index: per-user budget check (hot path)
            -- Query: SELECT SUM(cost) FROM usage_tracking WHERE user_id = ? AND timestamp > ?
            CREATE INDEX IF NOT EXISTS idx_usage_user_time
                ON usage_tracking(user_id, timestamp);

            -- Secondary index: per-team rollup (dashboard)
            -- Query: SELECT team, SUM(cost) FROM usage_tracking WHERE timestamp > ? GROUP BY team
            CREATE INDEX IF NOT EXISTS idx_usage_team_time
                ON usage_tracking(team, timestamp);
        """)
        conn.close()

        print(f"  Usage tracking database ready: {DB_PATH}")
        return True
    except Exception as e:
        print(f"  Error setting up usage tracking database: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
