# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

#!/usr/bin/env python3
"""Set up the SQLite database for the Text2SQL demo.

Imports and calls init_database() from the backend's text2sql.db module.
Idempotent: skips seeding if the database already exists.
"""

import sys
from pathlib import Path

# Add the backend directory to the path so we can import text2sql
BACKEND_DIR = str(Path(__file__).parent.parent / "backend")
sys.path.insert(0, BACKEND_DIR)


def main() -> bool:
    """Initialize the demo database. Returns True on success."""
    print("Setting up Text2SQL database...")

    try:
        from text2sql.db import DB_PATH, init_database

        already_exists = DB_PATH.exists()
        init_database()

        if already_exists:
            print(f"  Database already seeded: {DB_PATH}")
        else:
            print(f"  Database created and seeded: {DB_PATH}")

        return True
    except Exception as e:
        print(f"  Error setting up database: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
