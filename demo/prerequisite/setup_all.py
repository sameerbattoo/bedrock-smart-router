#!/usr/bin/env python3
"""Run all prerequisite setup scripts for the Bedrock Smart Router demo.

Order:
  1. Database (Text2SQL SQLite)
  2. Guardrail (Bedrock Guardrail with PII + content filtering)
"""

import sys
from pathlib import Path

# Ensure this script can find sibling modules
sys.path.insert(0, str(Path(__file__).parent))

from setup_database import main as setup_database
from setup_guardrail import main as setup_guardrail


def main():
    print("=" * 60)
    print("  Bedrock Smart Router Demo — Prerequisites")
    print("=" * 60)
    print()

    results = {}

    # 1. Database
    results["database"] = setup_database()
    print()

    # 2. Guardrail
    results["guardrail"] = setup_guardrail()
    print()

    # Summary
    print("=" * 60)
    print("  Summary")
    print("=" * 60)
    for name, success in results.items():
        status = "✓" if success else "✗"
        print(f"  {status} {name}")

    all_passed = all(results.values())
    if all_passed:
        print("\n  All prerequisites ready!")
    else:
        failed = [k for k, v in results.items() if not v]
        print(f"\n  Warning: {', '.join(failed)} failed. Demo may have limited functionality.")

    print()
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
