#!/usr/bin/env python3
"""Run all prompt generators to create/refresh the prompts/ JSON files."""
import os
import subprocess
import sys

GEN_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "generated_scripts")

print("Generating all benchmark prompts...")
print("=" * 50)

generators = sorted(f for f in os.listdir(GEN_DIR) if f.startswith("gen_") and f.endswith(".py"))

for gen_file in generators:
    path = os.path.join(GEN_DIR, gen_file)
    print(f"\nRunning {gen_file}...")
    result = subprocess.run([sys.executable, path], capture_output=True, text=True)
    if result.returncode == 0:
        print(result.stdout.strip())
    else:
        print(f"  ERROR: {result.stderr.strip()}")

print("\n" + "=" * 50)
print("Done! Prompt files are in benchmarks/prompts/")
