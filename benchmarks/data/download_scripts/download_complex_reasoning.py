# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

#!/usr/bin/env python3
"""Download 50 samples each from complex reasoning datasets:
1. GSM8K (multi-step math reasoning)
2. GPQA (graduate-level science questions)
3. MBPP (complex Python programming)

All labeled as "complex" for the complexity classifier training.

Requirements:
    pip install datasets
"""
import json
import os
import sys

try:
    from datasets import load_dataset
except ImportError:
    print("ERROR: 'datasets' library not installed. Run: pip install datasets")
    sys.exit(1)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_SIZE = 50


def download_gsm8k():
    """Download 50 complex math reasoning problems from GSM8K."""
    print("\n" + "=" * 60)
    print("1. Downloading openai/gsm8k (multi-step math reasoning)")
    print("=" * 60)

    out_dir = os.path.join(BASE_DIR, "gsm8k")
    os.makedirs(out_dir, exist_ok=True)

    ds = load_dataset("openai/gsm8k", "main", split="test")
    print(f"  Dataset size: {len(ds)} samples")

    # Pick the hardest problems (longest solutions = more steps)
    items = list(ds)
    # Sort by answer length (more steps = more complex)
    items.sort(key=lambda x: len(x.get("answer", "")), reverse=True)

    samples = []
    for i, item in enumerate(items[:SAMPLE_SIZE]):
        question = item.get("question", "")
        answer = item.get("answer", "")

        # Extract just the final numeric answer
        final_answer = ""
        if "####" in answer:
            final_answer = answer.split("####")[-1].strip()

        samples.append({
            "id": f"gsm8k_{i:03d}",
            "question": question,
            "full_solution": answer,
            "final_answer": final_answer,
            "task": "math_reasoning",
            "difficulty": "complex",
            "source": "openai/gsm8k",
        })

    meta_path = os.path.join(out_dir, "samples.json")
    with open(meta_path, "w") as f:
        json.dump(samples, f, indent=2)

    print(f"  Saved {len(samples)} samples to {out_dir}")
    print(f"  Sample Q: {samples[0]['question'][:80]}...")
    print(f"  Sample A: {samples[0]['final_answer']}")
    return len(samples)


def download_gpqa():
    """Download 50 graduate-level science questions from GPQA."""
    print("\n" + "=" * 60)
    print("2. Downloading GPQA (graduate-level science reasoning)")
    print("=" * 60)

    out_dir = os.path.join(BASE_DIR, "gpqa")
    os.makedirs(out_dir, exist_ok=True)

    # Try different GPQA sources
    ds = None
    for dataset_id in ["Idavidrein/gpqa", "casimiir/gpqa", "math-ai/gpqa"]:
        try:
            ds = load_dataset(dataset_id, split="train")
            print(f"  Loaded {dataset_id}: {len(ds)} samples")
            print(f"  Columns: {ds.column_names}")
            break
        except Exception as e:
            print(f"  {dataset_id} failed: {str(e)[:60]}")
            continue

    if ds is None:
        print("  All GPQA sources failed. Trying GPQA Diamond...")
        try:
            ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
            print(f"  Loaded GPQA Diamond: {len(ds)} samples")
        except Exception as e:
            print(f"  GPQA Diamond also failed: {e}")
            print("  Skipping GPQA.")
            return 0

    samples = []
    for i, item in enumerate(ds):
        if i >= SAMPLE_SIZE:
            break

        # GPQA has different column names depending on the version
        question = item.get("Question", item.get("question", ""))
        correct_answer = item.get("Correct Answer", item.get("correct_answer", ""))
        choices = []
        for key in ["Correct Answer", "Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3"]:
            val = item.get(key, "")
            if val:
                choices.append(val)

        if not question:
            continue

        samples.append({
            "id": f"gpqa_{i:03d}",
            "question": question,
            "correct_answer": correct_answer,
            "choices": choices,
            "task": "science_reasoning",
            "difficulty": "complex",
            "source": "gpqa",
        })

    meta_path = os.path.join(out_dir, "samples.json")
    with open(meta_path, "w") as f:
        json.dump(samples, f, indent=2)

    print(f"  Saved {len(samples)} samples to {out_dir}")
    if samples:
        print(f"  Sample Q: {samples[0]['question'][:80]}...")
    return len(samples)


def download_mbpp():
    """Download 50 complex Python programming problems from MBPP."""
    print("\n" + "=" * 60)
    print("3. Downloading google-research-datasets/mbpp (Python coding)")
    print("=" * 60)

    out_dir = os.path.join(BASE_DIR, "mbpp")
    os.makedirs(out_dir, exist_ok=True)

    ds = load_dataset("google-research-datasets/mbpp", "full", split="test")
    print(f"  Dataset size: {len(ds)} samples")
    print(f"  Columns: {ds.column_names}")

    # Sort by code length (longer solutions = more complex)
    items = list(ds)
    items.sort(key=lambda x: len(x.get("code", "")), reverse=True)

    samples = []
    for i, item in enumerate(items[:SAMPLE_SIZE]):
        prompt = item.get("text", item.get("prompt", ""))
        code = item.get("code", "")
        test_list = item.get("test_list", [])

        if not prompt:
            continue

        samples.append({
            "id": f"mbpp_{i:03d}",
            "prompt": prompt,
            "solution_code": code,
            "test_cases": test_list,
            "task": "code_generation",
            "difficulty": "complex",
            "source": "google-research-datasets/mbpp",
        })

    meta_path = os.path.join(out_dir, "samples.json")
    with open(meta_path, "w") as f:
        json.dump(samples, f, indent=2)

    print(f"  Saved {len(samples)} samples to {out_dir}")
    if samples:
        print(f"  Sample: {samples[0]['prompt'][:80]}...")
    return len(samples)


def main():
    print("Downloading complex reasoning datasets")
    print(f"Target: {SAMPLE_SIZE} samples per dataset")
    print(f"Output: {BASE_DIR}")

    total = 0
    total += download_gsm8k()
    total += download_gpqa()
    total += download_mbpp()

    print("\n" + "=" * 60)
    print(f"DONE: {total} complex reasoning samples downloaded")
    print("=" * 60)


if __name__ == "__main__":
    main()
