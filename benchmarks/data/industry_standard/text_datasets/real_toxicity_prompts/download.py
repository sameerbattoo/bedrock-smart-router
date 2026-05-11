#!/usr/bin/env python3
"""
Download the RealToxicityPrompts dataset.

Source: https://github.com/allenai/real-toxicity-prompts
Paper:  Gehman et al., "RealToxicityPrompts: Evaluating Neural Toxic
        Degeneration in Language Models", EMNLP Findings 2020.

99K prompts (sentence beginnings) with toxicity scores. The task is to
test whether models generate toxic completions.
"""
import json
import os
import sys

RAW_DIR = os.path.join(os.path.dirname(__file__), "raw")


def download():
    os.makedirs(RAW_DIR, exist_ok=True)

    out_path = os.path.join(RAW_DIR, "prompts.jsonl")
    if os.path.exists(out_path):
        with open(out_path) as f:
            count = sum(1 for _ in f)
        print(f"prompts.jsonl already exists ({count} examples), skipping.")
        return

    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' package required.  pip install datasets")
        sys.exit(1)

    print("Downloading RealToxicityPrompts from HuggingFace …")
    ds = load_dataset("allenai/real-toxicity-prompts", split="train")

    written = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for row in ds:
            prompt = row.get("prompt", {})
            text = (prompt.get("text") or "").strip()
            toxicity = prompt.get("toxicity")
            challenging = row.get("challenging", False)

            if not text or toxicity is None:
                continue

            f.write(json.dumps({
                "prompt_text": text,
                "prompt_toxicity": toxicity,
                "challenging": challenging,
            }, ensure_ascii=False) + "\n")
            written += 1

    print(f"  Saved {written} prompts to {out_path}")


if __name__ == "__main__":
    download()
