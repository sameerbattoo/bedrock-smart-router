#!/usr/bin/env python3
"""
Download the Natural Questions dataset.

Source: https://github.com/google-research-datasets/natural-questions
Paper:  Kwiatkowski et al., "Natural Questions: a Benchmark for Question
        Answering Research", TACL 2019.

Uses the sentence-transformers/natural-questions HuggingFace dataset which
provides a simplified format: (query, answer, title) with passage text.
"""
import json
import os
import sys

RAW_DIR = os.path.join(os.path.dirname(__file__), "raw")


def download():
    os.makedirs(RAW_DIR, exist_ok=True)

    out_path = os.path.join(RAW_DIR, "nq_dev.jsonl")
    if os.path.exists(out_path):
        with open(out_path) as f:
            count = sum(1 for _ in f)
        print(f"nq_dev.jsonl already exists ({count} examples), skipping.")
        return

    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' package required.  pip install datasets")
        sys.exit(1)

    print("Downloading Natural Questions from HuggingFace …")
    # sentence-transformers version has clean (query, answer) pairs
    # with passage context in the answer field
    ds = load_dataset("sentence-transformers/natural-questions", split="train")

    # The dataset is large (100K+). We take a random sample for our purposes.
    # Shuffle with seed for reproducibility
    ds = ds.shuffle(seed=42)

    written = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for row in ds:
            query = row.get("query", "").strip()
            answer = row.get("answer", "").strip()

            if not query or not answer:
                continue
            # Skip very short answers (likely noise)
            if len(answer) < 10:
                continue

            f.write(json.dumps({
                "question": query,
                "answer": answer,
            }, ensure_ascii=False) + "\n")
            written += 1
            if written >= 10000:
                break

    print(f"  Saved {written} examples to {out_path}")


if __name__ == "__main__":
    download()
