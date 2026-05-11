#!/usr/bin/env python3
"""
Download the BoolQ dataset (yes/no question answering).

Source: https://github.com/google-research-datasets/boolean-questions
Paper:  Clark et al., "BoolQ: Exploring the Surprising Difficulty of Natural
        Yes/No Questions", NAACL 2019.
License: Creative Commons Share-Alike 3.0

Downloads via HuggingFace datasets library and saves as JSONL.
"""
import json
import os
import sys

RAW_DIR = os.path.join(os.path.dirname(__file__), "raw")


def download():
    os.makedirs(RAW_DIR, exist_ok=True)

    dev_path = os.path.join(RAW_DIR, "dev.jsonl")
    if os.path.exists(dev_path):
        with open(dev_path) as f:
            count = sum(1 for _ in f)
        print(f"dev.jsonl already exists ({count} examples), skipping.")
        return

    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' package required.  pip install datasets")
        sys.exit(1)

    print("Downloading BoolQ from HuggingFace …")
    ds = load_dataset("google/boolq")

    # Save validation split as dev.jsonl
    val = ds["validation"]
    with open(dev_path, "w", encoding="utf-8") as f:
        for row in val:
            f.write(json.dumps({
                "question": row["question"],
                "passage": row["passage"],
                "answer": row["answer"],
            }, ensure_ascii=False) + "\n")
    print(f"  Saved {len(val)} validation examples to {dev_path}")

    # Also save train split
    train_path = os.path.join(RAW_DIR, "train.jsonl")
    train = ds["train"]
    with open(train_path, "w", encoding="utf-8") as f:
        for row in train:
            f.write(json.dumps({
                "question": row["question"],
                "passage": row["passage"],
                "answer": row["answer"],
            }, ensure_ascii=False) + "\n")
    print(f"  Saved {len(train)} training examples to {train_path}")


if __name__ == "__main__":
    download()
