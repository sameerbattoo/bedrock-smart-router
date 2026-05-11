#!/usr/bin/env python3
"""
Download the XSum dataset (extreme summarization of BBC articles).

Used as an open-access alternative to Gigaword for the Text Summarization
benchmark. Same task type and metrics as AWS Bedrock's Gigaword usage.

Source: https://huggingface.co/datasets/EdinburghNLP/xsum
Paper:  Narayan et al., "Don't Give Me the Details, Just the Summary!
        Topic-Aware Convolutional Neural Networks for Extreme Summarization",
        EMNLP 2018.
"""
import json
import os
import sys

RAW_DIR = os.path.join(os.path.dirname(__file__), "raw")


def download():
    os.makedirs(RAW_DIR, exist_ok=True)

    out_path = os.path.join(RAW_DIR, "test.jsonl")
    if os.path.exists(out_path):
        with open(out_path) as f:
            count = sum(1 for _ in f)
        print(f"test.jsonl already exists ({count} examples), skipping.")
        return

    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' package required.  pip install datasets")
        sys.exit(1)

    print("Downloading XSum from HuggingFace …")
    ds = load_dataset("EdinburghNLP/xsum", split="test")

    written = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for row in ds:
            document = (row.get("document") or "").strip()
            summary = (row.get("summary") or "").strip()

            if not document or not summary:
                continue

            f.write(json.dumps({
                "document": document,
                "summary": summary,
            }, ensure_ascii=False) + "\n")
            written += 1

    print(f"  Saved {written} examples to {out_path}")


if __name__ == "__main__":
    download()
