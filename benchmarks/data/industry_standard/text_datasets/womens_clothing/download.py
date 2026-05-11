#!/usr/bin/env python3
"""
Download the Women's E-Commerce Clothing Reviews dataset.

Source: https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews
License: CC0 Public Domain

Downloads via HuggingFace mirror and saves as JSONL.
"""
import json
import os
import sys

RAW_DIR = os.path.join(os.path.dirname(__file__), "raw")


def download():
    os.makedirs(RAW_DIR, exist_ok=True)

    out_path = os.path.join(RAW_DIR, "reviews.jsonl")
    if os.path.exists(out_path):
        with open(out_path) as f:
            count = sum(1 for _ in f)
        print(f"reviews.jsonl already exists ({count} examples), skipping.")
        return

    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' package required.  pip install datasets")
        sys.exit(1)

    print("Downloading Women's E-Commerce Clothing Reviews from HuggingFace …")
    ds = load_dataset("Censius-AI/ECommerce-Women-Clothing-Reviews", split="train")

    written = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for row in ds:
            review_text = (row.get("Review Text") or "").strip()
            rating = row.get("Rating")
            recommended = row.get("Recommended IND")
            department = row.get("Department Name") or ""
            class_name = row.get("Class Name") or ""
            title = (row.get("Title") or "").strip()

            # Skip rows without review text or rating
            if not review_text or rating is None:
                continue

            f.write(json.dumps({
                "review_text": review_text,
                "title": title,
                "rating": rating,
                "recommended": recommended,
                "department": department,
                "class_name": class_name,
            }, ensure_ascii=False) + "\n")
            written += 1

    print(f"  Saved {written} reviews to {out_path}")


if __name__ == "__main__":
    download()
