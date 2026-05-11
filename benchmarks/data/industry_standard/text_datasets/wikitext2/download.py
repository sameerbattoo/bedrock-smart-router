#!/usr/bin/env python3
"""
Download the WikiText-2 dataset (language modeling / text generation).

Source: https://huggingface.co/datasets/Salesforce/wikitext
Paper:  Merity et al., "Pointer Sentinel Mixture Models", 2016.
License: Creative Commons Attribution-ShareAlike (CC BY-SA 4.0)

WikiText-2 contains ~2M tokens from verified Good and Featured Wikipedia
articles. Used by Bedrock for Robustness evaluation under General text
generation.
"""
import json
import os
import sys

RAW_DIR = os.path.join(os.path.dirname(__file__), "raw")
MIN_TEXT_LEN = 80  # Skip short lines (headers, blank lines)


def download():
    os.makedirs(RAW_DIR, exist_ok=True)

    out_path = os.path.join(RAW_DIR, "wikitext2.jsonl")
    if os.path.exists(out_path):
        with open(out_path) as f:
            count = sum(1 for _ in f)
        print(f"wikitext2.jsonl already exists ({count} passages), skipping.")
        return

    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' package required.  pip install datasets")
        sys.exit(1)

    print("Downloading WikiText-2 (raw, test+validation) from HuggingFace …")
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")

    written = 0
    with open(out_path, "w", encoding="utf-8") as f:
        # Combine validation and test splits for more data
        for split_name in ["validation", "test"]:
            split = ds[split_name]
            for row in split:
                text = (row.get("text") or "").strip()
                # Skip headers (lines starting with ' = '), blank lines, short lines
                if not text or len(text) < MIN_TEXT_LEN:
                    continue
                if text.startswith("=") and text.endswith("="):
                    continue

                f.write(json.dumps({
                    "text": text,
                }, ensure_ascii=False) + "\n")
                written += 1

    print(f"  Saved {written} passages to {out_path}")


if __name__ == "__main__":
    download()
