#!/usr/bin/env python3
"""
Download the TriviaQA dataset (reading comprehension).

Source: https://nlp.cs.washington.edu/triviaqa/
Paper:  Joshi et al., "TriviaQA: A Large Scale Distantly Supervised Challenge
        Dataset for Reading Comprehension", ACL 2017.

Uses the HuggingFace `rc.nocontext` subset validation split — questions with
answer aliases but no evidence documents (we provide the question only, similar
to how Bedrock uses TriviaQA for QA evaluation).

We save a manageable sample as JSONL.
"""
import json
import os
import sys

RAW_DIR = os.path.join(os.path.dirname(__file__), "raw")
MAX_EXAMPLES = 5000  # More than enough; we only use ~600 in transform


def download():
    os.makedirs(RAW_DIR, exist_ok=True)

    out_path = os.path.join(RAW_DIR, "triviaqa_val.jsonl")
    if os.path.exists(out_path):
        with open(out_path) as f:
            count = sum(1 for _ in f)
        print(f"triviaqa_val.jsonl already exists ({count} examples), skipping.")
        return

    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' package required.  pip install datasets")
        sys.exit(1)

    print("Downloading TriviaQA (rc, validation) from HuggingFace …")
    ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation",
                       trust_remote_code=True)

    written = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for row in ds:
            question = (row.get("question") or "").strip()
            answer_value = (row.get("answer", {}).get("value") or "").strip()
            aliases = row.get("answer", {}).get("aliases", [])
            normalized = row.get("answer", {}).get("normalized_aliases", [])

            if not question or not answer_value:
                continue

            f.write(json.dumps({
                "question": question,
                "answer": answer_value,
                "aliases": aliases,
                "normalized_aliases": normalized,
            }, ensure_ascii=False) + "\n")
            written += 1
            if written >= MAX_EXAMPLES:
                break

    print(f"  Saved {written} examples to {out_path}")


if __name__ == "__main__":
    download()
