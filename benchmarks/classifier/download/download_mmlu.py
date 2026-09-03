# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

#!/usr/bin/env python3
"""MMLU (easy subjects) → datasets/mmlu_easy.json  (label: simple)

Source : cais/mmlu (HuggingFace)  |  License: MIT
Task    : multiple-choice knowledge questions from easier subjects — labeled "simple".
"""
from _common import save_dataset

# Easier, general-knowledge subjects → "simple" complexity.
EASY_SUBJECTS = [
    "elementary_mathematics", "high_school_geography",
    "high_school_government_and_politics", "miscellaneous",
    "marketing", "nutrition",
]


def main() -> None:
    print("MMLU (easy subjects) ← cais/mmlu (MIT)")
    from datasets import load_dataset

    samples: list[dict] = []
    for subj in EASY_SUBJECTS:
        try:
            ds = load_dataset("cais/mmlu", subj, split="test")
        except Exception as e:
            print(f"  (skip {subj}: {type(e).__name__})")
            continue
        for row in ds:
            q = (row.get("question") or "").strip()
            if not q:
                continue
            samples.append({"text": f"Answer this question:\n\n{q}", "label": "simple"})
            if len(samples) >= 200:
                break
        if len(samples) >= 200:
            break
    save_dataset("mmlu_easy.json", samples)


if __name__ == "__main__":
    main()
