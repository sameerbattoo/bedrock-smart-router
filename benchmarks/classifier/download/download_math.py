# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

#!/usr/bin/env python3
"""MATH → datasets/competition_math.json (label: complex)
      → datasets/cross_difficulty_math.json (label: <float z-score>)

Source : EleutherAI/hendrycks_math (HuggingFace)  |  License: MIT
Task    : competition mathematics.
  - competition_math.json : every problem labeled "complex".
  - cross_difficulty_math.json : problems scored by the dataset's own
    difficulty level (1-5), z-normalized, for the trainer's cross-difficulty
    binning (moderate/complex/reasoning).
"""
import re

from _common import MAX_PER_SOURCE, save_dataset, zscore

SUBJECTS = [
    "algebra", "counting_and_probability", "geometry",
    "intermediate_algebra", "number_theory", "prealgebra", "precalculus",
]


def main() -> None:
    print("MATH ← EleutherAI/hendrycks_math (MIT)")
    from datasets import load_dataset

    fixed: list[dict] = []
    cd_texts: list[str] = []
    cd_levels: list[float] = []
    for subj in SUBJECTS:
        try:
            ds = load_dataset("EleutherAI/hendrycks_math", subj, split="test")
        except Exception as e:
            print(f"  (skip {subj}: {type(e).__name__})")
            continue
        for row in ds:
            problem = (row.get("problem") or "").strip()
            if not problem:
                continue
            text = f"Solve this math problem:\n\n{problem}"
            # Fixed-label contribution (cap at 300 to match corpus balance).
            if len(fixed) < 300:
                fixed.append({"text": text, "label": "complex"})
            # Cross-difficulty contribution (uses the built-in level 1-5).
            m = re.search(r"(\d+)", str(row.get("level", "")))
            if m and len(cd_texts) < MAX_PER_SOURCE:
                cd_texts.append(text)
                cd_levels.append(float(m.group(1)))
        if len(fixed) >= 300 and len(cd_texts) >= MAX_PER_SOURCE:
            break

    save_dataset("competition_math.json", fixed)
    scores = zscore(cd_levels)
    save_dataset("cross_difficulty_math.json",
                 [{"text": t, "label": s} for t, s in zip(cd_texts, scores)])


if __name__ == "__main__":
    main()
