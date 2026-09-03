# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

#!/usr/bin/env python3
"""Big-Bench-Hard → datasets/cross_difficulty_bbh.json  (label: <float z-score>)

Source : lukaemon/bbh (HuggingFace)  |  License: MIT
Task    : hard reasoning tasks. Difficulty proxied by a per-task ordinal
          (multi-step reasoning tasks rank higher) plus input length,
          z-normalized for the trainer's cross-difficulty binning.
"""
from _common import MAX_PER_SOURCE, save_dataset, zscore

# Multi-step reasoning tasks → higher intrinsic difficulty.
HARD_TASKS = {
    "logical_deduction_seven_objects", "logical_deduction_five_objects",
    "multistep_arithmetic_two", "word_sorting", "dyck_languages",
    "tracking_shuffled_objects_seven_objects", "geometric_shapes",
    "temporal_sequences", "boolean_expressions",
}


def main() -> None:
    print("Big-Bench-Hard ← lukaemon/bbh (MIT)")
    from datasets import load_dataset, get_dataset_config_names

    try:
        configs = get_dataset_config_names("lukaemon/bbh")
    except Exception as e:
        print(f"  (could not list BBH configs: {type(e).__name__}); using a default subset")
        configs = ["boolean_expressions", "multistep_arithmetic_two",
                   "logical_deduction_five_objects", "word_sorting"]

    texts: list[str] = []
    raw: list[float] = []
    for cfg in configs:
        try:
            ds = load_dataset("lukaemon/bbh", cfg, split="test")
        except Exception:
            continue
        base = 2.0 if cfg in HARD_TASKS else 1.0
        for row in ds:
            q = (row.get("input") or "").strip()
            if not q:
                continue
            texts.append(f"Solve this reasoning task:\n\n{q}")
            raw.append(base + min(len(q) / 400.0, 3.0))
            if len(texts) >= MAX_PER_SOURCE:
                break
        if len(texts) >= MAX_PER_SOURCE:
            break

    scores = zscore(raw)
    save_dataset("cross_difficulty_bbh.json",
                 [{"text": t, "label": s} for t, s in zip(texts, scores)])


if __name__ == "__main__":
    main()
