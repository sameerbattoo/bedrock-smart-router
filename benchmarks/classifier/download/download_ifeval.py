# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

#!/usr/bin/env python3
"""IFEval → datasets/cross_difficulty_ifeval.json  (label: <float z-score>)

Source : google/IFEval (HuggingFace)  |  License: Apache-2.0
Task    : instruction-following with verifiable constraints. Difficulty
          proxied by the number of constraints imposed per prompt
          (len(instruction_id_list)), z-normalized.
"""
from _common import MAX_PER_SOURCE, save_dataset, zscore


def main() -> None:
    print("IFEval ← google/IFEval (Apache-2.0)")
    from datasets import load_dataset

    ds = load_dataset("google/IFEval", split="train")
    texts: list[str] = []
    raw: list[float] = []
    for row in ds:
        prompt = (row.get("prompt") or "").strip()
        if not prompt:
            continue
        texts.append(prompt)
        raw.append(float(len(row.get("instruction_id_list") or [])))
        if len(texts) >= MAX_PER_SOURCE:
            break

    scores = zscore(raw)
    save_dataset("cross_difficulty_ifeval.json",
                 [{"text": t, "label": s} for t, s in zip(texts, scores)])


if __name__ == "__main__":
    main()
