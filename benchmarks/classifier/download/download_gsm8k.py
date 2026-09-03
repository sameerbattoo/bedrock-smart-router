# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

#!/usr/bin/env python3
"""GSM8K → datasets/gsm8k.json (label: complex)
       → datasets/cross_difficulty_gsm8k.json (label: <float z-score>)

Source : openai/gsm8k, config "main" (HuggingFace)  |  License: MIT
Task    : grade-school multi-step math word problems.
  - gsm8k.json : every problem labeled "complex".
  - cross_difficulty_gsm8k.json : difficulty proxied by the number of
    reasoning steps in the reference solution (<<...>> calculator
    annotations, else solution line count), z-normalized.
"""
from _common import MAX_PER_SOURCE, save_dataset, zscore


def main() -> None:
    print("GSM8K ← openai/gsm8k (MIT)")
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="test")
    fixed: list[dict] = []
    cd_texts: list[str] = []
    cd_steps: list[float] = []
    for row in ds:
        q = (row.get("question") or "").strip()
        ans = row.get("answer") or ""
        if not q:
            continue
        text = f"Solve this problem step by step:\n\n{q}"
        if len(fixed) < 250:
            fixed.append({"text": text, "label": "complex"})
        steps = ans.count("<<") or (ans.count("\n") + 1)
        if len(cd_texts) < MAX_PER_SOURCE:
            cd_texts.append(text)
            cd_steps.append(float(steps))

    save_dataset("gsm8k.json", fixed)
    scores = zscore(cd_steps)
    save_dataset("cross_difficulty_gsm8k.json",
                 [{"text": t, "label": s} for t, s in zip(cd_texts, scores)])


if __name__ == "__main__":
    main()
