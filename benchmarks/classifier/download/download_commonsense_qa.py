# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

#!/usr/bin/env python3
"""CommonsenseQA → datasets/commonsense_qa.json  (label: moderate)

Source : tau/commonsense_qa (HuggingFace)  |  License: MIT
Task    : commonsense multiple-choice QA — labeled "moderate".
"""
from _common import save_dataset


def main() -> None:
    print("CommonsenseQA ← tau/commonsense_qa (MIT)")
    from datasets import load_dataset

    ds = load_dataset("tau/commonsense_qa", split="validation")
    samples: list[dict] = []
    for row in ds:
        q = (row.get("question") or "").strip()
        if not q:
            continue
        samples.append({"text": f"Answer this commonsense question:\n\n{q}", "label": "moderate"})
        if len(samples) >= 150:
            break
    save_dataset("commonsense_qa.json", samples)


if __name__ == "__main__":
    main()
