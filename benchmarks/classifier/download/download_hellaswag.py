# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

#!/usr/bin/env python3
"""HellaSwag → datasets/hellaswag.json  (label: moderate)

Source : Rowan/hellaswag (HuggingFace)  |  License: MIT
Task    : commonsense sentence completion — labeled "moderate" complexity.
"""
from _common import MAX_PER_SOURCE, save_dataset


def main() -> None:
    print("HellaSwag ← Rowan/hellaswag (MIT)")
    from datasets import load_dataset

    ds = load_dataset("Rowan/hellaswag", split="validation")
    samples: list[dict] = []
    for row in ds:
        ctx = (row.get("ctx") or "").strip()
        if not ctx:
            continue
        text = f"Complete this sentence with the most logical continuation:\n\n{ctx}"
        samples.append({"text": text, "label": "moderate"})
        if len(samples) >= 350:  # match original corpus contribution
            break
    save_dataset("hellaswag.json", samples)


if __name__ == "__main__":
    main()
