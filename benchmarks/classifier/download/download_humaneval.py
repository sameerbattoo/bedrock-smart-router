# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

#!/usr/bin/env python3
"""HumanEval → datasets/humaneval.json  (label: complex)

Source : openai/openai_humaneval (HuggingFace)  |  License: MIT
Task    : Python function-completion coding problems — labeled "complex".
"""
from _common import save_dataset


def main() -> None:
    print("HumanEval ← openai/openai_humaneval (MIT)")
    from datasets import load_dataset

    ds = load_dataset("openai/openai_humaneval", split="test")
    samples: list[dict] = []
    for row in ds:
        prompt = (row.get("prompt") or "").strip()
        if not prompt:
            continue
        samples.append({"text": f"Complete this Python function:\n\n{prompt}", "label": "complex"})
    save_dataset("humaneval.json", samples)


if __name__ == "__main__":
    main()
