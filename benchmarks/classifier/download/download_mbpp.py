# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

#!/usr/bin/env python3
"""MBPP → datasets/mbpp.json  (label: complex)

Source : google-research-datasets/mbpp, config "full" (HuggingFace)  |  License: CC-BY-4.0
Task    : "Mostly Basic Python Problems" coding tasks — labeled "complex".
"""
from _common import save_dataset


def main() -> None:
    print("MBPP ← google-research-datasets/mbpp (CC-BY-4.0)")
    from datasets import load_dataset

    ds = load_dataset("google-research-datasets/mbpp", "full", split="test")
    samples: list[dict] = []
    for row in ds:
        prompt = (row.get("text") or row.get("prompt") or "").strip()
        if not prompt:
            continue
        samples.append({"text": f"Write a Python function:\n\n{prompt}", "label": "complex"})
    save_dataset("mbpp.json", samples)


if __name__ == "__main__":
    main()
