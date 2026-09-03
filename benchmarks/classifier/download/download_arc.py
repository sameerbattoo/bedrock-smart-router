# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

#!/usr/bin/env python3
"""ARC-Challenge → datasets/arc_challenge.json  (label: complex)

Source : allenai/ai2_arc, config ARC-Challenge (HuggingFace)  |  License: CC-BY-SA-4.0
Task    : grade-school science questions requiring reasoning — labeled "complex".
"""
from _common import save_dataset


def main() -> None:
    print("ARC-Challenge ← allenai/ai2_arc (CC-BY-SA-4.0)")
    from datasets import load_dataset

    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    samples: list[dict] = []
    for row in ds:
        q = (row.get("question") or "").strip()
        if not q:
            continue
        samples.append({"text": f"Answer this science question:\n\n{q}", "label": "complex"})
        if len(samples) >= 200:
            break
    save_dataset("arc_challenge.json", samples)


if __name__ == "__main__":
    main()
