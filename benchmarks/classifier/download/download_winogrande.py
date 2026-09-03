# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

#!/usr/bin/env python3
"""WinoGrande → datasets/winogrande.json  (label: moderate)

Source : allenai/winogrande, config winogrande_xl (HuggingFace)  |  License: Apache-2.0
Task    : pronoun-resolution commonsense — labeled "moderate".
"""
from _common import save_dataset


def main() -> None:
    print("WinoGrande ← allenai/winogrande (Apache-2.0)")
    from datasets import load_dataset

    ds = load_dataset("allenai/winogrande", "winogrande_xl", split="validation")
    samples: list[dict] = []
    for row in ds:
        sentence = (row.get("sentence") or "").strip()
        o1 = row.get("option1", ""); o2 = row.get("option2", "")
        if not sentence:
            continue
        text = (f"Resolve the pronoun in this sentence:\n\n{sentence}\n"
                f"Option 1: {o1}\nOption 2: {o2}")
        samples.append({"text": text, "label": "moderate"})
        if len(samples) >= 150:
            break
    save_dataset("winogrande.json", samples)


if __name__ == "__main__":
    main()
