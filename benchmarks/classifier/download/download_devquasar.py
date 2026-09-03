# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

#!/usr/bin/env python3
"""DevQuasar LLM Router → datasets/devquasar_router.json  (label: simple|complex)

Source : DevQuasar/llm_router_dataset-synth (HuggingFace)  |  License: Apache-2.0
Task    : binary simple/complex routing prompts. Fields {prompt, label} where
          label "0"=simple, "1"=complex. The trainer further splits "complex"
          into moderate vs complex via keyword heuristics.
"""
from _common import MAX_PER_SOURCE, save_dataset


def main() -> None:
    print("DevQuasar ← DevQuasar/llm_router_dataset-synth (Apache-2.0)")
    from datasets import load_dataset

    ds = load_dataset("DevQuasar/llm_router_dataset-synth", split="train")
    label_map = {"0": "simple", "1": "complex", 0: "simple", 1: "complex"}
    samples: list[dict] = []
    for row in ds:
        prompt = (row.get("prompt") or "").strip()
        label = label_map.get(row.get("label")) or label_map.get(str(row.get("label")))
        if not prompt or label is None:
            continue
        samples.append({"text": prompt, "label": label})
        if len(samples) >= MAX_PER_SOURCE * 4:  # DevQuasar is the largest source
            break
    save_dataset("devquasar_router.json", samples)


if __name__ == "__main__":
    main()
