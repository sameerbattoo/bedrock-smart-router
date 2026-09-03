# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

#!/usr/bin/env python3
"""Download every classifier training dataset into benchmarks/classifier/datasets/.

Runs each per-source download script in turn. A failure in one source (e.g. a
transient HuggingFace error) is reported but does not abort the others.

Usage:
    python benchmarks/classifier/download/download_all.py

Sources (one script each; all permissively licensed, none gated):
    devquasar, hellaswag, math, gsm8k, arc, mmlu, winogrande,
    commonsense_qa, humaneval, mbpp, bbh, ifeval
"""
import importlib

MODULES = [
    "download_devquasar",
    "download_hellaswag",
    "download_math",
    "download_gsm8k",
    "download_arc",
    "download_mmlu",
    "download_winogrande",
    "download_commonsense_qa",
    "download_humaneval",
    "download_mbpp",
    "download_bbh",
    "download_ifeval",
]


def main() -> None:
    print("=" * 60)
    print("Downloading all classifier datasets → classifier/datasets/")
    print("=" * 60)
    ok, failed = 0, []
    for name in MODULES:
        try:
            mod = importlib.import_module(name)
            mod.main()
            ok += 1
        except Exception as e:
            print(f"  ERROR in {name}: {type(e).__name__}: {e}")
            failed.append(name)
    print("\n" + "=" * 60)
    print(f"Done. {ok}/{len(MODULES)} sources downloaded.")
    if failed:
        print("Failed:", ", ".join(failed))


if __name__ == "__main__":
    main()
