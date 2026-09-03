# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for the classifier dataset download scripts.

Each ``download_<source>.py`` script fetches one upstream dataset from the
HuggingFace Hub and writes a trainer-compatible JSON file into
``benchmarks/classifier/datasets/``.

Two output schemas are used by ``train_tfidf.py``:
  - Labeled files      : [{"text": str, "label": "simple"|"moderate"|"complex"}]
  - cross_difficulty_* : [{"text": str, "label": <float z-score>}]
"""
from __future__ import annotations

import json
import os

# benchmarks/classifier/datasets/
DATASETS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets"
)

# Per-source cap to keep the corpus balanced and downloads quick.
MAX_PER_SOURCE = 4000


def save_dataset(name: str, samples: list[dict]) -> None:
    """Write samples to datasets/<name>.json."""
    os.makedirs(DATASETS_DIR, exist_ok=True)
    path = os.path.join(DATASETS_DIR, name)
    with open(path, "w") as f:
        json.dump(samples, f, indent=2)
    print(f"  Saved {len(samples)} samples → datasets/{name}")


def zscore(values: list[float]) -> list[float]:
    """Z-normalize raw difficulty values (population std). Empty/constant → 0s."""
    n = len(values)
    if n == 0:
        return []
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    std = var ** 0.5
    if std == 0:
        return [0.0] * n
    return [(v - mean) / std for v in values]
