# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared utility functions used across multiple modules."""

from __future__ import annotations

from typing import Any


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors.

    Returns 0.0 for empty, mismatched, or zero-norm vectors.
    """
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token for English text."""
    return max(1, len(text) // 4)


def estimate_tokens_from_messages(
    messages: list[dict[str, Any]],
    system: list[dict[str, Any]] | None = None,
) -> int:
    """Estimate token count from Bedrock Converse-format messages."""
    total_chars = 0
    for msg in messages:
        content = msg.get("content", [])
        if isinstance(content, str):
            total_chars += len(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    if "text" in block:
                        total_chars += len(block["text"])
                    elif "image" in block:
                        total_chars += 4_000  # ~1K tokens per image
    if system:
        for block in system:
            if isinstance(block, dict) and "text" in block:
                total_chars += len(block["text"])
    return max(1, total_chars // 4)


def extract_text_from_messages(messages: list[dict[str, Any]]) -> str:
    """Extract all text content from Bedrock Converse-format messages."""
    parts: list[str] = []
    for msg in messages:
        content = msg.get("content", [])
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and "text" in block:
                    parts.append(block["text"])
    return "\n".join(parts)


def percentile(sorted_vals: list[float], pct: float) -> float:
    """Compute the *pct*-th percentile from a pre-sorted list.

    Uses nearest-rank interpolation.  Returns 0.0 for empty lists.
    """
    if not sorted_vals:
        return 0.0
    idx = int((len(sorted_vals) - 1) * pct / 100)
    return sorted_vals[min(idx, len(sorted_vals) - 1)]
