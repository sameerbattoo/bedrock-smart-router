# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Complexity Classifier — abstract base class for prompt complexity classification.

The complexity classifier determines how "hard" a user's request is, which
drives model tier selection. The router uses this to pick the right model:
simple → cheap/fast models, reasoning → expensive/capable models.

Built-in classifiers:
- ``HeuristicClassifier`` — 15 weighted keyword dimensions, zero dependencies
- ``MLComplexityClassifier`` — TF-IDF + LogisticRegression, requires numpy

Custom classifiers: subclass ``ComplexityClassifier`` and implement
``classify()`` and ``predict_proba_all()``. Everything else (floor logic,
message extraction, multimodal boost) is inherited.

Usage
-----
::

    from bedrock_smart_router.complexity_classifier import ComplexityClassifier

    class MyClassifier(ComplexityClassifier):
        def classify(self, text: str) -> tuple[str, float]:
            # Your classification logic here
            return ("moderate", 0.85)

        def predict_proba_all(self, text: str) -> dict[str, float]:
            return {"simple": 0.05, "moderate": 0.85, "complex": 0.08, "reasoning": 0.02}

    router = BedrockRouter.create({"classifier": MyClassifier()})
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

# Valid complexity labels (fixed vocabulary — the router maps these to tiers)
COMPLEXITY_LABELS = ("simple", "moderate", "complex", "reasoning")

# Complexity level ordering for floor comparison
COMPLEXITY_ORDER: dict[str, int] = {"simple": 0, "moderate": 1, "complex": 2, "reasoning": 3}
LEVEL_TO_LABEL: dict[int, str] = {0: "simple", 1: "moderate", 2: "complex", 3: "reasoning"}

# Default floor parameters
DEFAULT_FLOOR_CONFIDENCE_THRESHOLD = 0.7
DEFAULT_FLOOR_DAMPENING = 0.8

# Maximum complexity level the system prompt floor is allowed to push to.
# The system prompt floor can upgrade simple→moderate or moderate→complex,
# but CANNOT push into reasoning. Reasoning-tier classification should only
# be triggered by the user message content itself, not by role-assignment
# text in system prompts. ML models give noisy predictions on short system
# prompts (e.g., "You are a security architect" → reasoning at 76% confidence),
# so we cap the floor's reach at complex.
FLOOR_MAX_LEVEL = 2  # complex


class ComplexityClassifier(ABC):
    """Abstract base class for complexity classifiers.

    Subclass this to implement a custom classifier. You only need to
    implement two methods:

    - ``classify(text)`` — classify a single text string
    - ``predict_proba_all(text)`` — return per-class probabilities

    The base class provides:

    - ``classify_request(messages, system, tool_config)`` — full request
      classification with system prompt floor, message extraction, and
      multimodal payload boost
    - Helper methods for extracting text from Bedrock message formats
    - Floor logic (system prompt establishes minimum complexity)

    Parameters
    ----------
    floor_confidence_threshold : float
        Minimum confidence the system prompt floor must have to be applied.
        Default 0.7 — prevents low-confidence floor from overriding user message.
    floor_dampening : float
        Confidence dampening when floor overrides user message classification.
        Default 0.8 — signals the result was floor-influenced.
    """

    def __init__(
        self,
        floor_confidence_threshold: float = DEFAULT_FLOOR_CONFIDENCE_THRESHOLD,
        floor_dampening: float = DEFAULT_FLOOR_DAMPENING,
    ) -> None:
        self._floor_confidence_threshold = floor_confidence_threshold
        self._floor_dampening = floor_dampening

    @property
    def floor_confidence_threshold(self) -> float:
        """Minimum confidence for floor to apply."""
        return self._floor_confidence_threshold

    @property
    def floor_dampening(self) -> float:
        """Confidence dampening factor when floor is applied."""
        return self._floor_dampening

    # ── Abstract methods (must be implemented by subclasses) ────

    @abstractmethod
    def classify(self, text: str) -> tuple[str, float]:
        """Classify a single text string.

        Parameters
        ----------
        text : str
            The text to classify (typically the last user message).

        Returns
        -------
        tuple[str, float]
            A tuple of (label, confidence) where:
            - label is one of: "simple", "moderate", "complex", "reasoning"
            - confidence is a float between 0.0 and 1.0
        """
        ...

    @abstractmethod
    def predict_proba_all(self, text: str) -> dict[str, float]:
        """Return probability distribution across all complexity classes.

        Parameters
        ----------
        text : str
            The text to classify.

        Returns
        -------
        dict[str, float]
            Dictionary mapping each label to its probability.
            Values should sum to ~1.0.
            Example: {"simple": 0.7, "moderate": 0.2, "complex": 0.08, "reasoning": 0.02}
        """
        ...

    # ── Concrete methods (shared logic, inherited by all) ───────

    def classify_request(
        self,
        messages: list[dict[str, Any]],
        system: list[dict[str, Any]] | None = None,
        tool_config: dict[str, Any] | None = None,
    ) -> tuple[str, float]:
        """Classify a full Bedrock Converse request.

        This is the main entry point used by the router. It implements
        the shared classification pipeline:

        1. Extract the LAST USER MESSAGE (primary signal)
        2. Classify it using the subclass's ``classify()`` method
        3. Apply post-classify guard (hook for subclass-specific logic)
        4. Apply system prompt floor (only upgrades, never downgrades)
        5. Return the final (label, confidence)

        The system prompt + tools establish a complexity FLOOR — they can
        upgrade the classification but never downgrade it. This prevents
        complex system prompts from inflating simple follow-up questions.

        Parameters
        ----------
        messages : list[dict]
            Bedrock Converse messages (role + content blocks).
        system : list[dict], optional
            System prompt blocks.
        tool_config : dict, optional
            Tool configuration with tool specs.

        Returns
        -------
        tuple[str, float]
            Final (label, confidence) after floor application.
        """
        # 1. Extract and classify the LAST USER MESSAGE (primary signal)
        last_user_text = self.extract_last_user_text(messages)
        if last_user_text:
            user_label, user_conf = self.classify(last_user_text)
        else:
            # No text in user messages (e.g., only toolResult blocks)
            # Fall back to classifying the full context
            full_text = self.assemble_full_context(messages, system, tool_config)
            return self.classify(full_text)

        # 2. Post-classify guard (subclasses can override for custom logic)
        user_label, user_conf = self._post_classify_guard(user_label, user_conf)

        # 3. Apply system prompt floor
        return self._apply_floor(user_label, user_conf, system, tool_config)

    def _post_classify_guard(
        self, label: str, confidence: float,
    ) -> tuple[str, float]:
        """Hook for subclass-specific post-classification guards.

        Called after ``classify()`` but before ``_apply_floor()``.
        Override in subclasses to add custom logic (e.g., low-confidence
        fallback). Default implementation is a no-op pass-through.

        Parameters
        ----------
        label : str
            Classification label from ``classify()``.
        confidence : float
            Classification confidence from ``classify()``.

        Returns
        -------
        tuple[str, float]
            Possibly adjusted (label, confidence).
        """
        return label, confidence

    def classify_batch(self, texts: list[str]) -> list[tuple[str, float]]:
        """Classify a batch of text prompts.

        Default implementation iterates over ``classify()``. Subclasses
        can override for optimized batch inference.

        Parameters
        ----------
        texts : list[str]
            List of prompt texts to classify.

        Returns
        -------
        list[tuple[str, float]]
            List of (label, confidence) tuples for each input text.
        """
        return [self.classify(text) for text in texts]

    def _apply_floor(
        self,
        user_label: str,
        user_conf: float,
        system: list[dict[str, Any]] | None = None,
        tool_config: dict[str, Any] | None = None,
    ) -> tuple[str, float]:
        """Apply system prompt complexity floor.

        The floor only UPGRADES classification, never downgrades.
        It's capped at +1 level above the user's classification to
        prevent a "reasoning" system prompt from pushing a "simple"
        question all the way to "reasoning".

        Parameters
        ----------
        user_label : str
            The user message's classification label.
        user_conf : float
            The user message's classification confidence.
        system : list[dict], optional
            System prompt blocks.
        tool_config : dict, optional
            Tool configuration.

        Returns
        -------
        tuple[str, float]
            Final (label, confidence) — possibly upgraded by floor.
        """
        if not system and not tool_config:
            return user_label, user_conf

        floor_text = self.assemble_floor_context(system, tool_config)
        if not floor_text:
            return user_label, user_conf

        floor_label, floor_conf = self.classify(floor_text)

        # Apply floor: only upgrade, never downgrade
        # Only apply if floor is moderate+ (simple system prompts don't boost)
        user_level = COMPLEXITY_ORDER.get(user_label, 0)
        floor_level = COMPLEXITY_ORDER.get(floor_label, 0)

        if (
            floor_level > user_level
            and floor_level >= 1
            and floor_conf > self._floor_confidence_threshold
        ):
            # Floor is higher with high confidence — apply it but cap at +1 level.
            # Also cap at FLOOR_MAX_LEVEL: system prompt floor cannot push into
            # reasoning tier (see FLOOR_MAX_LEVEL comment for rationale).
            if user_level >= FLOOR_MAX_LEVEL:
                return user_label, user_conf
            capped_level = min(floor_level, user_level + 1, FLOOR_MAX_LEVEL)
            return LEVEL_TO_LABEL[capped_level], user_conf * self._floor_dampening

        return user_label, user_conf

    # ── Static helper methods (shared utilities) ────────────────

    @staticmethod
    def extract_last_user_text(messages: list[dict[str, Any]]) -> str:
        """Extract text from the last user message that has actual text content.

        Skips user messages that only contain toolResult blocks (from Strands
        agent loop tool calls) — these are not real user questions.

        Parameters
        ----------
        messages : list[dict]
            Bedrock Converse messages.

        Returns
        -------
        str
            The text content of the last real user message, or empty string.
        """
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", [])
                if isinstance(content, list):
                    parts = [
                        b["text"]
                        for b in content
                        if isinstance(b, dict) and "text" in b
                    ]
                    if parts:  # Only return if there's actual text
                        return " ".join(parts)
                elif isinstance(content, str) and content.strip():
                    return content
        return ""

    @staticmethod
    def assemble_floor_context(
        system: list[dict[str, Any]] | None = None,
        tool_config: dict[str, Any] | None = None,
    ) -> str:
        """Assemble system prompt + tool specs for floor calculation.

        Parameters
        ----------
        system : list[dict], optional
            System prompt blocks.
        tool_config : dict, optional
            Tool configuration.

        Returns
        -------
        str
            Combined text for floor classification.
        """
        parts: list[str] = []
        if system:
            for block in system:
                if isinstance(block, dict) and "text" in block:
                    parts.append(block["text"])
        if tool_config:
            tools = tool_config.get("tools", [])
            if tools:
                tool_names = [
                    t.get("toolSpec", {}).get("name", "")
                    for t in tools
                    if t.get("toolSpec", {}).get("name")
                ]
                if tool_names:
                    parts.append(f"[Tools: {', '.join(tool_names)}]")
        return "\n".join(parts)

    @staticmethod
    def assemble_full_context(
        messages: list[dict[str, Any]],
        system: list[dict[str, Any]] | None = None,
        tool_config: dict[str, Any] | None = None,
    ) -> str:
        """Assemble full context text from system + messages + tools.

        Used as fallback when no user text is available.

        Parameters
        ----------
        messages : list[dict]
            Bedrock Converse messages.
        system : list[dict], optional
            System prompt blocks.
        tool_config : dict, optional
            Tool configuration.

        Returns
        -------
        str
            Combined context string.
        """
        context_parts: list[str] = []

        # System prompt
        if system:
            for block in system:
                if isinstance(block, dict) and "text" in block:
                    context_parts.append(block["text"])

        # Tool specs
        if tool_config:
            tools = tool_config.get("tools", [])
            if tools:
                tool_names = []
                for t in tools:
                    spec = t.get("toolSpec", {})
                    name = spec.get("name", "")
                    desc = spec.get("description", "")
                    if name:
                        tool_names.append(f"{name}: {desc[:50]}" if desc else name)
                if tool_names:
                    context_parts.append(f"[Tools available: {', '.join(tool_names)}]")

        # Conversation messages
        for msg in messages:
            content = msg.get("content", [])
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and "text" in block:
                        context_parts.append(block["text"])

        return "\n\n".join(context_parts)
