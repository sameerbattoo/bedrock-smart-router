# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Request analysis orchestrator — delegates complexity classification and detects capabilities.

Orchestrates request analysis by combining two concerns:

1. **Complexity Classification** — Delegates to a pluggable classifier
   (heuristic or ML) to determine how "hard" the request is. The classifier
   scores the last user message and applies a system prompt floor.

2. **Capability Detection & Token Estimation** — Uses the full request
   context (all messages + system + tool_config) to detect:
   - Vision requirements (inline images)
   - Document support requirements (inline PDFs/documents)
   - Tool use requirements (tool_config or tool-related language)
   - Long context requirements (estimated tokens > 32K)
   - Multimodal payload boost (large images/documents bump complexity)

The ``RequestAnalyzer.analyze()`` method produces a ``RequestAnalysis``
object consumed by the router for model tier selection.

Classifier selection:
- ``classifier="heuristic"`` (default) — 15-dimension keyword scoring
- ``classifier="ml"`` — TF-IDF + Logistic Regression (requires numpy)
- Per-request override via ``classifier_override`` parameter
"""

from __future__ import annotations

import logging
from typing import Any

from bedrock_smart_router.models import Complexity, RequestAnalysis

# ── Import heuristic constants, keyword sets, utilities, and dataclasses ──
from bedrock_smart_router.heuristic_classifier import (
    HeuristicClassifier,
    REASONING_MARKERS, CODE_MARKERS, CODE_LANG_KEYWORDS, SIMPLE_INDICATORS,
    MULTI_STEP_PATTERNS, TOOL_USE_SIGNALS, MATH_SIGNALS,
    DATA_ANALYSIS_SIGNALS, CREATIVE_SIGNALS, AWS_SIGNALS, COMPLEX_QUESTION_PATTERNS,
    OUTPUT_FORMAT_SIGNALS, CONSTRAINT_SIGNALS,
    CONTEXT_REFERENCE_SIGNALS,
    PAYLOAD_THRESHOLD_5MB, PAYLOAD_THRESHOLD_1MB, PAYLOAD_THRESHOLD_100KB,
    _TABLE_PATTERN, _CSV_DATA, _PARAGRAPH_BREAK, _CODE_BLOCK,
    AnalyzerWeights, ComplexityThresholds, _count_matches, _kw_matches,
)

logger = logging.getLogger(__name__)

# ── Request-analysis-specific constants ─────────────────────────────

# Token estimation
CHARS_PER_TOKEN = 4

# Multimodal token estimates
TOKENS_PER_IMAGE = 750
TOKENS_PER_DOC_PAGE = 1500
BYTES_PER_DOC_PAGE = 3000

# Long context threshold (tokens)
LONG_CONTEXT_THRESHOLD = 32_000


# ── Token estimation ────────────────────────────────────────────────

from bedrock_smart_router.utils import estimate_tokens as _estimate_tokens


def _extract_text(messages: list[dict[str, Any]]) -> str:
    """Extract all text content from Bedrock Converse-format messages.

    Handles both message format (``[{"role": "user", "content": [{"text": "..."}]}]``)
    and system prompt format (``[{"text": "..."}]``).
    """
    parts: list[str] = []
    for msg in messages:
        # System prompt format: [{"text": "..."}]
        if "text" in msg and "content" not in msg:
            parts.append(msg["text"])
            continue
        # Message format: [{"role": "...", "content": [{"text": "..."}]}]
        content = msg.get("content", [])
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and "text" in block:
                    parts.append(block["text"])
    return "\n".join(parts)


def _extract_last_user_text(messages: list[dict[str, Any]]) -> str:
    """Extract text from only the last user message.

    In multi-turn conversations, complexity should be determined by
    the user's current request, not the accumulated conversation history
    or system prompt keywords.
    """
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", [])
            parts: list[str] = []
            if isinstance(content, str):
                return content
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and "text" in block:
                        parts.append(block["text"])
            return "\n".join(parts)
    return ""



# ── Main analyzer ───────────────────────────────────────────────────

class RequestAnalyzer:
    """Orchestrates request analysis: classification + capability detection.

    Delegates complexity scoring to a classifier (heuristic or ML),
    detects multimodal capabilities, estimates tokens, applies payload
    boost, and produces a ``RequestAnalysis`` for the router.

    No external API calls — runs in sub-millisecond time (heuristic)
    or single-digit milliseconds (ML).
    """

    def __init__(
        self,
        weights: AnalyzerWeights | None = None,
        thresholds: ComplexityThresholds | None = None,
        classifier: str = "heuristic",
    ) -> None:
        self.weights = weights or AnalyzerWeights()
        self.thresholds = thresholds or ComplexityThresholds()

        # Instantiate the heuristic classifier (always available for explain())
        self._heuristic_classifier = HeuristicClassifier(
            weights=self.weights, thresholds=self.thresholds,
        )

        # ML classifier: only enabled when explicitly requested via classifier="ml"
        self._ml_classifier = None
        self._default_classifier = classifier
        if classifier == "ml":
            try:
                from bedrock_smart_router.ml_classifier import MLComplexityClassifier
                self._ml_classifier = MLComplexityClassifier()
                logger.info("ML classifier enabled for complexity detection")
            except ImportError:
                raise ImportError(
                    "ML classifier requested but numpy is not installed. "
                    "Install with: pip install bedrock-smart-router[ml]"
                )

    def analyze(
        self,
        messages: list[dict[str, Any]],
        system: list[dict[str, Any]] | None = None,
        tool_config: dict[str, Any] | None = None,
        classifier_override: str | None = None,
    ) -> RequestAnalysis:
        """Analyze a request and return a ``RequestAnalysis``.

        Delegates complexity classification to the configured classifier
        (heuristic or ML). Full context (all messages + system) is used
        for capability detection and token estimation.
        """
        # ── Step 1: Classify complexity (delegate to classifier) ─
        use_ml = self._default_classifier == "ml"
        if classifier_override == "ml":
            use_ml = True
            if self._ml_classifier is None:
                try:
                    from bedrock_smart_router.ml_classifier import MLComplexityClassifier
                    self._ml_classifier = MLComplexityClassifier()
                except ImportError:
                    use_ml = False
        elif classifier_override == "heuristic":
            use_ml = False

        if use_ml and self._ml_classifier is not None:
            try:
                label, confidence = self._ml_classifier.classify_request(
                    messages, system=system, tool_config=tool_config,
                )
            except Exception:
                label, confidence = self._heuristic_classifier.classify_request(
                    messages, system=system, tool_config=tool_config,
                )
        else:
            label, confidence = self._heuristic_classifier.classify_request(
                messages, system=system, tool_config=tool_config,
            )

        # Map label to Complexity enum
        label_map = {
            "simple": Complexity.SIMPLE,
            "moderate": Complexity.MODERATE,
            "complex": Complexity.COMPLEX,
            "reasoning": Complexity.REASONING,
        }
        complexity = label_map.get(label, Complexity.MODERATE)
        composite = confidence

        # ── Step 2: Apply multimodal payload boost ──────────────
        payload_bytes = self._multimodal_payload_bytes(messages)
        if payload_bytes > 0:
            complexity_order = {"simple": 0, "moderate": 1, "complex": 2, "reasoning": 3}
            current_level = complexity_order.get(label, 1)
            if payload_bytes > PAYLOAD_THRESHOLD_5MB and current_level < 2:
                complexity = Complexity.COMPLEX
                composite = max(composite, 0.55)
            elif payload_bytes > PAYLOAD_THRESHOLD_1MB and current_level < 2:
                complexity = Complexity.MODERATE
                composite = max(composite, 0.35)
            elif payload_bytes > PAYLOAD_THRESHOLD_100KB and current_level < 1:
                complexity = Complexity.MODERATE
                composite = max(composite, 0.20)

        # ── Step 3: Detect capabilities needed ──────────────────
        full_user_text = _extract_text(messages)
        system_text = _extract_text(system) if system else ""
        full_text = f"{system_text}\n{full_user_text}".strip()
        last_user_text = _extract_last_user_text(messages)
        scoring_text_lower = last_user_text.lower()

        has_images = self._has_images(messages)
        has_documents = self._has_documents(messages)
        requires_tool = tool_config is not None or _count_matches(scoring_text_lower, TOOL_USE_SIGNALS) > 1

        # ── Step 2b: Tool presence boost ────────────────────────
        # If tools are explicitly provided via tool_config, ensure minimum
        # complexity of "moderate" — tool-calling requires a capable model.
        tool_boost_applied = False
        if tool_config and tool_config.get("tools"):
            complexity_order = {"simple": 0, "moderate": 1, "complex": 2, "reasoning": 3}
            current_level = complexity_order.get(label, 1)
            if current_level < 1:  # < moderate
                complexity = Complexity.MODERATE
                composite = max(composite, 0.20)
                tool_boost_applied = True

        est_input = _estimate_tokens(full_text)

        # Add estimated tokens for multimodal content
        if payload_bytes > 0:
            if has_documents:
                est_pages = max(1, payload_bytes // BYTES_PER_DOC_PAGE)
                est_input += int(est_pages * TOKENS_PER_DOC_PAGE)
            elif has_images:
                image_count = sum(
                    1 for msg in messages
                    for block in (msg.get("content", []) if isinstance(msg.get("content"), list) else [])
                    if isinstance(block, dict) and "image" in block
                )
                est_input += image_count * TOKENS_PER_IMAGE

        est_output = max(256, est_input // 3)

        # Detect code task from last user message
        code_hits = _count_matches(scoring_text_lower, CODE_MARKERS)
        lang_hits = _count_matches(scoring_text_lower, CODE_LANG_KEYWORDS)
        is_code = (code_hits + lang_hits) >= 2

        return RequestAnalysis(
            complexity=complexity,
            complexity_score=round(composite, 4),
            estimated_input_tokens=est_input,
            estimated_output_tokens=est_output,
            requires_vision=has_images,
            requires_document_support=has_documents,
            requires_tool_use=requires_tool,
            requires_long_context=est_input > LONG_CONTEXT_THRESHOLD,
            requires_extended_thinking=complexity == Complexity.REASONING,
            is_code_task=is_code,
            is_conversational=len(messages) > 2,
            is_multi_turn=len(messages) > 2,
            conversation_turn_count=len([m for m in messages if m.get("role") == "user"]),
            content_sensitivity="low",
            tool_boost_applied=tool_boost_applied,
        )

    # ── Internal helpers ────────────────────────────────────────

    @staticmethod
    def _has_images(messages: list[dict[str, Any]]) -> bool:
        for msg in messages:
            content = msg.get("content", [])
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and "image" in block:
                        return True
        return False

    @staticmethod
    def _has_documents(messages: list[dict[str, Any]]) -> bool:
        for msg in messages:
            content = msg.get("content", [])
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and "document" in block:
                        return True
        return False

    @staticmethod
    def _multimodal_payload_bytes(messages: list[dict[str, Any]]) -> int:
        """Sum the byte size of all inline image and document payloads."""
        total = 0
        for msg in messages:
            content = msg.get("content", [])
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict):
                    continue
                # Image bytes
                if "image" in block:
                    source = block["image"].get("source", {})
                    data = source.get("bytes")
                    if isinstance(data, (bytes, bytearray)):
                        total += len(data)
                # Document bytes
                if "document" in block:
                    source = block["document"].get("source", {})
                    data = source.get("bytes")
                    if isinstance(data, (bytes, bytearray)):
                        total += len(data)
        return total

    def explain(
        self,
        messages: list[dict[str, Any]],
        system: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Return detailed explanation of the complexity analysis.

        Includes which markers matched, dimension scores, and classification reasoning.
        Only called when RoutingConfig(explain=True).

        Scoring is based on the last user message only (consistent with
        ``analyze()``), but the explanation shows what was matched.
        """
        # Score based on last user message (same as analyze())
        last_user_text = _extract_last_user_text(messages)
        scoring_text_lower = last_user_text.lower()

        # Full text still used for token estimation reporting
        full_user_text = _extract_text(messages)
        system_text = _extract_text(system) if system else ""
        full_text = f"{system_text}\n{full_user_text}".strip()

        # Collect matched markers from last user message
        matched_markers: dict[str, list[str]] = {
            "reasoning": [kw for kw in REASONING_MARKERS if _kw_matches(kw, scoring_text_lower)],
            "code": [kw for kw in CODE_MARKERS if _kw_matches(kw, scoring_text_lower)],
            "code_languages": [kw for kw in CODE_LANG_KEYWORDS if _kw_matches(kw, scoring_text_lower)],
            "simple": [kw for kw in SIMPLE_INDICATORS if _kw_matches(kw, scoring_text_lower)],
            "multi_step": [kw for kw in MULTI_STEP_PATTERNS if _kw_matches(kw, scoring_text_lower)],
            "tool_use": [kw for kw in TOOL_USE_SIGNALS if _kw_matches(kw, scoring_text_lower)],
            "aws": [kw for kw in AWS_SIGNALS if _kw_matches(kw, scoring_text_lower)],
            "math": [kw for kw in MATH_SIGNALS if _kw_matches(kw, scoring_text_lower)],
            "creative": [kw for kw in CREATIVE_SIGNALS if _kw_matches(kw, scoring_text_lower)],
            "complex_questions": [kw for kw in COMPLEX_QUESTION_PATTERNS if _kw_matches(kw, scoring_text_lower)],
            "output_format": [kw for kw in OUTPUT_FORMAT_SIGNALS if _kw_matches(kw, scoring_text_lower)],
            "constraints": [kw for kw in CONSTRAINT_SIGNALS if _kw_matches(kw, scoring_text_lower)],
            "context_references": [kw for kw in CONTEXT_REFERENCE_SIGNALS if _kw_matches(kw, scoring_text_lower)],
            "data_analysis": [kw for kw in DATA_ANALYSIS_SIGNALS if _kw_matches(kw, scoring_text_lower)],
        }

        marker_counts = {k: len(v) for k, v in matched_markers.items()}
        # Add computed signals
        marker_counts["last_user_message_chars"] = len(last_user_text)
        marker_counts["full_context_chars"] = len(full_text)
        marker_counts["conversation_turns"] = len(messages)
        marker_counts["structural_signals"] = (
            (1 if _TABLE_PATTERN.search(last_user_text) else 0) +
            (1 if _CSV_DATA.search(last_user_text) else 0) +
            (1 if _CODE_BLOCK.search(last_user_text) else 0) +
            len(_PARAGRAPH_BREAK.findall(last_user_text))
        )

        # Get dimension scores from the heuristic classifier
        scores = self._heuristic_classifier._score_dimensions(scoring_text_lower, last_user_text)
        dimension_names = [
            "token_count", "code_presence", "reasoning_markers", "technical_depth",
            "simple_indicators", "structural_complexity", "tool_use", "domain_specificity",
            "conversation_depth", "multi_step", "question_complexity", "creative_open",
            "output_format", "constraint_density", "context_ratio",
        ]
        dimension_scores = {name: round(score, 4) for name, score in zip(dimension_names, scores)}

        # Compute user message composite score
        w = self.weights
        weight_list = [
            w.token_count, w.code_presence, w.reasoning_markers,
            w.technical_depth, w.simple_indicators, w.multi_step,
            w.tool_use, w.document_analysis, w.conversation_depth,
            w.aws_specificity, w.math_logical, w.creative_open,
            w.output_format, w.constraint_density, w.context_ratio,
        ]
        user_message_score = round(sum(s * wt for s, wt in zip(scores, weight_list)), 4)

        # System prompt floor analysis
        system_floor = 0.0
        system_floor_markers: dict[str, list[str]] = {}
        floor_applied = False
        if system_text:
            system_text_lower = system_text.lower()
            system_floor = round(self._heuristic_classifier._compute_system_floor(system_text_lower), 4)
            floor_applied = system_floor > user_message_score
            # Show which system prompt keywords contributed
            # Use _kw_matches for word-boundary-aware matching on short keywords
            system_floor_markers = {
                "reasoning": [kw for kw in REASONING_MARKERS if _kw_matches(kw, system_text_lower)],
                "code": [kw for kw in CODE_MARKERS if _kw_matches(kw, system_text_lower)],
                "aws": [kw for kw in AWS_SIGNALS if _kw_matches(kw, system_text_lower)],
                "math": [kw for kw in MATH_SIGNALS if _kw_matches(kw, system_text_lower)],
                "creative": [kw for kw in CREATIVE_SIGNALS if _kw_matches(kw, system_text_lower)],
                "constraints": [kw for kw in CONSTRAINT_SIGNALS if _kw_matches(kw, system_text_lower)],
                "complex_questions": [kw for kw in COMPLEX_QUESTION_PATTERNS if _kw_matches(kw, system_text_lower)],
                "data_analysis": [kw for kw in DATA_ANALYSIS_SIGNALS if _kw_matches(kw, system_text_lower)],
            }
            # Remove empty categories
            system_floor_markers = {k: v for k, v in system_floor_markers.items() if v}

        return {
            "matched_markers": matched_markers,
            "marker_counts": marker_counts,
            "dimension_scores": dimension_scores,
            "user_message_score": user_message_score,
            "system_prompt_floor": system_floor,
            "floor_applied": floor_applied,
            "final_score": round(max(user_message_score, system_floor), 4),
            "system_floor_markers": system_floor_markers,
            "text_length": len(last_user_text),
            "full_context_length": len(full_text),
            "estimated_tokens": _estimate_tokens(full_text),
            "multimodal_payload_bytes": self._multimodal_payload_bytes(messages),
        }
