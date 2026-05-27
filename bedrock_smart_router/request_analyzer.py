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
import math
from typing import Any

from bedrock_smart_router.models import Complexity, RequestAnalysis

# ── Import heuristic constants, keyword sets, utilities, and dataclasses ──
from bedrock_smart_router.heuristic_classifier import (
    REASONING_MARKERS, CODE_MARKERS, CODE_LANG_KEYWORDS, SIMPLE_INDICATORS,
    MULTI_STEP_PATTERNS, TOOL_USE_SIGNALS, DOCUMENT_SIGNALS, MATH_SIGNALS,
    DATA_ANALYSIS_SIGNALS, CREATIVE_SIGNALS, AWS_SIGNALS, COMPLEX_QUESTION_PATTERNS,
    SIMPLE_QUESTION_PATTERNS, OUTPUT_FORMAT_SIGNALS, CONSTRAINT_SIGNALS,
    CONTEXT_REFERENCE_SIGNALS, SYSTEM_FLOOR_FACTOR, PAYLOAD_BOOST_5MB,
    PAYLOAD_BOOST_1MB, PAYLOAD_BOOST_100KB, PAYLOAD_BOOST_SMALL,
    PAYLOAD_THRESHOLD_5MB, PAYLOAD_THRESHOLD_1MB, PAYLOAD_THRESHOLD_100KB,
    _TABLE_PATTERN, _CSV_DATA, _PARAGRAPH_BREAK, _NUMBERED_LIST, _CODE_BLOCK,
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

        # ML classifier: only enabled when explicitly requested via classifier="ml"
        self._ml_classifier = None
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

        Complexity scoring is based on the **last user message** only,
        so that multi-turn conversations and verbose system prompts
        don't inflate the complexity of simple follow-up messages.
        Full context (all messages + system) is still used for
        capability detection and token estimation.
        """
        # For complexity scoring — only the last user message determines
        # how "hard" the current request is.
        last_user_text = _extract_last_user_text(messages)
        scoring_text_lower = last_user_text.lower()

        # For capability detection and token estimation — full context
        full_user_text = _extract_text(messages)
        system_text = _extract_text(system) if system else ""
        full_text = f"{system_text}\n{full_user_text}".strip()

        # ── Per-dimension scores (0.0 – 1.0) ───────────────────
        scores = self._score_dimensions(scoring_text_lower, messages, tool_config)

        # ── Weighted composite ──────────────────────────────────
        w = self.weights
        weight_list = [
            w.token_count, w.code_presence, w.reasoning_markers,
            w.technical_depth, w.simple_indicators, w.multi_step,
            w.tool_use, w.document_analysis, w.conversation_depth,
            w.aws_specificity, w.math_logical, w.creative_open,
            w.output_format, w.constraint_density, w.context_ratio,
        ]
        composite = sum(s * wt for s, wt in zip(scores, weight_list))

        # ── Multimodal payload complexity boost ─────────────────
        # Large images/documents are inherently complex tasks that
        # need capable models.  Boost the composite directly so the
        # payload size influences tier selection.
        payload_bytes = self._multimodal_payload_bytes(messages)
        if payload_bytes > 0:
            if payload_bytes > PAYLOAD_THRESHOLD_5MB:
                composite += PAYLOAD_BOOST_5MB
            elif payload_bytes > PAYLOAD_THRESHOLD_1MB:
                composite += PAYLOAD_BOOST_1MB
            elif payload_bytes > PAYLOAD_THRESHOLD_100KB:
                composite += PAYLOAD_BOOST_100KB
            else:
                composite += PAYLOAD_BOOST_SMALL

        composite = max(0.0, min(1.0, composite))

        # ── System prompt complexity floor ──────────────────────
        # The system prompt establishes a baseline task complexity.
        # A complex system prompt (e.g. "you are a senior architect,
        # analyze trade-offs") means even short user messages like
        # "analyse for X" require a capable model.
        # We derive a floor from the system prompt's keyword density
        # and ensure the composite never drops below it.
        if system_text:
            system_floor = self._compute_system_floor(system_text.lower())
            composite = max(composite, system_floor)

        # ── Classify complexity ─────────────────────────────────
        # Use last user message for reasoning marker count too
        reasoning_count = _count_matches(scoring_text_lower, REASONING_MARKERS)

        # Determine which classifier to use (per-request override or default)
        use_ml = self._ml_classifier is not None
        if classifier_override == "ml":
            use_ml = True
            # Lazily initialize ML classifier if not already available
            if self._ml_classifier is None:
                try:
                    from bedrock_smart_router.ml_classifier import MLComplexityClassifier
                    self._ml_classifier = MLComplexityClassifier()
                except ImportError:
                    use_ml = False
        elif classifier_override == "heuristic":
            use_ml = False

        # If ML classifier is available, use it for complexity detection
        if use_ml and self._ml_classifier is not None:
            try:
                ml_label, ml_conf = self._ml_classifier.classify_request(
                    messages, system=system, tool_config=tool_config,
                )
                label_map = {
                    "simple": Complexity.SIMPLE,
                    "moderate": Complexity.MODERATE,
                    "complex": Complexity.COMPLEX,
                    "reasoning": Complexity.REASONING,
                }
                complexity = label_map.get(ml_label, Complexity.MODERATE)
                # Use ML confidence as the composite score (scaled to 0-1)
                composite = ml_conf

                # Apply multimodal payload boost (same as heuristic)
                # Large images/documents need capable models regardless of text complexity
                if payload_bytes > 0:
                    complexity_order = {"simple": 0, "moderate": 1, "complex": 2, "reasoning": 3}
                    current_level = complexity_order.get(ml_label, 1)
                    if payload_bytes > PAYLOAD_THRESHOLD_5MB and current_level < 2:
                        complexity = Complexity.COMPLEX
                    elif payload_bytes > PAYLOAD_THRESHOLD_100KB and current_level < 1:
                        complexity = Complexity.MODERATE

            except Exception:
                # Fall back to heuristic on any ML error
                complexity = self._classify(composite, reasoning_count)
        else:
            complexity = self._classify(composite, reasoning_count)

        # ── Detect capabilities needed ──────────────────────────
        has_images = self._has_images(messages)
        has_documents = self._has_documents(messages)
        requires_tool = tool_config is not None or scores[6] > 0.3
        est_input = _estimate_tokens(full_text)

        # Add estimated tokens for multimodal content.
        # Bedrock converts images/documents to tokens internally:
        # ~TOKENS_PER_IMAGE tokens per image, ~TOKENS_PER_DOC_PAGE tokens per document page.
        payload_bytes = self._multimodal_payload_bytes(messages)
        if payload_bytes > 0:
            if has_documents:
                # Estimate pages: ~BYTES_PER_DOC_PAGE bytes per page
                est_pages = max(1, payload_bytes // BYTES_PER_DOC_PAGE)
                est_input += int(est_pages * TOKENS_PER_DOC_PAGE)
            elif has_images:
                # Estimate ~TOKENS_PER_IMAGE tokens per image (conservative)
                image_count = sum(
                    1 for msg in messages
                    for block in (msg.get("content", []) if isinstance(msg.get("content"), list) else [])
                    if isinstance(block, dict) and "image" in block
                )
                est_input += image_count * TOKENS_PER_IMAGE

        est_output = max(256, est_input // 3)

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
            is_code_task=scores[1] > 0.3,
            is_conversational=len(messages) > 2,
            is_multi_turn=len(messages) > 2,
            conversation_turn_count=len([m for m in messages if m.get("role") == "user"]),
            content_sensitivity="low",
        )

    # ── Internal helpers ────────────────────────────────────────

    def _score_dimensions(
        self,
        text_lower: str,
        messages: list[dict[str, Any]],
        tool_config: dict[str, Any] | None,
    ) -> list[float]:
        """Return a list of 15 dimension scores in [0, 1].

        Dimensions are calibrated to produce well-separated distributions
        across simple/medium/complex prompts.

        Note: ``text_lower`` is the lowercased **last user message** only,
        ensuring complexity scoring reflects the current request rather
        than accumulated conversation history.
        """
        text_len = len(text_lower)

        # Use last user message for structural pattern detection too
        text_original = _extract_last_user_text(messages)

        # 1. Text length — log-scaled, strongest separation signal
        if text_len <= 20:
            token_score = 0.0
        else:
            token_score = min(1.0, max(0.0,
                (math.log(text_len) - math.log(20)) / (math.log(3000) - math.log(20))
            ))

        # 2. Code presence — aggressive scaling
        code_hits = _count_matches(text_lower, CODE_MARKERS)
        lang_hits = _count_matches(text_lower, CODE_LANG_KEYWORDS)
        code_score = min(1.0, (code_hits + lang_hits) * 0.35)

        # 3. Reasoning markers
        reasoning_hits = _count_matches(text_lower, REASONING_MARKERS)
        reasoning_score = min(1.0, reasoning_hits * 0.35)

        # 4. Technical keyword density (hits per 200 chars)
        total_tech = code_hits + lang_hits + reasoning_hits
        if text_len > 0:
            density = total_tech / max(1, text_len / 200)
            tech_score = min(1.0, density * 0.5)
        else:
            tech_score = 0.0

        # 5. Simple indicators — properly inverted
        simple_hits = _count_matches(text_lower, SIMPLE_INDICATORS)
        if text_len < 100 and simple_hits >= 1:
            simple_score = 0.0
        elif simple_hits >= 2:
            simple_score = 0.05
        elif simple_hits == 1:
            simple_score = 0.2
        else:
            simple_score = 0.5

        # 6. Structural complexity (tables, CSV, code blocks, paragraphs)
        struct_signals = 0
        if _TABLE_PATTERN.search(text_original):
            struct_signals += 2
        if _CSV_DATA.search(text_original):
            struct_signals += 2
        num_paragraphs = len(_PARAGRAPH_BREAK.findall(text_original))
        if num_paragraphs >= 3:
            struct_signals += 1
        if num_paragraphs >= 6:
            struct_signals += 1
        numbered = len(_NUMBERED_LIST.findall(text_original))
        if numbered >= 3:
            struct_signals += 1
        if _CODE_BLOCK.search(text_original):
            struct_signals += 2
        multi_score = min(1.0, struct_signals * 0.2)

        # 7. Tool use signals
        tool_hits = _count_matches(text_lower, TOOL_USE_SIGNALS)
        tool_score = min(1.0, tool_hits * 0.4)
        # Note: tool_config presence is used for capability detection
        # (requires_tool_use) but does NOT inflate complexity score.
        # An agent with tools attached can still receive simple messages.

        # 8. Domain specificity (AWS + math + data analysis)
        aws_hits = _count_matches(text_lower, AWS_SIGNALS)
        math_hits = _count_matches(text_lower, MATH_SIGNALS)
        data_hits = _count_matches(text_lower, DATA_ANALYSIS_SIGNALS)
        doc_score = min(1.0, (aws_hits + math_hits + data_hits) * 0.25)

        # 9. Conversation depth
        turn_count = len(messages)
        conv_score = min(1.0, (turn_count - 1) / 6) if turn_count > 1 else 0.0

        # 10. Multi-step patterns
        multi_hits = _count_matches(text_lower, MULTI_STEP_PATTERNS)
        aws_score = min(1.0, multi_hits * 0.25)

        # 11. Question complexity
        complex_q_hits = _count_matches(text_lower, COMPLEX_QUESTION_PATTERNS)
        simple_q_hits = _count_matches(text_lower, SIMPLE_QUESTION_PATTERNS)
        if complex_q_hits > 0 and simple_q_hits == 0:
            math_score = min(1.0, complex_q_hits * 0.4)
        elif simple_q_hits > 0 and complex_q_hits == 0:
            math_score = 0.0
        else:
            math_score = min(1.0, max(0, complex_q_hits - simple_q_hits) * 0.3)

        # 12. Creative / open-ended
        creative_hits = _count_matches(text_lower, CREATIVE_SIGNALS)
        creative_score = min(1.0, creative_hits * 0.35)

        # 13. Output format constraints — structured output requests
        format_hits = _count_matches(text_lower, OUTPUT_FORMAT_SIGNALS)
        format_score = min(1.0, format_hits * 0.4)

        # 14. Constraint density — more constraints = harder task
        constraint_hits = _count_matches(text_lower, CONSTRAINT_SIGNALS)
        constraint_score = min(1.0, constraint_hits * 0.2)

        # 15. Context ratio — references to external context vs instruction
        #     High context references + short instruction = extraction task (medium)
        #     High context references + long instruction = complex analysis
        context_hits = _count_matches(text_lower, CONTEXT_REFERENCE_SIGNALS)
        if context_hits > 0:
            # Scale by both presence and text length
            context_score = min(1.0, context_hits * 0.2 + (0.2 if text_len > 500 else 0.0))
        else:
            context_score = 0.0

        return [
            token_score, code_score, reasoning_score, tech_score,
            simple_score, multi_score, tool_score, doc_score,
            conv_score, aws_score, math_score, creative_score,
            format_score, constraint_score, context_score,
        ]

    def _classify(self, score: float, reasoning_count: int) -> Complexity:
        t = self.thresholds
        if reasoning_count >= t.reasoning_marker_count or score >= t.complex_max:
            return Complexity.REASONING
        if score >= t.moderate_max:
            return Complexity.COMPLEX
        if score >= t.simple_max:
            return Complexity.MODERATE
        return Complexity.SIMPLE

    # ── System prompt floor scaling factor ──────────────────────
    # What fraction of the system prompt's raw complexity score
    # becomes the minimum floor for any user message.
    # Uses module-level SYSTEM_FLOOR_FACTOR constant.

    def _compute_system_floor(self, system_text_lower: str) -> float:
        """Derive a complexity floor from the system prompt.

        Scores the system prompt across keyword dimensions (reasoning,
        code, AWS, math, creative, constraints) and returns a fraction
        of that score as the minimum complexity for any user message.

        A simple system prompt ("You are a helpful assistant") → floor ~0.0
        A complex system prompt ("You are a senior architect, analyze
        trade-offs, design well-architected solutions") → floor ~0.10-0.15
        """
        # Count keyword hits in system prompt
        reasoning_hits = _count_matches(system_text_lower, REASONING_MARKERS)
        code_hits = _count_matches(system_text_lower, CODE_MARKERS)
        code_lang_hits = _count_matches(system_text_lower, CODE_LANG_KEYWORDS)
        aws_hits = _count_matches(system_text_lower, AWS_SIGNALS)
        math_hits = _count_matches(system_text_lower, MATH_SIGNALS)
        creative_hits = _count_matches(system_text_lower, CREATIVE_SIGNALS)
        constraint_hits = _count_matches(system_text_lower, CONSTRAINT_SIGNALS)
        complex_q_hits = _count_matches(system_text_lower, COMPLEX_QUESTION_PATTERNS)
        data_hits = _count_matches(system_text_lower, DATA_ANALYSIS_SIGNALS)

        # Compute a raw system complexity score (0-1)
        # Weight the most indicative dimensions
        raw = min(1.0, (
            reasoning_hits * 0.12
            + (code_hits + code_lang_hits) * 0.08
            + aws_hits * 0.06
            + math_hits * 0.10
            + creative_hits * 0.08
            + constraint_hits * 0.05
            + complex_q_hits * 0.10
            + data_hits * 0.08
        ))

        # Return a fraction as the floor
        return raw * SYSTEM_FLOOR_FACTOR

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

        # Get dimension scores (uses scoring_text_lower = last user message)
        scores = self._score_dimensions(scoring_text_lower, messages, None)
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
            system_floor = round(self._compute_system_floor(system_text_lower), 4)
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
