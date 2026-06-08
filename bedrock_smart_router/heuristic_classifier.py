"""Heuristic Complexity Classifier — 15-dimension keyword-based scoring.

Zero-dependency, sub-millisecond complexity classification using weighted
keyword matching across 15 scoring dimensions. No API calls, no ML models.

Scoring Strategy
----------------
Complexity is determined by two signals combined via ``max()``:

1. **User Message Score** — The last user message is scored across 15
   keyword/pattern dimensions.  Only the last user message is used,
   NOT the full conversation history or system prompt.  This prevents
   multi-turn conversations and verbose system prompts from inflating
   the complexity of simple follow-up messages like "Hi" or "Thanks".

2. **System Prompt Floor** — The system prompt establishes a baseline
   task complexity.  A complex system prompt (e.g. "You are a senior
   architect, analyze trade-offs, design well-architected solutions")
   means even short user messages require a capable model.  The floor
   is computed as ``system_prompt_keyword_score × SYSTEM_FLOOR_FACTOR
   (0.30)``.

The final score is ``max(user_message_score, system_prompt_floor)``.

This design ensures:
- "Hi" with a complex system prompt → MODERATE (floor applies)
- "Hi" with no system prompt → SIMPLE (no floor)
- "Design a DR architecture" → COMPLEX (user message score dominates)
- Short follow-ups in multi-turn don't inherit prior turn complexity

Scoring Dimensions (15)
------------------------
1.  Token count (text length, log-scaled)
2.  Code presence (code markers + language keywords)
3.  Reasoning markers (analytical/logical keywords)
4.  Technical depth (keyword density per 200 chars)
5.  Simple indicators (greetings, basic questions — inverted)
6.  Structural complexity (tables, CSV, code blocks, paragraphs)
7.  Tool use signals
8.  Domain specificity (AWS + math + data analysis)
9.  Conversation depth (turn count)
10. Multi-step patterns (sequential instructions)
11. Question complexity (complex vs simple question patterns)
12. Creative/open-ended signals
13. Output format constraints (JSON, YAML, structured output)
14. Constraint density (must/should/exactly/at least)
15. Context references (references to external context)
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

from bedrock_smart_router.complexity_classifier import (
    ComplexityClassifier,
    COMPLEXITY_ORDER,
    LEVEL_TO_LABEL,
    DEFAULT_FLOOR_CONFIDENCE_THRESHOLD,
    DEFAULT_FLOOR_DAMPENING,
)

# ── Extracted numeric constants ─────────────────────────────────────

# System prompt floor: fraction of system prompt complexity used as minimum
SYSTEM_FLOOR_FACTOR = 0.30

# Multimodal payload complexity boosts added to composite score
PAYLOAD_BOOST_5MB = 0.30
PAYLOAD_BOOST_1MB = 0.20
PAYLOAD_BOOST_100KB = 0.10
PAYLOAD_BOOST_SMALL = 0.05

# Payload size thresholds (bytes)
PAYLOAD_THRESHOLD_5MB = 5_000_000
PAYLOAD_THRESHOLD_1MB = 1_000_000
PAYLOAD_THRESHOLD_100KB = 100_000


# ── Keyword / pattern sets ──────────────────────────────────────────

REASONING_MARKERS = {
    "step by step", "step-by-step", "analyze", "analyse", "analysis",
    "evaluate", "compare and contrast",
    "prove", "derive", "reason through", "think through", "work through",
    "explain why", "explain how", "trade-off", "tradeoff", "pros and cons",
    "critically", "systematically", "deduce", "infer", "hypothesize",
    "build a", "design a", "architect", "implement a", "construct",
    "optimize", "refactor", "for each", "for every",
    "calculate the", "compute the", "determine the",
    "showing", "demonstrating", "comprehensive",
}

CODE_MARKERS = {
    "```", "def ", "class ", "function ", "import ", "const ", "let ", "var ",
    "return ", "if __name__", "async def", "lambda ", "=>", "public static",
    "private ", "protected ", "#include", "package ", "func ", "fn ",
    "write a function", "write a program", "implement a", "code that",
    "write code", "write a script", "write a class", "write a method",
}

CODE_LANG_KEYWORDS = {
    "python", "javascript", "typescript", "java", "rust", "golang", "go ",
    "c++", "c#", "ruby", "swift", "kotlin", "scala", "sql", "html", "css",
    "react", "angular", "vue", "django", "flask", "fastapi", "spring",
    "terraform", "dockerfile", "yaml", "json schema",
}

SIMPLE_INDICATORS = {
    "hello", "hi ", "hey ", "thanks", "thank you", "yes", "no", "ok",
    "what is", "what's", "define ", "who is", "when was", "where is",
    "how old", "how many", "how much", "translate",
}

MULTI_STEP_PATTERNS = {
    "first,", "first ", "then,", "then ", "next,", "next ", "finally,",
    "step 1", "step 2", "1.", "2.", "3.", "after that", "followed by",
    "once you", "before you", "make sure to",
}

TOOL_USE_SIGNALS = {
    "function call", "tool_use", "tool use", "json schema", "structured output",
    "json output", "return json", "api call", "execute", "run the",
    "call the function", "invoke",
}

DOCUMENT_SIGNALS = {
    "document", "pdf", "attached", "file", "spreadsheet", "csv",
    "the following text", "the above", "this article", "this paper",
    "summarize the", "extract from", "based on the",
}

MATH_SIGNALS = {
    "equation", "formula", "calculate", "compute", "integral", "derivative",
    "probability", "optimize", "minimize", "maximize", "proof", "theorem",
    "algorithm", "complexity", "big-o", "matrix", "vector", "linear algebra",
}

DATA_ANALYSIS_SIGNALS = {
    "cohort", "retention", "funnel", "segmentation", "rfm",
    "churn", "lifetime value", "clv", "ltv",
    "window function", "partition by", "over (", "over(",
    "ntile", "percentile", "lag(", "lead(", "row_number",
    "dense_rank", "rank()", "cte",
    "regr_slope", "stddev", "variance", "correlation",
    "pivot", "unpivot", "rollup", "cube", "grouping sets",
    "generate_series", "date_trunc", "interval",
    "subquery", "nested query", "self join", "cross join",
    "full outer", "lateral join",
    "month-over-month", "year-over-year", "yoy", "mom",
    "forecast", "trend", "anomaly", "outlier",
    "waterfall", "basket analysis", "market basket",
    "running total", "moving average", "cumulative",
    "top 5", "top 10", "top n", "bottom 5", "bottom 10",
    "group by", "having", "case when",
}

CREATIVE_SIGNALS = {
    "write a story", "write a poem", "imagine", "creative", "brainstorm",
    "come up with", "invent", "fiction", "narrative", "compose",
    "design a", "create a", "generate ideas",
}

AWS_SIGNALS = {
    "aws", "amazon web services", "s3", "ec2", "lambda", "dynamodb",
    "cloudformation", "cdk", "iam", "vpc", "ecs", "eks", "sagemaker",
    "bedrock", "cloudwatch", "sns", "sqs", "api gateway", "route 53",
    "rds", "aurora", "redshift", "kinesis", "step functions",
    "arn:", "arn:aws:",
}

COMPLEX_QUESTION_PATTERNS = {
    "how would", "how can i", "how do i", "how to implement",
    "what are the tradeoffs", "what are the pros", "what approach",
    "design a", "build a", "create a system", "architect",
    "optimize", "debug", "troubleshoot", "refactor",
    "compare", "evaluate", "analyze the",
}

SIMPLE_QUESTION_PATTERNS = {
    "what is", "what's", "who is", "when was", "where is",
    "how old", "how many", "how much", "define ",
    "what does", "is it", "can you",
}

# ── Output format constraint signals ───────────────────────────────

OUTPUT_FORMAT_SIGNALS = {
    "return as json", "return json", "output as json", "json format",
    "format as", "output format", "in the format", "formatted as",
    "as a table", "as a list", "as bullet points", "as markdown",
    "```json", "```yaml", "```xml", "```csv",
    "structured output", "json schema", "output schema",
    "respond with json", "reply in json", "answer in json",
    "return a json", "provide json", "give me json",
    "xml format", "yaml format", "csv format",
    "following format", "this format", "exact format",
    "schema:", "fields:", "columns:",
}

# ── Constraint density signals ─────────────────────────────────────

CONSTRAINT_SIGNALS = {
    "must be", "must not", "must include", "must have",
    "should be", "should not", "should include",
    "no more than", "no less than", "no longer than",
    "at least", "at most", "exactly", "precisely",
    "without using", "only use", "do not use", "don't use",
    "limited to", "restricted to", "confined to",
    "between", "within", "not exceeding",
    "ensure that", "make sure", "guarantee",
    "required", "mandatory", "necessary",
    "exclude", "avoid", "never",
    "maximum", "minimum",
}

# ── Context reference signals ──────────────────────────────────────

CONTEXT_REFERENCE_SIGNALS = {
    "the above", "the following", "the below",
    "given the", "based on the", "according to the",
    "from the", "in the", "using the",
    "this document", "this text", "this article", "this paper",
    "the provided", "the attached", "the given",
    "extract from", "summarize the", "analyze the",
    "refer to", "as shown", "as described",
}

# ── Structural complexity patterns ──────────────────────────────────

_TABLE_PATTERN = re.compile(r'[\|\+][-=+|]+[\|\+]|(\w{1,50}\s*[,\t]\s*){3,}')
_CSV_DATA = re.compile(r'^[^,\n]+(?:,[^,\n]+){2,}$', re.MULTILINE)
_PARAGRAPH_BREAK = re.compile(r'\n\s*\n')
_NUMBERED_LIST = re.compile(r'^\s*\d+[\.\)]\s', re.MULTILINE)
_CODE_BLOCK = re.compile(r'```[\s\S]*?```|^    \S', re.MULTILINE)


# ── Dimension weights (must sum to 1.0) ─────────────────────────────

@dataclass
class AnalyzerWeights:
    """Configurable weights for the 15 scoring dimensions."""

    token_count: float = 0.3784
    code_presence: float = 0.0573
    reasoning_markers: float = 0.0813
    technical_depth: float = 0.0486
    simple_indicators: float = 0.0072
    multi_step: float = 0.0010
    tool_use: float = 0.0418
    document_analysis: float = 0.1265
    conversation_depth: float = 0.0097
    aws_specificity: float = 0.0257
    math_logical: float = 0.0257
    creative_open: float = 0.0962
    # New dimensions
    output_format: float = 0.0987
    constraint_density: float = 0.0010
    context_ratio: float = 0.0010


# ── Complexity thresholds ───────────────────────────────────────────

@dataclass
class ComplexityThresholds:
    """Score boundaries for complexity classification."""

    simple_max: float = 0.125
    moderate_max: float = 0.350
    complex_max: float = 0.500
    reasoning_marker_count: int = 4  # Auto-promote to reasoning if >= N markers


# ── Utility functions ───────────────────────────────────────────────

def _count_matches(text_lower: str, keywords: set[str]) -> int:
    """Count how many keywords appear in the lowered text.

    For short keywords (≤3 chars), uses word boundary matching to avoid
    false positives from substring matches (e.g., "rds" in "words").
    """
    count = 0
    for kw in keywords:
        if _kw_matches(kw, text_lower):
            count += 1
    return count


def _kw_matches(kw: str, text_lower: str) -> bool:
    """Check if a keyword matches in text, with word boundary for short keywords."""
    if len(kw) <= 3:
        idx = text_lower.find(kw)
        while idx != -1:
            before_ok = (idx == 0 or not text_lower[idx - 1].isalnum())
            after_ok = (idx + len(kw) >= len(text_lower) or not text_lower[idx + len(kw)].isalnum())
            if before_ok and after_ok:
                return True
            idx = text_lower.find(kw, idx + 1)
        return False
    return kw in text_lower


# ── Classifier class ────────────────────────────────────────────────

class HeuristicClassifier(ComplexityClassifier):
    """Heuristic complexity classifier using 15 weighted keyword dimensions.

    Zero dependencies, sub-millisecond inference. Scores text across
    15 dimensions (code presence, reasoning markers, technical depth, etc.)
    and classifies based on configurable thresholds.

    Parameters
    ----------
    weights : AnalyzerWeights, optional
        Custom weights for the 15 scoring dimensions.
    thresholds : ComplexityThresholds, optional
        Custom score boundaries for classification.
    floor_confidence_threshold : float
        Not used by heuristic (floor is keyword-based, not confidence-based).
        Kept for interface compatibility.
    floor_dampening : float
        Not used by heuristic (floor is applied via max() on composite score).
        Kept for interface compatibility.
    """

    def __init__(
        self,
        weights: AnalyzerWeights | None = None,
        thresholds: ComplexityThresholds | None = None,
        floor_confidence_threshold: float = DEFAULT_FLOOR_CONFIDENCE_THRESHOLD,
        floor_dampening: float = DEFAULT_FLOOR_DAMPENING,
    ) -> None:
        super().__init__(
            floor_confidence_threshold=floor_confidence_threshold,
            floor_dampening=floor_dampening,
        )
        self.weights = weights or AnalyzerWeights()
        self.thresholds = thresholds or ComplexityThresholds()

    def classify(self, text: str) -> tuple[str, float]:
        """Classify a single text using 15-dimension heuristic scoring.

        Parameters
        ----------
        text : str
            The text to classify.

        Returns
        -------
        tuple[str, float]
            (label, composite_score) where composite_score is the weighted
            sum of all 15 dimensions (0.0 to 1.0).
        """
        text_lower = text.lower()
        scores = self._score_dimensions(text_lower, text)
        composite = self._compute_composite(scores)
        reasoning_count = _count_matches(text_lower, REASONING_MARKERS)
        label = self._threshold_classify(composite, reasoning_count)
        return label, composite

    def predict_proba_all(self, text: str) -> dict[str, float]:
        """Synthesize probability distribution from the composite score.

        The heuristic doesn't produce natural probabilities like an ML model.
        Instead, we synthesize a distribution based on where the composite
        score falls relative to the thresholds.

        Returns
        -------
        dict[str, float]
            Synthesized probabilities for each class.
        """
        text_lower = text.lower()
        scores = self._score_dimensions(text_lower, text)
        composite = self._compute_composite(scores)
        return self._score_to_probabilities(composite)

    def classify_request(
        self,
        messages: list[dict[str, Any]],
        system: list[dict[str, Any]] | None = None,
        tool_config: dict[str, Any] | None = None,
    ) -> tuple[str, float]:
        """Classify a full request using heuristic scoring + system floor.

        The heuristic classifier uses its own floor mechanism (keyword-based
        system prompt floor via max()) rather than the base class's
        classify-the-floor-text approach. This is because the heuristic
        floor is computed from keyword density in the system prompt, not
        from classifying the system prompt as a standalone text.

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
        tuple[str, float]
            (label, composite_score) after floor application.
        """
        # Extract last user message for scoring
        last_user_text = self.extract_last_user_text(messages)
        if not last_user_text:
            # No user text — use full context
            last_user_text = self.assemble_full_context(messages, system, tool_config)

        text_lower = last_user_text.lower()

        # Score dimensions
        scores = self._score_dimensions(text_lower, last_user_text, messages, tool_config)
        composite = self._compute_composite(scores)

        # Apply system prompt floor (keyword-based, not classify-based).
        # Guard: don't let the system prompt floor push a complex classification
        # into reasoning — reasoning should only be triggered by the user message
        # itself, not by role-assignment text in the system prompt.
        if system:
            from bedrock_smart_router.request_analyzer import _extract_text
            from bedrock_smart_router.complexity_classifier import FLOOR_MAX_LEVEL
            system_text = _extract_text(system)
            if system_text:
                system_floor = self._compute_system_floor(system_text.lower())
                # Cap the floor so it cannot push into reasoning tier
                max_floor_threshold = self.thresholds.complex_max - 0.001
                capped_floor = min(system_floor, max_floor_threshold)
                composite = max(composite, capped_floor)

        # Classify
        reasoning_count = _count_matches(text_lower, REASONING_MARKERS)
        label = self._threshold_classify(composite, reasoning_count)
        return label, composite

    # ── Scoring engine ──────────────────────────────────────────

    def _score_dimensions(
        self,
        text_lower: str,
        text_original: str,
        messages: list[dict[str, Any]] | None = None,
        tool_config: dict[str, Any] | None = None,
    ) -> list[float]:
        """Score text across 15 dimensions. Returns list of scores in [0, 1]."""
        text_len = len(text_lower)

        # 1. Text length — log-scaled
        if text_len <= 20:
            token_score = 0.0
        else:
            token_score = min(1.0, max(0.0,
                (math.log(text_len) - math.log(20)) / (math.log(3000) - math.log(20))
            ))

        # 2. Code presence
        code_hits = _count_matches(text_lower, CODE_MARKERS)
        lang_hits = _count_matches(text_lower, CODE_LANG_KEYWORDS)
        code_score = min(1.0, (code_hits + lang_hits) * 0.35)

        # 3. Reasoning markers
        reasoning_hits = _count_matches(text_lower, REASONING_MARKERS)
        reasoning_score = min(1.0, reasoning_hits * 0.35)

        # 4. Technical keyword density
        total_tech = code_hits + lang_hits + reasoning_hits
        if text_len > 0:
            density = total_tech / max(1, text_len / 200)
            tech_score = min(1.0, density * 0.5)
        else:
            tech_score = 0.0

        # 5. Simple indicators (inverted)
        simple_hits = _count_matches(text_lower, SIMPLE_INDICATORS)
        if text_len < 100 and simple_hits >= 1:
            simple_score = 0.0
        elif simple_hits >= 2:
            simple_score = 0.05
        elif simple_hits == 1:
            simple_score = 0.2
        else:
            simple_score = 0.5

        # 6. Structural complexity
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

        # 8. Domain specificity (AWS + math + data analysis)
        aws_hits = _count_matches(text_lower, AWS_SIGNALS)
        math_hits = _count_matches(text_lower, MATH_SIGNALS)
        data_hits = _count_matches(text_lower, DATA_ANALYSIS_SIGNALS)
        doc_score = min(1.0, (aws_hits + math_hits + data_hits) * 0.25)

        # 9. Conversation depth
        turn_count = len(messages) if messages else 1
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

        # 13. Output format constraints
        format_hits = _count_matches(text_lower, OUTPUT_FORMAT_SIGNALS)
        format_score = min(1.0, format_hits * 0.4)

        # 14. Constraint density
        constraint_hits = _count_matches(text_lower, CONSTRAINT_SIGNALS)
        constraint_score = min(1.0, constraint_hits * 0.2)

        # 15. Context ratio
        context_hits = _count_matches(text_lower, CONTEXT_REFERENCE_SIGNALS)
        if context_hits > 0:
            context_score = min(1.0, context_hits * 0.2 + (0.2 if text_len > 500 else 0.0))
        else:
            context_score = 0.0

        return [
            token_score, code_score, reasoning_score, tech_score,
            simple_score, multi_score, tool_score, doc_score,
            conv_score, aws_score, math_score, creative_score,
            format_score, constraint_score, context_score,
        ]

    def _compute_composite(self, scores: list[float]) -> float:
        """Compute weighted composite from dimension scores."""
        w = self.weights
        weight_list = [
            w.token_count, w.code_presence, w.reasoning_markers,
            w.technical_depth, w.simple_indicators, w.multi_step,
            w.tool_use, w.document_analysis, w.conversation_depth,
            w.aws_specificity, w.math_logical, w.creative_open,
            w.output_format, w.constraint_density, w.context_ratio,
        ]
        composite = sum(s * wt for s, wt in zip(scores, weight_list))
        return max(0.0, min(1.0, composite))

    def _threshold_classify(self, score: float, reasoning_count: int) -> str:
        """Classify based on score thresholds."""
        t = self.thresholds
        if reasoning_count >= t.reasoning_marker_count or score >= t.complex_max:
            return "reasoning"
        if score >= t.moderate_max:
            return "complex"
        if score >= t.simple_max:
            return "moderate"
        return "simple"

    def _compute_system_floor(self, system_text_lower: str) -> float:
        """Derive a complexity floor from the system prompt keywords."""
        reasoning_hits = _count_matches(system_text_lower, REASONING_MARKERS)
        code_hits = _count_matches(system_text_lower, CODE_MARKERS)
        code_lang_hits = _count_matches(system_text_lower, CODE_LANG_KEYWORDS)
        aws_hits = _count_matches(system_text_lower, AWS_SIGNALS)
        math_hits = _count_matches(system_text_lower, MATH_SIGNALS)
        creative_hits = _count_matches(system_text_lower, CREATIVE_SIGNALS)
        constraint_hits = _count_matches(system_text_lower, CONSTRAINT_SIGNALS)
        complex_q_hits = _count_matches(system_text_lower, COMPLEX_QUESTION_PATTERNS)
        data_hits = _count_matches(system_text_lower, DATA_ANALYSIS_SIGNALS)

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
        return raw * SYSTEM_FLOOR_FACTOR

    def _score_to_probabilities(self, composite: float) -> dict[str, float]:
        """Synthesize a probability distribution from the composite score.

        Maps the composite score to a soft distribution across classes
        based on distance from thresholds.
        """
        t = self.thresholds
        # Create soft boundaries using sigmoid-like transitions
        probs = {
            "simple": max(0.0, 1.0 - composite / t.simple_max) if composite < t.moderate_max else 0.05,
            "moderate": 0.0,
            "complex": 0.0,
            "reasoning": 0.0,
        }

        if composite < t.simple_max:
            probs["simple"] = 0.7 + 0.3 * (1.0 - composite / t.simple_max)
            probs["moderate"] = 0.2 * (composite / t.simple_max)
            probs["complex"] = 0.05
            probs["reasoning"] = 0.05 - probs["moderate"] * 0.1
        elif composite < t.moderate_max:
            progress = (composite - t.simple_max) / (t.moderate_max - t.simple_max)
            probs["simple"] = 0.15 * (1.0 - progress)
            probs["moderate"] = 0.5 + 0.2 * progress
            probs["complex"] = 0.25 * progress
            probs["reasoning"] = 0.1 * progress
        elif composite < t.complex_max:
            progress = (composite - t.moderate_max) / (t.complex_max - t.moderate_max)
            probs["simple"] = 0.05
            probs["moderate"] = 0.2 * (1.0 - progress)
            probs["complex"] = 0.5 + 0.2 * progress
            probs["reasoning"] = 0.25 * progress
        else:
            probs["simple"] = 0.02
            probs["moderate"] = 0.05
            probs["complex"] = 0.25
            probs["reasoning"] = 0.68

        # Normalize to sum to 1.0
        total = sum(probs.values())
        if total > 0:
            probs = {k: v / total for k, v in probs.items()}

        return probs
