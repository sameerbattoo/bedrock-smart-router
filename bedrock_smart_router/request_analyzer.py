"""Zero-API-call request complexity analyzer.

Classifies incoming requests across 15 scoring dimensions to determine
the appropriate model tier, entirely locally with sub-millisecond overhead.

Scoring Strategy
----------------
Complexity is determined by two signals combined via ``max()``:

1. **User Message Score** — The last user message is scored across 15
   keyword/pattern dimensions (token count, code presence, reasoning
   markers, technical depth, etc.).  Only the last user message is used,
   NOT the full conversation history or system prompt.  This prevents
   multi-turn conversations and verbose system prompts from inflating
   the complexity of simple follow-up messages like "Hi" or "Thanks".

2. **System Prompt Floor** — The system prompt establishes a baseline
   task complexity.  A complex system prompt (e.g. "You are a senior
   architect, analyze trade-offs, design well-architected solutions")
   means even short user messages require a capable model because the
   system prompt defines what the model must do.  The floor is computed
   as ``system_prompt_keyword_score × SYSTEM_FLOOR_FACTOR (0.30)``.

The final score is ``max(user_message_score, system_prompt_floor)``.

This design ensures:
- "Hi" with a complex system prompt → MODERATE (floor applies)
- "Hi" with no system prompt → SIMPLE (no floor)
- "Design a DR architecture" → COMPLEX (user message score dominates)
- Short follow-ups in multi-turn don't inherit prior turn complexity

Capability Detection (separate from complexity)
-----------------------------------------------
Full context (all messages + system + tool_config) is still used for:
- Token estimation (for cost prediction)
- Capability requirements (vision, documents, tool use, long context)
- Conversation metadata (turn count, multi-turn flag)
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any

from bedrock_smart_router.models import Complexity, RequestAnalysis

logger = logging.getLogger(__name__)

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

# Token estimation
CHARS_PER_TOKEN = 4

# Multimodal token estimates
TOKENS_PER_IMAGE = 750
TOKENS_PER_DOC_PAGE = 1500
BYTES_PER_DOC_PAGE = 3000

# Long context threshold (tokens)
LONG_CONTEXT_THRESHOLD = 32_000

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

_TABLE_PATTERN = re.compile(r'[\|\+][-=+|]+[\|\+]|(\w+\s*[,\t]\s*){3,}')
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


def _count_matches(text_lower: str, keywords: set[str]) -> int:
    """Count how many keywords appear in the lowered text."""
    return sum(1 for kw in keywords if kw in text_lower)


# ── Main analyzer ───────────────────────────────────────────────────

class RequestAnalyzer:
    """Classifies requests using 15 local scoring dimensions.

    No API calls — runs in sub-millisecond time.
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

        # If ML classifier is available, use it for complexity detection
        if self._ml_classifier is not None:
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
            "reasoning": [kw for kw in REASONING_MARKERS if kw in scoring_text_lower],
            "code": [kw for kw in CODE_MARKERS if kw in scoring_text_lower],
            "code_languages": [kw for kw in CODE_LANG_KEYWORDS if kw in scoring_text_lower],
            "simple": [kw for kw in SIMPLE_INDICATORS if kw in scoring_text_lower],
            "multi_step": [kw for kw in MULTI_STEP_PATTERNS if kw in scoring_text_lower],
            "tool_use": [kw for kw in TOOL_USE_SIGNALS if kw in scoring_text_lower],
            "aws": [kw for kw in AWS_SIGNALS if kw in scoring_text_lower],
            "math": [kw for kw in MATH_SIGNALS if kw in scoring_text_lower],
            "creative": [kw for kw in CREATIVE_SIGNALS if kw in scoring_text_lower],
            "complex_questions": [kw for kw in COMPLEX_QUESTION_PATTERNS if kw in scoring_text_lower],
            "output_format": [kw for kw in OUTPUT_FORMAT_SIGNALS if kw in scoring_text_lower],
            "constraints": [kw for kw in CONSTRAINT_SIGNALS if kw in scoring_text_lower],
            "context_references": [kw for kw in CONTEXT_REFERENCE_SIGNALS if kw in scoring_text_lower],
            "data_analysis": [kw for kw in DATA_ANALYSIS_SIGNALS if kw in scoring_text_lower],
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
            system_floor_markers = {
                "reasoning": [kw for kw in REASONING_MARKERS if kw in system_text_lower],
                "code": [kw for kw in CODE_MARKERS if kw in system_text_lower],
                "aws": [kw for kw in AWS_SIGNALS if kw in system_text_lower],
                "math": [kw for kw in MATH_SIGNALS if kw in system_text_lower],
                "creative": [kw for kw in CREATIVE_SIGNALS if kw in system_text_lower],
                "constraints": [kw for kw in CONSTRAINT_SIGNALS if kw in system_text_lower],
                "complex_questions": [kw for kw in COMPLEX_QUESTION_PATTERNS if kw in system_text_lower],
                "data_analysis": [kw for kw in DATA_ANALYSIS_SIGNALS if kw in system_text_lower],
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
