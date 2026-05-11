"""Zero-API-call request complexity analyzer.

Classifies incoming requests across 15 scoring dimensions to determine
the appropriate model tier, entirely locally with sub-millisecond overhead.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

from bedrock_smart_router.models import Complexity, RequestAnalysis

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
    moderate_max: float = 0.200
    complex_max: float = 0.350
    reasoning_marker_count: int = 2  # Auto-promote to reasoning if >= N markers


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
    ) -> None:
        self.weights = weights or AnalyzerWeights()
        self.thresholds = thresholds or ComplexityThresholds()

    def analyze(
        self,
        messages: list[dict[str, Any]],
        system: list[dict[str, Any]] | None = None,
        tool_config: dict[str, Any] | None = None,
    ) -> RequestAnalysis:
        """Analyze a request and return a ``RequestAnalysis``."""
        user_text = _extract_text(messages)
        system_text = _extract_text(system) if system else ""
        full_text = f"{system_text}\n{user_text}".strip()
        text_lower = full_text.lower()

        # ── Per-dimension scores (0.0 – 1.0) ───────────────────
        scores = self._score_dimensions(text_lower, messages, tool_config)

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
            if payload_bytes > 5_000_000:       # > 5MB
                composite += 0.30
            elif payload_bytes > 1_000_000:     # > 1MB
                composite += 0.20
            elif payload_bytes > 100_000:       # > 100KB
                composite += 0.10
            else:                               # < 100KB
                composite += 0.05

        composite = max(0.0, min(1.0, composite))

        # ── Classify complexity ─────────────────────────────────
        reasoning_count = _count_matches(text_lower, REASONING_MARKERS)
        complexity = self._classify(composite, reasoning_count)

        # ── Detect capabilities needed ──────────────────────────
        has_images = self._has_images(messages)
        has_documents = self._has_documents(messages)
        requires_tool = tool_config is not None or scores[6] > 0.3
        est_input = _estimate_tokens(full_text)

        # Add estimated tokens for multimodal content.
        # Bedrock converts images/documents to tokens internally:
        # ~750 tokens per image, ~1500 tokens per document page (~3KB/page).
        payload_bytes = self._multimodal_payload_bytes(messages)
        if payload_bytes > 0:
            if has_documents:
                # Estimate pages: ~3KB per page, ~1500 tokens per page
                est_pages = max(1, payload_bytes // 3000)
                est_input += int(est_pages * 1500)
            elif has_images:
                # Estimate ~750 tokens per image (conservative)
                image_count = sum(
                    1 for msg in messages
                    for block in (msg.get("content", []) if isinstance(msg.get("content"), list) else [])
                    if isinstance(block, dict) and "image" in block
                )
                est_input += image_count * 750

        est_output = max(256, est_input // 3)

        return RequestAnalysis(
            complexity=complexity,
            complexity_score=round(composite, 4),
            estimated_input_tokens=est_input,
            estimated_output_tokens=est_output,
            requires_vision=has_images,
            requires_document_support=has_documents,
            requires_tool_use=requires_tool,
            requires_long_context=est_input > 32_000,
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
        """
        text_len = len(text_lower)

        # Also get the original (non-lowered) text for structural patterns
        text_original = _extract_text(messages)

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
        if tool_config:
            tool_score = max(tool_score, 0.6)

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
