"""Zero-API-call request complexity analyzer.

Classifies incoming requests across 12 scoring dimensions to determine
the appropriate model tier, entirely locally with sub-millisecond overhead.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from bedrock_smart_router.models import Complexity, RequestAnalysis

# ── Keyword / pattern sets ──────────────────────────────────────────

REASONING_MARKERS = {
    "step by step", "step-by-step", "analyze", "evaluate", "compare and contrast",
    "prove", "derive", "reason through", "think through", "work through",
    "explain why", "explain how", "trade-off", "tradeoff", "pros and cons",
    "critically", "systematically", "deduce", "infer", "hypothesize",
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


# ── Dimension weights (must sum to 1.0) ─────────────────────────────

@dataclass
class AnalyzerWeights:
    """Configurable weights for the 12 scoring dimensions."""

    token_count: float = 0.07
    code_presence: float = 0.12
    reasoning_markers: float = 0.14
    technical_depth: float = 0.10
    simple_indicators: float = 0.05
    multi_step: float = 0.08
    tool_use: float = 0.09
    document_analysis: float = 0.08
    conversation_depth: float = 0.06
    aws_specificity: float = 0.06
    math_logical: float = 0.08
    creative_open: float = 0.07


# ── Complexity thresholds ───────────────────────────────────────────

@dataclass
class ComplexityThresholds:
    """Score boundaries for complexity classification."""

    simple_max: float = 0.25
    moderate_max: float = 0.55
    complex_max: float = 0.80
    reasoning_marker_count: int = 2  # Auto-promote to reasoning if >= N markers


# ── Token estimation ────────────────────────────────────────────────

def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token for English text."""
    return max(1, len(text) // 4)


def _extract_text(messages: list[dict[str, Any]]) -> str:
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


def _count_matches(text_lower: str, keywords: set[str]) -> int:
    """Count how many keywords appear in the lowered text."""
    return sum(1 for kw in keywords if kw in text_lower)


# ── Main analyzer ───────────────────────────────────────────────────

class RequestAnalyzer:
    """Classifies requests using 12 local scoring dimensions.

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
        ]
        composite = sum(s * wt for s, wt in zip(scores, weight_list))
        composite = max(0.0, min(1.0, composite))

        # ── Classify complexity ─────────────────────────────────
        reasoning_count = _count_matches(text_lower, REASONING_MARKERS)
        complexity = self._classify(composite, reasoning_count)

        # ── Detect capabilities needed ──────────────────────────
        has_images = self._has_images(messages)
        requires_tool = tool_config is not None or scores[6] > 0.3
        est_input = _estimate_tokens(full_text)
        est_output = max(256, est_input // 3)

        return RequestAnalysis(
            complexity=complexity,
            complexity_score=round(composite, 4),
            estimated_input_tokens=est_input,
            estimated_output_tokens=est_output,
            requires_vision=has_images,
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
        """Return a list of 12 dimension scores in [0, 1]."""
        text_len = len(text_lower)

        # 1. Token count
        token_score = min(1.0, text_len / 20_000)

        # 2. Code presence
        code_hits = _count_matches(text_lower, CODE_MARKERS)
        lang_hits = _count_matches(text_lower, CODE_LANG_KEYWORDS)
        code_score = min(1.0, (code_hits * 0.2 + lang_hits * 0.15))

        # 3. Reasoning markers
        reasoning_hits = _count_matches(text_lower, REASONING_MARKERS)
        reasoning_score = min(1.0, reasoning_hits * 0.25)

        # 4. Technical depth
        tech_density = (code_hits + lang_hits + reasoning_hits) / max(1, text_len / 500)
        tech_score = min(1.0, tech_density * 0.5)

        # 5. Simple indicators (inverted — more simple = lower complexity)
        simple_hits = _count_matches(text_lower, SIMPLE_INDICATORS)
        simple_score = max(0.0, 1.0 - simple_hits * 0.2)

        # 6. Multi-step patterns
        multi_hits = _count_matches(text_lower, MULTI_STEP_PATTERNS)
        multi_score = min(1.0, multi_hits * 0.2)

        # 7. Tool use signals
        tool_hits = _count_matches(text_lower, TOOL_USE_SIGNALS)
        tool_score = min(1.0, tool_hits * 0.25)
        if tool_config:
            tool_score = max(tool_score, 0.5)

        # 8. Document analysis
        doc_hits = _count_matches(text_lower, DOCUMENT_SIGNALS)
        doc_score = min(1.0, doc_hits * 0.2)

        # 9. Conversation depth
        turn_count = len(messages)
        conv_score = min(1.0, turn_count / 10)

        # 10. AWS specificity
        aws_hits = _count_matches(text_lower, AWS_SIGNALS)
        aws_score = min(1.0, aws_hits * 0.15)

        # 11. Math / logical
        math_hits = _count_matches(text_lower, MATH_SIGNALS)
        math_score = min(1.0, math_hits * 0.25)

        # 12. Creative / open-ended
        creative_hits = _count_matches(text_lower, CREATIVE_SIGNALS)
        creative_score = min(1.0, creative_hits * 0.25)

        return [
            token_score, code_score, reasoning_score, tech_score,
            simple_score, multi_score, tool_score, doc_score,
            conv_score, aws_score, math_score, creative_score,
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
