# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Intent extractor — auto-extract canonical intent and variables from queries.

Uses a cheap Bedrock model (default: Nova Micro) to decompose a user
query into a parameterised intent and a dict of variable values.  This
enables the semantic cache to match queries that differ only in their
parameter values without the caller having to extract variables manually.

Supports both single-turn queries and multi-turn conversations.  For
multi-turn, the full conversation history is resolved into a single
self-contained query before extraction.

Usage::

    extractor = IntentExtractor(region="us-west-2")

    # Single-turn
    result = extractor.extract("Count users by geo for 2026 with sales > $200")
    # result.intent = "Count users by geography for a given year with sales above a threshold"
    # result.variables = {"year": "2026", "sales_threshold": "200"}

    # Multi-turn
    result = extractor.extract_from_messages([
        {"role": "user", "content": [{"text": "show me users by geo"}]},
        {"role": "assistant", "content": [{"text": "Here are users by geography..."}]},
        {"role": "user", "content": [{"text": "now for 2026 with sales > $200"}]},
    ])
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_EXTRACTION_MODEL = "amazon.nova-micro-v1:0"

# Retry defaults for extraction calls (lighter than the main router retries)
_MAX_RETRIES = 2
_BACKOFF_BASE = 0.3
_BACKOFF_MULTIPLIER = 2.0
_BACKOFF_MAX = 4.0

_RETRYABLE_ERRORS = frozenset({
    "ThrottlingException",
    "ServiceUnavailableException",
    "InternalServerException",
    "ModelTimeoutException",
})

_EXTRACTION_SYSTEM_PROMPT = """\
You are a query analysis assistant. Your job is to extract two things from the user's query:

1. **intent**: Rewrite the query as a generic template by replacing specific values \
(dates, numbers, names, categories, thresholds, limits) with descriptive placeholders. \
The intent should capture WHAT the user wants, not the specific parameter values.

2. **variables**: A JSON object mapping each placeholder name to its actual value from the query.

Rules:
- The intent must be a complete, self-contained sentence (not a fragment).
- Variable names should be descriptive snake_case (e.g. "year", "sales_threshold", "category").
- If the query has no extractable variables, return an empty variables object.
- If the input is a multi-turn conversation, resolve it into a single self-contained query first, \
then extract intent and variables from that resolved query.
- Return ONLY valid JSON with exactly two keys: "intent" and "variables".

Examples:

Query: "Show me the top 10 customers in Electronics for Q3 2025"
Output: {"intent": "Show the top N customers in a category for a time period", "variables": {"count": "10", "category": "Electronics", "time_period": "Q3 2025"}}

Query: "How many users signed up last month?"
Output: {"intent": "How many users signed up in a time period", "variables": {"time_period": "last month"}}

Query: "What is Amazon S3?"
Output: {"intent": "What is Amazon S3", "variables": {}}

Query: "Compare sales between New York and London for 2024"
Output: {"intent": "Compare sales between two cities for a year", "variables": {"city_1": "New York", "city_2": "London", "year": "2024"}}

Query: "Find orders over $500 placed in January by premium customers"
Output: {"intent": "Find orders above a dollar amount placed in a month by a customer tier", "variables": {"amount": "500", "month": "January", "customer_tier": "premium"}}

Query: "Count users distributed by geography for 2026 who have overall sales of more than $200"
Output: {"intent": "Count users distributed by geography for a year with overall sales above a threshold", "variables": {"year": "2026", "sales_threshold": "200"}}

Conversation:
User: "show me the users by geo"
Assistant: "Here are users distributed by geography: ..."
User: "Ok now show me this data for 2026 for overall sales more than $200"
Output: {"intent": "Count users distributed by geography for a year with overall sales above a threshold", "variables": {"year": "2026", "sales_threshold": "200"}}

Conversation:
User: "What were our top products?"
Assistant: "Here are the top products across all categories..."
User: "Now filter this for Electronics only"
User: "And just for 2025"
Output: {"intent": "Show top products for a category in a year", "variables": {"category": "Electronics", "year": "2025"}}
"""


@dataclass
class ExtractionResult:
    """Result of intent extraction."""

    intent: str
    variables: dict[str, str]
    raw_query: str = ""
    source: str = "single-turn"  # "single-turn" | "multi-turn"


@dataclass
class IntentExtractorConfig:
    """Configuration for the intent extractor."""

    model_id: str = DEFAULT_EXTRACTION_MODEL
    max_retries: int = _MAX_RETRIES
    backoff_base: float = _BACKOFF_BASE
    backoff_multiplier: float = _BACKOFF_MULTIPLIER
    backoff_max: float = _BACKOFF_MAX
    cache_max_entries: int = 500


class IntentExtractor:
    """Extracts canonical intent and variables from queries using a Bedrock model.

    Results are cached in an LRU dict keyed by the hash of the input
    text, so repeated identical queries don't incur additional LLM calls.
    """

    def __init__(
        self,
        config: IntentExtractorConfig | None = None,
        boto_session: Any | None = None,
        region: str = "us-west-2",
    ) -> None:
        self.config = config or IntentExtractorConfig()
        self._session = boto_session
        self._region = region
        self._cache: OrderedDict[str, ExtractionResult] = OrderedDict()
        self._client: Any | None = None

    def _get_client(self) -> Any:
        if self._client is None:
            if self._session is None:
                import boto3
                self._session = boto3.Session(region_name=self._region)
            self._client = self._session.client(
                "bedrock-runtime", region_name=self._region,
            )
        return self._client

    def extract(self, query_text: str) -> ExtractionResult:
        """Extract intent and variables from a single-turn query.

        Args:
            query_text: The user's query string.

        Returns:
            ExtractionResult with the canonical intent and variables.
        """
        cache_key = self._cache_key(query_text)
        cached = self._cache.get(cache_key)
        if cached is not None:
            self._cache.move_to_end(cache_key)
            return cached

        messages = [{"role": "user", "content": [{"text": query_text}]}]
        result = self._call_model(messages)
        result.raw_query = query_text
        result.source = "single-turn"

        self._put_cache(cache_key, result)
        return result

    def extract_from_messages(
        self,
        messages: list[dict[str, Any]],
    ) -> ExtractionResult:
        """Extract intent and variables from a multi-turn conversation.

        Resolves the full conversation history into a single
        self-contained query, then extracts intent and variables.

        Args:
            messages: Bedrock Converse-format message list.

        Returns:
            ExtractionResult with the resolved intent and variables.
        """
        # Build a text representation for caching
        text_repr = self._messages_to_text(messages)
        cache_key = self._cache_key(text_repr)
        cached = self._cache.get(cache_key)
        if cached is not None:
            self._cache.move_to_end(cache_key)
            return cached

        # Build the extraction prompt with conversation context
        conversation_text = self._format_conversation(messages)
        extraction_messages = [
            {"role": "user", "content": [{"text": conversation_text}]},
        ]
        result = self._call_model(extraction_messages)
        result.raw_query = text_repr
        result.source = "multi-turn"

        self._put_cache(cache_key, result)
        return result

    def _call_model(
        self,
        messages: list[dict[str, Any]],
    ) -> ExtractionResult:
        """Call the extraction model with retries and parse the response."""
        client = self._get_client()
        last_exc: Exception | None = None

        for attempt in range(1 + self.config.max_retries):
            try:
                response = client.converse(
                    modelId=self.config.model_id,
                    messages=messages,
                    system=[{"text": _EXTRACTION_SYSTEM_PROMPT}],
                    inferenceConfig={"maxTokens": 512, "temperature": 0.0},
                )
                return self._parse_response(response)
            except Exception as exc:
                last_exc = exc
                error_code = self._get_error_code(exc)

                if error_code not in _RETRYABLE_ERRORS:
                    logger.warning(
                        "Intent extraction non-retryable error: %s", error_code,
                    )
                    raise

                if attempt >= self.config.max_retries:
                    logger.warning(
                        "Intent extraction max retries (%d) exhausted: %s",
                        self.config.max_retries, error_code,
                    )
                    raise

                delay = min(
                    self.config.backoff_base
                    * (self.config.backoff_multiplier ** attempt),
                    self.config.backoff_max,
                )
                logger.info(
                    "Intent extraction retry %d/%d after %s, backoff %.2fs",
                    attempt + 1, self.config.max_retries, error_code, delay,
                )
                time.sleep(delay)

        raise last_exc  # type: ignore[misc]

    @staticmethod
    def _parse_response(response: dict[str, Any]) -> ExtractionResult:
        """Parse the model response into an ExtractionResult."""
        output = response.get("output", {}).get("message", {})
        content = output.get("content", [])
        text = ""
        for block in content:
            if "text" in block:
                text += block["text"]

        # Try to parse as JSON
        text = text.strip()
        # Handle markdown code blocks
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(
                line for line in lines
                if not line.strip().startswith("```")
            ).strip()

        try:
            data = json.loads(text)
            intent = str(data.get("intent", ""))
            variables = data.get("variables", {})
            # Ensure all variable values are strings
            variables = {str(k): str(v) for k, v in variables.items()}
            return ExtractionResult(intent=intent, variables=variables)
        except (json.JSONDecodeError, AttributeError) as exc:
            logger.warning(
                "Failed to parse extraction response as JSON: %s — text: %s",
                exc, text[:200],
            )
            # Fallback: use the raw text as intent, no variables
            return ExtractionResult(intent=text, variables={})

    @staticmethod
    def _format_conversation(messages: list[dict[str, Any]]) -> str:
        """Format a Bedrock message list into a readable conversation string."""
        parts: list[str] = []
        parts.append("Conversation:")
        for msg in messages:
            role = msg.get("role", "user").capitalize()
            content_blocks = msg.get("content", [])
            text_parts = []
            for block in content_blocks:
                if isinstance(block, dict) and "text" in block:
                    text_parts.append(block["text"])
                elif isinstance(block, str):
                    text_parts.append(block)
            text = " ".join(text_parts)
            parts.append(f"{role}: {text}")
        return "\n".join(parts)

    @staticmethod
    def _messages_to_text(messages: list[dict[str, Any]]) -> str:
        """Create a deterministic text representation for cache keying."""
        parts = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", [])
            for block in content:
                if isinstance(block, dict) and "text" in block:
                    parts.append(f"{role}:{block['text']}")
        return "|".join(parts)

    @staticmethod
    def _cache_key(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()[:20]

    def _put_cache(self, key: str, result: ExtractionResult) -> None:
        if len(self._cache) >= self.config.cache_max_entries:
            self._cache.popitem(last=False)  # Evict oldest
        self._cache[key] = result

    @staticmethod
    def _get_error_code(exc: Exception) -> str:
        if hasattr(exc, "response"):
            return exc.response.get("Error", {}).get("Code", type(exc).__name__)
        return type(exc).__name__
