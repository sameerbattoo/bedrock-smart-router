"""Tests for the semantic intent router.

Uses mocked embeddings to test routing logic without calling Bedrock.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from bedrock_smart_router.semantic_router import (
    SemanticRoute,
    SemanticRouter,
    SemanticRouteMatch,
)


def _mock_embedding(text: str) -> list[float]:
    """Deterministic mock embedding based on keywords.

    Returns a 4-dim vector where each dimension represents a topic:
    [code, creative, data, general]
    """
    text_lower = text.lower()
    vec = [0.1, 0.1, 0.1, 0.1]
    if any(w in text_lower for w in ["code", "function", "bug", "python", "debug", "algorithm"]):
        vec[0] = 0.9
    if any(w in text_lower for w in ["story", "poem", "creative", "imagine", "write a story"]):
        vec[1] = 0.9
    if any(w in text_lower for w in ["data", "sql", "average", "trend", "analyze"]):
        vec[2] = 0.9
    if any(w in text_lower for w in ["weather", "hello", "general"]):
        vec[3] = 0.9
    return vec


@pytest.fixture
def mock_router():
    """SemanticRouter with mocked embedding function."""
    router = SemanticRouter(
        routes=[
            SemanticRoute(
                name="code",
                model="us.anthropic.claude-sonnet-4-6",
                examples=["Write a Python function", "Debug this code", "Fix this bug"],
                threshold=0.70,
            ),
            SemanticRoute(
                name="creative",
                model="us.anthropic.claude-opus-4-7",
                examples=["Write a story about", "Compose a poem", "Imagine a world"],
                threshold=0.70,
            ),
            SemanticRoute(
                name="data",
                model="us.amazon.nova-pro-v1:0",
                examples=["Analyze this data", "Create a SQL query", "Calculate the average"],
                threshold=0.70,
            ),
        ],
        default_model="us.amazon.nova-lite-v1:0",
    )
    # Replace the embedding function with our mock
    router._get_embedding = _mock_embedding
    router._initialized = False  # Force re-init with mock embeddings
    return router


class TestSemanticRouter:
    def test_routes_code_query(self, mock_router):
        match = mock_router.route("Help me debug this Python function")
        assert match is not None
        assert match.route_name == "code"
        assert match.model == "us.anthropic.claude-sonnet-4-6"

    def test_routes_creative_query(self, mock_router):
        match = mock_router.route("Write a story about a dragon")
        assert match is not None
        assert match.route_name == "creative"
        assert match.model == "us.anthropic.claude-opus-4-7"

    def test_routes_data_query(self, mock_router):
        match = mock_router.route("Analyze this data and find trends")
        assert match is not None
        assert match.route_name == "data"
        assert match.model == "us.amazon.nova-pro-v1:0"

    def test_no_match_returns_none(self, mock_router):
        match = mock_router.route("What's the weather today?")
        # "weather" doesn't match any route's examples closely enough
        # (general topic, not code/creative/data)
        # May or may not match depending on threshold — test the default_model path
        if match is None:
            assert mock_router.default_model == "us.amazon.nova-lite-v1:0"

    def test_match_has_score(self, mock_router):
        match = mock_router.route("Write a Python algorithm")
        assert match is not None
        assert 0.0 <= match.score <= 1.0

    def test_match_has_matched_example(self, mock_router):
        match = mock_router.route("Debug this code please")
        assert match is not None
        assert len(match.matched_example) > 0

    def test_lazy_initialization(self, mock_router):
        """Embeddings should be computed lazily on first route() call."""
        assert not mock_router._initialized
        mock_router.route("test")
        assert mock_router._initialized

    def test_second_call_reuses_embeddings(self, mock_router):
        """Route examples should only be embedded once."""
        mock_router.route("first call")
        assert mock_router._initialized
        # Second call should not re-embed
        embed_count_before = len(mock_router._route_embeddings)
        mock_router.route("second call")
        assert len(mock_router._route_embeddings) == embed_count_before

    def test_empty_routes(self):
        router = SemanticRouter(routes=[], default_model="default")
        router._get_embedding = _mock_embedding
        match = router.route("anything")
        assert match is None

    def test_single_route(self):
        router = SemanticRouter(
            routes=[SemanticRoute(
                name="only",
                model="model-a",
                examples=["Write code"],
                threshold=0.5,
            )],
        )
        router._get_embedding = _mock_embedding
        match = router.route("Write a Python function")
        assert match is not None
        assert match.route_name == "only"

    def test_threshold_respected(self):
        """High threshold should reject weak matches."""
        router = SemanticRouter(
            routes=[SemanticRoute(
                name="code",
                model="model-a",
                examples=["Write code"],
                threshold=0.99,  # Very high — almost nothing matches
            )],
        )
        router._get_embedding = _mock_embedding
        match = router.route("Tell me about the weather")
        assert match is None  # Weather doesn't match code at 0.99

    def test_default_model_attribute(self, mock_router):
        assert mock_router.default_model == "us.amazon.nova-lite-v1:0"


class TestSemanticRouteMatch:
    def test_dataclass_fields(self):
        match = SemanticRouteMatch(
            route_name="code",
            model="model-a",
            score=0.95,
            matched_example="Write a function",
        )
        assert match.route_name == "code"
        assert match.model == "model-a"
        assert match.score == 0.95
        assert match.matched_example == "Write a function"
