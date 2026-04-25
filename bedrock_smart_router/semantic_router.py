"""Semantic intent router (optional — requires embeddings extra).

Routes requests to specific models based on embedding similarity to
predefined intent examples.  Useful for directing specialized queries
to specialized models (e.g., code questions to a code-tuned model).

Install with::

    pip install bedrock-smart-router[embeddings]

Usage::

    from bedrock_smart_router.semantic_router import SemanticRoute, SemanticRouter

    routes = [
        SemanticRoute(
            name="code",
            model="us.anthropic.claude-sonnet-4-6",
            examples=["Write a function", "Debug this code", "Explain this algorithm"],
        ),
        SemanticRoute(
            name="creative",
            model="us.anthropic.claude-opus-4-7",
            examples=["Write a story", "Compose a poem", "Brainstorm ideas"],
        ),
    ]
    router = SemanticRouter(routes=routes)
    match = router.route("Help me fix this Python bug")
    # match.route_name == "code", match.model == "us.anthropic.claude-sonnet-4-6"
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Max parallel Bedrock embedding calls during route initialization.
# Keeps us under typical Titan Embed RPM quotas while still being fast.
_INIT_MAX_WORKERS = 10


@dataclass
class SemanticRoute:
    """A named route with example utterances and a target model."""

    name: str
    model: str
    examples: list[str] = field(default_factory=list)
    threshold: float = 0.80  # Minimum similarity to match


@dataclass
class SemanticRouteMatch:
    """Result of a semantic route match."""

    route_name: str
    model: str
    score: float
    matched_example: str


class SemanticRouter:
    """Embedding-based intent router.

    Computes embeddings for all route examples at init time, then
    matches incoming queries by cosine similarity.
    """

    def __init__(
        self,
        routes: list[SemanticRoute] | None = None,
        embedding_model: str = "amazon.titan-embed-text-v2:0",
        boto_session: Any | None = None,
        region: str = "us-west-2",
        default_model: str | None = None,
    ) -> None:
        self.routes = routes or []
        self.embedding_model = embedding_model
        self.default_model = default_model
        self._session = boto_session
        self._region = region
        self._client: Any | None = None
        self._route_embeddings: list[tuple[SemanticRoute, str, list[float]]] = []
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Compute embeddings for all route examples (lazy, concurrent).

        Uses a thread pool to call the Bedrock embedding API in
        parallel (up to ``_INIT_MAX_WORKERS`` concurrent calls).
        For 40 examples at ~100ms each, this takes ~400ms instead of 4s.
        """
        if self._initialized:
            return

        all_examples: list[tuple[SemanticRoute, str]] = [
            (route, example)
            for route in self.routes
            for example in route.examples
        ]

        if not all_examples:
            self._initialized = True
            return

        # Fire all embedding calls concurrently
        with ThreadPoolExecutor(max_workers=_INIT_MAX_WORKERS) as pool:
            embeddings = list(pool.map(
                lambda pair: self._get_embedding(pair[1]),
                all_examples,
            ))

        for (route, example), emb in zip(all_examples, embeddings):
            self._route_embeddings.append((route, example, emb))

        self._initialized = True
        logger.info(
            "Semantic router initialized with %d examples across %d routes",
            len(self._route_embeddings), len(self.routes),
        )

    def _get_embedding(self, text: str) -> list[float]:
        if self._client is None:
            if self._session is None:
                import boto3
                self._session = boto3.Session(region_name=self._region)
            self._client = self._session.client("bedrock-runtime", region_name=self._region)
        import json
        resp = self._client.invoke_model(
            modelId=self.embedding_model,
            body=json.dumps({"inputText": text}),
        )
        body = json.loads(resp["body"].read())
        return body.get("embedding", [])

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        from bedrock_smart_router.utils import cosine_similarity
        return cosine_similarity(a, b)

    def route(self, query: str) -> SemanticRouteMatch | None:
        """Match a query to the best route by embedding similarity.

        Returns *None* if no route exceeds its threshold.
        """
        self._ensure_initialized()
        query_emb = self._get_embedding(query)

        best_score = 0.0
        best_route: SemanticRoute | None = None
        best_example = ""

        for route, example, emb in self._route_embeddings:
            score = self._cosine_similarity(query_emb, emb)
            if score > best_score:
                best_score = score
                best_route = route
                best_example = example

        if best_route and best_score >= best_route.threshold:
            return SemanticRouteMatch(
                route_name=best_route.name,
                model=best_route.model,
                score=round(best_score, 4),
                matched_example=best_example,
            )

        return None
