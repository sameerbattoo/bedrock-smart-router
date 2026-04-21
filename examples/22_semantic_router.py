"""Semantic Router — route queries to specialized models by intent.

Define routes with example utterances.  The router embeds the incoming
query and matches it to the closest route using cosine similarity.
Each route maps to a specific model optimized for that type of task.

Cost: ~$0.0001 per embedding call, ~50-100ms latency overhead.
The embedding is computed once per query (not per route).

Demonstrates:
  - Defining routes with example utterances
  - Routing queries to specialized models
  - Handling no-match with a default model
  - Combining semantic router with the smart router (preferred_model)
  - Per-route threshold tuning
  - Full pipeline: intent → cache → route
"""

from bedrock_smart_router import BedrockRouter, RoutingConfig
from bedrock_smart_router.semantic_router import (
    SemanticRoute,
    SemanticRouter,
)

router = BedrockRouter.create()


# ═══════════════════════════════════════════════════════════════════
# Example 1: Define routes with example utterances
# ═══════════════════════════════════════════════════════════════════

intent_router = SemanticRouter(
    routes=[
        SemanticRoute(
            name="code",
            model="us.anthropic.claude-sonnet-4-6",
            examples=[
                "Write a Python function",
                "Debug this code",
                "Explain this algorithm",
                "Fix this bug",
                "Refactor this class",
            ],
            threshold=0.80,
        ),
        SemanticRoute(
            name="creative",
            model="us.anthropic.claude-opus-4-7",
            examples=[
                "Write a story about",
                "Compose a poem",
                "Brainstorm ideas for",
                "Create a narrative",
                "Imagine a world where",
            ],
            threshold=0.80,
        ),
        SemanticRoute(
            name="data",
            model="us.amazon.nova-pro-v1:0",
            examples=[
                "Analyze this data",
                "Create a SQL query",
                "Summarize these numbers",
                "What trends do you see",
                "Calculate the average",
            ],
            threshold=0.80,
        ),
    ],
    embedding_model="amazon.titan-embed-text-v2:0",
    default_model="us.amazon.nova-lite-v1:0",
    region="us-west-2",
)


# ═══════════════════════════════════════════════════════════════════
# Example 2: Route queries to specialized models
# ═══════════════════════════════════════════════════════════════════

queries = [
    "Help me fix this Python bug in my sort function",
    "Write me a short story about a robot learning to paint",
    "What's the average revenue per quarter from this dataset?",
    "What's the weather today?",
]

print("Semantic routing:")
for query in queries:
    match = intent_router.route(query)
    if match:
        print(f"  '{query[:55]:55s}' → {match.route_name:8s} ({match.model})")
    else:
        print(f"  '{query[:55]:55s}' → DEFAULT  ({intent_router.default_model})")


# ═══════════════════════════════════════════════════════════════════
# Example 3: Combine with smart router using preferred_model
# ═══════════════════════════════════════════════════════════════════
# The semantic router picks the EXACT model.  Pass it to the smart
# router via preferred_model — the smart router uses that model as
# primary and builds a fallback chain around it.  You get the intent
# router's model choice PLUS the smart router's reliability features
# (retries, fallbacks, caching, metrics, observability).

query = "Write a recursive fibonacci function in Python"
match = intent_router.route(query)

if match:
    response = router.converse(
        messages=[{"role": "user", "content": [{"text": query}]}],
        routing=RoutingConfig(preferred_model=match.model),
    )
    d = response["routing_decision"]
    print(f"\nIntent + Smart Router:")
    print(f"  Intent: {match.route_name} (score={match.score:.2f})")
    print(f"  Model:  {d.selected_model}")
    print(f"  Cost:   ${d.actual_cost:.6f}")
else:
    response = router.converse(
        messages=[{"role": "user", "content": [{"text": query}]}],
    )
    print(f"\nNo match → smart router picked: {response['routing_decision'].selected_model}")


# ═══════════════════════════════════════════════════════════════════
# Example 4: No match — falls back to default model
# ═══════════════════════════════════════════════════════════════════

match = intent_router.route("What's the weather today?")
if match is None:
    print(f"\nNo intent match → use default: {intent_router.default_model}")
    response = router.converse(
        messages=[{"role": "user", "content": [{"text": "What's the weather?"}]}],
    )
    print(f"  Smart router picked: {response['routing_decision'].selected_model}")


# ═══════════════════════════════════════════════════════════════════
# Example 5: Full pipeline — intent → cache → smart router
# ═══════════════════════════════════════════════════════════════════

from bedrock_smart_router.semantic_cache import SemanticCache, SemanticCacheConfig

cache = SemanticCache(
    config=SemanticCacheConfig(enabled=True, threshold=0.90),
    region="us-west-2",
)


def full_pipeline(query: str) -> dict:
    """Semantic router → semantic cache → smart router."""
    match = intent_router.route(query)
    intent = match.route_name if match else "default"

    cached = cache.get(query)
    if cached is not None:
        print(f"  [{intent}] Cache HIT: '{query[:40]}'")
        return cached

    response = router.converse(
        messages=[{"role": "user", "content": [{"text": query}]}],
        routing=RoutingConfig(preferred_model=match.model) if match else None,
    )

    cache.put(query, response)
    model = response["routing_decision"].selected_model
    print(f"  [{intent}] Cache MISS → {model}: '{query[:40]}'")
    return response


print(f"\nFull pipeline:")
full_pipeline("Write a Python class for a linked list")
full_pipeline("Create a linked list implementation in Python")  # Cache hit
full_pipeline("What's the average salary in the dataset?")
full_pipeline("Calculate the mean salary from this data")        # Cache hit
