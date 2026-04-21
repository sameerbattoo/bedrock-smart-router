"""Semantic Cache & Intent Routing — embedding-based intelligence.

These are optional modules that use Bedrock Titan Embeddings to add
semantic understanding to caching and routing.  Unlike the default
exact-match cache and heuristic complexity classifier, these use
vector similarity to match requests by meaning, not by exact text.

When to use:
  - Semantic cache: customer support, FAQ bots, any workload where
    users ask the same thing in different words
  - Semantic router: when different types of queries should go to
    different specialized models

Cost: ~$0.0001 per embedding call, ~50-100ms latency overhead.

These modules call bedrock-runtime:InvokeModel with Titan Embeddings
and are NOT wired into the router's converse() flow — you use them
directly alongside the router.

Demonstrates:
  - Semantic cache: store and retrieve by meaning
  - Semantic router: route by intent to specialized models
  - Combining both with the router
"""

from bedrock_smart_router import BedrockRouter

router = BedrockRouter.create()


# ═══════════════════════════════════════════════════════════════════
# Example 1: Semantic Cache — cache by meaning, not exact text
# ═══════════════════════════════════════════════════════════════════
# The exact-match cache only hits when the request is identical.
# The semantic cache hits when the request has the same MEANING.
#
# "How do I reset my password?" → cache miss (first time)
# "I forgot my password, help"  → cache HIT (same meaning, 0.96 similarity)

from bedrock_smart_router.semantic_cache import SemanticCache, SemanticCacheConfig

semantic_cache = SemanticCache(
    config=SemanticCacheConfig(
        enabled=True,
        threshold=0.92,          # Cosine similarity threshold (0.0-1.0)
        embedding_model="amazon.titan-embed-text-v2:0",
        max_entries=5000,
        ttl_seconds=3600,
    ),
    region="us-west-2",
)

# First request — cache miss, call the router
query1 = "How do I reset my password?"
cached = semantic_cache.get(query1)
if cached is None:
    response = router.converse(
        messages=[{"role": "user", "content": [{"text": query1}]}],
    )
    semantic_cache.put(query1, response)
    print(f"Query 1: cache MISS → called Bedrock")

# Second request — different words, same meaning → cache HIT
query2 = "I forgot my password and need to change it"
cached = semantic_cache.get(query2)
if cached is not None:
    print(f"Query 2: cache HIT (semantic match)")
    print(f"  Hit rate: {semantic_cache.hit_rate:.0%}")
else:
    print(f"Query 2: cache MISS (similarity below threshold)")


# ═══════════════════════════════════════════════════════════════════
# Example 2: Semantic Router — route by intent to specialized models
# ═══════════════════════════════════════════════════════════════════
# Define routes with example utterances.  The router embeds the
# incoming query and matches it to the closest route.

from bedrock_smart_router.semantic_router import (
    SemanticRoute,
    SemanticRouter,
)

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
    default_model="us.amazon.nova-lite-v1:0",  # Fallback if no route matches
    region="us-west-2",
)

# Route a code question
match = intent_router.route("Help me fix this Python bug in my sort function")
if match:
    print(f"\nCode query → route={match.route_name}, model={match.model}")
    print(f"  Score: {match.score}, matched: '{match.matched_example}'")

# Route a creative question
match = intent_router.route("Write me a short story about a robot")
if match:
    print(f"Creative query → route={match.route_name}, model={match.model}")

# Route a data question
match = intent_router.route("What's the average revenue per quarter?")
if match:
    print(f"Data query → route={match.route_name}, model={match.model}")

# No match — falls back to default
match = intent_router.route("What's the weather today?")
if match is None:
    print(f"No match → use default model: {intent_router.default_model}")


# ═══════════════════════════════════════════════════════════════════
# Example 3: Combine semantic router with the Bedrock Smart Router
# ═══════════════════════════════════════════════════════════════════
# Use the semantic router to pick the model, then pass it to the
# smart router via preferred_family or a direct model override.

from bedrock_smart_router import RoutingConfig

query = "Write a recursive fibonacci function in Python"

# Step 1: Semantic router picks the intent
match = intent_router.route(query)

# Step 2: Use the matched model with the smart router
if match:
    response = router.converse(
        messages=[{"role": "user", "content": [{"text": query}]}],
        routing=RoutingConfig(
            # The semantic router suggested this model's family
            preferred_family=match.model.split(".")[1],  # "anthropic"
        ),
    )
    d = response["routing_decision"]
    print(f"\nCombined: intent={match.route_name} → model={d.selected_model}")
    print(f"  Cost: ${d.actual_cost:.6f}")


# ═══════════════════════════════════════════════════════════════════
# Example 4: Semantic cache + smart router — full pipeline
# ═══════════════════════════════════════════════════════════════════
# Check semantic cache first, then route if miss.

def smart_converse(query: str) -> dict:
    """Semantic cache → smart router → cache store."""
    # Check semantic cache
    cached = semantic_cache.get(query)
    if cached is not None:
        print(f"  Semantic cache HIT for: '{query[:50]}...'")
        return cached

    # Cache miss — route and call Bedrock
    response = router.converse(
        messages=[{"role": "user", "content": [{"text": query}]}],
    )

    # Store in semantic cache for future similar queries
    semantic_cache.put(query, response)
    print(f"  Semantic cache MISS → stored for: '{query[:50]}...'")
    return response


print(f"\nFull pipeline:")
smart_converse("How do I create an S3 bucket?")
smart_converse("What are the steps to make a new S3 bucket?")  # Semantic hit
smart_converse("Tell me about creating buckets in S3")          # Semantic hit
