"""Semantic Router — route queries to specialized models by intent.

Define routes with example utterances.  The router embeds the incoming
query and matches it to the closest route using cosine similarity.
Each route maps to a specific model optimized for that type of task.

Use case: code questions → code-tuned model, creative writing →
creative model, data analysis → analytical model.

Cost: ~$0.0001 per embedding call, ~50-100ms latency overhead.
The embedding is computed once per query (not per route).

Demonstrates:
  - Defining routes with example utterances
  - Routing queries to specialized models
  - Handling no-match with a default model
  - Combining semantic router with the smart router
  - Per-route threshold tuning
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
# Each route has a name, a target model, and example queries that
# represent the intent.  More examples = better matching accuracy.

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
                "Write unit tests for",
                "Optimize this SQL query",
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
                "Write a song about",
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
                "Build a dashboard for",
            ],
            threshold=0.80,
        ),
        SemanticRoute(
            name="aws",
            model="us.anthropic.claude-sonnet-4-6",
            examples=[
                "How do I configure a VPC",
                "Set up an S3 bucket policy",
                "Create a Lambda function",
                "Configure IAM roles",
                "Deploy with CloudFormation",
                "Set up an ECS cluster",
            ],
            threshold=0.82,
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
    "How do I set up cross-account IAM access?",
    "What's the weather today?",  # No match — falls back to default
]

print("Semantic routing:")
for query in queries:
    match = intent_router.route(query)
    if match:
        print(f"  '{query[:55]:55s}' → {match.route_name:8s} ({match.model})")
        print(f"    Score: {match.score:.2f}, matched: '{match.matched_example}'")
    else:
        print(f"  '{query[:55]:55s}' → DEFAULT  ({intent_router.default_model})")


# ═══════════════════════════════════════════════════════════════════
# Example 3: Combine semantic router with the smart router
# ═══════════════════════════════════════════════════════════════════
# The semantic router picks the intent and target model family.
# The smart router handles the actual invocation with all its
# features (fallbacks, caching, metrics, etc.).

query = "Write a recursive fibonacci function in Python"
match = intent_router.route(query)

if match:
    # Extract the model family from the matched route
    family = match.model.split(".")[1]  # e.g. "anthropic"
    response = router.converse(
        messages=[{"role": "user", "content": [{"text": query}]}],
        routing=RoutingConfig(preferred_family=family),
    )
    d = response["routing_decision"]
    print(f"\nCombined routing:")
    print(f"  Intent: {match.route_name} (score={match.score:.2f})")
    print(f"  Model:  {d.selected_model}")
    print(f"  Cost:   ${d.actual_cost:.6f}")
else:
    # No intent match — let the smart router decide
    response = router.converse(
        messages=[{"role": "user", "content": [{"text": query}]}],
    )
    print(f"\nNo intent match, smart router picked: {response['routing_decision'].selected_model}")


# ═══════════════════════════════════════════════════════════════════
# Example 4: Per-route threshold tuning
# ═══════════════════════════════════════════════════════════════════
# Each route can have its own threshold.  Code routes need higher
# thresholds (code is specific), creative routes can be lower.
#
# Recommended thresholds:
#   Code:     0.82–0.90 (specific, avoid false matches)
#   Creative: 0.75–0.85 (broader, more flexible matching)
#   Data:     0.80–0.88 (moderate specificity)
#   AWS:      0.82–0.90 (service-specific terminology)

print(f"\nThreshold guide:")
print(f"  Code routes:     0.82–0.90 (specific)")
print(f"  Creative routes: 0.75–0.85 (flexible)")
print(f"  Data routes:     0.80–0.88 (moderate)")
print(f"  AWS routes:      0.82–0.90 (service-specific)")


# ═══════════════════════════════════════════════════════════════════
# Example 5: Full pipeline — semantic router + semantic cache
# ═══════════════════════════════════════════════════════════════════
# Route by intent, check semantic cache, call Bedrock if miss.

from bedrock_smart_router.semantic_cache import SemanticCache, SemanticCacheConfig

cache = SemanticCache(
    config=SemanticCacheConfig(enabled=True, threshold=0.90),
    region="us-west-2",
)


def full_pipeline(query: str) -> dict:
    """Semantic router → semantic cache → smart router."""
    # Step 1: Route by intent
    match = intent_router.route(query)
    family = match.model.split(".")[1] if match else None
    intent = match.route_name if match else "default"

    # Step 2: Check semantic cache
    cached = cache.get(query)
    if cached is not None:
        print(f"  [{intent}] Cache HIT: '{query[:40]}'")
        return cached

    # Step 3: Call smart router with intent-based family preference
    response = router.converse(
        messages=[{"role": "user", "content": [{"text": query}]}],
        routing=RoutingConfig(preferred_family=family) if family else None,
    )

    # Step 4: Store in semantic cache
    cache.put(query, response)
    model = response["routing_decision"].selected_model
    print(f"  [{intent}] Cache MISS → {model}: '{query[:40]}'")
    return response


print(f"\nFull pipeline (intent → cache → route):")
full_pipeline("Write a Python class for a linked list")
full_pipeline("Create a linked list implementation in Python")  # Cache hit
full_pipeline("What's the average salary in the dataset?")
full_pipeline("Calculate the mean salary from this data")        # Cache hit
