"""Auto-Extracting Semantic Cache — automatic intent + variable extraction.

The standard semantic cache matches queries by embedding similarity, but
the caller must manually extract and pass variables to prevent false hits
on parameterised queries. The auto-extracting mode uses a cheap Bedrock
model (Nova Micro) to automatically decompose each query into a canonical
intent and variables — no manual extraction needed.

Demonstrates:
  1. Manual mode (existing) — caller passes variables explicitly
  2. Auto-extract single-turn — variables extracted automatically
  3. Auto-extract multi-turn — conversation resolved into single query
  4. Auto-extract with converse() — full router pipeline
  5. Auto-extract with converse_stream() — streaming pipeline
  6. Auto-extract with Strands agent — SmartRouterModel integration

Requirements:
    pip install bedrock-smart-router[strands]
"""

from bedrock_smart_router import BedrockRouter
from bedrock_smart_router.semantic_cache import SemanticCache, SemanticCacheConfig


# ═══════════════════════════════════════════════════════════════════
# Example 1: Manual mode (existing behavior)
# ═══════════════════════════════════════════════════════════════════
# The caller must extract variables and pass them explicitly.
# This works but requires the developer to know which parts of the
# query are parameters.

print("=" * 60)
print("1. Manual variable-aware caching")
print("=" * 60)

manual_cache = SemanticCache(
    config=SemanticCacheConfig(
        threshold=0.75,  # Lower threshold for raw query matching (no auto-extract)
        embedding_model="amazon.titan-embed-text-v2:0",
    ),
    region="us-west-2",
)

# Store with explicit variables
manual_cache.put(
    "How do I reset my password?",
    {"answer": "Go to Settings > Security > Reset Password"},
    variables={"action": "reset"},
)

# Same intent + same variables → HIT
hit = manual_cache.get(
    "I forgot my password, how can I reset it?",
    variables={"action": "reset"},
)
print(f"  Same variables:      {'HIT' if hit else 'MISS'}")

# Same intent + different variables → MISS
hit = manual_cache.get(
    "How do I change my password?",
    variables={"action": "change"},
)
print(f"  Different variables: {'HIT' if hit else 'MISS'}")
# Note: Manual mode embeds the raw query text, so rephrased queries
# may miss even with the same variables. This is why auto-extract
# mode exists — it normalises the intent before embedding.


# ═══════════════════════════════════════════════════════════════════
# Example 2: Auto-extract single-turn
# ═══════════════════════════════════════════════════════════════════
# The cache automatically extracts intent and variables using a cheap
# Bedrock model (Nova Micro). No manual variable passing needed.

print(f"\n{'=' * 60}")
print("2. Auto-extract single-turn")
print("=" * 60)

auto_cache = SemanticCache(
    config=SemanticCacheConfig(
        threshold=0.85,
        auto_extract=True,                            # Enable auto-extraction
        extraction_model="us.amazon.nova-micro-v1:0", # Cheapest model
    ),
    region="us-west-2",
)

# Store — variables extracted automatically
auto_cache.put(
    "Count users by geography for 2026 with sales > $200",
    {"result": "42 users across 5 regions"},
)
print("  Stored: 'Count users by geography for 2026 with sales > $200'")

# Different wording, same intent + same variables → HIT
hit = auto_cache.get("Show user distribution by geo, year 2026, sales over $200")
print(f"  Same intent+vars:      {'HIT' if hit else 'MISS'}")

# Same intent but different variables → MISS
hit = auto_cache.get("Count users by geography for 2025 with sales > $100")
print(f"  Different variables:   {'HIT' if hit else 'MISS'}")

# No variables at all → different intent
hit = auto_cache.get("What is Amazon S3?")
print(f"  Completely different:  {'HIT' if hit else 'MISS'}")

print(f"  Stats: {auto_cache.stats}")


# ═══════════════════════════════════════════════════════════════════
# Example 3: Auto-extract multi-turn
# ═══════════════════════════════════════════════════════════════════
# A multi-turn conversation is resolved into a single self-contained
# query before extraction. This means a single-turn cached entry can
# match a multi-turn conversation with the same intent.

print(f"\n{'=' * 60}")
print("3. Auto-extract multi-turn")
print("=" * 60)

multi_cache = SemanticCache(
    config=SemanticCacheConfig(
        threshold=0.85,
        auto_extract=True,
        extraction_model="us.amazon.nova-micro-v1:0",
    ),
    region="us-west-2",
)

# Store from a single-turn query
multi_cache.put(
    "Count users by geography for 2026 with sales > $200",
    {"result": "42 users across 5 regions"},
)
print("  Stored single-turn: 'Count users by geography for 2026 with sales > $200'")

# Lookup from a multi-turn conversation with the same intent
messages = [
    {"role": "user", "content": [{"text": "show me the users by geo"}]},
    {"role": "assistant", "content": [{"text": "Here are users distributed by geography: ..."}]},
    {"role": "user", "content": [{"text": "Ok now show me this data for 2026 for overall sales more than $200"}]},
]
hit = multi_cache.get(messages=messages)
print(f"  Multi-turn lookup:     {'HIT' if hit else 'MISS'}")


# ═══════════════════════════════════════════════════════════════════
# Example 3b: Auto-extract with FAISS vector store
# ═══════════════════════════════════════════════════════════════════
# FAISS provides fast approximate nearest-neighbor search, much better
# than brute-force at scale (100K+ entries). Same auto-extract logic,
# just a different vector store backend.
#
# Install: pip install bedrock-smart-router[faiss]

print(f"\n{'=' * 60}")
print("3b. Auto-extract with FAISS backend")
print("=" * 60)

faiss_cache = SemanticCache(
    config=SemanticCacheConfig(
        threshold=0.85,
        auto_extract=True,
        extraction_model="us.amazon.nova-micro-v1:0",
        vector_store_backend="faiss",       # Use FAISS instead of brute-force
        embedding_dimension=1024,           # Must match Titan v2 output
    ),
    region="us-west-2",
)

# Store a few entries
faiss_cache.put("Count users by geo for 2026 with sales > $200",
                {"result": "42 users across 5 regions"})
faiss_cache.put("Show top products for Electronics in Q3 2025",
                {"result": "Top 10 electronics products"})
faiss_cache.put("What is the average order value for 2026?",
                {"result": "$127.50"})

print(f"  Stored 3 entries in FAISS")

# Same intent, same variables → HIT
hit = faiss_cache.get("Show user distribution by geography, year 2026, sales over $200")
print(f"  Same intent+vars (users):    {'HIT' if hit else 'MISS'}")

# Different variables → MISS
hit = faiss_cache.get("Count users by geo for 2025 with sales > $100")
print(f"  Different vars (users):      {'HIT' if hit else 'MISS'}")

# Different topic → MISS
hit = faiss_cache.get("What is Amazon S3?")
print(f"  Different topic:             {'HIT' if hit else 'MISS'}")

# Multi-turn → HIT (resolves to same intent as stored single-turn)
hit = faiss_cache.get(messages=[
    {"role": "user", "content": [{"text": "show me the top products"}]},
    {"role": "assistant", "content": [{"text": "For which category and period?"}]},
    {"role": "user", "content": [{"text": "Electronics, Q3 2025"}]},
])
print(f"  Multi-turn (products):       {'HIT' if hit else 'MISS'}")

print(f"  Backend: {faiss_cache.stats['backend']}, entries: {faiss_cache.stats['entries']}")


# ═══════════════════════════════════════════════════════════════════
# Example 4: Auto-extract with converse()
# ═══════════════════════════════════════════════════════════════════
# Integrate the auto-extracting semantic cache with the smart router.

print(f"\n{'=' * 60}")
print("4. Auto-extract with converse()")
print("=" * 60)

router = BedrockRouter.create({"region": "us-west-2"})
converse_cache = SemanticCache(
    config=SemanticCacheConfig(
        threshold=0.85,
        auto_extract=True,
        extraction_model="us.amazon.nova-micro-v1:0",
    ),
    region="us-west-2",
)


def smart_converse(query: str) -> dict:
    """Semantic cache → smart router → cache store."""
    cached = converse_cache.get(query)
    if cached is not None:
        print(f"  Cache HIT: '{query[:60]}'")
        return cached

    response = router.converse(
        messages=[{"role": "user", "content": [{"text": query}]}],
    )
    converse_cache.put(query, response)
    d = response["routing_decision"]
    print(f"  Cache MISS → {d.selected_model}, cost: ${d.actual_cost:.6f}")
    return response


smart_converse("Count users by geography for 2026 with sales > $200")
smart_converse("Show user distribution by geo, year 2026, sales over $200")  # HIT
smart_converse("Count users by geography for 2025 with sales > $100")        # MISS (different vars)


# ═══════════════════════════════════════════════════════════════════
# Example 5: Auto-extract with converse_stream()
# ═══════════════════════════════════════════════════════════════════

print(f"\n{'=' * 60}")
print("5. Auto-extract with converse_stream()")
print("=" * 60)

stream_cache = SemanticCache(
    config=SemanticCacheConfig(
        threshold=0.70,  # Lower threshold for rephrased queries
        auto_extract=True,
        extraction_model="us.amazon.nova-micro-v1:0",
    ),
    region="us-west-2",
)


def smart_stream(query: str) -> dict | None:
    """Semantic cache → streaming router → cache store."""
    cached = stream_cache.get(query)
    if cached is not None:
        print(f"  Cache HIT: '{query[:60]}'")
        return cached

    full_text = []
    decision = None
    for event in router.converse_stream(
        messages=[{"role": "user", "content": [{"text": query}]}],
    ):
        if "contentBlockDelta" in event:
            delta = event["contentBlockDelta"]["delta"]
            if "text" in delta:
                full_text.append(delta["text"])
        elif "routing_decision" in event:
            decision = event["routing_decision"]

    response = {"text": "".join(full_text)}
    stream_cache.put(query, response)
    if decision:
        print(f"  Cache MISS → {decision.selected_model}")
    return response


smart_stream("How many orders were placed in January 2026?")
smart_stream("What is the order count for January 2026?")  # HIT (same intent + variables)


# ═══════════════════════════════════════════════════════════════════
# Example 6: Auto-extract with Strands agent
# ═══════════════════════════════════════════════════════════════════

print(f"\n{'=' * 60}")
print("6. Auto-extract with Strands agent")
print("=" * 60)

try:
    from strands import Agent
    from bedrock_smart_router.strands_model import SmartRouterModel

    strands_cache = SemanticCache(
        config=SemanticCacheConfig(
                threshold=0.85,
            auto_extract=True,
                extraction_model="us.amazon.nova-micro-v1:0",
        ),
        region="us-west-2",
    )

    model = SmartRouterModel(router_config={"region": "us-west-2"})
    agent = Agent(model=model)

    def agent_with_cache(query: str) -> str:
        """Check semantic cache before calling the agent."""
        cached = strands_cache.get(query)
        if cached is not None:
            print(f"  Cache HIT: '{query[:60]}'")
            return cached["text"]

        response = agent(query)
        text = str(response)
        strands_cache.put(query, {"text": text})
        d = model.last_routing_decision
        print(f"  Cache MISS → {d.selected_model}, cost: ${d.actual_cost:.6f}")
        return text

    agent_with_cache("What is Amazon DynamoDB?")
    agent_with_cache("Tell me about DynamoDB")  # HIT

    # ═══════════════════════════════════════════════════════════════
    # Example 7: Multi-turn Strands agent with semantic cache
    # ═══════════════════════════════════════════════════════════════
    # This is the key scenario for agent chatbots.
    #
    # User1 asks a complete question in one turn → cached.
    # User2 builds up the same question across multiple turns using
    # the Strands agent's native conversation history (agent.messages).
    # The cache resolves User2's multi-turn history into the same
    # intent as User1's single-turn query → HIT.

    print(f"\n{'=' * 60}")
    print("7. Multi-turn Strands agent — chatbot scenario")
    print("=" * 60)

    multi_turn_cache = SemanticCache(
        config=SemanticCacheConfig(
                threshold=0.85,
            auto_extract=True,
                extraction_model="us.amazon.nova-micro-v1:0",
        ),
        region="us-west-2",
    )

    # --- User1: Single-turn, complete question → store in cache ---
    print("  User1 (single-turn):")
    model_u1 = SmartRouterModel(router_config={"region": "us-west-2"}, max_tokens=256)
    agent_u1 = Agent(model=model_u1, system_prompt="Answer concisely in 2-3 sentences.")

    query1 = "Count users distributed by geography for 2026 who have overall sales of more than $200"
    cached = multi_turn_cache.get(query1)
    if cached is None:
        response1 = agent_u1(query1)
        multi_turn_cache.put(query1, {"text": str(response1)})
        d = model_u1.last_routing_decision
        print(f"    Cache MISS → {d.selected_model}, cost: ${d.actual_cost:.6f}")

    # --- User2: Multi-turn conversation via Strands agent ---
    # User2 builds up the same question across 2 turns.
    # The agent maintains conversation history in agent.messages.
    print("  User2 (multi-turn via Strands agent):")
    model_u2 = SmartRouterModel(router_config={"region": "us-west-2"}, max_tokens=256)
    agent_u2 = Agent(model=model_u2, system_prompt="Answer concisely in 2-3 sentences.")

    # Turn 1: vague question
    agent_u2("Show me the users by geography")
    print(f"    Turn 1: asked 'Show me the users by geography'")
    print(f"    Agent history: {len(agent_u2.messages)} messages")

    # Turn 2: refine with specific parameters
    # Before calling the agent, check the cache using the FULL
    # conversation history from agent.messages + the new user message.
    new_query = "Ok now show me this data for 2026 for overall sales more than $200"
    # Build the full message history as the agent would see it
    full_messages = list(agent_u2.messages) + [
        {"role": "user", "content": [{"text": new_query}]},
    ]

    cached = multi_turn_cache.get(messages=full_messages)
    if cached is not None:
        print(f"    Turn 2: Cache HIT — multi-turn matched User1's single-turn! ✅")
        print(f"    (Saved a full Bedrock call for User2)")
    else:
        # Cache miss — let the agent handle it
        response2 = agent_u2(new_query)
        multi_turn_cache.put(messages=agent_u2.messages,
                            response={"text": str(response2)})
        print(f"    Turn 2: Cache MISS")

    # --- User3: Multi-turn with DIFFERENT variables ---
    print("  User3 (multi-turn, different variables):")
    model_u3 = SmartRouterModel(router_config={"region": "us-west-2"}, max_tokens=256)
    agent_u3 = Agent(model=model_u3, system_prompt="Answer concisely in 2-3 sentences.")

    agent_u3("Show me the users by geography")
    full_messages_u3 = list(agent_u3.messages) + [
        {"role": "user", "content": [{"text": "Now filter for 2025 with sales over $500"}]},
    ]
    cached = multi_turn_cache.get(messages=full_messages_u3)
    if cached is None:
        print(f"    Cache MISS — different variables (2025/$500 vs 2026/$200) ✅")
    else:
        print(f"    Cache HIT (unexpected)")

except ImportError:
    print("  Skipped — install strands-agents: pip install bedrock-smart-router[strands]")
