"""Benchmark configuration — models, region, strategies, and judge prompts."""

# ── Region ──────────────────────────────────────────────────────────
REGION = "us-west-2"

# ── Baseline Models ─────────────────────────────────────────────────
MODELS = {
    "sonnet": {
        "model_id": "us.anthropic.claude-sonnet-4-6",
        "display_name": "Claude Sonnet 4.6",
    },
    "haiku": {
        "model_id": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
        "display_name": "Claude Haiku 4.5",
    },
    "nova-pro": {
        "model_id": "us.amazon.nova-pro-v1:0",
        "display_name": "Amazon Nova Pro",
    },
    "opus": {
        "model_id": "us.anthropic.claude-opus-4-7",
        "display_name": "Claude Opus 4.7",
    },
}

# ── Judge Model ─────────────────────────────────────────────────────
JUDGE_MODEL_ID = "us.anthropic.claude-sonnet-4-6"

# ── Smart Router Strategies ─────────────────────────────────────────
ROUTER_STRATEGIES = {
    "router-default": {
        "strategy": "balanced",
        "display_name": "Smart Router (Default/Balanced)",
    },
    "router-budget": {
        "strategy": "cost-optimized",
        "display_name": "Smart Router (Budget)",
    },
    "router-quality": {
        "strategy": "quality-optimized",
        "display_name": "Smart Router (Quality)",
    },
}

# ── All Runners (baselines + router strategies) ─────────────────────
ALL_RUNNERS = list(MODELS.keys()) + list(ROUTER_STRATEGIES.keys())

# ── Prompt Categories ───────────────────────────────────────────────
PROMPT_CATEGORIES = [
    "text_to_sql",
    "document_extraction",
    "log_analysis",
    "anomaly_detection",
    "code_generation",
    "summarization",
]

# ── Burst Test Config ───────────────────────────────────────────────
BURST_CONCURRENCY_LEVELS = [10, 25, 50]
BURST_PROMPT = "Explain what Amazon S3 is in one sentence."

# ── Judge Prompts ───────────────────────────────────────────────────
JUDGE_SYSTEM_PROMPT = """You are an expert evaluator. Score the following AI response on a scale of 1-10.
Consider: accuracy, completeness, relevance, clarity, and helpfulness.

System prompt given to the AI: {system_prompt}

User prompt: {user_prompt}

AI Response: {response}

IMPORTANT: Respond with ONLY a raw JSON object, no markdown fences, no code blocks, no extra text.
Your entire response must be exactly: {{"score": <1-10>, "reasoning": "<brief explanation>"}}"""

JUDGE_SYSTEM_PROMPT_WITH_ANSWER = """You are an expert evaluator. Score the following AI response on a scale of 1-10.
Consider: accuracy, completeness, relevance, clarity, and helpfulness.
Pay special attention to how well the response matches the expected answer.

System prompt given to the AI: {system_prompt}

User prompt: {user_prompt}

Expected answer: {expected_answer}

AI Response: {response}

IMPORTANT: Respond with ONLY a raw JSON object, no markdown fences, no code blocks, no extra text.
Your entire response must be exactly: {{"score": <1-10>, "reasoning": "<brief explanation>"}}"""
