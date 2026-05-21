"""Example 31: ML-based complexity classifier.

Demonstrates the optional ML classifier that provides more accurate
complexity detection than the default heuristic — especially for
complex and reasoning tasks.

Requirements:
    pip install bedrock-smart-router[ml]

The ML classifier:
- Uses TF-IDF + Logistic Regression (pure numpy inference)
- 80% accuracy vs 63% for the heuristic on diverse test sets
- 0.1ms inference time (after 20ms model load)
- 2.4MB model file, numpy-only runtime dependency
"""

import sys
sys.path.insert(0, ".")

from bedrock_smart_router import BedrockRouter, RoutingConfig

# ── Example 1: Enable ML classifier via config ──────────────────────
# Just add "classifier": "ml" to your router config

router = BedrockRouter.create({
    "region": "us-west-2",
    "classifier": "ml",  # Use ML instead of heuristic
})

print("=" * 60)
print("ML Classifier — Enabled via config")
print("=" * 60)

# Test with different complexity levels
test_prompts = [
    ("What is S3?", "simple"),
    ("Compare REST and GraphQL APIs with pros and cons", "moderate"),
    ("Design a distributed system for real-time fraud detection at 1M TPS with sub-100ms latency", "complex"),
    ("Prove by induction that the sum of first n squares equals n(n+1)(2n+1)/6", "reasoning"),
]

for prompt, expected in test_prompts:
    response = router.converse(
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        routing=RoutingConfig(explain=True),
    )
    d = response["routing_decision"]
    # Extract response text
    resp_text = ""
    for block in response.get("output", {}).get("message", {}).get("content", []):
        if "text" in block:
            resp_text = block["text"]

    print(f"\n  Prompt: {prompt[:60]}...")
    print(f"  Expected: {expected}")
    print(f"  Detected: {d.complexity_detected} (score: {d.complexity_score:.3f})")
    print(f"  Model:    {d.selected_model}")
    print(f"  TTFT:     {d.ttft_ms or 0:.0f}ms")
    print(f"  Latency:  {d.latency_ms:.0f}ms")
    print(f"  Routing:  {d.routing_decision_ms:.2f}ms")
    print(f"  Cost:     ${d.actual_cost:.6f}")
    print(f"  Response: {resp_text[:500]}{'...' if len(resp_text) > 120 else ''}")

    # ML explain shows probabilities
    if d.explanation and d.explanation.get("complexity", {}).get("classifier") == "ml":
        probs = d.explanation["complexity"].get("probabilities", {})
        print(f"  Probs:    {', '.join(f'{k}={v:.2f}' for k, v in sorted(probs.items(), key=lambda x: -x[1]))}")


# ── Example 2: Direct classifier usage (without router) ────────────
print("\n" + "=" * 60)
print("ML Classifier — Direct Usage")
print("=" * 60)

from bedrock_smart_router.ml_classifier import MLComplexityClassifier

clf = MLComplexityClassifier()

# Simple classification
label, confidence = clf.classify("What is Python?")
print(f"\n  classify('What is Python?') → {label} ({confidence:.2f})")

# Full request classification (with system prompt + tools)
label, confidence = clf.classify_request(
    messages=[{"role": "user", "content": [{"text": "Design a microservices architecture"}]}],
    system=[{"text": "You are a principal engineer. Design for scale."}],
    tool_config={"tools": [{"toolSpec": {"name": "diagram", "description": "Generate diagrams"}}]},
)
print(f"  classify_request(system+user+tools) → {label} ({confidence:.2f})")

# All probabilities
probs = clf.predict_proba_all("Prove that sqrt(2) is irrational")
print(f"  predict_proba_all('Prove sqrt(2)...') → {probs}")

# Batch classification
results = clf.classify_batch([
    "Hello!",
    "Write a REST API with auth",
    "Design a distributed consensus protocol",
    "Prove the halting problem is undecidable",
])
print(f"\n  Batch results:")
for text, (label, conf) in zip(["Hello!", "REST API", "Consensus protocol", "Halting problem"], results):
    print(f"    {text:<25} → {label:<10} ({conf:.2f})")


# ── Example 3: Compare ML vs Heuristic ─────────────────────────────
print("\n" + "=" * 60)
print("ML vs Heuristic Comparison")
print("=" * 60)

from bedrock_smart_router.request_analyzer import RequestAnalyzer

heuristic = RequestAnalyzer(classifier="heuristic")
ml_analyzer = RequestAnalyzer(classifier="ml")

prompts = [
    "What is AWS Lambda?",
    "Implement a B-tree with insert, search, and range queries",
    "Prove the FLP impossibility theorem step by step",
]

for prompt in prompts:
    msgs = [{"role": "user", "content": [{"text": prompt}]}]
    h_result = heuristic.analyze(msgs)
    m_result = ml_analyzer.analyze(msgs)
    match = "✓" if h_result.complexity == m_result.complexity else "≠"
    print(f"\n  {prompt[:55]}...")
    print(f"    Heuristic: {h_result.complexity.value} (score={h_result.complexity_score:.3f})")
    print(f"    ML:        {m_result.complexity.value} (score={m_result.complexity_score:.3f}) {match}")
