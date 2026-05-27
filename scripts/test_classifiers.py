#!/usr/bin/env python3
"""Test script for comparing Heuristic and ML classifiers side-by-side.

Runs both classifiers against a comprehensive set of test cases covering:
- Simple queries (greetings, factual lookups)
- Moderate queries (summarization, code tasks)
- Complex queries (architecture, multi-step analysis)
- Reasoning queries (proofs, logical deduction)
- Edge cases (empty, very short, ambiguous, adversarial)
- System prompt floor scenarios
- Multi-turn conversations
- Tool-use contexts

Usage:
    python scripts/test_classifiers.py
"""

import time
from dataclasses import dataclass
from typing import Any

from bedrock_smart_router.heuristic_classifier import HeuristicClassifier
from bedrock_smart_router.ml_classifier import MLComplexityClassifier
from bedrock_smart_router.complexity_classifier import ComplexityClassifier, COMPLEXITY_LABELS


@dataclass
class TestCase:
    prompt: str
    expected: str  # Expected complexity level
    system: str | None = None
    tool_config: dict | None = None
    category: str = ""
    notes: str = ""


# ═══════════════════════════════════════════════════════════════════
# Test Cases
# ═══════════════════════════════════════════════════════════════════

TEST_CASES = [
    # ── Simple ──────────────────────────────────────────────────
    TestCase("Hello", "simple", category="Simple", notes="Greeting"),
    TestCase("Hi, how are you?", "simple", category="Simple", notes="Greeting"),
    TestCase("What is Python?", "simple", category="Simple", notes="Factual lookup"),
    TestCase("What time is it?", "simple", category="Simple", notes="Factual"),
    TestCase("Tell me a joke", "simple", category="Simple", notes="Simple request"),
    TestCase("Thanks!", "simple", category="Simple", notes="Acknowledgment"),
    TestCase("Who is the president?", "simple", category="Simple", notes="Factual"),
    TestCase("What is 2+2?", "simple", category="Simple", notes="Trivial math"),
    TestCase("Translate hello to Spanish", "simple", category="Simple", notes="Translation"),
    TestCase("What does API stand for?", "simple", category="Simple", notes="Definition"),

    # ── Moderate ────────────────────────────────────────────────
    TestCase("Summarize the key points of this article about climate change", "moderate",
             category="Moderate", notes="Summarization"),
    TestCase("Write a Python function to sort a list using merge sort", "moderate",
             category="Moderate", notes="Code generation"),
    TestCase("Explain how REST APIs work with examples", "moderate",
             category="Moderate", notes="Explanation with examples"),
    TestCase("Compare AWS Lambda vs EC2 for web hosting", "moderate",
             category="Moderate", notes="Comparison"),
    TestCase("Write a SQL query to find the top 5 customers by revenue", "moderate",
             category="Moderate", notes="SQL generation"),
    TestCase("Create a Python decorator for caching function results", "moderate",
             category="Moderate", notes="Code pattern"),
    TestCase("Explain the difference between TCP and UDP with use cases", "moderate",
             category="Moderate", notes="Technical explanation"),

    # ── Complex ─────────────────────────────────────────────────
    TestCase(
        "Design a distributed microservices architecture with event sourcing and CQRS patterns "
        "for a high-throughput e-commerce platform handling 10K orders per second",
        "complex", category="Complex", notes="Architecture design"),
    TestCase(
        "Implement a B-tree with insertion, deletion, and rebalancing in Python "
        "with full test coverage and O(log n) guarantees",
        "complex", category="Complex", notes="Data structure implementation"),
    TestCase(
        "Analyze the trade-offs between consistency and availability in distributed databases "
        "considering CAP theorem, PACELC, and real-world systems like DynamoDB vs Spanner",
        "complex", category="Complex", notes="Deep analysis"),
    TestCase(
        "Write a Terraform module for a multi-region active-active deployment on AWS "
        "with Route53 failover, Aurora Global Database, and CloudFront distribution",
        "complex", category="Complex", notes="Infrastructure as code"),
    TestCase(
        "Design a real-time fraud detection system using Kinesis, SageMaker, and Step Functions "
        "that processes 1M transactions per hour with sub-second latency",
        "complex", category="Complex", notes="System design"),

    # ── Reasoning ───────────────────────────────────────────────
    TestCase(
        "Prove that the halting problem is undecidable using a diagonal argument. "
        "Show each step of the proof and explain why the contradiction arises.",
        "reasoning", category="Reasoning", notes="Mathematical proof"),
    TestCase(
        "Solve this step by step: If all A are B, some B are C, and no C are D, "
        "what can we conclude about the relationship between A and D? "
        "Prove your answer using formal logic.",
        "reasoning", category="Reasoning", notes="Logical deduction"),
    TestCase(
        "Analyze the time complexity of the following recursive algorithm step by step, "
        "derive the recurrence relation, solve it using the Master theorem, "
        "and prove the tight bound: T(n) = 3T(n/4) + n*log(n)",
        "reasoning", category="Reasoning", notes="Algorithm analysis"),

    # ── Edge Cases ──────────────────────────────────────────────
    TestCase("", "simple", category="Edge", notes="Empty string"),
    TestCase("a", "simple", category="Edge", notes="Single character"),
    TestCase("?", "simple", category="Edge", notes="Single punctuation"),
    TestCase("ok", "simple", category="Edge", notes="Two characters"),
    TestCase("yes", "simple", category="Edge", notes="Affirmative"),
    TestCase("hmm", "simple", category="Edge", notes="Filler word"),
    TestCase("do it", "simple", category="Edge", notes="Vague command"),
    TestCase("help", "simple", category="Edge", notes="Single word request"),
    TestCase("x" * 100, "simple", category="Edge", notes="Repeated character (100x)"),
    TestCase("lol what even is this", "simple", category="Edge", notes="Informal/ambiguous"),
    TestCase(
        "You are is flying and what dates?", "simple",
        category="Edge", notes="Grammatically broken (should NOT be reasoning)"),
    TestCase(
        "asdf jkl; qwerty uiop", "simple",
        category="Edge", notes="Random keyboard mashing"),

    # ── System Prompt Floor ─────────────────────────────────────
    TestCase(
        "Hi", "moderate",
        system="You are a senior solutions architect. Analyze complex distributed systems and provide detailed trade-off analysis.",
        category="Floor", notes="Simple msg + complex system prompt → floor upgrade"),
    TestCase(
        "What is S3?", "moderate",
        system="You are a senior solutions architect. Analyze complex distributed systems and provide detailed trade-off analysis.",
        category="Floor", notes="Simple question + complex system prompt"),
    TestCase(
        "Hello", "simple",
        system="You are a helpful assistant.",
        category="Floor", notes="Simple msg + simple system prompt → no floor"),
    TestCase(
        "Design a system", "complex",
        system="You are a helpful assistant.",
        category="Floor", notes="Complex msg + simple system prompt → msg dominates"),

    # ── Tool Use Context ────────────────────────────────────────
    TestCase(
        "What is the weather?", "simple",
        tool_config={"tools": [{"toolSpec": {"name": "get_weather", "description": "Get weather data"}}]},
        category="Tools", notes="Simple query with tools attached"),
    TestCase(
        "Search for recent papers on transformer architectures and summarize the key findings",
        "moderate",
        tool_config={"tools": [
            {"toolSpec": {"name": "search_papers", "description": "Search academic papers"}},
            {"toolSpec": {"name": "summarize", "description": "Summarize text"}},
        ]},
        category="Tools", notes="Moderate query with tools"),
]


# ═══════════════════════════════════════════════════════════════════
# Test Runner
# ═══════════════════════════════════════════════════════════════════

def is_acceptable(got: str, expected: str) -> bool:
    """Check if classification is acceptable (within ±1 level)."""
    order = {"simple": 0, "moderate": 1, "complex": 2, "reasoning": 3}
    return abs(order.get(got, 0) - order.get(expected, 0)) <= 1


def run_classifier(clf: ComplexityClassifier, name: str, test_cases: list[TestCase]) -> dict:
    """Run all test cases through a classifier and report results."""
    results = {"exact": 0, "acceptable": 0, "failed": 0, "total": len(test_cases)}
    failures = []
    timings = []

    for tc in test_cases:
        msgs = [{"role": "user", "content": [{"text": tc.prompt}]}]
        system = [{"text": tc.system}] if tc.system else None

        t0 = time.perf_counter()
        label, conf = clf.classify_request(msgs, system=system, tool_config=tc.tool_config)
        elapsed_us = (time.perf_counter() - t0) * 1_000_000
        timings.append(elapsed_us)

        if label == tc.expected:
            results["exact"] += 1
        elif is_acceptable(label, tc.expected):
            results["acceptable"] += 1
        else:
            results["failed"] += 1
            failures.append((tc, label, conf))

    results["exact_pct"] = results["exact"] / results["total"] * 100
    results["acceptable_pct"] = (results["exact"] + results["acceptable"]) / results["total"] * 100
    results["avg_latency_us"] = sum(timings) / len(timings)
    results["p99_latency_us"] = sorted(timings)[int(len(timings) * 0.99)]
    results["failures"] = failures
    return results


def print_report(name: str, results: dict):
    """Print a formatted report for a classifier."""
    print(f"\n{'═' * 70}")
    print(f"  {name}")
    print(f"{'═' * 70}")
    print(f"  Exact match:  {results['exact']}/{results['total']} ({results['exact_pct']:.1f}%)")
    print(f"  Acceptable:   {results['exact'] + results['acceptable']}/{results['total']} ({results['acceptable_pct']:.1f}%)")
    print(f"  Failed:       {results['failed']}/{results['total']}")
    print(f"  Avg latency:  {results['avg_latency_us']:.0f}µs")
    print(f"  P99 latency:  {results['p99_latency_us']:.0f}µs")

    if results["failures"]:
        print(f"\n  Failures:")
        for tc, got, conf in results["failures"]:
            print(f"    ✗ [{tc.category}] \"{tc.prompt[:60]}{'...' if len(tc.prompt) > 60 else ''}\"")
            print(f"      Expected: {tc.expected}, Got: {got} (conf={conf:.4f}) — {tc.notes}")


def print_comparison_table(heuristic_clf, ml_clf, test_cases):
    """Print a side-by-side comparison table."""
    print(f"\n{'═' * 100}")
    print(f"  Side-by-Side Comparison")
    print(f"{'═' * 100}")
    print(f"  {'Prompt':<50} {'Expected':<10} {'Heuristic':<12} {'ML':<12} {'Match'}")
    print(f"  {'-'*50} {'-'*10} {'-'*12} {'-'*12} {'-'*5}")

    for tc in test_cases:
        msgs = [{"role": "user", "content": [{"text": tc.prompt}]}]
        system = [{"text": tc.system}] if tc.system else None

        h_label, h_conf = heuristic_clf.classify_request(msgs, system=system, tool_config=tc.tool_config)
        m_label, m_conf = ml_clf.classify_request(msgs, system=system, tool_config=tc.tool_config)

        h_ok = "✓" if is_acceptable(h_label, tc.expected) else "✗"
        m_ok = "✓" if is_acceptable(m_label, tc.expected) else "✗"
        agree = "=" if h_label == m_label else "≠"

        short_prompt = tc.prompt[:48] + ".." if len(tc.prompt) > 48 else tc.prompt
        print(f"  {short_prompt:<50} {tc.expected:<10} {h_label:<12} {m_label:<12} {agree}")


def test_predict_proba(clf: ComplexityClassifier, name: str):
    """Test that predict_proba_all returns valid distributions."""
    print(f"\n  {name} — predict_proba_all validation:")
    test_texts = ["Hello", "Write a Python function", "Design a distributed system"]
    all_valid = True
    for text in test_texts:
        probs = clf.predict_proba_all(text)
        keys_ok = set(probs.keys()) == set(COMPLEXITY_LABELS)
        sum_ok = abs(sum(probs.values()) - 1.0) < 0.02
        all_positive = all(v >= 0 for v in probs.values())
        valid = keys_ok and sum_ok and all_positive
        if not valid:
            all_valid = False
            print(f"    ✗ \"{text[:40]}\": keys={keys_ok}, sum={sum(probs.values()):.3f}, positive={all_positive}")
    if all_valid:
        print(f"    ✓ All distributions valid (4 classes, sum≈1.0, all≥0)")


def test_custom_classifier():
    """Test that a custom classifier works with the interface."""
    print(f"\n{'═' * 70}")
    print(f"  Custom Classifier Test")
    print(f"{'═' * 70}")

    class WordCountClassifier(ComplexityClassifier):
        """Simple word-count-based classifier for testing."""
        def classify(self, text: str) -> tuple[str, float]:
            words = len(text.split())
            if words > 50:
                return ("reasoning", 0.8)
            elif words > 20:
                return ("complex", 0.75)
            elif words > 8:
                return ("moderate", 0.7)
            return ("simple", 0.9)

        def predict_proba_all(self, text: str) -> dict[str, float]:
            label, conf = self.classify(text)
            probs = {l: 0.05 for l in COMPLEXITY_LABELS}
            probs[label] = conf
            total = sum(probs.values())
            return {k: v / total for k, v in probs.items()}

    clf = WordCountClassifier()

    # Test basic classification
    label, conf = clf.classify("Hello world")
    assert label == "simple", f"Expected simple, got {label}"
    print(f"  ✓ classify('Hello world') = {label} ({conf:.2f})")

    # Test classify_request inherits floor logic
    msgs = [{"role": "user", "content": [{"text": "Hi"}]}]
    system = [{"text": "You are a complex distributed systems architect analyzing trade-offs"}]
    label, conf = clf.classify_request(msgs, system=system)
    print(f"  ✓ classify_request with floor = {label} ({conf:.4f})")

    # Test predict_proba_all
    probs = clf.predict_proba_all("Hello")
    assert abs(sum(probs.values()) - 1.0) < 0.01
    print(f"  ✓ predict_proba_all sums to 1.0")

    print(f"  ✓ Custom classifier works with inherited floor logic")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║  Complexity Classifier Comparison Test                              ║")
    print("║  Heuristic (15-dimension keyword) vs ML (TF-IDF + LogReg)          ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")

    heuristic = HeuristicClassifier()
    ml = MLComplexityClassifier()

    # Run both classifiers
    h_results = run_classifier(heuristic, "Heuristic", TEST_CASES)
    m_results = run_classifier(ml, "ML", TEST_CASES)

    # Print reports
    print_report("Heuristic Classifier (15 dimensions, keyword-based)", h_results)
    print_report("ML Classifier (TF-IDF + LogReg, 35K training samples)", m_results)

    # Side-by-side comparison
    print_comparison_table(heuristic, ml, TEST_CASES)

    # Probability distribution validation
    print(f"\n{'═' * 70}")
    print(f"  Probability Distribution Validation")
    print(f"{'═' * 70}")
    test_predict_proba(heuristic, "Heuristic")
    test_predict_proba(ml, "ML")

    # Custom classifier test
    test_custom_classifier()

    # Summary
    print(f"\n{'═' * 70}")
    print(f"  Summary")
    print(f"{'═' * 70}")
    print(f"  {'Metric':<30} {'Heuristic':<15} {'ML':<15}")
    print(f"  {'-'*30} {'-'*15} {'-'*15}")
    print(f"  {'Exact match':<30} {h_results['exact_pct']:.1f}%{'':<10} {m_results['exact_pct']:.1f}%")
    print(f"  {'Acceptable (±1 level)':<30} {h_results['acceptable_pct']:.1f}%{'':<10} {m_results['acceptable_pct']:.1f}%")
    print(f"  {'Avg latency':<30} {h_results['avg_latency_us']:.0f}µs{'':<10} {m_results['avg_latency_us']:.0f}µs")
    print(f"  {'P99 latency':<30} {h_results['p99_latency_us']:.0f}µs{'':<10} {m_results['p99_latency_us']:.0f}µs")
    print()
