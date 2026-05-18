#!/usr/bin/env python3
"""Test script for the ML-based complexity classifier.

Compares ML classifier predictions against the heuristic RequestAnalyzer
across diverse prompts spanning all 4 complexity categories.
"""

import time
import sys
sys.path.insert(0, ".")

from bedrock_smart_router.ml_classifier import MLComplexityClassifier
from bedrock_smart_router.request_analyzer import RequestAnalyzer


def make_messages(text: str) -> list[dict]:
    """Wrap text into Bedrock Converse message format."""
    return [{"role": "user", "content": [{"text": text}]}]


def main():
    print("=" * 80)
    print("ML Complexity Classifier — Test & Comparison")
    print("=" * 80)

    # Initialize classifiers
    ml_classifier = MLComplexityClassifier()
    heuristic = RequestAnalyzer()

    # Test prompts across all 4 categories
    test_prompts = [
        # ── Simple prompts ──
        ("What is the capital of France?", "simple"),
        ("Hello, how are you?", "simple"),
        ("Translate 'hello' to Spanish", "simple"),
        ("What year was Python created?", "simple"),
        ("Define machine learning in one sentence", "simple"),

        # ── Moderate prompts ──
        ("Summarize the key differences between SQL and NoSQL databases", "moderate"),
        ("Write a short Python function to reverse a string", "moderate"),
        ("Explain how HTTP cookies work and why they are used", "moderate"),
        ("List the pros and cons of microservices architecture", "moderate"),
        ("What are the main differences between TCP and UDP protocols?", "moderate"),

        # ── Complex prompts ──
        ("Design a scalable real-time notification system that handles 1M concurrent users, "
         "supports multiple channels (push, email, SMS), includes rate limiting, "
         "and provides exactly-once delivery guarantees. Include the database schema, "
         "message queue architecture, and failure recovery mechanisms.", "complex"),
        ("Implement a complete REST API in Python with FastAPI that includes "
         "authentication, rate limiting, pagination, caching with Redis, "
         "comprehensive error handling, and OpenAPI documentation", "complex"),
        ("Analyze the trade-offs between event sourcing and traditional CRUD "
         "for a financial trading platform. Consider consistency, auditability, "
         "performance under load, and disaster recovery strategies.", "complex"),
        ("Write a distributed cache implementation with consistent hashing, "
         "replication, and automatic failover. Include the hash ring, "
         "virtual nodes, and gossip protocol for membership.", "complex"),
        ("Design a CI/CD pipeline for a multi-service application deployed on "
         "Kubernetes with canary deployments, automated rollbacks, security scanning, "
         "and infrastructure as code using Terraform and ArgoCD.", "complex"),

        # ── Reasoning prompts ──
        ("Prove that the halting problem is undecidable using a diagonalization "
         "argument. Then explain step by step why this implies that no general "
         "algorithm can determine if an arbitrary program will terminate.", "reasoning"),
        ("Analyze step by step: A company has 3 data centers. Each can handle "
         "10K requests/sec. Design a load balancing strategy that minimizes "
         "latency while maintaining 99.99% availability. Prove mathematically "
         "that your approach satisfies the SLA under the given failure model.", "reasoning"),
        ("Derive the time complexity of the following recursive algorithm step by step, "
         "prove it using the Master Theorem, then optimize it using dynamic programming "
         "and prove the optimality of your solution: T(n) = 3T(n/2) + n^2", "reasoning"),
        ("Think through this systematically: Given a distributed system with "
         "eventual consistency, prove that it's impossible to achieve both "
         "linearizability and availability during a network partition. "
         "Then derive the optimal consistency level for a banking application.", "reasoning"),
        ("Evaluate and compare three different approaches to solving the "
         "traveling salesman problem: branch and bound, genetic algorithms, "
         "and ant colony optimization. For each, derive the expected time "
         "complexity, prove correctness bounds, and analyze trade-offs "
         "between solution quality and computational cost.", "reasoning"),
    ]

    # ── Run ML classifier with timing ──
    print(f"\n{'Prompt (truncated)':<55} {'Expected':<12} {'ML Pred':<12} {'Conf':<8} {'Heuristic':<12} {'Match'}")
    print("-" * 110)

    ml_times = []
    heuristic_times = []
    ml_correct = 0
    heuristic_correct = 0

    for prompt, expected in test_prompts:
        # ML classification with timing
        t0 = time.perf_counter()
        ml_label, ml_conf = ml_classifier.classify(prompt)
        ml_time = time.perf_counter() - t0
        ml_times.append(ml_time)

        # Heuristic classification with timing
        t0 = time.perf_counter()
        analysis = heuristic.analyze(make_messages(prompt))
        heuristic_time = time.perf_counter() - t0
        heuristic_times.append(heuristic_time)
        heuristic_label = analysis.complexity.value

        # Check correctness
        ml_match = "✓" if ml_label == expected else "✗"
        if ml_label == expected:
            ml_correct += 1
        if heuristic_label == expected:
            heuristic_correct += 1

        # Display
        truncated = prompt[:52] + "..." if len(prompt) > 55 else prompt
        print(f"{truncated:<55} {expected:<12} {ml_label:<12} {ml_conf:<8.4f} {heuristic_label:<12} {ml_match}")

    # ── Summary statistics ──
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    n = len(test_prompts)
    print(f"\nML Classifier Accuracy:        {ml_correct}/{n} ({100*ml_correct/n:.1f}%)")
    print(f"Heuristic Classifier Accuracy: {heuristic_correct}/{n} ({100*heuristic_correct/n:.1f}%)")

    print(f"\nML Classifier Timing:")
    print(f"  First call (includes load): {ml_times[0]*1000:.2f} ms")
    print(f"  Average (after load):       {sum(ml_times[1:])*1000/(n-1):.3f} ms")
    print(f"  Min:                        {min(ml_times[1:])*1000:.3f} ms")
    print(f"  Max:                        {max(ml_times[1:])*1000:.3f} ms")

    print(f"\nHeuristic Classifier Timing:")
    print(f"  Average:                    {sum(heuristic_times)*1000/n:.3f} ms")
    print(f"  Min:                        {min(heuristic_times)*1000:.3f} ms")
    print(f"  Max:                        {max(heuristic_times)*1000:.3f} ms")

    # ── Detailed probability view for a few examples ──
    print("\n" + "=" * 80)
    print("DETAILED PROBABILITIES (selected prompts)")
    print("=" * 80)

    detail_prompts = [
        "What is Python?",
        "Write a function to sort a list",
        "Design a distributed system with consensus protocol",
        "Prove step by step that P != NP implies one-way functions exist",
    ]

    for prompt in detail_prompts:
        probs = ml_classifier.predict_proba_all(prompt)
        label, conf = ml_classifier.classify(prompt)
        print(f"\n  \"{prompt[:70]}...\"" if len(prompt) > 70 else f"\n  \"{prompt}\"")
        print(f"  Prediction: {label} ({conf:.4f})")
        print(f"  Probabilities: {', '.join(f'{k}={v:.4f}' for k, v in sorted(probs.items(), key=lambda x: -x[1]))}")


if __name__ == "__main__":
    main()
