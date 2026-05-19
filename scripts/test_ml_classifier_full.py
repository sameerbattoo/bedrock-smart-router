#!/usr/bin/env python3
"""Full test of ML classifier with system prompts, tools, and multi-turn conversations.

Tests classify_request() with realistic Bedrock Converse API payloads.
"""
import time
import sys
sys.path.insert(0, ".")

from bedrock_smart_router.ml_classifier import MLComplexityClassifier
from bedrock_smart_router.request_analyzer import RequestAnalyzer


def msg(role, text):
    return {"role": role, "content": [{"text": text}]}


def main():
    print("=" * 90)
    print("ML Classifier — Full Context Test (system + tools + multi-turn)")
    print("=" * 90)

    clf = MLComplexityClassifier()
    heuristic = RequestAnalyzer()

    # Test cases: (description, messages, system, tool_config, expected)
    test_cases = [
        # ── SIMPLE: basic questions with/without system prompts ──
        (
            "Simple greeting with system prompt",
            [msg("user", "Hi, how are you?")],
            [{"text": "You are a friendly assistant."}],
            None,
            "simple",
        ),
        (
            "Simple factual question",
            [msg("user", "What is Amazon S3?")],
            [{"text": "You are an AWS expert. Answer concisely."}],
            None,
            "simple",
        ),
        (
            "Simple translation",
            [msg("user", "Translate 'good morning' to Japanese")],
            None,
            None,
            "simple",
        ),
        (
            "Simple with tools available (but simple question)",
            [msg("user", "What time is it?")],
            [{"text": "You are a helpful assistant."}],
            {"tools": [{"toolSpec": {"name": "current_time", "description": "Get current time"}}]},
            "simple",
        ),
        (
            "Simple multi-turn follow-up",
            [
                msg("user", "What is DynamoDB?"),
                msg("assistant", "DynamoDB is a fully managed NoSQL database service by AWS."),
                msg("user", "Thanks!"),
            ],
            None,
            None,
            "simple",
        ),

        # ── MODERATE: summarization, explanations, moderate code ──
        (
            "Moderate: explain with system prompt",
            [msg("user", "Explain how Kubernetes pods work and their lifecycle")],
            [{"text": "You are a DevOps engineer. Use clear examples."}],
            None,
            "moderate",
        ),
        (
            "Moderate: code with context",
            [msg("user", "Write a Python function to parse CSV files and handle missing values")],
            [{"text": "You are a senior Python developer. Write production-ready code with error handling."}],
            None,
            "moderate",
        ),
        (
            "Moderate: SQL query with tools",
            [msg("user", "Show me the top 10 customers by revenue this quarter")],
            [{"text": "You are a data analyst. Query the database to answer questions."}],
            {"tools": [{"toolSpec": {"name": "query_database", "description": "Execute SQL queries against the database"}}]},
            "moderate",
        ),
        (
            "Moderate: multi-turn conversation building up",
            [
                msg("user", "I need help with my AWS architecture"),
                msg("assistant", "I'd be happy to help. What services are you using?"),
                msg("user", "We have an ECS cluster with 3 services. Can you explain how to set up service discovery between them?"),
            ],
            [{"text": "You are an AWS solutions architect."}],
            None,
            "moderate",
        ),
        (
            "Moderate: comparison task",
            [msg("user", "Compare REST vs GraphQL APIs. When would you choose one over the other?")],
            [{"text": "You are a senior backend engineer. Keep under 500 words."}],
            None,
            "moderate",
        ),

        # ── COMPLEX: architecture, multi-tool, deep analysis ──
        (
            "Complex: system design with constraints",
            [msg("user", "Design a real-time fraud detection system that processes 1M transactions per second with sub-100ms latency. Include the data pipeline, ML model serving, and alerting.")],
            [{"text": "You are a principal engineer at a fintech company. Design for scale and reliability."}],
            None,
            "complex",
        ),
        (
            "Complex: multi-tool orchestration",
            [msg("user", "Research the latest AWS Lambda pricing changes, create a cost comparison spreadsheet, and generate a migration plan from EC2 to Lambda for our API")],
            [{"text": "You are a cloud architect with access to documentation and analysis tools."}],
            {"tools": [
                {"toolSpec": {"name": "search_docs", "description": "Search AWS documentation"}},
                {"toolSpec": {"name": "calculator", "description": "Perform calculations"}},
                {"toolSpec": {"name": "file_write", "description": "Write files to disk"}},
            ]},
            "complex",
        ),
        (
            "Complex: code with architecture",
            [msg("user", "Implement a distributed rate limiter using Redis that supports sliding window, token bucket, and leaky bucket algorithms. Include cluster mode support and graceful degradation.")],
            [{"text": "You are a distributed systems engineer. Write production-grade code with tests."}],
            None,
            "complex",
        ),
        (
            "Complex: multi-turn building to complex task",
            [
                msg("user", "I'm building a multi-tenant SaaS platform"),
                msg("assistant", "Great! What's the tech stack and scale you're targeting?"),
                msg("user", "Python/FastAPI, PostgreSQL, targeting 10K tenants. I need you to design the complete data isolation strategy, tenant provisioning pipeline, and billing integration with Stripe. Include the database schema, API design, and deployment architecture on AWS."),
            ],
            [{"text": "You are a SaaS platform architect. Design for enterprise-grade multi-tenancy."}],
            {"tools": [
                {"toolSpec": {"name": "diagram", "description": "Generate architecture diagrams"}},
                {"toolSpec": {"name": "file_write", "description": "Write code files"}},
            ]},
            "complex",
        ),
        (
            "Complex: security audit",
            [msg("user", "Perform a comprehensive security audit of this IAM policy and suggest improvements for least-privilege access. Consider cross-account access patterns, service control policies, and permission boundaries.\n\n{\"Version\": \"2012-10-17\", \"Statement\": [{\"Effect\": \"Allow\", \"Action\": \"*\", \"Resource\": \"*\"}]}")],
            [{"text": "You are a cloud security architect specializing in AWS IAM."}],
            None,
            "complex",
        ),

        # ── REASONING: proofs, step-by-step logic, mathematical ──
        (
            "Reasoning: mathematical proof",
            [msg("user", "Prove by induction that for all n >= 1, the sum 1 + 2 + ... + n = n(n+1)/2. Then use this to derive the formula for the sum of first n squares.")],
            [{"text": "You are a mathematics professor. Show all steps rigorously."}],
            None,
            "reasoning",
        ),
        (
            "Reasoning: algorithm correctness",
            [msg("user", "Prove that Dijkstra's algorithm correctly finds shortest paths in a graph with non-negative edge weights. Use a loop invariant argument and show the inductive step in detail.")],
            [{"text": "You are an algorithms researcher. Prove correctness formally."}],
            None,
            "reasoning",
        ),
        (
            "Reasoning: distributed systems theory",
            [msg("user", "Prove the FLP impossibility result: in an asynchronous distributed system with even one crash failure, consensus is impossible. Walk through the bivalency argument step by step.")],
            [{"text": "You are a distributed systems theorist."}],
            None,
            "reasoning",
        ),
        (
            "Reasoning: multi-turn building to proof",
            [
                msg("user", "I'm studying computational complexity"),
                msg("assistant", "Great topic! What aspect are you working on?"),
                msg("user", "I need to understand the Cook-Levin theorem. Prove step by step that SAT is NP-complete by showing that any problem in NP can be reduced to SAT in polynomial time. Show the construction of the Boolean formula from the Turing machine computation."),
            ],
            [{"text": "You are a theoretical computer science professor. Be rigorous and formal."}],
            None,
            "reasoning",
        ),
        (
            "Reasoning: logic puzzle with systematic deduction",
            [msg("user", "Think step by step: Five houses in a row, each a different color. Each owner has different nationality, drink, smoke, pet. Given these 15 clues, determine who owns the fish. Show your complete deductive reasoning at each step, explaining why you can eliminate each possibility.")],
            None,
            None,
            "reasoning",
        ),
    ]

    print(f"\n{'Description':<50} {'Expected':<11} {'ML':<11} {'Conf':<7} {'Heur':<11} {'ML✓'} {'ML ms':<7} {'H ms':<7}")
    print("-" * 115)

    ml_correct = 0
    heur_correct = 0
    ml_times = []
    heur_times = []

    for desc, messages, system, tools, expected in test_cases:
        t0 = time.perf_counter()
        ml_label, ml_conf = clf.classify_request(messages, system=system, tool_config=tools)
        ml_time = (time.perf_counter() - t0) * 1000
        ml_times.append(ml_time)

        # Heuristic
        t0 = time.perf_counter()
        analysis = heuristic.analyze(messages, system, tools)
        heur_time = (time.perf_counter() - t0) * 1000
        heur_times.append(heur_time)
        heur_label = analysis.complexity.value

        ml_ok = "✓" if ml_label == expected else "✗"
        if ml_label == expected:
            ml_correct += 1
        if heur_label == expected:
            heur_correct += 1

        print(f"{desc[:49]:<50} {expected:<11} {ml_label:<11} {ml_conf:<7.3f} {heur_label:<11} {ml_ok}  {ml_time:<7.3f} {heur_time:<7.3f}")

    n = len(test_cases)
    print(f"\n{'=' * 115}")
    print(f"ML Accuracy:        {ml_correct}/{n} ({100*ml_correct/n:.1f}%)")
    print(f"Heuristic Accuracy: {heur_correct}/{n} ({100*heur_correct/n:.1f}%)")
    print(f"\nML Timing:        first={ml_times[0]:.2f}ms | avg (after load)={sum(ml_times[1:])/(n-1):.3f}ms | min={min(ml_times[1:]):.3f}ms | max={max(ml_times[1:]):.3f}ms")
    print(f"Heuristic Timing: avg={sum(heur_times)/n:.3f}ms | min={min(heur_times):.3f}ms | max={max(heur_times):.3f}ms")
    print(f"{'=' * 115}")


if __name__ == "__main__":
    main()
