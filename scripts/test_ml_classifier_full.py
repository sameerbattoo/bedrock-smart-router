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

        # ── Demo templates: Simple ──
        (
            "Demo: What is Amazon S3 in one sentence?",
            [msg("user", "What is Amazon S3 in one sentence?")],
            [{"text": "Respond in markdown format."}],
            None,
            "simple",
        ),
        (
            "Demo: Define serverless computing",
            [msg("user", "Define serverless computing.")],
            [{"text": "Respond in markdown format."}],
            None,
            "simple",
        ),
        (
            "Demo: Three types of cloud computing",
            [msg("user", "What are the three main types of cloud computing services?")],
            [{"text": "Respond in markdown format."}],
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

        # ── Demo templates: Medium ──
        (
            "Demo: Python decorators with retry",
            [msg("user", "Explain how Python decorators work. Include a practical example of a retry decorator with exponential backoff.")],
            [{"text": "You are a senior Python developer. Use markdown with code blocks. Keep under 500 words."}],
            None,
            "moderate",
        ),
        (
            "Demo: SQL query with CTEs",
            [msg("user", "Write a SQL query to find the top 5 customers by total revenue in the last 90 days, including their most purchased product category. Use CTEs for clarity.")],
            [{"text": "You are a data engineer. Use markdown code blocks. Keep under 500 words."}],
            None,
            "moderate",
        ),
        (
            "Demo: Dockerfile multi-stage",
            [msg("user", "Write a Dockerfile for a Python FastAPI application with multi-stage build, non-root user, and health check endpoint.")],
            [{"text": "You are a DevOps engineer. Use markdown with code blocks. Keep under 500 words."}],
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

        # ── Demo templates: Complex ──
        (
            "Demo: Fraud detection system design",
            [msg("user", "Design a real-time fraud detection system that processes 1 million transactions per second with sub-100ms latency. Include the data pipeline architecture, ML model serving strategy, feature store design, and alerting system.")],
            [{"text": "You are a principal engineer. Use markdown with headings and bullet points. Keep under 800 words."}],
            None,
            "complex",
        ),
        (
            "Demo: B-tree implementation",
            [msg("user", "Implement a B-tree in Python with insert, search, and range query operations. The tree should support configurable order (minimum degree). Include proper node splitting and rebalancing. Add type hints and docstrings.")],
            [{"text": "You are a computer science professor. Use markdown code blocks. Keep under 150 lines."}],
            None,
            "complex",
        ),
        (
            "Demo: Zero-trust security architecture",
            [msg("user", "Design a zero-trust security architecture for a multi-account AWS organization. Cover identity federation, network segmentation, data encryption, secrets management, and incident response automation.")],
            [{"text": "You are a cloud security architect. Use markdown with headings and lists. Keep under 800 words."}],
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

        # ── Demo templates: Reasoning ──
        (
            "Demo: Sum of squares proof + cubes",
            [msg("user", "Prove that for every positive integer n, the sum 1² + 2² + 3² + ... + n² equals n(n+1)(2n+1)/6. Then derive the closed-form formula for the sum of cubes 1³ + 2³ + ... + n³ and prove it by induction step by step.")],
            [{"text": "You are a mathematics professor. Think through each step systematically. Prove your answer rigorously using formal logic. Show all intermediate steps. Keep under 1000 words."}],
            None,
            "reasoning",
        ),
        (
            "Demo: LIS algorithm design",
            [msg("user", "Design an algorithm to find the longest increasing subsequence in an array of n integers. Compare and contrast the brute force O(2^n), dynamic programming O(n²), and patience sorting O(n log n) approaches. For each, prove the time complexity, explain why it works, and analyze the space trade-offs.")],
            [{"text": "You are an algorithms researcher. Analyze each approach systematically, evaluate trade-offs, and reason through the complexity analysis step by step. Prove correctness. Keep under 1000 words."}],
            None,
            "reasoning",
        ),
        (
            "Demo: CAP theorem flash sales",
            [msg("user", "A global e-commerce platform needs to handle flash sales with 10x traffic spikes while maintaining strong consistency for inventory counts. Analyze step by step: Why can't you have both strong consistency and high availability during a network partition? Evaluate three approaches (pessimistic locking, optimistic concurrency with CRDTs, saga pattern) and prove which guarantees each provides.")],
            [{"text": "You are a distributed systems architect. Reason through each design decision systematically, analyze the pros and cons of each approach, and explain why certain trade-offs are unavoidable. Think step by step. Keep under 1000 words."}],
            None,
            "reasoning",
        ),
        (
            "Demo: Einstein's logic puzzle",
            [msg("user", "Five houses in a row are painted different colors. Each owner has a different nationality, drinks a different beverage, smokes a different brand, and keeps a different pet. Given: The Brit lives in the red house. The Swede keeps dogs. The Dane drinks tea. The green house is left of the white house. The green house owner drinks coffee. The Pall Mall smoker keeps birds. The yellow house owner smokes Dunhill. The middle house owner drinks milk. The Norwegian lives in the first house. The Blend smoker lives next to the cat owner. The horse owner lives next to the Dunhill smoker. The Blue Master smoker drinks beer. The German smokes Prince. The Norwegian lives next to the blue house. The Blend smoker has a neighbor who drinks water. Who keeps the fish? Show your complete reasoning.")],
            [{"text": "You are a logic and reasoning expert. Work through this problem step by step, showing your deductive reasoning at each stage. Explain why you can eliminate each possibility. Keep under 1000 words."}],
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
