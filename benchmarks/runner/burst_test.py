# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

#!/usr/bin/env python3
"""Burst/concurrency test: compares single-model boto3 vs Smart Router under load.

Tests how well the router handles throttling by spreading load across models,
vs direct boto3 which hits a single model's rate limit.

Usage:
    python benchmarks/burst_test.py
    python benchmarks/burst_test.py --levels 10,25,50
    python benchmarks/burst_test.py --model sonnet
"""

import argparse
import json
import os
import sys
import time
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import boto3

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from benchmarks.runner.config import BURST_CONCURRENCY_LEVELS, BURST_PROMPT, MODELS, REGION
from bedrock_smart_router import BedrockRouter, RoutingConfig


def make_request_baseline(client, model_id, prompt):
    """Single baseline request."""
    t_start = time.perf_counter()
    try:
        response = client.converse(
            modelId=model_id,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
        )
        latency = (time.perf_counter() - t_start) * 1000
        return {"success": True, "latency_ms": latency, "model": model_id}
    except Exception as e:
        latency = (time.perf_counter() - t_start) * 1000
        return {
            "success": False,
            "latency_ms": latency,
            "error": str(e),
            "error_type": type(e).__name__,
            "model": model_id,
        }


def make_request_router(router, prompt):
    """Single router request."""
    t_start = time.perf_counter()
    try:
        response = router.converse(
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            routing=RoutingConfig(strategy="balanced"),
        )
        latency = (time.perf_counter() - t_start) * 1000
        decision = response.get("routing_decision")
        return {
            "success": True,
            "latency_ms": latency,
            "model": decision.selected_model if decision else "unknown",
            "fallback_used": decision.fallback_used if decision else False,
        }
    except Exception as e:
        latency = (time.perf_counter() - t_start) * 1000
        return {
            "success": False,
            "latency_ms": latency,
            "error": str(e),
            "error_type": type(e).__name__,
        }


def run_burst(executor_fn, concurrency, prompt, num_requests=None):
    """Run a burst of concurrent requests and collect metrics."""
    num_requests = num_requests or concurrency
    results = []

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(executor_fn, prompt) for _ in range(num_requests)]
        for future in as_completed(futures):
            results.append(future.result())

    # Calculate metrics
    latencies = [r["latency_ms"] for r in results]
    successes = [r for r in results if r["success"]]
    failures = [r for r in results if not r["success"]]

    metrics = {
        "total_requests": len(results),
        "successes": len(successes),
        "failures": len(failures),
        "success_rate": len(successes) / len(results) * 100,
        "latency_p50": round(statistics.median(latencies), 1),
        "latency_p95": round(sorted(latencies)[int(len(latencies) * 0.95)], 1) if latencies else 0,
        "latency_p99": round(sorted(latencies)[int(len(latencies) * 0.99)], 1) if latencies else 0,
        "latency_avg": round(statistics.mean(latencies), 1),
        "latency_min": round(min(latencies), 1),
        "latency_max": round(max(latencies), 1),
    }

    if failures:
        error_types = {}
        for f in failures:
            et = f.get("error_type", "unknown")
            error_types[et] = error_types.get(et, 0) + 1
        metrics["error_types"] = error_types

    # Model distribution (for router)
    models_used = {}
    for r in successes:
        m = r.get("model", "unknown")
        models_used[m] = models_used.get(m, 0) + 1
    if models_used:
        metrics["models_used"] = models_used

    fallbacks = sum(1 for r in successes if r.get("fallback_used"))
    if fallbacks:
        metrics["fallback_count"] = fallbacks

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Burst/concurrency test")
    parser.add_argument("--levels", type=str, default=None,
                        help="Comma-separated concurrency levels (default: 10,25,50)")
    parser.add_argument("--model", type=str, default="sonnet",
                        help="Baseline model to test (default: sonnet)")
    parser.add_argument("--prompt", type=str, default=BURST_PROMPT,
                        help="Prompt to use for burst test")
    args = parser.parse_args()

    levels = [int(x) for x in args.levels.split(",")] if args.levels else BURST_CONCURRENCY_LEVELS
    model_key = args.model
    model_id = MODELS[model_key]["model_id"]
    model_name = MODELS[model_key]["display_name"]

    print(f"Burst Test Configuration:")
    print(f"  Region: {REGION}")
    print(f"  Baseline model: {model_name} ({model_id})")
    print(f"  Concurrency levels: {levels}")
    print(f"  Prompt: {args.prompt[:60]}...")
    print()

    # Setup
    session = boto3.Session(region_name=REGION)
    client = session.client("bedrock-runtime")
    router = BedrockRouter.create({"region": REGION})

    all_results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "region": REGION,
            "baseline_model": model_id,
            "concurrency_levels": levels,
            "prompt": args.prompt,
        },
        "results": [],
    }

    # Warm up
    print("Warming up (1 request each)...")
    make_request_baseline(client, model_id, args.prompt)
    make_request_router(router, args.prompt)
    print()

    for level in levels:
        print(f"{'='*60}")
        print(f"Concurrency Level: {level}")
        print(f"{'='*60}")

        # Baseline
        print(f"\n  Baseline ({model_name})...")
        baseline_fn = lambda p: make_request_baseline(client, model_id, p)
        baseline_metrics = run_burst(baseline_fn, level, args.prompt)
        baseline_metrics["runner"] = f"baseline-{model_key}"
        baseline_metrics["concurrency"] = level
        all_results["results"].append(baseline_metrics)

        print(f"    Success rate: {baseline_metrics['success_rate']:.1f}%")
        print(f"    Latency p50/p95/p99: {baseline_metrics['latency_p50']}/{baseline_metrics['latency_p95']}/{baseline_metrics['latency_p99']} ms")
        if baseline_metrics.get("error_types"):
            print(f"    Errors: {baseline_metrics['error_types']}")

        # Small delay between tests
        time.sleep(2)

        # Router
        print(f"\n  Smart Router (balanced)...")
        router_fn = lambda p: make_request_router(router, p)
        router_metrics = run_burst(router_fn, level, args.prompt)
        router_metrics["runner"] = "router-default"
        router_metrics["concurrency"] = level
        all_results["results"].append(router_metrics)

        print(f"    Success rate: {router_metrics['success_rate']:.1f}%")
        print(f"    Latency p50/p95/p99: {router_metrics['latency_p50']}/{router_metrics['latency_p95']}/{router_metrics['latency_p99']} ms")
        if router_metrics.get("models_used"):
            print(f"    Models used: {router_metrics['models_used']}")
        if router_metrics.get("fallback_count"):
            print(f"    Fallbacks: {router_metrics['fallback_count']}")
        if router_metrics.get("error_types"):
            print(f"    Errors: {router_metrics['error_types']}")

        # Comparison
        print(f"\n  Comparison:")
        sr_diff = router_metrics["success_rate"] - baseline_metrics["success_rate"]
        print(f"    Success rate: Router {'+' if sr_diff >= 0 else ''}{sr_diff:.1f}% vs baseline")
        lat_diff = router_metrics["latency_p50"] - baseline_metrics["latency_p50"]
        print(f"    Latency p50: Router {'+' if lat_diff >= 0 else ''}{lat_diff:.0f}ms vs baseline")

        time.sleep(3)

    # Save results
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(results_dir, f"burst_test_{timestamp}.json")

    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
