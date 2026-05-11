#!/usr/bin/env python3
"""Benchmark runner: compares baseline models (direct boto3) vs Smart Router.

Usage:
    python benchmarks/run_benchmark.py                    # Run all
    python benchmarks/run_benchmark.py --category text_to_sql
    python benchmarks/run_benchmark.py --runner sonnet
    python benchmarks/run_benchmark.py --limit 10         # First 10 prompts per category
    python benchmarks/run_benchmark.py --runner router-default --category code_generation
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

import boto3

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmarks.runner.config import (
    ALL_RUNNERS,
    MODELS,
    REGION,
    ROUTER_STRATEGIES,
)

from bedrock_smart_router import BedrockRouter, RoutingConfig


def load_prompts(category=None):
    """Load prompts from JSON files."""
    prompts_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "generated")
    all_prompts = []
    for filename in sorted(os.listdir(prompts_dir)):
        if not filename.endswith(".json"):
            continue
        cat = filename.replace(".json", "")
        if category and cat != category:
            continue
        with open(os.path.join(prompts_dir, filename)) as f:
            prompts = json.load(f)
            all_prompts.extend(prompts)
    return all_prompts


def build_messages(prompt):
    """Build Bedrock converse API messages from a prompt dict."""
    user_text = prompt["user_prompt"]
    if prompt.get("context"):
        user_text = f"{prompt['context']}\n\n{prompt['user_prompt']}"
    messages = [{"role": "user", "content": [{"text": user_text}]}]
    system = [{"text": prompt["system_prompt"]}] if prompt.get("system_prompt") else None
    return messages, system


def run_baseline(client, model_id, messages, system):
    """Run a single prompt through a baseline model via boto3."""
    kwargs = {"modelId": model_id, "messages": messages}
    if system:
        kwargs["system"] = system

    t_start = time.perf_counter()
    response = client.converse(**kwargs)
    latency_ms = (time.perf_counter() - t_start) * 1000

    usage = response.get("usage", {})
    output_text = ""
    if response.get("output", {}).get("message", {}).get("content"):
        output_text = response["output"]["message"]["content"][0].get("text", "")

    return {
        "response_text": output_text,
        "latency_ms": round(latency_ms, 1),
        "input_tokens": usage.get("inputTokens", 0),
        "output_tokens": usage.get("outputTokens", 0),
        "model_used": model_id,
        "stop_reason": response.get("stopReason", ""),
    }


def run_router(router, strategy, messages, system):
    """Run a single prompt through the Smart Router."""
    t_start = time.perf_counter()
    response = router.converse(
        messages=messages,
        system=system,
        routing=RoutingConfig(strategy=strategy),
    )
    latency_ms = (time.perf_counter() - t_start) * 1000

    decision = response.get("routing_decision")
    output_text = ""
    if response.get("output", {}).get("message", {}).get("content"):
        output_text = response["output"]["message"]["content"][0].get("text", "")

    return {
        "response_text": output_text,
        "latency_ms": round(latency_ms, 1),
        "input_tokens": decision.input_tokens if decision else 0,
        "output_tokens": decision.output_tokens if decision else 0,
        "model_used": decision.selected_model if decision else "unknown",
        "actual_cost": decision.actual_cost if decision else 0,
        "complexity": decision.complexity_detected if decision else "unknown",
        "fallback_used": decision.fallback_used if decision else False,
        "stop_reason": decision.stop_reason if decision else "",
    }


def estimate_cost(model_id, input_tokens, output_tokens):
    """Estimate cost for baseline models (approximate pricing)."""
    # Approximate per-1M-token pricing (input/output)
    pricing = {
        "us.anthropic.claude-sonnet-4-6": (3.0, 15.0),
        "us.anthropic.claude-haiku-4-5-20251001-v1:0": (0.80, 4.0),
        "us.amazon.nova-pro-v1:0": (0.80, 3.20),
        "us.anthropic.claude-opus-4-7": (15.0, 75.0),
    }
    rates = pricing.get(model_id, (3.0, 15.0))
    cost = (input_tokens * rates[0] + output_tokens * rates[1]) / 1_000_000
    return round(cost, 6)


def main():
    parser = argparse.ArgumentParser(description="Run benchmark comparison")
    parser.add_argument("--category", type=str, help="Run only this category")
    parser.add_argument("--runner", type=str, help="Run only this runner")
    parser.add_argument("--limit", type=int, help="Limit prompts per category")
    parser.add_argument("--output", type=str, help="Output file path")
    args = parser.parse_args()

    # Load prompts
    prompts = load_prompts(args.category)
    if args.limit:
        # Limit per category
        by_cat = {}
        for p in prompts:
            by_cat.setdefault(p["category"], []).append(p)
        prompts = []
        for cat_prompts in by_cat.values():
            prompts.extend(cat_prompts[:args.limit])

    print(f"Loaded {len(prompts)} prompts")

    # Determine runners
    runners = ALL_RUNNERS
    if args.runner:
        runners = [args.runner]

    # Setup clients
    session = boto3.Session(region_name=REGION)
    client = session.client("bedrock-runtime")
    router = BedrockRouter.create({"region": REGION})

    # Results storage
    results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "region": REGION,
            "total_prompts": len(prompts),
            "runners": runners,
        },
        "results": [],
    }

    total_runs = len(prompts) * len(runners)
    completed = 0

    for runner in runners:
        print(f"\n{'='*60}")
        print(f"Runner: {runner}")
        print(f"{'='*60}")

        for i, prompt in enumerate(prompts):
            completed += 1
            print(f"  [{completed}/{total_runs}] {prompt['id']} ({prompt['category']}/{prompt['difficulty']})...", end=" ", flush=True)

            messages, system = build_messages(prompt)
            result_entry = {
                "prompt_id": prompt["id"],
                "category": prompt["category"],
                "difficulty": prompt["difficulty"],
                "runner": runner,
            }

            try:
                if runner in MODELS:
                    # Baseline model
                    model_id = MODELS[runner]["model_id"]
                    run_result = run_baseline(client, model_id, messages, system)
                    run_result["actual_cost"] = estimate_cost(
                        model_id, run_result["input_tokens"], run_result["output_tokens"]
                    )
                elif runner in ROUTER_STRATEGIES:
                    # Smart Router
                    strategy = ROUTER_STRATEGIES[runner]["strategy"]
                    run_result = run_router(router, strategy, messages, system)
                else:
                    print(f"SKIP (unknown runner)")
                    continue

                result_entry.update(run_result)
                result_entry["success"] = True
                print(f"OK ({run_result['latency_ms']:.0f}ms, {run_result['model_used']})")

            except Exception as e:
                result_entry["success"] = False
                result_entry["error"] = str(e)
                result_entry["error_type"] = type(e).__name__
                print(f"FAIL ({type(e).__name__}: {str(e)[:80]})")

            results["results"].append(result_entry)

    # Save results
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(results_dir, exist_ok=True)

    if args.output:
        output_path = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(results_dir, f"benchmark_{timestamp}.json")

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Results saved to: {output_path}")
    print(f"Total runs: {len(results['results'])}")
    successes = sum(1 for r in results["results"] if r.get("success"))
    print(f"Successes: {successes}/{len(results['results'])}")


if __name__ == "__main__":
    main()
