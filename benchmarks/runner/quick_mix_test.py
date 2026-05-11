#!/usr/bin/env python3
"""Quick test: 1 simple + 1 medium + 1 complex prompt across key runners + judge.
Runs: sonnet, haiku, nova-pro, router-default, router-quality
Then judges all responses and prints a comparison table.
"""
import sys
import os
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import boto3
from bedrock_smart_router import BedrockRouter, RoutingConfig
from benchmarks.runner.config import MODELS, REGION, JUDGE_MODEL_ID, JUDGE_SYSTEM_PROMPT_WITH_ANSWER

# Load 1 prompt per difficulty from text_to_sql
prompts_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "generated", "text_to_sql.json")
all_prompts = json.load(open(prompts_path))

test_prompts = []
for diff in ["simple", "medium", "complex"]:
    for p in all_prompts:
        if p["difficulty"] == diff:
            test_prompts.append(p)
            break

print(f"Test prompts: {len(test_prompts)}")
for p in test_prompts:
    print(f"  {p['id']} ({p['difficulty']}): {p['user_prompt'][:70]}...")

# Setup
session = boto3.Session(region_name=REGION)
client = session.client("bedrock-runtime")

# Runners to test
runners = {
    "sonnet": MODELS["sonnet"]["model_id"],
    "haiku": MODELS["haiku"]["model_id"],
    "nova-pro": MODELS["nova-pro"]["model_id"],
}
router_strategies = {
    "router-default": "balanced",
    "router-quality": "quality-optimized",
}

results = []


def build_msgs(prompt):
    user_text = prompt["user_prompt"]
    if prompt.get("context"):
        user_text = f"{prompt['context']}\n\n{prompt['user_prompt']}"
    return [{"role": "user", "content": [{"text": user_text}]}], [{"text": prompt["system_prompt"]}]


def judge(prompt, response_text):
    """Score with Sonnet judge."""
    judge_prompt = JUDGE_SYSTEM_PROMPT_WITH_ANSWER.format(
        system_prompt=prompt["system_prompt"],
        user_prompt=prompt["user_prompt"],
        expected_answer=prompt.get("expected_answer", "N/A"),
        response=response_text,
    )
    resp = client.converse(
        modelId=JUDGE_MODEL_ID,
        messages=[{"role": "user", "content": [{"text": judge_prompt}]}],
    )
    text = resp["output"]["message"]["content"][0]["text"].strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0]
    try:
        return json.loads(text)["score"]
    except Exception:
        return 0


# Run baselines
print("\n" + "=" * 70)
print("RUNNING BASELINES")
print("=" * 70)

for runner_name, model_id in runners.items():
    for p in test_prompts:
        messages, system = build_msgs(p)
        print(f"  {runner_name}/{p['difficulty']}...", end=" ", flush=True)
        try:
            t0 = time.perf_counter()
            resp = client.converse(modelId=model_id, messages=messages, system=system)
            latency = (time.perf_counter() - t0) * 1000
            usage = resp.get("usage", {})
            text = resp["output"]["message"]["content"][0]["text"]
            results.append({
                "runner": runner_name, "difficulty": p["difficulty"], "prompt_id": p["id"],
                "latency_ms": round(latency), "model": model_id,
                "input_tokens": usage.get("inputTokens", 0),
                "output_tokens": usage.get("outputTokens", 0),
                "response_text": text,
            })
            print(f"{latency:.0f}ms")
        except Exception as e:
            print(f"FAIL: {e}")
            results.append({"runner": runner_name, "difficulty": p["difficulty"], "prompt_id": p["id"], "error": str(e)})

# Run router (fresh instance, no cache)
print("\n" + "=" * 70)
print("RUNNING SMART ROUTER")
print("=" * 70)

for strat_name, strategy in router_strategies.items():
    router = BedrockRouter.create({"region": REGION, "cache": {"enabled": False}})
    for p in test_prompts:
        messages, system = build_msgs(p)
        print(f"  {strat_name}/{p['difficulty']}...", end=" ", flush=True)
        try:
            t0 = time.perf_counter()
            resp = router.converse(messages=messages, system=system, routing=RoutingConfig(strategy=strategy))
            latency = (time.perf_counter() - t0) * 1000
            decision = resp.get("routing_decision")
            text = resp["output"]["message"]["content"][0]["text"]
            results.append({
                "runner": strat_name, "difficulty": p["difficulty"], "prompt_id": p["id"],
                "latency_ms": round(latency), "model": decision.selected_model if decision else "?",
                "input_tokens": decision.input_tokens if decision else 0,
                "output_tokens": decision.output_tokens if decision else 0,
                "actual_cost": decision.actual_cost if decision else 0,
                "complexity": decision.complexity_detected if decision else "?",
                "response_text": text,
            })
            print(f"{latency:.0f}ms -> {decision.selected_model if decision else '?'}")
        except Exception as e:
            print(f"FAIL: {e}")
            results.append({"runner": strat_name, "difficulty": p["difficulty"], "prompt_id": p["id"], "error": str(e)})

# Judge all responses
print("\n" + "=" * 70)
print("JUDGING RESPONSES")
print("=" * 70)

for r in results:
    if "response_text" not in r:
        r["score"] = 0
        continue
    prompt = next(p for p in test_prompts if p["id"] == r["prompt_id"])
    print(f"  Judging {r['runner']}/{r['difficulty']}...", end=" ", flush=True)
    try:
        score = judge(prompt, r["response_text"])
        r["score"] = score
        print(f"{score}/10")
    except Exception as e:
        r["score"] = 0
        print(f"ERROR: {e}")
    time.sleep(0.3)

# Print comparison table
print("\n" + "=" * 70)
print("RESULTS: LATENCY + COST + ACCURACY")
print("=" * 70)
print(f"\n{'Runner':<18} {'Difficulty':<10} {'Score':<8} {'Latency':<10} {'Model Selected':<30}")
print("-" * 76)

for diff in ["simple", "medium", "complex"]:
    for r in sorted([x for x in results if x["difficulty"] == diff], key=lambda x: x["runner"]):
        model_short = r.get("model", "?").split(".")[-1][:28] if r.get("model") else "error"
        score_str = f"{r.get('score', 0)}/10" if "score" in r else "err"
        lat_str = f"{r.get('latency_ms', 0)}ms" if "latency_ms" in r else "err"
        print(f"{r['runner']:<18} {diff:<10} {score_str:<8} {lat_str:<10} {model_short}")
    print()

# Summary
print("=" * 70)
print("SUMMARY BY RUNNER (averaged across difficulties)")
print("=" * 70)
runner_names = list(runners.keys()) + list(router_strategies.keys())
print(f"\n{'Runner':<18} {'Avg Score':<12} {'Avg Latency':<14} {'Notes'}")
print("-" * 70)
for rn in runner_names:
    rr = [x for x in results if x["runner"] == rn and "score" in x]
    if not rr:
        continue
    avg_score = sum(x["score"] for x in rr) / len(rr)
    avg_lat = sum(x.get("latency_ms", 0) for x in rr) / len(rr)
    models = set(x.get("model", "?").split(".")[-1][:20] for x in rr)
    print(f"{rn:<18} {avg_score:<12.1f} {avg_lat:<14.0f} models: {', '.join(models)}")

# Save full results
output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "quick_mix_judged.json")
with open(output_path, "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nFull results saved to: {output_path}")
