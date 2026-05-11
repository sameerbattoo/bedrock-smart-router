#!/usr/bin/env python3
"""Test the ML classifier-enhanced router vs baselines.

Runs 1 simple + 1 medium + 1 complex prompt through:
- Baselines: Sonnet, Opus, Nova Pro, Haiku (direct boto3)
- Router (keyword heuristic): current default
- Router (ML classifier): uses ONNX model for complexity detection

Compares: latency, cost, accuracy (via judge), and model selection.
"""
import sys
import os
import json
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import boto3
from bedrock_smart_router import BedrockRouter, RoutingConfig
from bedrock_smart_router.request_analyzer import RequestAnalyzer, _extract_text
from benchmarks.runner.config import MODELS, REGION, JUDGE_MODEL_ID, JUDGE_SYSTEM_PROMPT_WITH_ANSWER

# ── ML Classifier (ONNX) ────────────────────────────────────────
import onnxruntime as ort
from tokenizers import Tokenizer

ONNX_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "classifier", "onnx_model")


class MLComplexityClassifier:
    """ONNX-based complexity classifier. Replaces keyword heuristic."""

    MAX_TOKENS = 256

    def __init__(self):
        self.session = ort.InferenceSession(os.path.join(ONNX_DIR, "model.onnx"))
        self.tokenizer = Tokenizer.from_file(os.path.join(ONNX_DIR, "tokenizer.json"))
        self.tokenizer.enable_padding(length=self.MAX_TOKENS)
        self.tokenizer.enable_truncation(max_length=self.MAX_TOKENS)

        with open(os.path.join(ONNX_DIR, "classifier_weights.json")) as f:
            clf_data = json.load(f)
        self.weights = np.array(clf_data["weights"])
        self.bias = np.array(clf_data["bias"])
        self.classes = clf_data["classes"]

    def _build_classify_text(self, system_prompt: str, user_prompt: str, context: str = "") -> str:
        """Build text for classification: prioritize system+user, append context if room.

        Strategy: system_prompt + user_prompt always included first.
        Context (schemas, documents, etc.) appended only if there's token budget left.
        """
        # Priority text: system prompt + user prompt (the actual instruction)
        priority_text = f"{system_prompt}\n\n{user_prompt}".strip()

        if not context:
            return priority_text

        # Check how many tokens the priority text uses
        priority_encoded = self.tokenizer.encode(priority_text)
        priority_tokens = len([t for t in priority_encoded.ids if t != 0])  # non-padding tokens

        # If priority text already fills most of the budget, skip context
        remaining_budget = self.MAX_TOKENS - priority_tokens - 10  # 10 token buffer
        if remaining_budget <= 20:
            return priority_text

        # Append as much context as fits (truncate context, not the question)
        # Rough estimate: 1 token ≈ 4 chars
        max_context_chars = remaining_budget * 4
        truncated_context = context[:max_context_chars]

        return f"{priority_text}\n\n{truncated_context}"

    def classify(self, system_prompt: str = "", user_prompt: str = "", context: str = "", text: str = "") -> tuple:
        """Classify complexity. Returns (label, confidence, probabilities, elapsed_ms).

        Args:
            system_prompt: The system instruction
            user_prompt: The user's question/request
            context: Injected data (schemas, documents) - lower priority
            text: Full text override (if provided, ignores other args)
        """
        t0 = time.perf_counter()

        if text:
            classify_text = text
        else:
            classify_text = self._build_classify_text(system_prompt, user_prompt, context)

        # Tokenize
        encoded = self.tokenizer.encode(classify_text)
        input_ids = np.array([encoded.ids], dtype=np.int64)
        attention_mask = np.array([encoded.attention_mask], dtype=np.int64)

        # Run ONNX model
        output = self.session.run(None, {"input_ids": input_ids, "attention_mask": attention_mask})

        # Mean pooling
        token_embeddings = output[0]  # (1, seq_len, 384)
        mask_expanded = np.expand_dims(attention_mask, -1)
        sum_embeddings = np.sum(token_embeddings * mask_expanded, axis=1)
        sum_mask = np.sum(mask_expanded, axis=1)
        embedding = sum_embeddings / sum_mask

        # Normalize
        norm = np.linalg.norm(embedding, axis=1, keepdims=True)
        embedding = embedding / norm

        # Classify (logistic regression)
        logits = embedding @ self.weights.T + self.bias
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        pred_idx = np.argmax(probs, axis=1)[0]

        elapsed_ms = (time.perf_counter() - t0) * 1000
        label = self.classes[pred_idx]
        confidence = float(probs[0][pred_idx])
        prob_dict = {c: float(p) for c, p in zip(self.classes, probs[0])}

        return label, confidence, prob_dict, elapsed_ms


# ── Test Setup ───────────────────────────────────────────────────

# Load test prompts (1 simple, 1 medium, 1 complex from text_to_sql)
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
    print(f"  {p['id']} ({p['difficulty']}): {p['user_prompt'][:60]}...")

# Setup
session = boto3.Session(region_name=REGION)
client = session.client("bedrock-runtime")
ml_classifier = MLComplexityClassifier()
keyword_analyzer = RequestAnalyzer()

# ── Helper Functions ─────────────────────────────────────────────

def build_msgs(prompt):
    user_text = prompt["user_prompt"]
    if prompt.get("context"):
        user_text = f"{prompt['context']}\n\n{prompt['user_prompt']}"
    return [{"role": "user", "content": [{"text": user_text}]}], [{"text": prompt["system_prompt"]}]


def judge(prompt, response_text):
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


def estimate_cost(model_id, input_tokens, output_tokens):
    pricing = {
        "us.anthropic.claude-sonnet-4-6": (3.0, 15.0),
        "us.anthropic.claude-haiku-4-5-20251001-v1:0": (0.80, 4.0),
        "us.amazon.nova-pro-v1:0": (0.80, 3.20),
        "us.anthropic.claude-opus-4-7": (15.0, 75.0),
        "global.anthropic.claude-opus-4-7": (15.0, 75.0),
        "us.amazon.nova-micro-v1:0": (0.035, 0.14),
        "us.amazon.nova-lite-v1:0": (0.06, 0.24),
        "us.deepseek.r1-v1:0": (1.35, 5.40),
    }
    rates = pricing.get(model_id, (3.0, 15.0))
    return (input_tokens * rates[0] + output_tokens * rates[1]) / 1_000_000


# ── Run Tests ────────────────────────────────────────────────────

results = []

# 1. Show ML classifier predictions
print("\n" + "=" * 70)
print("ML CLASSIFIER PREDICTIONS")
print("=" * 70)
for p in test_prompts:
    messages, system = build_msgs(p)
    label, conf, probs, clf_ms = ml_classifier.classify(
        system_prompt=p.get("system_prompt", ""),
        user_prompt=p.get("user_prompt", ""),
        context=p.get("context", ""),
    )

    # Compare with keyword
    kw_result = keyword_analyzer.analyze(messages, system)

    print(f"\n  {p['id']} (labeled: {p['difficulty']})")
    print(f"    ML classifier: {label} ({conf:.0%}) [{clf_ms:.1f}ms]")
    print(f"    Keyword heuristic: {kw_result.complexity.value} (score: {kw_result.complexity_score:.3f})")
    print(f"    Probabilities: {', '.join(f'{k}={v:.2f}' for k,v in sorted(probs.items()))}")

# 2. Run baselines
print("\n" + "=" * 70)
print("RUNNING BASELINES")
print("=" * 70)

for model_key, model_info in MODELS.items():
    model_id = model_info["model_id"]
    for p in test_prompts:
        messages, system = build_msgs(p)
        print(f"  {model_key}/{p['difficulty']}...", end=" ", flush=True)
        try:
            t0 = time.perf_counter()
            resp = client.converse(modelId=model_id, messages=messages, system=system)
            latency = (time.perf_counter() - t0) * 1000
            usage = resp.get("usage", {})
            text = resp["output"]["message"]["content"][0]["text"]
            cost = estimate_cost(model_id, usage.get("inputTokens", 0), usage.get("outputTokens", 0))
            results.append({
                "runner": model_key, "difficulty": p["difficulty"], "prompt_id": p["id"],
                "latency_ms": round(latency), "model": model_id, "cost": cost,
                "input_tokens": usage.get("inputTokens", 0),
                "output_tokens": usage.get("outputTokens", 0),
                "response_text": text,
            })
            print(f"{latency:.0f}ms")
        except Exception as e:
            print(f"FAIL: {e}")
            results.append({"runner": model_key, "difficulty": p["difficulty"], "prompt_id": p["id"], "error": str(e)})

# 3. Run router with keyword heuristic
print("\n" + "=" * 70)
print("RUNNING ROUTER (KEYWORD HEURISTIC)")
print("=" * 70)

router_kw = BedrockRouter.create({"region": REGION, "cache": {"enabled": False}})
for p in test_prompts:
    messages, system = build_msgs(p)
    print(f"  router-keyword/{p['difficulty']}...", end=" ", flush=True)
    try:
        t0 = time.perf_counter()
        resp = router_kw.converse(messages=messages, system=system, routing=RoutingConfig(strategy="balanced"))
        latency = (time.perf_counter() - t0) * 1000
        decision = resp.get("routing_decision")
        text = resp["output"]["message"]["content"][0]["text"]
        results.append({
            "runner": "router-keyword", "difficulty": p["difficulty"], "prompt_id": p["id"],
            "latency_ms": round(latency), "model": decision.selected_model if decision else "?",
            "cost": decision.actual_cost if decision else 0,
            "input_tokens": decision.input_tokens if decision else 0,
            "output_tokens": decision.output_tokens if decision else 0,
            "complexity_detected": decision.complexity_detected if decision else "?",
            "response_text": text,
        })
        print(f"{latency:.0f}ms -> {decision.selected_model if decision else '?'} (complexity: {decision.complexity_detected if decision else '?'})")
    except Exception as e:
        print(f"FAIL: {e}")
        results.append({"runner": "router-keyword", "difficulty": p["difficulty"], "prompt_id": p["id"], "error": str(e)})

# 4. Run router with ML classifier override
print("\n" + "=" * 70)
print("RUNNING ROUTER (ML CLASSIFIER)")
print("=" * 70)

# For ML classifier, we override the complexity before routing
# by using the appropriate strategy based on ML prediction
for p in test_prompts:
    messages, system = build_msgs(p)
    ml_label, ml_conf, _, clf_ms = ml_classifier.classify(
        system_prompt=p.get("system_prompt", ""),
        user_prompt=p.get("user_prompt", ""),
        context=p.get("context", ""),
    )

    # Map ML label to routing strategy
    strategy_map = {
        "simple": "cost-optimized",
        "medium": "balanced",
        "complex": "quality-optimized",
    }
    strategy = strategy_map[ml_label]

    print(f"  router-ml/{p['difficulty']}... (ML says: {ml_label})", end=" ", flush=True)
    router_ml = BedrockRouter.create({"region": REGION, "cache": {"enabled": False}})
    try:
        t0 = time.perf_counter()
        resp = router_ml.converse(messages=messages, system=system, routing=RoutingConfig(strategy=strategy))
        latency = (time.perf_counter() - t0) * 1000
        # Add classifier overhead
        total_latency = latency + clf_ms
        decision = resp.get("routing_decision")
        text = resp["output"]["message"]["content"][0]["text"]
        results.append({
            "runner": "router-ml", "difficulty": p["difficulty"], "prompt_id": p["id"],
            "latency_ms": round(total_latency), "model": decision.selected_model if decision else "?",
            "cost": decision.actual_cost if decision else 0,
            "input_tokens": decision.input_tokens if decision else 0,
            "output_tokens": decision.output_tokens if decision else 0,
            "ml_prediction": ml_label,
            "ml_confidence": ml_conf,
            "classifier_ms": round(clf_ms, 1),
            "response_text": text,
        })
        print(f"{total_latency:.0f}ms -> {decision.selected_model if decision else '?'}")
    except Exception as e:
        print(f"FAIL: {e}")
        results.append({"runner": "router-ml", "difficulty": p["difficulty"], "prompt_id": p["id"], "error": str(e)})

# 5. Judge all responses
print("\n" + "=" * 70)
print("JUDGING RESPONSES")
print("=" * 70)

for r in results:
    if "response_text" not in r:
        r["score"] = 0
        continue
    prompt = next(p for p in test_prompts if p["id"] == r["prompt_id"])
    print(f"  {r['runner']}/{r['difficulty']}...", end=" ", flush=True)
    try:
        score = judge(prompt, r["response_text"])
        r["score"] = score
        print(f"{score}/10")
    except Exception as e:
        r["score"] = 0
        print(f"ERROR: {e}")
    time.sleep(0.3)

# ── Results Table ────────────────────────────────────────────────
print("\n" + "=" * 70)
print("RESULTS: LATENCY + COST + ACCURACY")
print("=" * 70)
print(f"\n{'Runner':<18} {'Diff':<8} {'Score':<7} {'Latency':<9} {'Cost':<12} {'Model'}")
print("-" * 85)

for diff in ["simple", "medium", "complex"]:
    for r in sorted([x for x in results if x["difficulty"] == diff], key=lambda x: x["runner"]):
        if "error" in r:
            continue
        model_short = r.get("model", "?").split(".")[-1][:25]
        score_str = f"{r.get('score', 0)}/10"
        lat_str = f"{r.get('latency_ms', 0)}ms"
        cost_str = f"${r.get('cost', 0):.6f}"
        print(f"{r['runner']:<18} {diff:<8} {score_str:<7} {lat_str:<9} {cost_str:<12} {model_short}")
    print()

# ── Summary ──────────────────────────────────────────────────────
print("=" * 70)
print("SUMMARY (averaged across all difficulties)")
print("=" * 70)

runners = ["sonnet", "opus", "nova-pro", "haiku", "router-keyword", "router-ml"]
print(f"\n{'Runner':<18} {'Avg Score':<11} {'Avg Latency':<13} {'Avg Cost':<12}")
print("-" * 55)
for rn in runners:
    rr = [x for x in results if x["runner"] == rn and "score" in x and "error" not in x]
    if not rr:
        continue
    avg_score = sum(x["score"] for x in rr) / len(rr)
    avg_lat = sum(x.get("latency_ms", 0) for x in rr) / len(rr)
    avg_cost = sum(x.get("cost", 0) for x in rr) / len(rr)
    print(f"{rn:<18} {avg_score:<11.1f} {avg_lat:<13.0f} ${avg_cost:.6f}")

# Save
output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "ml_classifier_test.json")
with open(output_path, "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nResults saved to: {output_path}")
