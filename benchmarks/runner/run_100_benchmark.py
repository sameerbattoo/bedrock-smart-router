# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

#!/usr/bin/env python3
"""100-prompt benchmark: ML classifier router vs baselines.

Runs 100 prompts (mix of simple/medium/complex across all categories) through:
- Baselines: Sonnet, Haiku, Nova Pro, Opus
- Router (keyword heuristic)
- Router (ML classifier)

Then judges all responses and generates a report.

Usage:
    python benchmarks/runner/run_100_benchmark.py
    python benchmarks/runner/run_100_benchmark.py --skip-judge   # Run models only, judge later
    python benchmarks/runner/run_100_benchmark.py --judge-only   # Judge existing results
"""
import sys
import os
import json
import time
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import boto3
from bedrock_smart_router import BedrockRouter, RoutingConfig
from bedrock_smart_router.request_analyzer import RequestAnalyzer
from benchmarks.runner.config import MODELS, REGION, JUDGE_MODEL_ID, JUDGE_SYSTEM_PROMPT_WITH_ANSWER

import onnxruntime as ort
from tokenizers import Tokenizer

ONNX_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "classifier", "onnx_model")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
PROMPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "generated")
os.makedirs(RESULTS_DIR, exist_ok=True)

OUTPUT_FILE = os.path.join(RESULTS_DIR, "benchmark_100_full.json")


class MLClassifier:
    """ONNX complexity classifier with smart truncation."""

    def __init__(self):
        self.session = ort.InferenceSession(os.path.join(ONNX_DIR, "model.onnx"))
        self.tokenizer = Tokenizer.from_file(os.path.join(ONNX_DIR, "tokenizer.json"))
        self.tokenizer.enable_padding(length=256)
        self.tokenizer.enable_truncation(max_length=256)
        with open(os.path.join(ONNX_DIR, "classifier_weights.json")) as f:
            d = json.load(f)
        self.weights = np.array(d["weights"])
        self.bias = np.array(d["bias"])
        self.classes = d["classes"]

    def classify(self, system_prompt="", user_prompt="", context=""):
        """Classify with smart truncation: prioritize system+user, append context if room."""
        t0 = time.perf_counter()
        priority = f"{system_prompt}\n\n{user_prompt}".strip()
        if context:
            enc = self.tokenizer.encode(priority)
            used = sum(1 for t in enc.ids if t != 0)
            remaining = 256 - used - 10
            if remaining > 20:
                priority += "\n\n" + context[:remaining * 4]

        encoded = self.tokenizer.encode(priority)
        ids = np.array([encoded.ids], dtype=np.int64)
        mask = np.array([encoded.attention_mask], dtype=np.int64)
        out = self.session.run(None, {"input_ids": ids, "attention_mask": mask})

        emb = out[0]
        m = np.expand_dims(mask, -1)
        pooled = np.sum(emb * m, axis=1) / np.sum(m, axis=1)
        pooled = pooled / np.linalg.norm(pooled, axis=1, keepdims=True)

        logits = pooled @ self.weights.T + self.bias
        exp_l = np.exp(logits - np.max(logits))
        probs = exp_l / exp_l.sum(axis=1, keepdims=True)
        idx = np.argmax(probs[0])
        elapsed = (time.perf_counter() - t0) * 1000
        return self.classes[idx], float(probs[0][idx]), elapsed


def load_100_prompts():
    """Load 100 prompts: ~17 per category, balanced across difficulties, plus 10 multimodal."""
    all_prompts = []
    for f in sorted(os.listdir(PROMPTS_DIR)):
        if f.endswith(".json"):
            with open(os.path.join(PROMPTS_DIR, f)) as fh:
                all_prompts.extend(json.load(fh))

    # Take ~15 per category (5 simple, 5 medium, 5 complex from each)
    by_cat = {}
    for p in all_prompts:
        by_cat.setdefault(p["category"], {"simple": [], "medium": [], "complex": []})
        by_cat[p["category"]][p["difficulty"]].append(p)

    selected = []
    for cat, diffs in sorted(by_cat.items()):
        selected.extend(diffs["simple"][:5])
        selected.extend(diffs["medium"][:5])
        selected.extend(diffs["complex"][:5])

    # Add 5 image prompts and 5 PDF prompts
    multimodal_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "industry_standard")

    # 5 image prompts (from DocVQA - they have questions + answers)
    docvqa_path = os.path.join(multimodal_dir, "multimodal", "docvqa", "samples.json")
    if os.path.exists(docvqa_path):
        docvqa = json.load(open(docvqa_path))[:5]
        img_dir = os.path.join(multimodal_dir, "multimodal", "docvqa", "images")
        for s in docvqa:
            img_path = os.path.join(img_dir, os.path.basename(s["image_path"]))
            if os.path.exists(img_path):
                selected.append({
                    "id": s["id"],
                    "category": "multimodal_image",
                    "difficulty": "medium",
                    "system_prompt": "You are a document analysis expert. Answer the question based on the document image.",
                    "user_prompt": s["question"],
                    "context": "",
                    "expected_answer": s["answers"][0] if s.get("answers") else "",
                    "_image_path": img_path,
                    "_is_multimodal": "image",
                })

    # 5 PDF prompts (from dude_pdfs)
    pdf_path = os.path.join(multimodal_dir, "pdfs", "dude_pdfs", "samples.json")
    if os.path.exists(pdf_path):
        pdfs = json.load(open(pdf_path))[:5]
        pdf_dir = os.path.join(multimodal_dir, "pdfs", "dude_pdfs", "pdfs")
        for s in pdfs:
            full_pdf_path = os.path.join(pdf_dir, os.path.basename(s["pdf_path"]))
            if os.path.exists(full_pdf_path):
                selected.append({
                    "id": s["id"],
                    "category": "multimodal_pdf",
                    "difficulty": "complex",
                    "system_prompt": "You are a document analysis expert. Read the PDF and answer the question.",
                    "user_prompt": s["question"],
                    "context": "",
                    "expected_answer": s.get("first_page_preview", "")[:200],
                    "_pdf_path": full_pdf_path,
                    "_is_multimodal": "pdf",
                })

    return selected[:110]  # 90 text + 5 image + 5 pdf = 100, with buffer


def build_msgs(p):
    """Build Bedrock converse messages, including image/PDF if multimodal."""
    content_blocks = []

    # Add image if present
    if p.get("_is_multimodal") == "image" and p.get("_image_path"):
        with open(p["_image_path"], "rb") as f:
            img_bytes = f.read()
        content_blocks.append({
            "image": {
                "format": "png",
                "source": {"bytes": img_bytes},
            }
        })

    # Add PDF if present
    if p.get("_is_multimodal") == "pdf" and p.get("_pdf_path"):
        with open(p["_pdf_path"], "rb") as f:
            pdf_bytes = f.read()
        content_blocks.append({
            "document": {
                "format": "pdf",
                "name": os.path.basename(p["_pdf_path"]).replace(".pdf", ""),
                "source": {"bytes": pdf_bytes},
            }
        })

    # Add text
    text = p["user_prompt"]
    if p.get("context"):
        text = f"{p['context']}\n\n{p['user_prompt']}"
    content_blocks.append({"text": text})

    messages = [{"role": "user", "content": content_blocks}]
    system = [{"text": p["system_prompt"]}] if p.get("system_prompt") else None
    return messages, system


def estimate_cost(model_id, in_tok, out_tok):
    rates = {
        "us.anthropic.claude-sonnet-4-6": (3.0, 15.0),
        "us.anthropic.claude-haiku-4-5-20251001-v1:0": (0.80, 4.0),
        "us.amazon.nova-pro-v1:0": (0.80, 3.20),
        "us.anthropic.claude-opus-4-7": (15.0, 75.0),
        "global.anthropic.claude-opus-4-7": (15.0, 75.0),
        "us.amazon.nova-micro-v1:0": (0.035, 0.14),
        "us.amazon.nova-lite-v1:0": (0.06, 0.24),
        "us.deepseek.r1-v1:0": (1.35, 5.40),
    }
    r = rates.get(model_id, (3.0, 15.0))
    return (in_tok * r[0] + out_tok * r[1]) / 1_000_000


def run_models(prompts):
    """Run all 6 runners on all prompts — runners in parallel, prompts sequential per runner."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading

    ml_clf = MLClassifier()
    all_results = []
    lock = threading.Lock()
    progress = {"done": 0, "total": len(prompts) * 6}

    def run_baseline(model_key, prompts_list):
        """Run a single baseline model on all prompts."""
        model_id = MODELS[model_key]["model_id"]
        client = boto3.Session(region_name=REGION).client("bedrock-runtime")
        results = []
        for p in prompts_list:
            if p.get("_is_multimodal") == "pdf" and model_key == "nova-pro":
                results.append({"runner": model_key, "prompt_id": p["id"], "category": p["category"],
                    "difficulty": p["difficulty"], "error": "model does not support PDF", "success": False})
                with lock:
                    progress["done"] += 1
                    print(f"    [{progress['done']}/{progress['total']}] {model_key}/{p['id']} SKIP (no PDF)", flush=True)
                continue
            msgs, sys = build_msgs(p)
            try:
                t0 = time.perf_counter()
                resp = client.converse(modelId=model_id, messages=msgs, system=sys)
                lat = (time.perf_counter() - t0) * 1000
                usage = resp.get("usage", {})
                text = resp["output"]["message"]["content"][0]["text"]
                cost = estimate_cost(model_id, usage.get("inputTokens", 0), usage.get("outputTokens", 0))
                results.append({"runner": model_key, "prompt_id": p["id"], "category": p["category"],
                    "difficulty": p["difficulty"], "latency_ms": round(lat), "model": model_id,
                    "cost": cost, "response_text": text, "success": True})
                with lock:
                    progress["done"] += 1
                    print(f"    [{progress['done']}/{progress['total']}] {model_key}/{p['id']} ({p['difficulty']}) {lat:.0f}ms", flush=True)
            except Exception as e:
                results.append({"runner": model_key, "prompt_id": p["id"], "category": p["category"],
                    "difficulty": p["difficulty"], "error": str(e)[:200], "success": False})
                with lock:
                    progress["done"] += 1
                    print(f"    [{progress['done']}/{progress['total']}] {model_key}/{p['id']} FAIL: {str(e)[:60]}", flush=True)
        return results

    def run_router_keyword(prompts_list):
        """Run keyword-heuristic router on all prompts."""
        router = BedrockRouter.create({"region": REGION, "cache": {"enabled": False}})
        results = []
        for p in prompts_list:
            msgs, sys = build_msgs(p)
            try:
                t0 = time.perf_counter()
                resp = router.converse(messages=msgs, system=sys, routing=RoutingConfig(strategy="balanced"))
                lat = (time.perf_counter() - t0) * 1000
                dec = resp.get("routing_decision")
                text = resp["output"]["message"]["content"][0]["text"]
                results.append({"runner": "router-keyword", "prompt_id": p["id"], "category": p["category"],
                    "difficulty": p["difficulty"], "latency_ms": round(lat),
                    "model": dec.selected_model if dec else "?", "cost": dec.actual_cost if dec else 0,
                    "complexity_detected": dec.complexity_detected if dec else "?",
                    "response_text": text, "success": True})
                with lock:
                    progress["done"] += 1
                    print(f"    [{progress['done']}/{progress['total']}] router-kw/{p['id']} ({p['difficulty']}) {lat:.0f}ms -> {dec.selected_model.split('.')[-1][:20] if dec else '?'}", flush=True)
            except Exception as e:
                results.append({"runner": "router-keyword", "prompt_id": p["id"], "category": p["category"],
                    "difficulty": p["difficulty"], "error": str(e)[:200], "success": False})
                with lock:
                    progress["done"] += 1
                    print(f"    [{progress['done']}/{progress['total']}] router-kw/{p['id']} FAIL: {str(e)[:60]}", flush=True)
        return results

    def run_router_ml(prompts_list):
        """Run ML-classifier router on all prompts."""
        results = []
        for p in prompts_list:
            msgs, sys = build_msgs(p)
            ml_label, ml_conf, clf_ms = ml_clf.classify(
                system_prompt=p.get("system_prompt", ""),
                user_prompt=p.get("user_prompt", ""),
                context=p.get("context", ""),
            )
            strategy_map = {"simple": "cost-optimized", "medium": "balanced", "complex": "quality-optimized"}
            strategy = strategy_map[ml_label]
            router_ml = BedrockRouter.create({"region": REGION, "cache": {"enabled": False}})
            try:
                t0 = time.perf_counter()
                resp = router_ml.converse(messages=msgs, system=sys, routing=RoutingConfig(strategy=strategy))
                lat = (time.perf_counter() - t0) * 1000 + clf_ms
                dec = resp.get("routing_decision")
                text = resp["output"]["message"]["content"][0]["text"]
                results.append({"runner": "router-ml", "prompt_id": p["id"], "category": p["category"],
                    "difficulty": p["difficulty"], "latency_ms": round(lat),
                    "model": dec.selected_model if dec else "?", "cost": dec.actual_cost if dec else 0,
                    "ml_prediction": ml_label, "ml_confidence": ml_conf,
                    "response_text": text, "success": True})
                with lock:
                    progress["done"] += 1
                    print(f"    [{progress['done']}/{progress['total']}] router-ml/{p['id']} ({p['difficulty']}) {lat:.0f}ms -> {dec.selected_model.split('.')[-1][:20] if dec else '?'} [ML:{ml_label}]", flush=True)
            except Exception as e:
                results.append({"runner": "router-ml", "prompt_id": p["id"], "category": p["category"],
                    "difficulty": p["difficulty"], "error": str(e)[:200], "success": False})
                with lock:
                    progress["done"] += 1
                    print(f"    [{progress['done']}/{progress['total']}] router-ml/{p['id']} FAIL: {str(e)[:60]}", flush=True)
        return results

    # Run all 6 runners in parallel
    print(f"\n  Running 6 runners in parallel ({len(prompts)} prompts each)...")
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {
            executor.submit(run_baseline, "sonnet", prompts): "sonnet",
            executor.submit(run_baseline, "haiku", prompts): "haiku",
            executor.submit(run_baseline, "nova-pro", prompts): "nova-pro",
            executor.submit(run_baseline, "opus", prompts): "opus",
            executor.submit(run_router_keyword, prompts): "router-keyword",
            executor.submit(run_router_ml, prompts): "router-ml",
        }
        for future in as_completed(futures):
            runner_name = futures[future]
            try:
                results = future.result()
                all_results.extend(results)
                print(f"\n  ✓ {runner_name} complete ({len(results)} results)", flush=True)
            except Exception as e:
                print(f"\n  ✗ {runner_name} FAILED: {e}", flush=True)

    return all_results


def judge_results(results, prompts_map):
    """Judge all successful results with incremental saving."""
    client = boto3.Session(region_name=REGION).client("bedrock-runtime")
    to_judge = [r for r in results if r.get("success") and r.get("response_text") and r.get("score", 0) == 0]
    print(f"\n  Judging {len(to_judge)} responses...", flush=True)

    for i, r in enumerate(to_judge):
        p = prompts_map.get(r["prompt_id"])
        if not p:
            r["score"] = 0
            continue
        try:
            judge_prompt = JUDGE_SYSTEM_PROMPT_WITH_ANSWER.format(
                system_prompt=p["system_prompt"], user_prompt=p["user_prompt"],
                expected_answer=p.get("expected_answer", "N/A"), response=r["response_text"])
            resp = client.converse(modelId=JUDGE_MODEL_ID,
                messages=[{"role": "user", "content": [{"text": judge_prompt}]}])
            text = resp["output"]["message"]["content"][0]["text"].strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0]
            r["score"] = json.loads(text).get("score", 0)
            print(f"    [{i+1}/{len(to_judge)}] {r['runner']}/{r['prompt_id']}: {r['score']}/10", flush=True)
        except Exception as e:
            r["score"] = 0
            print(f"    [{i+1}/{len(to_judge)}] {r['runner']}/{r['prompt_id']}: ERROR {str(e)[:40]}", flush=True)

        # Save every 20 results
        if (i + 1) % 20 == 0:
            with open(OUTPUT_FILE, "w") as f:
                json.dump(results, f, indent=2, default=str)
            print(f"    (saved progress: {i+1}/{len(to_judge)})", flush=True)

        time.sleep(0.3)


def print_report(results):
    """Print summary report."""
    runners = ["sonnet", "opus", "nova-pro", "haiku", "router-keyword", "router-ml"]
    print("\n" + "=" * 70)
    print("FINAL RESULTS (100 prompts)")
    print("=" * 70)
    print(f"\n{'Runner':<18} {'Score':<9} {'Latency':<11} {'Cost':<12} {'Success'}")
    print("-" * 62)
    for rn in runners:
        rr = [x for x in results if x["runner"] == rn and x.get("success")]
        if not rr:
            continue
        scored = [x for x in rr if "score" in x and x["score"] > 0]
        avg_score = sum(x["score"] for x in scored) / len(scored) if scored else 0
        avg_lat = sum(x.get("latency_ms", 0) for x in rr) / len(rr)
        avg_cost = sum(x.get("cost", 0) for x in rr) / len(rr)
        print(f"{rn:<18} {avg_score:<9.1f} {avg_lat:<11.0f} ${avg_cost:<11.6f} {len(rr)}/100")

    # By difficulty
    print(f"\n{'Runner':<18} {'Simple':<9} {'Medium':<9} {'Complex':<9}")
    print("-" * 45)
    for rn in runners:
        row = f"{rn:<18}"
        for diff in ["simple", "medium", "complex"]:
            scored = [x for x in results if x["runner"] == rn and x.get("difficulty") == diff and x.get("score", 0) > 0]
            avg = sum(x["score"] for x in scored) / len(scored) if scored else 0
            row += f" {avg:<9.1f}"
        print(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-judge", action="store_true")
    parser.add_argument("--judge-only", action="store_true")
    args = parser.parse_args()

    prompts = load_100_prompts()
    prompts_map = {p["id"]: p for p in prompts}
    print(f"Loaded {len(prompts)} prompts")
    by_diff = {}
    for p in prompts:
        by_diff.setdefault(p["difficulty"], []).append(p)
    for d, ps in sorted(by_diff.items()):
        print(f"  {d}: {len(ps)}")

    if args.judge_only:
        with open(OUTPUT_FILE) as f:
            results = json.load(f)
        judge_results(results, prompts_map)
    else:
        print("\nRunning models...")
        results = run_models(prompts)
        # Save intermediate
        with open(OUTPUT_FILE, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nModel runs saved to {OUTPUT_FILE}")

        if not args.skip_judge:
            judge_results(results, prompts_map)

    # Save final
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print_report(results)
    print(f"\nFull results: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
