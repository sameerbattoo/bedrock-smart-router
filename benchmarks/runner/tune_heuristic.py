# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

#!/usr/bin/env python3
"""Test and tune the heuristic complexity classifier.

Evaluates the RequestAnalyzer's heuristic scoring against all labeled data
(2,545 samples from training_data.json + 295 generated prompts), then performs
grid search over weights and thresholds to find optimal parameters.

Usage:
    python benchmarks/runner/tune_heuristic.py              # Full evaluation + tuning
    python benchmarks/runner/tune_heuristic.py --eval-only  # Just evaluate current settings
    python benchmarks/runner/tune_heuristic.py --verbose    # Show all misclassifications
    python benchmarks/runner/tune_heuristic.py --tune-weights --iterations 500
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import statistics
import time
from dataclasses import fields
from typing import Any

# Setup path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BENCHMARKS_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(BENCHMARKS_DIR)
sys.path.insert(0, PROJECT_ROOT)

from bedrock_smart_router.request_analyzer import RequestAnalyzer, AnalyzerWeights, ComplexityThresholds
from bedrock_smart_router.models import Complexity


# ─── Data Loading ───────────────────────────────────────────────────────────

def load_training_data() -> list[dict[str, Any]]:
    """Load the labeled samples from classifier training data."""
    path = os.path.join(BENCHMARKS_DIR, "classifier", "training_data.json")
    if not os.path.exists(path):
        print(f"  WARNING: {path} not found, skipping.")
        return []
    with open(path) as f:
        data = json.load(f)
    print(f"  Loaded {len(data)} samples from training_data.json")
    return data


def load_generated_prompts() -> list[dict[str, Any]]:
    """Load the 295 hand-labeled generated prompts."""
    prompts_dir = os.path.join(BENCHMARKS_DIR, "data", "generated")
    if not os.path.exists(prompts_dir):
        print(f"  WARNING: {prompts_dir} not found, skipping.")
        return []
    samples = []
    for filename in sorted(os.listdir(prompts_dir)):
        if not filename.endswith(".json") or filename.startswith("_"):
            continue
        with open(os.path.join(prompts_dir, filename)) as f:
            prompts = json.load(f)
        for p in prompts:
            text_parts = []
            if p.get("system_prompt"):
                text_parts.append(p["system_prompt"])
            if p.get("context"):
                text_parts.append(p["context"])
            if p.get("user_prompt"):
                text_parts.append(p["user_prompt"])
            samples.append({
                "text": "\n\n".join(text_parts),
                "label": p["difficulty"],
                "source": f"generated/{filename}",
                "id": p["id"],
                "category": p.get("category", "unknown"),
            })
    print(f"  Loaded {len(samples)} samples from generated prompts")
    return samples


def load_all_data() -> list[dict[str, Any]]:
    """Load and deduplicate all labeled data."""
    print("\nLoading labeled data...")
    training = load_training_data()
    generated = load_generated_prompts()

    # Deduplicate by text prefix
    seen_texts = set()
    all_data = []
    for item in training:
        key = item["text"][:200]
        if key not in seen_texts:
            seen_texts.add(key)
            all_data.append(item)
    for item in generated:
        key = item["text"][:200]
        if key not in seen_texts:
            seen_texts.add(key)
            all_data.append(item)

    print(f"  Total unique samples: {len(all_data)}")
    dist = {}
    for item in all_data:
        dist[item["label"]] = dist.get(item["label"], 0) + 1
    print(f"  Distribution: {dist}")
    return all_data


# ─── Evaluation ─────────────────────────────────────────────────────────────

LABEL_TO_COMPLEXITY = {
    "simple": [Complexity.SIMPLE],
    "medium": [Complexity.MODERATE],
    "complex": [Complexity.COMPLEX, Complexity.REASONING],
}

COMPLEXITY_TO_LABEL = {
    Complexity.SIMPLE: "simple",
    Complexity.MODERATE: "medium",
    Complexity.COMPLEX: "complex",
    Complexity.REASONING: "complex",
}


def evaluate_fast(
    analyzer: RequestAnalyzer,
    precomputed: list[tuple[str, str]],
) -> float:
    """Fast evaluation using precomputed messages — returns accuracy only."""
    correct = 0
    for text, true_label in precomputed:
        messages = [{"role": "user", "content": [{"text": text}]}]
        result = analyzer.analyze(messages)
        if result.complexity in LABEL_TO_COMPLEXITY.get(true_label, []):
            correct += 1
    return correct / len(precomputed) if precomputed else 0


def evaluate_full(
    analyzer: RequestAnalyzer,
    data: list[dict[str, Any]],
    collect_errors: bool = True,
) -> dict[str, Any]:
    """Full evaluation with metrics."""
    correct = 0
    total = 0
    misclassifications = []

    tp = {"simple": 0, "medium": 0, "complex": 0}
    fp = {"simple": 0, "medium": 0, "complex": 0}
    fn = {"simple": 0, "medium": 0, "complex": 0}
    confusion = {
        "simple": {"simple": 0, "medium": 0, "complex": 0},
        "medium": {"simple": 0, "medium": 0, "complex": 0},
        "complex": {"simple": 0, "medium": 0, "complex": 0},
    }

    start = time.perf_counter()
    for item in data:
        text = item["text"][:4000]
        messages = [{"role": "user", "content": [{"text": text}]}]
        result = analyzer.analyze(messages)

        predicted_label = COMPLEXITY_TO_LABEL[result.complexity]
        true_label = item["label"]
        total += 1

        if result.complexity in LABEL_TO_COMPLEXITY.get(true_label, []):
            correct += 1
            tp[true_label] += 1
        else:
            fp[predicted_label] += 1
            fn[true_label] += 1
            if collect_errors:
                misclassifications.append({
                    "id": item.get("id", "?"),
                    "source": item.get("source", "?"),
                    "category": item.get("category", "?"),
                    "true_label": true_label,
                    "predicted": predicted_label,
                    "score": result.complexity_score,
                    "text_preview": text[:100],
                })
        confusion[true_label][predicted_label] += 1

    elapsed = time.perf_counter() - start

    # Per-class metrics
    per_class_acc = {}
    per_class_f1 = {}
    for label in ["simple", "medium", "complex"]:
        class_total = sum(confusion[label].values())
        per_class_acc[label] = confusion[label][label] / class_total if class_total > 0 else 0.0
        precision = tp[label] / (tp[label] + fp[label]) if (tp[label] + fp[label]) > 0 else 0
        recall = tp[label] / (tp[label] + fn[label]) if (tp[label] + fn[label]) > 0 else 0
        per_class_f1[label] = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "accuracy": correct / total if total > 0 else 0,
        "per_class_accuracy": per_class_acc,
        "per_class_f1": per_class_f1,
        "confusion": confusion,
        "misclassifications": misclassifications,
        "avg_latency_us": (elapsed / total) * 1_000_000 if total > 0 else 0,
    }


def print_eval(result: dict, title: str) -> None:
    """Pretty-print evaluation results."""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")
    print(f"\n  Overall Accuracy: {result['accuracy']:.1%}")
    print(f"  Avg Latency: {result['avg_latency_us']:.0f} µs/classification")

    print(f"\n  Per-Class Accuracy:")
    for label in ["simple", "medium", "complex"]:
        print(f"    {label:>8}: {result['per_class_accuracy'][label]:.1%}")

    print(f"\n  Per-Class F1 Score:")
    macro_f1 = sum(result["per_class_f1"].values()) / 3
    for label in ["simple", "medium", "complex"]:
        print(f"    {label:>8}: {result['per_class_f1'][label]:.3f}")
    print(f"    {'macro':>8}: {macro_f1:.3f}")

    print(f"\n  Confusion Matrix (rows=true, cols=predicted):")
    print(f"    {'':>10} {'simple':>8} {'medium':>8} {'complex':>8}")
    for true_label in ["simple", "medium", "complex"]:
        row = f"    {true_label:>10}"
        for pred_label in ["simple", "medium", "complex"]:
            count = result["confusion"][true_label][pred_label]
            row += f" {count:>8}"
        print(row)


# ─── Threshold Tuning ───────────────────────────────────────────────────────

def tune_thresholds(
    precomputed: list[tuple[str, str]],
    weights: AnalyzerWeights | None = None,
) -> tuple[ComplexityThresholds, float]:
    """Grid search over threshold values to maximize accuracy.

    Uses a two-phase approach: coarse grid then fine-tuning around the best.
    """
    print("\n  Phase 1: Coarse grid search...")

    best_accuracy = 0.0
    best_thresholds = ComplexityThresholds()
    tested = 0

    # Coarse grid (step=0.02) focused on low-score region
    simple_range = [round(x * 0.02, 3) for x in range(2, 12)]    # 0.04 to 0.22
    moderate_range = [round(x * 0.02, 3) for x in range(4, 18)]  # 0.08 to 0.34
    reasoning_counts = [2, 3, 4, 5, 6]

    for s_max in simple_range:
        for m_max in moderate_range:
            if m_max <= s_max + 0.01:
                continue
            for r_count in reasoning_counts:
                c_max = min(0.95, m_max + 0.15)
                thresholds = ComplexityThresholds(
                    simple_max=s_max,
                    moderate_max=m_max,
                    complex_max=c_max,
                    reasoning_marker_count=r_count,
                )
                analyzer = RequestAnalyzer(weights=weights, thresholds=thresholds)
                acc = evaluate_fast(analyzer, precomputed)
                tested += 1

                if acc > best_accuracy:
                    best_accuracy = acc
                    best_thresholds = thresholds

    print(f"  Coarse: tested {tested} combos, best={best_accuracy:.1%}")

    # Phase 2: Fine-tune around best (step=0.005)
    print("  Phase 2: Fine-tuning around best...")
    base_s = best_thresholds.simple_max
    base_m = best_thresholds.moderate_max
    base_r = best_thresholds.reasoning_marker_count

    fine_s = [round(base_s + d * 0.005, 4) for d in range(-4, 5)]
    fine_m = [round(base_m + d * 0.005, 4) for d in range(-4, 5)]
    fine_r = [max(2, base_r + d) for d in range(-1, 2)]

    for s_max in fine_s:
        if s_max <= 0.01:
            continue
        for m_max in fine_m:
            if m_max <= s_max + 0.005:
                continue
            for r_count in fine_r:
                c_max = min(0.95, m_max + 0.15)
                thresholds = ComplexityThresholds(
                    simple_max=s_max,
                    moderate_max=m_max,
                    complex_max=c_max,
                    reasoning_marker_count=r_count,
                )
                analyzer = RequestAnalyzer(weights=weights, thresholds=thresholds)
                acc = evaluate_fast(analyzer, precomputed)
                tested += 1

                if acc > best_accuracy:
                    best_accuracy = acc
                    best_thresholds = thresholds

    print(f"  Total tested: {tested} combinations")
    print(f"  Best accuracy: {best_accuracy:.1%}")
    print(f"  Best thresholds: simple_max={best_thresholds.simple_max}, "
          f"moderate_max={best_thresholds.moderate_max}, "
          f"complex_max={best_thresholds.complex_max}, "
          f"reasoning_marker_count={best_thresholds.reasoning_marker_count}")
    return best_thresholds, best_accuracy


# ─── Weight Tuning ──────────────────────────────────────────────────────────

def tune_weights(
    precomputed: list[tuple[str, str]],
    thresholds: ComplexityThresholds,
    iterations: int = 300,
) -> tuple[AnalyzerWeights, float]:
    """Hill-climbing random search over weight space."""
    import random
    random.seed(42)

    print(f"\n  Random search over weight space ({iterations} iterations)...")

    weight_names = [f.name for f in fields(AnalyzerWeights)]
    base_weights = AnalyzerWeights()
    base_values = [getattr(base_weights, name) for name in weight_names]

    # Evaluate baseline
    analyzer = RequestAnalyzer(weights=base_weights, thresholds=thresholds)
    best_accuracy = evaluate_fast(analyzer, precomputed)
    best_values = base_values[:]
    print(f"  Baseline accuracy: {best_accuracy:.1%}")

    stagnant = 0
    for i in range(iterations):
        # Adaptive perturbation: larger steps when stagnant
        sigma = 0.05 if stagnant < 20 else 0.10
        noise = [random.gauss(0, sigma) for _ in best_values]
        new_values = [max(0.001, v + n) for v, n in zip(best_values, noise)]
        total = sum(new_values)
        new_values = [v / total for v in new_values]

        new_weights = AnalyzerWeights(**dict(zip(weight_names, new_values)))
        analyzer = RequestAnalyzer(weights=new_weights, thresholds=thresholds)
        acc = evaluate_fast(analyzer, precomputed)

        if acc > best_accuracy:
            best_accuracy = acc
            best_values = new_values
            stagnant = 0
            print(f"    Iter {i+1}: new best = {best_accuracy:.1%}")
        else:
            stagnant += 1

    best_weights = AnalyzerWeights(**dict(zip(weight_names, best_values)))
    print(f"\n  Best accuracy after weight tuning: {best_accuracy:.1%}")
    print(f"  Best weights:")
    for name in weight_names:
        print(f"    {name}: {getattr(best_weights, name):.4f}")
    return best_weights, best_accuracy


# ─── Dimension Analysis ─────────────────────────────────────────────────────

def analyze_dimensions(data: list[dict[str, Any]]) -> None:
    """Show per-dimension score distributions by label."""
    print(f"\n{'=' * 70}")
    print("  DIMENSION ANALYSIS (avg score per label)")
    print(f"{'=' * 70}")

    analyzer = RequestAnalyzer()
    dim_names = [
        "token_count", "code_presence", "reasoning_markers", "technical_depth",
        "simple_indicators", "multi_step", "tool_use", "document_analysis",
        "conversation_depth", "aws_specificity", "math_logical", "creative_open",
    ]

    dim_scores = {label: {d: [] for d in dim_names} for label in ["simple", "medium", "complex"]}

    for item in data:
        text = item["text"][:4000]
        messages = [{"role": "user", "content": [{"text": text}]}]
        text_lower = text.lower()
        scores = analyzer._heuristic_classifier._score_dimensions(text_lower, text)
        label = item["label"]
        for dim_name, score in zip(dim_names, scores):
            dim_scores[label][dim_name].append(score)

    print(f"\n  {'Dimension':<22} {'Simple':>8} {'Medium':>8} {'Complex':>8} {'Δ(C-S)':>8} {'Useful?':>8}")
    print(f"  {'-'*22} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    for dim in dim_names:
        s_avg = statistics.mean(dim_scores["simple"][dim]) if dim_scores["simple"][dim] else 0
        m_avg = statistics.mean(dim_scores["medium"][dim]) if dim_scores["medium"][dim] else 0
        c_avg = statistics.mean(dim_scores["complex"][dim]) if dim_scores["complex"][dim] else 0
        delta = c_avg - s_avg
        useful = "✓" if delta > 0.05 else "✗" if delta < -0.02 else "~"
        print(f"  {dim:<22} {s_avg:>8.3f} {m_avg:>8.3f} {c_avg:>8.3f} {delta:>+8.3f} {useful:>8}")

    # Recommendations
    print(f"\n  Recommendations:")
    print(f"    - 'technical_depth' is the strongest signal (Δ=+0.265). Increase its weight.")
    print(f"    - 'simple_indicators' is nearly constant (~0.85). It adds noise, reduce weight.")
    print(f"    - 'conversation_depth' is constant (0.1) for single-turn data. Low value for this dataset.")
    print(f"    - 'document_analysis' is inversely correlated (higher for simple). Consider inverting or reducing.")


# ─── Error Analysis ─────────────────────────────────────────────────────────

def error_analysis(result: dict, verbose: bool = False) -> None:
    """Analyze misclassification patterns."""
    misclassifications = result["misclassifications"]
    print(f"\n{'=' * 70}")
    print(f"  ERROR ANALYSIS ({len(misclassifications)} misclassifications)")
    print(f"{'=' * 70}")

    if not misclassifications:
        print("  No misclassifications!")
        return

    by_type: dict[str, list] = {}
    for m in misclassifications:
        key = f"{m['true_label']} → {m['predicted']}"
        by_type.setdefault(key, []).append(m)

    print(f"\n  Error type breakdown:")
    for key, items in sorted(by_type.items(), key=lambda x: -len(x[1])):
        scores = [i["score"] for i in items]
        print(f"    {key}: {len(items)} errors (score range: {min(scores):.3f} - {max(scores):.3f})")

    by_source: dict[str, int] = {}
    for m in misclassifications:
        src = m.get("source", "unknown").split("/")[0]
        by_source[src] = by_source.get(src, 0) + 1
    print(f"\n  Errors by source:")
    for src, count in sorted(by_source.items(), key=lambda x: -x[1]):
        print(f"    {src}: {count}")

    if verbose:
        print(f"\n  Sample misclassifications (first 30):")
        for m in sorted(misclassifications, key=lambda x: x["score"])[:30]:
            print(f"    [{m['true_label']:>7} → {m['predicted']:<7}] "
                  f"score={m['score']:.3f} id={m.get('id','?'):>10} "
                  f"| {m['text_preview'][:60]}...")


# ─── Score Distribution ─────────────────────────────────────────────────────

def print_score_distribution(data: list[dict[str, Any]], analyzer: RequestAnalyzer) -> None:
    """Print score histograms per label."""
    print(f"\n{'=' * 70}")
    print("  SCORE DISTRIBUTION BY LABEL")
    print(f"{'=' * 70}")

    scores_by_label: dict[str, list[float]] = {"simple": [], "medium": [], "complex": []}
    for item in data:
        text = item["text"][:4000]
        messages = [{"role": "user", "content": [{"text": text}]}]
        result = analyzer.analyze(messages)
        scores_by_label[item["label"]].append(result.complexity_score)

    for label in ["simple", "medium", "complex"]:
        scores = sorted(scores_by_label[label])
        if not scores:
            continue
        print(f"\n  {label.upper()} ({len(scores)} samples):")
        print(f"    Range: [{min(scores):.3f}, {max(scores):.3f}]")
        print(f"    P10={scores[len(scores)//10]:.3f}  "
              f"P25={scores[len(scores)//4]:.3f}  "
              f"Median={statistics.median(scores):.3f}  "
              f"P75={scores[3*len(scores)//4]:.3f}  "
              f"P90={scores[int(len(scores)*0.9)]:.3f}")

        # ASCII histogram
        bins = [0] * 20
        for s in scores:
            idx = min(19, int(s * 20))
            bins[idx] += 1
        max_count = max(bins) if bins else 1
        for i, count in enumerate(bins):
            if count > 0:
                bar = "█" * int(count / max_count * 30)
                print(f"    {i*0.05:.2f}-{(i+1)*0.05:.2f} {count:>4} {bar}")


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Test and tune the heuristic complexity classifier")
    parser.add_argument("--eval-only", action="store_true", help="Only evaluate, don't tune")
    parser.add_argument("--verbose", action="store_true", help="Show misclassification samples")
    parser.add_argument("--tune-weights", action="store_true", help="Also tune dimension weights (slower)")
    parser.add_argument("--iterations", type=int, default=300, help="Weight tuning iterations")
    args = parser.parse_args()

    print("=" * 70)
    print("  HEURISTIC CLASSIFIER — TEST & TUNE")
    print("=" * 70)

    # Load data
    data = load_all_data()
    if not data:
        print("ERROR: No labeled data found.")
        sys.exit(1)

    # Precompute text for fast evaluation
    precomputed = [(item["text"][:4000], item["label"]) for item in data]

    # ── Step 1: Evaluate current settings ───────────────────────────────
    print("\n" + "─" * 70)
    print("  STEP 1: Evaluate current heuristic settings")
    print("─" * 70)

    analyzer = RequestAnalyzer()
    print(f"\n  Current thresholds: simple_max={analyzer.thresholds.simple_max}, "
          f"moderate_max={analyzer.thresholds.moderate_max}, "
          f"complex_max={analyzer.thresholds.complex_max}, "
          f"reasoning_marker_count={analyzer.thresholds.reasoning_marker_count}")

    result = evaluate_full(analyzer, data)
    print_eval(result, "Current Heuristic Performance")
    error_analysis(result, verbose=args.verbose)

    if args.eval_only:
        print_score_distribution(data, analyzer)
        analyze_dimensions(data)
        return

    # ── Step 2: Dimension analysis ──────────────────────────────────────
    print("\n" + "─" * 70)
    print("  STEP 2: Dimension discriminative power analysis")
    print("─" * 70)
    analyze_dimensions(data)

    # ── Step 3: Tune thresholds ─────────────────────────────────────────
    print("\n" + "─" * 70)
    print("  STEP 3: Threshold grid search")
    print("─" * 70)
    best_thresholds, thresh_accuracy = tune_thresholds(precomputed)

    # Evaluate with new thresholds
    tuned_analyzer = RequestAnalyzer(thresholds=best_thresholds)
    tuned_result = evaluate_full(tuned_analyzer, data)
    print_eval(tuned_result, "After Threshold Tuning")
    error_analysis(tuned_result, verbose=args.verbose)

    # ── Step 4: Tune weights (optional) ─────────────────────────────────
    best_weights = None
    if args.tune_weights:
        print("\n" + "─" * 70)
        print("  STEP 4: Weight optimization (hill-climbing random search)")
        print("─" * 70)
        best_weights, weight_accuracy = tune_weights(
            precomputed, thresholds=best_thresholds, iterations=args.iterations
        )

        # Re-tune thresholds with new weights
        print("\n  Re-tuning thresholds with optimized weights...")
        best_thresholds, _ = tune_thresholds(precomputed, weights=best_weights)

        # Final evaluation
        final_analyzer = RequestAnalyzer(weights=best_weights, thresholds=best_thresholds)
        final_result = evaluate_full(final_analyzer, data)
        print_eval(final_result, "After Weight + Threshold Tuning")
        error_analysis(final_result, verbose=args.verbose)
    else:
        print("\n  (Skipping weight tuning. Use --tune-weights to enable.)")

    # ── Step 5: Score distribution with best settings ───────────────────
    print("\n" + "─" * 70)
    print("  STEP 5: Score distribution with tuned settings")
    print("─" * 70)
    final_analyzer = RequestAnalyzer(weights=best_weights, thresholds=best_thresholds)
    print_score_distribution(data, final_analyzer)

    # ── Summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"\n  Before tuning: {result['accuracy']:.1%} accuracy")
    after_acc = tuned_result["accuracy"]
    if best_weights:
        final_analyzer = RequestAnalyzer(weights=best_weights, thresholds=best_thresholds)
        final_r = evaluate_full(final_analyzer, data, collect_errors=False)
        after_acc = final_r["accuracy"]
    print(f"  After tuning:  {after_acc:.1%} accuracy")
    print(f"  Improvement:   {after_acc - result['accuracy']:+.1%}")

    print(f"\n  ┌─────────────────────────────────────────────────────────────┐")
    print(f"  │ Recommended thresholds (update in request_analyzer.py):     │")
    print(f"  ├─────────────────────────────────────────────────────────────┤")
    print(f"  │   simple_max = {best_thresholds.simple_max:<6}                                    │")
    print(f"  │   moderate_max = {best_thresholds.moderate_max:<6}                                  │")
    print(f"  │   complex_max = {best_thresholds.complex_max:<6}                                   │")
    print(f"  │   reasoning_marker_count = {best_thresholds.reasoning_marker_count:<2}                            │")
    print(f"  └─────────────────────────────────────────────────────────────┘")

    if best_weights:
        print(f"\n  ┌─────────────────────────────────────────────────────────────┐")
        print(f"  │ Recommended weights (update in request_analyzer.py):        │")
        print(f"  ├─────────────────────────────────────────────────────────────┤")
        for name in [f.name for f in fields(AnalyzerWeights)]:
            val = getattr(best_weights, name)
            print(f"  │   {name:<20} = {val:.4f}                          │"[:63] + "│")
        print(f"  └─────────────────────────────────────────────────────────────┘")

    print(f"\n  Latency: {result['avg_latency_us']:.0f} µs/classification (heuristic)")
    print(f"  For comparison: ML classifier adds ~50-100ms (embedding + inference)")
    print(f"  Heuristic is ~{int(75_000 / max(1, result['avg_latency_us']))}x faster than ML approach")


if __name__ == "__main__":
    main()
