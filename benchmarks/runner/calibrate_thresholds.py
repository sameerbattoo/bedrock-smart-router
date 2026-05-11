#!/usr/bin/env python3
"""Calibrate complexity thresholds using labeled benchmark prompts.

Runs the analyzer on all 295 prompts (which have known difficulty labels),
shows the score distribution by difficulty, and recommends optimal thresholds.

Usage:
    python benchmarks/calibrate_thresholds.py
"""
import sys
import os
import json
import statistics

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bedrock_smart_router.request_analyzer import RequestAnalyzer

analyzer = RequestAnalyzer()

# Load all prompts
prompts_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "generated")
all_prompts = []
for filename in sorted(os.listdir(prompts_dir)):
    if filename.endswith(".json") and not filename.startswith("_"):
        with open(os.path.join(prompts_dir, filename)) as f:
            all_prompts.extend(json.load(f))

print(f"Loaded {len(all_prompts)} prompts")
print(f"Current thresholds: simple_max={analyzer.thresholds.simple_max}, "
      f"moderate_max={analyzer.thresholds.moderate_max}, "
      f"complex_max={analyzer.thresholds.complex_max}")
print()

# Run analyzer on all prompts
scores_by_difficulty = {"simple": [], "medium": [], "complex": []}
scores_by_category = {}
misclassifications = []

for p in all_prompts:
    user_text = p["user_prompt"]
    if p.get("context"):
        user_text = f"{p['context']}\n\n{p['user_prompt']}"
    messages = [{"role": "user", "content": [{"text": user_text}]}]
    system = [{"text": p["system_prompt"]}] if p.get("system_prompt") else None

    result = analyzer.analyze(messages, system)
    score = result.complexity_score
    classified = result.complexity.value

    labeled = p["difficulty"]
    scores_by_difficulty[labeled].append(score)

    cat = p["category"]
    scores_by_category.setdefault(cat, {"simple": [], "medium": [], "complex": []})
    scores_by_category[cat][labeled].append(score)

    # Check misclassification
    # Map: simple->simple, medium->moderate, complex->complex/reasoning
    expected_map = {"simple": "simple", "medium": "moderate", "complex": ["complex", "reasoning"]}
    expected = expected_map[labeled]
    if isinstance(expected, list):
        is_correct = classified in expected
    else:
        is_correct = classified == expected
    if not is_correct:
        misclassifications.append({
            "id": p["id"], "category": cat, "labeled": labeled,
            "classified": classified, "score": score,
        })

# Print distribution
print("=" * 70)
print("SCORE DISTRIBUTION BY LABELED DIFFICULTY")
print("=" * 70)

for diff in ["simple", "medium", "complex"]:
    scores = scores_by_difficulty[diff]
    if not scores:
        continue
    print(f"\n  {diff.upper()} ({len(scores)} prompts):")
    print(f"    Min:    {min(scores):.4f}")
    print(f"    P25:    {sorted(scores)[len(scores)//4]:.4f}")
    print(f"    Median: {statistics.median(scores):.4f}")
    print(f"    P75:    {sorted(scores)[3*len(scores)//4]:.4f}")
    print(f"    Max:    {max(scores):.4f}")
    print(f"    Mean:   {statistics.mean(scores):.4f}")
    if len(scores) > 1:
        print(f"    StdDev: {statistics.stdev(scores):.4f}")

    # Histogram
    buckets = [0]*10
    for s in scores:
        idx = min(9, int(s * 10))
        buckets[idx] += 1
    print(f"    Distribution: ", end="")
    for i, count in enumerate(buckets):
        bar = "#" * count
        if count > 0:
            print(f"\n      {i*0.1:.1f}-{(i+1)*0.1:.1f}: {bar} ({count})", end="")
    print()

# Print by category
print("\n\n" + "=" * 70)
print("MEDIAN SCORES BY CATEGORY AND DIFFICULTY")
print("=" * 70)
print(f"\n  {'Category':<25} {'Simple':<10} {'Medium':<10} {'Complex':<10}")
print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10}")
for cat in sorted(scores_by_category.keys()):
    row = f"  {cat:<25}"
    for diff in ["simple", "medium", "complex"]:
        scores = scores_by_category[cat][diff]
        if scores:
            row += f" {statistics.median(scores):<10.4f}"
        else:
            row += f" {'N/A':<10}"
    print(row)

# Misclassification analysis
print("\n\n" + "=" * 70)
print(f"MISCLASSIFICATION ANALYSIS (current thresholds)")
print("=" * 70)

total = len(all_prompts)
correct = total - len(misclassifications)
print(f"\n  Accuracy: {correct}/{total} ({correct/total*100:.1f}%)")
print(f"  Misclassified: {len(misclassifications)}")

# Group misclassifications
by_type = {}
for m in misclassifications:
    key = f"{m['labeled']} classified as {m['classified']}"
    by_type.setdefault(key, []).append(m)

print(f"\n  Breakdown:")
for key, items in sorted(by_type.items(), key=lambda x: -len(x[1])):
    print(f"    {key}: {len(items)} prompts")
    for item in items[:3]:
        print(f"      - {item['id']} (score={item['score']:.4f})")
    if len(items) > 3:
        print(f"      ... and {len(items)-3} more")

# Recommend thresholds
print("\n\n" + "=" * 70)
print("RECOMMENDED THRESHOLDS")
print("=" * 70)

simple_scores = sorted(scores_by_difficulty["simple"])
medium_scores = sorted(scores_by_difficulty["medium"])
complex_scores = sorted(scores_by_difficulty["complex"])

# Optimal simple_max: between max(simple) and min(medium)
# Or use P90 of simple as the boundary
if simple_scores and medium_scores:
    simple_p90 = simple_scores[int(len(simple_scores) * 0.9)]
    medium_p10 = medium_scores[int(len(medium_scores) * 0.1)]
    recommended_simple_max = (simple_p90 + medium_p10) / 2
    print(f"\n  simple_max:")
    print(f"    Simple P90: {simple_p90:.4f}")
    print(f"    Medium P10: {medium_p10:.4f}")
    print(f"    Recommended: {recommended_simple_max:.4f}")

if medium_scores and complex_scores:
    medium_p90 = medium_scores[int(len(medium_scores) * 0.9)]
    complex_p10 = complex_scores[int(len(complex_scores) * 0.1)]
    recommended_moderate_max = (medium_p90 + complex_p10) / 2
    print(f"\n  moderate_max:")
    print(f"    Medium P90: {medium_p90:.4f}")
    print(f"    Complex P10: {complex_p10:.4f}")
    print(f"    Recommended: {recommended_moderate_max:.4f}")

print(f"\n  Current:     simple_max={analyzer.thresholds.simple_max}, moderate_max={analyzer.thresholds.moderate_max}")
if simple_scores and medium_scores and complex_scores:
    print(f"  Recommended: simple_max={recommended_simple_max:.4f}, moderate_max={recommended_moderate_max:.4f}")
