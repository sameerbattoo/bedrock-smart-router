# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

#!/usr/bin/env python3
"""Analyze benchmark results and generate a detailed markdown report.

Usage:
    python benchmarks/analyze_results.py results/benchmark_judged.json
    python benchmarks/analyze_results.py results/  # Analyze all JSON files in directory
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmarks.runner.config import MODELS, ROUTER_STRATEGIES


def load_results(filepath):
    with open(filepath) as f:
        return json.load(f)


def get_display_name(runner):
    if runner in MODELS:
        return MODELS[runner]["display_name"]
    if runner in ROUTER_STRATEGIES:
        return ROUTER_STRATEGIES[runner]["display_name"]
    return runner


def analyze_and_report(all_data, output_path):
    """Analyze results from one or more benchmark runs and write markdown report."""

    # Merge all results
    all_results = []
    metadata_list = []
    for data in all_data:
        if isinstance(data, list):
            all_results.extend(data)
        elif isinstance(data, dict):
            all_results.extend(data.get("results", []))
            metadata_list.append(data.get("metadata", {}))

    if not all_results:
        print("No results to analyze.")
        return

    # Normalize field names (quick_mix_test uses different keys)
    for r in all_results:
        if "success" not in r:
            r["success"] = "error" not in r
        if "judge_score" not in r and "score" in r:
            r["judge_score"] = r["score"]
        if "model_used" not in r and "model" in r:
            r["model_used"] = r["model"]
        if "actual_cost" not in r:
            # Estimate cost from tokens if available
            from benchmarks.runner.config import MODELS as _MODELS
            model_id = r.get("model_used", r.get("model", ""))
            input_t = r.get("input_tokens", 0)
            output_t = r.get("output_tokens", 0)
            pricing = {
                "us.anthropic.claude-sonnet-4-6": (3.0, 15.0),
                "us.anthropic.claude-haiku-4-5-20251001-v1:0": (0.80, 4.0),
                "us.amazon.nova-pro-v1:0": (0.80, 3.20),
                "us.anthropic.claude-opus-4-7": (15.0, 75.0),
                "global.anthropic.claude-opus-4-7": (15.0, 75.0),
                "us.amazon.nova-micro-v1:0": (0.035, 0.14),
                "us.deepseek.r1-v1:0": (1.35, 5.40),
            }
            rates = pricing.get(model_id, (3.0, 15.0))
            r["actual_cost"] = (input_t * rates[0] + output_t * rates[1]) / 1_000_000

    # Group by runner
    by_runner = defaultdict(list)
    for r in all_results:
        if r.get("success"):
            by_runner[r["runner"]].append(r)

    runner_order = [k for k in list(MODELS.keys()) + list(ROUTER_STRATEGIES.keys()) if k in by_runner]

    # Start building markdown
    lines = []
    lines.append("# Benchmark Analysis Report")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Total results:** {len(all_results)}")
    lines.append(f"**Successful:** {sum(1 for r in all_results if r.get('success'))}")
    lines.append(f"**Failed:** {sum(1 for r in all_results if not r.get('success'))}")
    lines.append(f"**Runners:** {', '.join(runner_order)}")
    lines.append("")

    # Metadata
    if metadata_list:
        m = metadata_list[0]
        lines.append(f"**Region:** {m.get('region', 'unknown')}")
        if m.get("judge_model"):
            lines.append(f"**Judge model:** {m.get('judge_model')}")
        lines.append("")

    # ── Overall Summary Table ────────────────────────────────────
    lines.append("## Overall Summary")
    lines.append("")
    has_scores = any(r.get("judge_score") for r in all_results)

    header = "| Runner | Count | Avg Latency |"
    separator = "|--------|-------|-------------|"
    if has_scores:
        header += " Avg Score |"
        separator += "-----------|"
    header += " Avg Cost | Success Rate |"
    separator += "----------|--------------|"

    lines.append(header)
    lines.append(separator)

    for runner in runner_order:
        runs = by_runner[runner]
        count = len(runs)
        avg_lat = sum(r.get("latency_ms", 0) for r in runs) / count
        avg_cost = sum(r.get("actual_cost", 0) for r in runs) / count
        total_for_runner = sum(1 for r in all_results if r["runner"] == runner)
        success_rate = count / total_for_runner * 100 if total_for_runner else 0
        display = get_display_name(runner)

        row = f"| {display} | {count} | {avg_lat:.0f}ms |"
        if has_scores:
            scored = [r for r in runs if r.get("judge_score")]
            avg_score = sum(r["judge_score"] for r in scored) / len(scored) if scored else 0
            row += f" {avg_score:.2f}/10 |"
        row += f" ${avg_cost:.6f} | {success_rate:.1f}% |"
        lines.append(row)

    lines.append("")

    # ── Score by Category ────────────────────────────────────────
    if has_scores:
        categories = sorted(set(r["category"] for r in all_results if r.get("category")))

        lines.append("## Quality Score by Category")
        lines.append("")
        header = "| Category |"
        separator = "|----------|"
        for runner in runner_order:
            short = get_display_name(runner)[:12]
            header += f" {short} |"
            separator += "------|"
        lines.append(header)
        lines.append(separator)

        for cat in categories:
            row = f"| {cat} |"
            for runner in runner_order:
                runs = [r for r in by_runner[runner] if r["category"] == cat and r.get("judge_score")]
                if runs:
                    avg = sum(r["judge_score"] for r in runs) / len(runs)
                    row += f" {avg:.1f} |"
                else:
                    row += " - |"
            lines.append(row)
        lines.append("")

        # ── Score by Difficulty ──────────────────────────────────
        lines.append("## Quality Score by Difficulty")
        lines.append("")
        header = "| Difficulty |"
        separator = "|------------|"
        for runner in runner_order:
            short = get_display_name(runner)[:12]
            header += f" {short} |"
            separator += "------|"
        lines.append(header)
        lines.append(separator)

        for diff in ["simple", "medium", "complex"]:
            row = f"| {diff} |"
            for runner in runner_order:
                runs = [r for r in by_runner[runner] if r.get("difficulty") == diff and r.get("judge_score")]
                if runs:
                    avg = sum(r["judge_score"] for r in runs) / len(runs)
                    row += f" {avg:.1f} |"
                else:
                    row += " - |"
            lines.append(row)
        lines.append("")

    # ── Cost Analysis ────────────────────────────────────────────
    lines.append("## Cost Analysis")
    lines.append("")
    lines.append("| Runner | Total Cost | Avg/Prompt | vs Sonnet Savings |")
    lines.append("|--------|-----------|-----------|-------------------|")

    sonnet_total = sum(r.get("actual_cost", 0) for r in by_runner.get("sonnet", []))
    for runner in runner_order:
        runs = by_runner[runner]
        total_cost = sum(r.get("actual_cost", 0) for r in runs)
        avg_cost = total_cost / len(runs) if runs else 0
        if sonnet_total > 0 and runner != "sonnet":
            savings = (sonnet_total - total_cost) / sonnet_total * 100
            savings_str = f"{savings:+.1f}%"
        else:
            savings_str = "baseline" if runner == "sonnet" else "N/A"
        display = get_display_name(runner)
        lines.append(f"| {display} | ${total_cost:.4f} | ${avg_cost:.6f} | {savings_str} |")
    lines.append("")

    # ── Latency Analysis ─────────────────────────────────────────
    lines.append("## Latency Analysis")
    lines.append("")
    lines.append("| Runner | Avg | Min | Max | p50 | p95 |")
    lines.append("|--------|-----|-----|-----|-----|-----|")

    for runner in runner_order:
        runs = by_runner[runner]
        lats = sorted(r.get("latency_ms", 0) for r in runs)
        if not lats:
            continue
        avg = sum(lats) / len(lats)
        p50 = lats[len(lats) // 2]
        p95 = lats[int(len(lats) * 0.95)]
        display = get_display_name(runner)
        lines.append(f"| {display} | {avg:.0f}ms | {min(lats):.0f}ms | {max(lats):.0f}ms | {p50:.0f}ms | {p95:.0f}ms |")
    lines.append("")

    # ── Router Model Selection ───────────────────────────────────
    router_runners = [r for r in runner_order if r in ROUTER_STRATEGIES]
    if router_runners:
        lines.append("## Router Model Selection Distribution")
        lines.append("")
        for runner in router_runners:
            runs = by_runner[runner]
            models_used = defaultdict(int)
            for r in runs:
                models_used[r.get("model_used", "unknown")] += 1

            display = get_display_name(runner)
            lines.append(f"### {display}")
            lines.append("")
            lines.append("| Model | Count | Percentage |")
            lines.append("|-------|-------|------------|")
            for model, count in sorted(models_used.items(), key=lambda x: -x[1]):
                pct = count / len(runs) * 100
                lines.append(f"| {model} | {count} | {pct:.1f}% |")
            lines.append("")

    # ── Fallback Analysis ────────────────────────────────────────
    fallback_results = [r for r in all_results if r.get("fallback_used")]
    if fallback_results:
        lines.append("## Fallback Events")
        lines.append("")
        lines.append(f"Total fallbacks triggered: {len(fallback_results)}")
        lines.append("")
        lines.append("| Prompt | Runner | Primary Failed | Fallback Model |")
        lines.append("|--------|--------|----------------|----------------|")
        for r in fallback_results[:20]:
            lines.append(f"| {r.get('prompt_id', '?')} | {r['runner']} | - | {r.get('model_used', '?')} |")
        lines.append("")

    # ── Errors ───────────────────────────────────────────────────
    errors = [r for r in all_results if not r.get("success")]
    if errors:
        lines.append("## Errors")
        lines.append("")
        lines.append(f"Total failures: {len(errors)}")
        lines.append("")

        error_types = defaultdict(int)
        for e in errors:
            error_types[e.get("error_type", "unknown")] += 1

        lines.append("| Error Type | Count |")
        lines.append("|-----------|-------|")
        for et, count in sorted(error_types.items(), key=lambda x: -x[1]):
            lines.append(f"| {et} | {count} |")
        lines.append("")

        lines.append("### Error Details (first 10)")
        lines.append("")
        for e in errors[:10]:
            lines.append(f"- **{e.get('prompt_id', '?')}** ({e['runner']}): {e.get('error', 'unknown')[:100]}")
        lines.append("")

    # ── Key Insights ─────────────────────────────────────────────
    lines.append("## Key Insights")
    lines.append("")

    if has_scores:
        scored_runners = {r: [x for x in by_runner[r] if x.get("judge_score")] for r in runner_order}
        scored_runners = {k: v for k, v in scored_runners.items() if v}

        if scored_runners:
            best_quality = max(scored_runners.keys(), key=lambda r: sum(x["judge_score"] for x in scored_runners[r]) / len(scored_runners[r]))
            best_score = sum(r["judge_score"] for r in scored_runners[best_quality]) / len(scored_runners[best_quality])
            lines.append(f"- **Best quality:** {get_display_name(best_quality)} ({best_score:.2f}/10)")

    cheapest = min(runner_order, key=lambda r: sum(x.get("actual_cost", 0) for x in by_runner[r]) / len(by_runner[r]) if by_runner[r] else float("inf"))
    cheapest_cost = sum(r.get("actual_cost", 0) for r in by_runner[cheapest]) / len(by_runner[cheapest])
    lines.append(f"- **Cheapest:** {get_display_name(cheapest)} (${cheapest_cost:.6f}/prompt)")

    fastest = min(runner_order, key=lambda r: sum(x.get("latency_ms", 0) for x in by_runner[r]) / len(by_runner[r]) if by_runner[r] else float("inf"))
    fastest_lat = sum(r.get("latency_ms", 0) for r in by_runner[fastest]) / len(by_runner[fastest])
    lines.append(f"- **Fastest:** {get_display_name(fastest)} ({fastest_lat:.0f}ms avg)")

    # Router vs Sonnet comparison
    if "router-default" in by_runner and "sonnet" in by_runner:
        lines.append("")
        lines.append("### Router (Default) vs Sonnet Comparison")
        lines.append("")
        r_runs = by_runner["router-default"]
        s_runs = by_runner["sonnet"]
        r_cost = sum(x.get("actual_cost", 0) for x in r_runs) / len(r_runs)
        s_cost = sum(x.get("actual_cost", 0) for x in s_runs) / len(s_runs)
        r_lat = sum(x.get("latency_ms", 0) for x in r_runs) / len(r_runs)
        s_lat = sum(x.get("latency_ms", 0) for x in s_runs) / len(s_runs)

        lines.append(f"| Metric | Router (Default) | Sonnet | Difference |")
        lines.append(f"|--------|-----------------|--------|------------|")
        lines.append(f"| Avg Cost | ${r_cost:.6f} | ${s_cost:.6f} | {(r_cost - s_cost) / s_cost * 100:+.1f}% |")
        lines.append(f"| Avg Latency | {r_lat:.0f}ms | {s_lat:.0f}ms | {(r_lat - s_lat) / s_lat * 100:+.1f}% |")

        if has_scores:
            r_scored = [x for x in r_runs if x.get("judge_score")]
            s_scored = [x for x in s_runs if x.get("judge_score")]
            if r_scored and s_scored:
                r_score = sum(x["judge_score"] for x in r_scored) / len(r_scored)
                s_score = sum(x["judge_score"] for x in s_scored) / len(s_scored)
                lines.append(f"| Avg Quality | {r_score:.2f}/10 | {s_score:.2f}/10 | {r_score - s_score:+.2f} |")
        lines.append("")

    # Write the report
    report_content = "\n".join(lines)

    with open(output_path, "w") as f:
        f.write(report_content)

    print(f"Report written to: {output_path}")
    print(f"  {len(all_results)} results analyzed across {len(runner_order)} runners")

    # Also print summary to stdout
    print("\n--- Quick Summary ---")
    for runner in runner_order:
        runs = by_runner[runner]
        avg_lat = sum(r.get("latency_ms", 0) for r in runs) / len(runs)
        avg_cost = sum(r.get("actual_cost", 0) for r in runs) / len(runs)
        scored = [r for r in runs if r.get("judge_score")]
        avg_score = sum(r["judge_score"] for r in scored) / len(scored) if scored else 0
        score_str = f"{avg_score:.1f}/10" if scored else "N/A"
        print(f"  {get_display_name(runner):<30} Score={score_str:<8} Cost=${avg_cost:.6f}  Latency={avg_lat:.0f}ms")


def main():
    parser = argparse.ArgumentParser(description="Analyze benchmark results and generate markdown report")
    parser.add_argument("input", help="Path to results JSON file or directory containing JSON files")
    parser.add_argument("--output", type=str, help="Output markdown file path (default: results/REPORT.md)")
    args = parser.parse_args()

    # Load results
    all_data = []
    if os.path.isdir(args.input):
        for f in sorted(os.listdir(args.input)):
            if f.endswith(".json"):
                all_data.append(load_results(os.path.join(args.input, f)))
    else:
        all_data.append(load_results(args.input))

    if not all_data:
        print("No results files found.")
        return

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
        os.makedirs(results_dir, exist_ok=True)
        output_path = os.path.join(results_dir, "REPORT.md")

    analyze_and_report(all_data, output_path)


if __name__ == "__main__":
    main()
