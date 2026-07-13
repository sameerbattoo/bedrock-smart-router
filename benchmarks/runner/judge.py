# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

#!/usr/bin/env python3
"""LLM-as-Judge: scores benchmark responses using Sonnet 4.6.

Usage:
    python benchmarks/judge.py results/benchmark_20240315_120000.json
    python benchmarks/judge.py results/benchmark_20240315_120000.json --limit 50
"""

import argparse
import json
import os
import sys
import time

import boto3

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmarks.runner.config import JUDGE_MODEL_ID, JUDGE_SYSTEM_PROMPT_WITH_ANSWER, REGION


def load_prompts_map():
    """Load all prompts into a dict keyed by prompt_id."""
    prompts_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "generated")
    prompts_map = {}
    for filename in os.listdir(prompts_dir):
        if not filename.endswith(".json"):
            continue
        with open(os.path.join(prompts_dir, filename)) as f:
            for prompt in json.load(f):
                prompts_map[prompt["id"]] = prompt
    return prompts_map


def judge_response(client, prompt, response_text):
    """Score a response using the judge model."""
    judge_prompt = JUDGE_SYSTEM_PROMPT_WITH_ANSWER.format(
        system_prompt=prompt["system_prompt"],
        user_prompt=prompt["user_prompt"],
        expected_answer=prompt.get("expected_answer", "N/A"),
        response=response_text,
    )

    response = client.converse(
        modelId=JUDGE_MODEL_ID,
        messages=[{"role": "user", "content": [{"text": judge_prompt}]}],
    )

    output_text = response["output"]["message"]["content"][0]["text"]

    # Parse JSON response
    try:
        # Handle potential markdown fences
        text = output_text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0]
        result = json.loads(text)
        return {
            "score": result.get("score", 0),
            "reasoning": result.get("reasoning", ""),
        }
    except (json.JSONDecodeError, KeyError):
        return {
            "score": 0,
            "reasoning": f"Failed to parse judge response: {output_text[:200]}",
            "parse_error": True,
        }


def main():
    parser = argparse.ArgumentParser(description="Score benchmark results with LLM judge")
    parser.add_argument("results_file", help="Path to benchmark results JSON")
    parser.add_argument("--limit", type=int, help="Limit number of judgments")
    parser.add_argument("--output", type=str, help="Output file (default: adds _judged suffix)")
    args = parser.parse_args()

    # Load results
    with open(args.results_file) as f:
        data = json.load(f)

    results = data["results"]
    prompts_map = load_prompts_map()

    # Filter to successful results that have response text
    to_judge = [
        r for r in results
        if r.get("success") and r.get("response_text")
    ]

    if args.limit:
        to_judge = to_judge[:args.limit]

    print(f"Judging {len(to_judge)} responses using {JUDGE_MODEL_ID}...")

    # Setup client
    client = boto3.Session(region_name=REGION).client("bedrock-runtime")

    judged_count = 0
    for i, result in enumerate(to_judge):
        prompt = prompts_map.get(result["prompt_id"])
        if not prompt:
            print(f"  [{i+1}/{len(to_judge)}] {result['prompt_id']} - SKIP (prompt not found)")
            continue

        print(f"  [{i+1}/{len(to_judge)}] {result['prompt_id']} ({result['runner']})...", end=" ", flush=True)

        try:
            judgment = judge_response(client, prompt, result["response_text"])
            result["judge_score"] = judgment["score"]
            result["judge_reasoning"] = judgment["reasoning"]
            if judgment.get("parse_error"):
                result["judge_parse_error"] = True
            judged_count += 1
            print(f"Score: {judgment['score']}/10")
        except Exception as e:
            result["judge_score"] = 0
            result["judge_reasoning"] = f"Judge error: {str(e)}"
            result["judge_error"] = True
            print(f"ERROR: {str(e)[:60]}")

        # Rate limiting - avoid throttling the judge model
        time.sleep(0.5)

    # Save judged results
    if args.output:
        output_path = args.output
    else:
        base, ext = os.path.splitext(args.results_file)
        output_path = f"{base}_judged{ext}"

    data["metadata"]["judge_model"] = JUDGE_MODEL_ID
    data["metadata"]["judged_count"] = judged_count
    data["metadata"]["judge_timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nJudged {judged_count} responses")
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
