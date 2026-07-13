# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

#!/usr/bin/env python3
"""Prepare training data for the complexity classifier.

Combines:
1. Our 295 custom prompts (benchmarks/prompts/) - hand-labeled simple/medium/complex
2. Industry standard multimodal datasets (ind_standard_datasets/) - labeled by task type
3. Existing NLP datasets (datasets/) - labeled by task complexity mapping

Outputs a single training_data.json with format:
[{"text": "full prompt text", "label": "simple|medium|complex", "source": "..."}]
"""
import json
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BENCHMARKS_DIR = os.path.dirname(BASE_DIR)
OUTPUT_PATH = os.path.join(BASE_DIR, "training_data.json")

training_data = []


def add_custom_prompts():
    """Load our 295 hand-labeled prompts."""
    prompts_dir = os.path.join(BENCHMARKS_DIR, "data", "generated")
    count = 0
    for filename in sorted(os.listdir(prompts_dir)):
        if not filename.endswith(".json") or filename.startswith("_"):
            continue
        with open(os.path.join(prompts_dir, filename)) as f:
            prompts = json.load(f)
        for p in prompts:
            # Build the full prompt text as the model would see it
            text_parts = []
            if p.get("system_prompt"):
                text_parts.append(p["system_prompt"])
            if p.get("context"):
                text_parts.append(p["context"])
            if p.get("user_prompt"):
                text_parts.append(p["user_prompt"])
            full_text = "\n\n".join(text_parts)

            # Map difficulty to our 3 labels
            label = p["difficulty"]  # already simple/medium/complex

            training_data.append({
                "text": full_text,
                "label": label,
                "source": f"custom/{filename}",
                "id": p["id"],
            })
            count += 1
    print(f"  Custom prompts: {count}")


def add_multimodal_datasets():
    """Load industry standard multimodal datasets and assign complexity labels."""
    ind_dir = os.path.join(BENCHMARKS_DIR, "data", "industry_standard")
    count = 0

    # Invoice extraction - simple task (structured extraction)
    invoices_path = os.path.join(ind_dir, "multimodal", "invoices", "samples.json")
    if os.path.exists(invoices_path):
        with open(invoices_path) as f:
            samples = json.load(f)
        for s in samples:
            text = f"Extract invoice information from the attached image.\nGround truth fields: {list(s.get('ground_truth', {}).keys())}"
            training_data.append({
                "text": text,
                "label": "simple",  # Structured extraction is simple
                "source": "ind_standard/invoices",
                "id": s["id"],
            })
            count += 1

    # DocVQA - varies by question complexity
    docvqa_path = os.path.join(ind_dir, "multimodal", "docvqa", "samples.json")
    if os.path.exists(docvqa_path):
        with open(docvqa_path) as f:
            samples = json.load(f)
        for s in samples:
            question = s.get("question", "")
            # Heuristic: short questions with "what is" = simple, longer = medium
            word_count = len(question.split())
            if word_count <= 8:
                label = "simple"
            elif word_count <= 15:
                label = "medium"
            else:
                label = "complex"
            text = f"Look at the document image and answer: {question}"
            training_data.append({
                "text": text,
                "label": label,
                "source": "ind_standard/docvqa",
                "id": s["id"],
            })
            count += 1

    # ChartQA - reasoning about visual data
    chartqa_path = os.path.join(ind_dir, "multimodal", "chartqa", "samples.json")
    if os.path.exists(chartqa_path):
        with open(chartqa_path) as f:
            samples = json.load(f)
        for s in samples:
            question = s.get("question", "")
            # Chart questions requiring counting/comparison = medium, reasoning = complex
            lower_q = question.lower()
            if any(w in lower_q for w in ["how many", "what is the", "which"]):
                label = "simple"
            elif any(w in lower_q for w in ["compare", "difference", "trend", "change"]):
                label = "complex"
            else:
                label = "medium"
            text = f"Analyze the chart image and answer: {question}"
            training_data.append({
                "text": text,
                "label": label,
                "source": "ind_standard/chartqa",
                "id": s["id"],
            })
            count += 1

    # PDF documents - multi-page understanding
    dude_path = os.path.join(ind_dir, "pdfs", "dude_pdfs", "samples.json")
    if os.path.exists(dude_path):
        with open(dude_path) as f:
            samples = json.load(f)
        for s in samples:
            question = s.get("question", "")
            num_pages = s.get("num_pages", 1)
            # Multi-page docs are inherently more complex
            if num_pages <= 3:
                label = "simple"
            elif num_pages <= 10:
                label = "medium"
            else:
                label = "complex"
            text = f"Read the {num_pages}-page PDF document and answer: {question}"
            training_data.append({
                "text": text,
                "label": label,
                "source": "ind_standard/dude_pdfs",
                "id": s["id"],
            })
            count += 1

    print(f"  Multimodal datasets: {count}")


def add_complex_reasoning_datasets():
    """Load complex reasoning datasets (GSM8K, GPQA, MBPP)."""
    ind_dir = os.path.join(BENCHMARKS_DIR, "data", "industry_standard")
    count = 0

    # GSM8K - multi-step math reasoning
    gsm8k_path = os.path.join(ind_dir, "reasoning", "gsm8k", "samples.json")
    if os.path.exists(gsm8k_path):
        with open(gsm8k_path) as f:
            samples = json.load(f)
        for s in samples:
            text = f"Solve this math problem step by step:\n\n{s['question']}"
            training_data.append({
                "text": text,
                "label": "complex",
                "source": "ind_standard/gsm8k",
                "id": s["id"],
            })
            count += 1

    # GPQA - graduate-level science
    gpqa_path = os.path.join(ind_dir, "reasoning", "gpqa", "samples.json")
    if os.path.exists(gpqa_path):
        with open(gpqa_path) as f:
            samples = json.load(f)
        for s in samples:
            text = f"Answer this graduate-level science question:\n\n{s['question']}"
            training_data.append({
                "text": text,
                "label": "complex",
                "source": "ind_standard/gpqa",
                "id": s["id"],
            })
            count += 1

    # MBPP - complex Python coding
    mbpp_path = os.path.join(ind_dir, "reasoning", "mbpp", "samples.json")
    if os.path.exists(mbpp_path):
        with open(mbpp_path) as f:
            samples = json.load(f)
        for s in samples:
            text = f"Write a Python function:\n\n{s['prompt']}"
            training_data.append({
                "text": text,
                "label": "complex",
                "source": "ind_standard/mbpp",
                "id": s["id"],
            })
            count += 1

    print(f"  Complex reasoning datasets: {count}")


def add_more_training_data():
    """Load additional datasets (ARC, HellaSwag, WinoGrande, MMLU, Alpaca)."""
    ind_dir = os.path.join(BENCHMARKS_DIR, "data", "industry_standard")
    count = 0

    # Map dataset folder -> (subdir, label, text_builder)
    additional = {
        "competition_math": ("reasoning", "complex", lambda s: f"Solve this competition math problem:\n\n{s.get('question', '')}"),
        "arc_challenge": ("reasoning", "complex", lambda s: f"Answer this science question:\n\n{s.get('question', '')}"),
        "humaneval": ("reasoning", "complex", lambda s: f"Complete this Python function:\n\n{s.get('prompt', '')}"),
        "hellaswag": ("auxiliary", "medium", lambda s: f"Complete this sentence with the most logical continuation:\n\n{s.get('context', '')}"),
        "winogrande": ("auxiliary", "medium", lambda s: f"Resolve the pronoun in this sentence:\n\n{s.get('sentence', '')}\nOption 1: {s.get('option1', '')}\nOption 2: {s.get('option2', '')}"),
        "mmlu_easy": ("auxiliary", "simple", lambda s: f"Answer this question:\n\n{s.get('question', '')}"),
        "alpaca_simple": ("auxiliary", "simple", lambda s: s.get("instruction", "")),
        "commonsense_qa": ("auxiliary", "medium", lambda s: f"Answer this commonsense question:\n\n{s.get('question', '')}"),
    }

    for folder, (subdir, label, text_fn) in additional.items():
        samples_path = os.path.join(ind_dir, subdir, folder, "samples.json")
        if not os.path.exists(samples_path):
            continue
        with open(samples_path) as f:
            samples = json.load(f)
        for s in samples:
            text = text_fn(s)
            if not text or len(text) < 10:
                continue
            training_data.append({
                "text": text,
                "label": label,
                "source": f"ind_standard/{folder}",
                "id": s.get("id", f"{folder}_{count}"),
            })
            count += 1

    print(f"  Additional datasets: {count}")


def add_existing_datasets():
    """Load prompts from the existing datasets/ folder (BoolQ, TriviaQA, etc.)."""
    datasets_dir = os.path.join(BENCHMARKS_DIR, "data", "industry_standard")
    count = 0

    # Task type -> complexity mapping
    task_complexity = {
        "boolq": "simple",           # Yes/no with passage
        "triviaqa": "medium",        # Open-domain QA
        "natural_questions": "medium",  # RAG-style QA
        "trex": "simple",            # Factual extraction
        "xsum": "medium",            # Summarization
        "wikitext2": "simple",       # Text continuation
        "womens_clothing": "simple", # Classification
        "bold": "medium",            # Open-ended generation
        "real_toxicity_prompts": "simple",  # Generation
    }

    for dataset_name, complexity in task_complexity.items():
        output_dir = os.path.join(datasets_dir, "text_datasets", dataset_name, "output")
        if not os.path.exists(output_dir):
            continue

        # Find the modeleval JSON file
        for filename in os.listdir(output_dir):
            if filename.endswith("_modeleval.json"):
                filepath = os.path.join(output_dir, filename)
                try:
                    with open(filepath) as f:
                        data = json.load(f)
                except (json.JSONDecodeError, IOError):
                    continue

                # Extract prompts from evaluations
                for evaluation in data.get("evaluations", []):
                    system_prompt = evaluation.get("systemPrompt", "")
                    user_prompts = evaluation.get("userPrompts", [])
                    variable_sets = evaluation.get("promptVariableSets", [])

                    # Take up to 50 samples per evaluation
                    for j, var_set in enumerate(variable_sets[:50]):
                        if not user_prompts:
                            continue
                        # Render the first user prompt template with variables
                        prompt_template = user_prompts[0]
                        rendered = prompt_template
                        for key, value in var_set.items():
                            rendered = rendered.replace(f"{{{{{key}}}}}", str(value))

                        full_text = f"{system_prompt}\n\n{rendered}" if system_prompt else rendered

                        training_data.append({
                            "text": full_text,
                            "label": complexity,
                            "source": f"datasets/{dataset_name}",
                            "id": f"{dataset_name}_{j:03d}",
                        })
                        count += 1

    print(f"  Existing datasets: {count}")


def main():
    print("=" * 60)
    print("Preparing training data for complexity classifier")
    print("=" * 60)

    print("\nLoading data sources:")
    add_custom_prompts()
    add_multimodal_datasets()
    add_complex_reasoning_datasets()
    add_more_training_data()
    add_existing_datasets()

    # Summary
    print(f"\n{'='*60}")
    print(f"Total training samples: {len(training_data)}")

    label_counts = {}
    source_counts = {}
    for item in training_data:
        label_counts[item["label"]] = label_counts.get(item["label"], 0) + 1
        src = item["source"].split("/")[0]
        source_counts[src] = source_counts.get(src, 0) + 1

    print(f"\nBy label:")
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count} ({count/len(training_data)*100:.1f}%)")

    print(f"\nBy source:")
    for src, count in sorted(source_counts.items(), key=lambda x: -x[1]):
        print(f"  {src}: {count}")

    # Save
    with open(OUTPUT_PATH, "w") as f:
        json.dump(training_data, f, indent=2)

    print(f"\nSaved to: {OUTPUT_PATH}")
    print(f"File size: {os.path.getsize(OUTPUT_PATH) / 1024:.1f} KB")


if __name__ == "__main__":
    main()
