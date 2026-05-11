#!/usr/bin/env python3
"""Download additional datasets to improve classifier training.

Targets:
- More COMPLEX samples: MATH (competition math), ARC-Challenge, HumanEval
- More MEDIUM samples: HellaSwag (commonsense reasoning), WinoGrande
- More SIMPLE samples: MMLU (easy subset), simple QA

Goal: Get to ~3000+ balanced samples.

Requirements:
    pip install datasets
"""
import json
import os
import sys

try:
    from datasets import load_dataset
except ImportError:
    print("ERROR: pip install datasets")
    sys.exit(1)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def download_math():
    """Competition-level math problems (COMPLEX)."""
    print("\n" + "=" * 60)
    print("1. hendrycks/competition_math (competition math - COMPLEX)")
    print("=" * 60)

    out_dir = os.path.join(BASE_DIR, "competition_math")
    os.makedirs(out_dir, exist_ok=True)

    ds = load_dataset("lighteval/MATH", split="test")
    print(f"  Size: {len(ds)}, Columns: {ds.column_names}")

    # Take hardest problems (level 4-5)
    items = [item for item in ds if item.get("level", "") in ["Level 4", "Level 5"]]
    print(f"  Level 4-5: {len(items)} problems")

    samples = []
    for i, item in enumerate(items[:100]):
        samples.append({
            "id": f"math_{i:03d}",
            "question": item.get("problem", ""),
            "solution": item.get("solution", ""),
            "level": item.get("level", ""),
            "type": item.get("type", ""),
            "task": "competition_math",
            "difficulty": "complex",
        })

    with open(os.path.join(out_dir, "samples.json"), "w") as f:
        json.dump(samples, f, indent=2)
    print(f"  Saved {len(samples)} samples")
    return len(samples)


def download_arc_challenge():
    """Science reasoning questions (COMPLEX)."""
    print("\n" + "=" * 60)
    print("2. allenai/ai2_arc Challenge (science reasoning - COMPLEX)")
    print("=" * 60)

    out_dir = os.path.join(BASE_DIR, "arc_challenge")
    os.makedirs(out_dir, exist_ok=True)

    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    print(f"  Size: {len(ds)}, Columns: {ds.column_names}")

    samples = []
    for i, item in enumerate(ds):
        if i >= 100:
            break
        question = item.get("question", "")
        choices = item.get("choices", {})
        answer_key = item.get("answerKey", "")

        # Format choices
        choice_texts = choices.get("text", [])
        choice_labels = choices.get("label", [])
        formatted_choices = "\n".join(f"  {l}) {t}" for l, t in zip(choice_labels, choice_texts))

        samples.append({
            "id": f"arc_{i:03d}",
            "question": f"{question}\n{formatted_choices}",
            "answer_key": answer_key,
            "task": "science_reasoning",
            "difficulty": "complex",
        })

    with open(os.path.join(out_dir, "samples.json"), "w") as f:
        json.dump(samples, f, indent=2)
    print(f"  Saved {len(samples)} samples")
    return len(samples)


def download_hellaswag():
    """Commonsense reasoning - sentence completion (MEDIUM)."""
    print("\n" + "=" * 60)
    print("3. Rowan/hellaswag (commonsense reasoning - MEDIUM)")
    print("=" * 60)

    out_dir = os.path.join(BASE_DIR, "hellaswag")
    os.makedirs(out_dir, exist_ok=True)

    ds = load_dataset("Rowan/hellaswag", split="validation")
    print(f"  Size: {len(ds)}, Columns: {ds.column_names}")

    samples = []
    for i, item in enumerate(ds):
        if i >= 150:
            break
        context = item.get("ctx", item.get("ctx_a", ""))
        endings = item.get("endings", [])
        label = item.get("label", 0)

        if not context or not endings:
            continue

        samples.append({
            "id": f"hswag_{i:03d}",
            "context": context,
            "endings": endings,
            "correct_ending": endings[int(label)] if int(label) < len(endings) else "",
            "task": "commonsense_reasoning",
            "difficulty": "medium",
        })

    with open(os.path.join(out_dir, "samples.json"), "w") as f:
        json.dump(samples, f, indent=2)
    print(f"  Saved {len(samples)} samples")
    return len(samples)


def download_winogrande():
    """Commonsense coreference resolution (MEDIUM)."""
    print("\n" + "=" * 60)
    print("4. allenai/winogrande (coreference resolution - MEDIUM)")
    print("=" * 60)

    out_dir = os.path.join(BASE_DIR, "winogrande")
    os.makedirs(out_dir, exist_ok=True)

    ds = load_dataset("allenai/winogrande", "winogrande_xl", split="validation")
    print(f"  Size: {len(ds)}, Columns: {ds.column_names}")

    samples = []
    for i, item in enumerate(ds):
        if i >= 150:
            break
        sentence = item.get("sentence", "")
        option1 = item.get("option1", "")
        option2 = item.get("option2", "")
        answer = item.get("answer", "")

        if not sentence:
            continue

        samples.append({
            "id": f"wino_{i:03d}",
            "sentence": sentence,
            "option1": option1,
            "option2": option2,
            "answer": answer,
            "task": "coreference_resolution",
            "difficulty": "medium",
        })

    with open(os.path.join(out_dir, "samples.json"), "w") as f:
        json.dump(samples, f, indent=2)
    print(f"  Saved {len(samples)} samples")
    return len(samples)


def download_mmlu_easy():
    """MMLU easy questions - elementary/high school level (SIMPLE)."""
    print("\n" + "=" * 60)
    print("5. cais/mmlu easy subsets (simple knowledge - SIMPLE)")
    print("=" * 60)

    out_dir = os.path.join(BASE_DIR, "mmlu_easy")
    os.makedirs(out_dir, exist_ok=True)

    # Use elementary and high school subjects
    easy_subjects = [
        "elementary_mathematics", "high_school_biology",
        "high_school_geography", "high_school_us_history",
    ]

    samples = []
    for subject in easy_subjects:
        try:
            ds = load_dataset("cais/mmlu", subject, split="test")
            print(f"  {subject}: {len(ds)} samples")
            for i, item in enumerate(ds):
                if len(samples) >= 200:
                    break
                question = item.get("question", "")
                choices = item.get("choices", [])
                answer = item.get("answer", 0)

                if not question:
                    continue

                formatted = question
                if choices:
                    formatted += "\n" + "\n".join(f"  {chr(65+j)}) {c}" for j, c in enumerate(choices))

                samples.append({
                    "id": f"mmlu_{len(samples):03d}",
                    "question": formatted,
                    "answer_idx": answer,
                    "subject": subject,
                    "task": "knowledge_qa",
                    "difficulty": "simple",
                })
        except Exception as e:
            print(f"  {subject} failed: {e}")

    with open(os.path.join(out_dir, "samples.json"), "w") as f:
        json.dump(samples, f, indent=2)
    print(f"  Saved {len(samples)} total MMLU samples")
    return len(samples)


def download_alpaca_simple():
    """Simple instruction-following from Alpaca (SIMPLE)."""
    print("\n" + "=" * 60)
    print("6. tatsu-lab/alpaca (simple instructions - SIMPLE)")
    print("=" * 60)

    out_dir = os.path.join(BASE_DIR, "alpaca_simple")
    os.makedirs(out_dir, exist_ok=True)

    ds = load_dataset("tatsu-lab/alpaca", split="train")
    print(f"  Size: {len(ds)}, Columns: {ds.column_names}")

    # Filter for short, simple instructions (no input context = simpler)
    simple_items = [item for item in ds if not item.get("input", "") and len(item.get("instruction", "")) < 100]
    print(f"  Simple (no input, short instruction): {len(simple_items)}")

    import random
    random.seed(42)
    random.shuffle(simple_items)

    samples = []
    for i, item in enumerate(simple_items[:150]):
        samples.append({
            "id": f"alpaca_{i:03d}",
            "instruction": item.get("instruction", ""),
            "output": item.get("output", ""),
            "task": "instruction_following",
            "difficulty": "simple",
        })

    with open(os.path.join(out_dir, "samples.json"), "w") as f:
        json.dump(samples, f, indent=2)
    print(f"  Saved {len(samples)} samples")
    return len(samples)


def main():
    print("Downloading additional training data")
    print(f"Output: {BASE_DIR}")

    total = 0
    for fn in [download_math, download_arc_challenge, download_hellaswag, download_winogrande, download_mmlu_easy, download_alpaca_simple]:
        try:
            total += fn()
        except Exception as e:
            print(f"  FAILED: {e}")
            continue

    print("\n" + "=" * 60)
    print(f"DONE: {total} additional samples downloaded")
    print("=" * 60)
    print(f"  Complex: ~200 (math + arc)")
    print(f"  Medium: ~300 (hellaswag + winogrande)")
    print(f"  Simple: ~350 (mmlu + alpaca)")


if __name__ == "__main__":
    main()
