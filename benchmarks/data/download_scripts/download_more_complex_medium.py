# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

#!/usr/bin/env python3
"""Download more complex and medium samples to balance the training data.

Current distribution: simple=1044, medium=1013, complex=488
Target: Get complex to ~800+ and medium to ~1200+

Downloads:
- COMPLEX: More GSM8K (200), more ARC (100), BigBenchHard (100), HumanEval (50)
- MEDIUM: More HellaSwag (200), PIQA (150), SocialIQA (150)

Requirements: pip install datasets
"""
import json
import os
import sys

try:
    from datasets import load_dataset
except ImportError:
    print("ERROR: pip install datasets")
    sys.exit(1)

BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "industry_standard")


def save_samples(subdir, name, samples):
    out_dir = os.path.join(BASE_DIR, subdir, name)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "samples.json")
    
    # Append to existing if present
    existing = []
    if os.path.exists(path):
        with open(path) as f:
            existing = json.load(f)
    
    # Deduplicate by id
    existing_ids = {s["id"] for s in existing}
    new_samples = [s for s in samples if s["id"] not in existing_ids]
    combined = existing + new_samples
    
    with open(path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"    Saved {len(combined)} total ({len(new_samples)} new) to {subdir}/{name}/")
    return len(new_samples)


# ═══════════════════════════════════════════════════════════════
# COMPLEX SAMPLES
# ═══════════════════════════════════════════════════════════════

def download_more_gsm8k():
    """Get 200 more GSM8K problems (different from the first 50)."""
    print("\n  More GSM8K (math reasoning)...")
    ds = load_dataset("openai/gsm8k", "main", split="test")
    items = list(ds)
    # Sort by answer length, skip first 50 (already downloaded)
    items.sort(key=lambda x: len(x.get("answer", "")), reverse=True)
    items = items[50:250]  # Next 200

    samples = []
    for i, item in enumerate(items):
        question = item.get("question", "")
        answer = item.get("answer", "")
        final = answer.split("####")[-1].strip() if "####" in answer else ""
        samples.append({
            "id": f"gsm8k_{50+i:03d}",
            "question": question,
            "full_solution": answer,
            "final_answer": final,
            "task": "math_reasoning",
            "difficulty": "complex",
            "source": "openai/gsm8k",
        })
    return save_samples("reasoning", "gsm8k", samples)


def download_more_arc():
    """Get 100 more ARC-Challenge problems."""
    print("\n  More ARC-Challenge (science reasoning)...")
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    items = list(ds)[100:200]  # Skip first 100

    samples = []
    for i, item in enumerate(items):
        question = item.get("question", "")
        choices = item.get("choices", {})
        choice_texts = choices.get("text", [])
        choice_labels = choices.get("label", [])
        formatted = question + "\n" + "\n".join(f"  {l}) {t}" for l, t in zip(choice_labels, choice_texts))
        samples.append({
            "id": f"arc_{100+i:03d}",
            "question": formatted,
            "answer_key": item.get("answerKey", ""),
            "task": "science_reasoning",
            "difficulty": "complex",
        })
    return save_samples("reasoning", "arc_challenge", samples)


def download_more_math():
    """Get 200 more competition math problems."""
    print("\n  More competition math...")
    samples = []
    for subject in ["algebra", "geometry", "number_theory", "counting_and_probability", "precalculus", "intermediate_algebra"]:
        try:
            ds = load_dataset("EleutherAI/hendrycks_math", subject, split="test")
            hard = [item for item in ds if "Level 4" in item.get("level", "") or "Level 5" in item.get("level", "")]
            for item in hard[20:60]:  # Skip first 20 (already have), take next 40
                if len(samples) >= 200:
                    break
                samples.append({
                    "id": f"math_{100+len(samples):03d}",
                    "question": item.get("problem", ""),
                    "solution": item.get("solution", ""),
                    "level": item.get("level", ""),
                    "type": subject,
                    "task": "competition_math",
                    "difficulty": "complex",
                })
        except Exception as e:
            print(f"    {subject}: {e}")
    return save_samples("reasoning", "competition_math", samples)


def download_humaneval():
    """HumanEval coding problems (COMPLEX)."""
    print("\n  HumanEval (complex coding)...")
    try:
        ds = load_dataset("openai/openai_humaneval", split="test")
        print(f"    Size: {len(ds)}")
    except Exception:
        try:
            ds = load_dataset("openai_humaneval", split="test")
        except Exception as e:
            print(f"    Failed: {e}")
            return 0

    samples = []
    for i, item in enumerate(ds):
        if i >= 100:
            break
        samples.append({
            "id": f"humaneval_{i:03d}",
            "prompt": item.get("prompt", ""),
            "canonical_solution": item.get("canonical_solution", ""),
            "test": item.get("test", ""),
            "entry_point": item.get("entry_point", ""),
            "task": "code_generation",
            "difficulty": "complex",
            "source": "openai_humaneval",
        })
    return save_samples("reasoning", "humaneval", samples)


# ═══════════════════════════════════════════════════════════════
# MEDIUM SAMPLES
# ═══════════════════════════════════════════════════════════════

def download_more_hellaswag():
    """Get 200 more HellaSwag (commonsense reasoning - MEDIUM)."""
    print("\n  More HellaSwag (commonsense)...")
    ds = load_dataset("Rowan/hellaswag", split="validation")
    items = list(ds)[150:350]  # Skip first 150

    samples = []
    for i, item in enumerate(items):
        context = item.get("ctx", "")
        endings = item.get("endings", [])
        label = item.get("label", 0)
        if not context or not endings:
            continue
        samples.append({
            "id": f"hswag_{150+i:03d}",
            "context": context,
            "endings": endings,
            "correct_ending": endings[int(label)] if int(label) < len(endings) else "",
            "task": "commonsense_reasoning",
            "difficulty": "medium",
        })
    return save_samples("auxiliary", "hellaswag", samples)


def download_piqa():
    """PIQA - physical intuition QA (MEDIUM)."""
    print("\n  PIQA (physical intuition)...")
    ds = load_dataset("ybisk/piqa", split="validation")
    print(f"    Size: {len(ds)}")

    samples = []
    for i, item in enumerate(ds):
        if i >= 150:
            break
        goal = item.get("goal", "")
        sol1 = item.get("sol1", "")
        sol2 = item.get("sol2", "")
        label = item.get("label", 0)
        samples.append({
            "id": f"piqa_{i:03d}",
            "goal": goal,
            "solution1": sol1,
            "solution2": sol2,
            "correct": label,
            "task": "physical_reasoning",
            "difficulty": "medium",
        })
    return save_samples("auxiliary", "piqa", samples)


def download_social_iqa():
    """Social IQA - social reasoning (MEDIUM)."""
    print("\n  Social IQA (social reasoning)...")
    try:
        ds = load_dataset("allenai/social_i_qa", split="validation")
        print(f"    Size: {len(ds)}")
    except Exception as e:
        print(f"    Failed: {e}")
        return 0

    samples = []
    for i, item in enumerate(ds):
        if i >= 150:
            break
        context = item.get("context", "")
        question = item.get("question", "")
        answerA = item.get("answerA", "")
        answerB = item.get("answerB", "")
        answerC = item.get("answerC", "")
        label = item.get("label", "1")
        samples.append({
            "id": f"siqa_{i:03d}",
            "context": context,
            "question": question,
            "choices": [answerA, answerB, answerC],
            "correct": int(label) - 1 if label.isdigit() else 0,
            "task": "social_reasoning",
            "difficulty": "medium",
        })
    return save_samples("auxiliary", "social_iqa", samples)


def download_commonsense_qa():
    """CommonsenseQA - requires world knowledge (MEDIUM)."""
    print("\n  CommonsenseQA (world knowledge)...")
    try:
        ds = load_dataset("tau/commonsense_qa", split="validation")
        print(f"    Size: {len(ds)}")
    except Exception as e:
        print(f"    Failed: {e}")
        return 0

    samples = []
    for i, item in enumerate(ds):
        if i >= 150:
            break
        question = item.get("question", "")
        choices = item.get("choices", {})
        choice_texts = choices.get("text", [])
        choice_labels = choices.get("label", [])
        answer_key = item.get("answerKey", "")
        formatted = question + "\n" + "\n".join(f"  {l}) {t}" for l, t in zip(choice_labels, choice_texts))
        samples.append({
            "id": f"csqa_{i:03d}",
            "question": formatted,
            "answer_key": answer_key,
            "task": "commonsense_qa",
            "difficulty": "medium",
        })
    return save_samples("auxiliary", "commonsense_qa", samples)


def main():
    print("=" * 60)
    print("Downloading more COMPLEX and MEDIUM samples")
    print("=" * 60)

    total = 0

    print("\n--- COMPLEX ---")
    for fn in [download_more_gsm8k, download_more_arc, download_more_math, download_humaneval]:
        try:
            total += fn()
        except Exception as e:
            print(f"    FAILED: {e}")

    print("\n--- MEDIUM ---")
    for fn in [download_more_hellaswag, download_piqa, download_social_iqa, download_commonsense_qa]:
        try:
            total += fn()
        except Exception as e:
            print(f"    FAILED: {e}")

    print(f"\n{'='*60}")
    print(f"DONE: {total} new samples added")
    print("=" * 60)
    print("\nRun these to update the classifier:")
    print("  python benchmarks/classifier/prepare_data.py")
    print("  python benchmarks/classifier/train.py")


if __name__ == "__main__":
    main()
