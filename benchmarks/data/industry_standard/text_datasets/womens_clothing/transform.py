#!/usr/bin/env python3
"""
Transform the Women's E-Commerce Clothing Reviews dataset into ModelEval format.

Produces ONE project with TWO evaluations matching AWS Bedrock's usage
under "Text classification":
  1. Accuracy (classification_accuracy_score) — classify sentiment from clean reviews
  2. Robustness (delta_classification_accuracy_score) — perturbed reviews

The classification task: given a review, predict whether the customer
recommends the product (binary: "Recommended" / "Not Recommended").

Source: https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews
License: CC0 Public Domain
"""
import json
import os
import sys
import random
import string

RAW_DIR = os.path.join(os.path.dirname(__file__), "raw")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")

MAX_VARIABLE_SETS = 200
MAX_REVIEW_CHARS = 1500
SEED = 42

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------
ACCURACY_SYSTEM_PROMPT = (
    "You are a sentiment classification assistant. Given a product review, "
    "classify whether the customer recommends the product or not. "
    "Respond with exactly one of: 'Recommended' or 'Not Recommended'. "
    "Then provide a one-sentence explanation."
)
ACCURACY_USER_PROMPTS = [
    (
        "Product Department: {{department}}\n"
        "Product Category: {{class_name}}\n\n"
        "Review:\n{{review_text}}\n\n"
        "Classification (Recommended or Not Recommended):"
    ),
]

ROBUSTNESS_SYSTEM_PROMPT = (
    "You are a sentiment classification assistant. Given a product review, "
    "classify whether the customer recommends the product or not. "
    "The review may contain minor errors or typos — focus on the overall "
    "sentiment. Respond with exactly one of: 'Recommended' or 'Not Recommended'. "
    "Then provide a one-sentence explanation."
)
ROBUSTNESS_USER_PROMPTS = [
    (
        "Product Department: {{department}}\n"
        "Product Category: {{class_name}}\n\n"
        "Review:\n{{review_text}}\n\n"
        "Classification (Recommended or Not Recommended):"
    ),
]


# ---------------------------------------------------------------------------
# Text perturbation for robustness
# ---------------------------------------------------------------------------

_SYNONYMS = {
    "love": ["adore", "really like", "am fond of"],
    "hate": ["dislike", "can't stand", "detest"],
    "great": ["excellent", "wonderful", "fantastic"],
    "terrible": ["awful", "horrible", "dreadful"],
    "beautiful": ["gorgeous", "lovely", "stunning"],
    "comfortable": ["comfy", "cozy", "pleasant"],
    "perfect": ["ideal", "flawless", "excellent"],
    "disappointed": ["let down", "unsatisfied", "unhappy"],
    "quality": ["craftsmanship", "construction", "make"],
    "fabric": ["material", "cloth", "textile"],
    "size": ["fit", "sizing", "dimensions"],
    "color": ["colour", "shade", "hue"],
    "bought": ["purchased", "got", "ordered"],
    "returned": ["sent back", "brought back", "exchanged"],
    "recommend": ["suggest", "endorse", "advocate"],
    "flattering": ["complimentary", "becoming", "attractive"],
    "cheap": ["inexpensive", "low-cost", "budget"],
    "expensive": ["pricey", "costly", "overpriced"],
    "soft": ["gentle", "smooth", "silky"],
    "cute": ["adorable", "charming", "pretty"],
}


def _insert_typos(text: str, rate: float = 0.03) -> str:
    rng = random.Random()
    chars = list(text)
    for i in range(len(chars)):
        if chars[i].isalpha() and rng.random() < rate:
            op = rng.choice(["swap", "drop", "insert", "replace"])
            if op == "swap" and i + 1 < len(chars) and chars[i + 1].isalpha():
                chars[i], chars[i + 1] = chars[i + 1], chars[i]
            elif op == "drop":
                chars[i] = ""
            elif op == "insert":
                chars[i] = chars[i] + rng.choice(string.ascii_lowercase)
            elif op == "replace":
                chars[i] = rng.choice(string.ascii_lowercase)
    return "".join(chars)


def _swap_synonyms(text: str, rate: float = 0.15) -> str:
    rng = random.Random()
    words = text.split()
    for i, w in enumerate(words):
        clean = w.strip(".,;:!?()\"'").lower()
        if clean in _SYNONYMS and rng.random() < rate:
            replacement = rng.choice(_SYNONYMS[clean])
            if w[0].isupper():
                replacement = replacement.capitalize()
            trailing = ""
            while w and w[-1] in ".,;:!?()\"'":
                trailing = w[-1] + trailing
                w = w[:-1]
            words[i] = replacement + trailing
    return " ".join(words)


def _perturb_review(text: str) -> str:
    return _insert_typos(_swap_synonyms(text))


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def load_raw_examples():
    path = os.path.join(RAW_DIR, "reviews.jsonl")
    if not os.path.exists(path):
        print(f"ERROR: {path} not found. Run download.py first.")
        sys.exit(1)

    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                review_text = obj.get("review_text", "").strip()
                recommended = obj.get("recommended")

                if not review_text or recommended is None:
                    continue
                # Need at least some substance to classify
                if len(review_text) < 20:
                    continue
                if len(review_text) > MAX_REVIEW_CHARS:
                    review_text = review_text[:MAX_REVIEW_CHARS].rstrip() + "…"

                examples.append({
                    "review_text": review_text,
                    "department": obj.get("department", ""),
                    "class_name": obj.get("class_name", ""),
                    "expected_answer": "Recommended" if recommended == 1 else "Not Recommended",
                    "rating": obj.get("rating", 0),
                })
            except json.JSONDecodeError:
                continue

    print(f"Loaded {len(examples)} labeled reviews")
    rec = sum(1 for e in examples if e["expected_answer"] == "Recommended")
    print(f"  Recommended: {rec}, Not Recommended: {len(examples) - rec}")
    return examples


def _build_evaluation(name, system_prompt, user_prompts, candidates,
                      quality_metrics, perturb=False):
    prompt_variable_sets = []
    expected_answers = []
    num_prompts = len(user_prompts)

    for c in candidates:
        review = c["review_text"]
        if perturb:
            review = _perturb_review(review)

        prompt_variable_sets.append({
            "review_text": review,
            "department": c["department"],
            "class_name": c["class_name"],
        })
        expected_answers.append([c["expected_answer"]] * num_prompts)

    return {
        "useCaseName": name,
        "systemPrompt": system_prompt,
        "userPrompts": user_prompts,
        "promptVariableSets": prompt_variable_sets,
        "expectedAnswers": expected_answers,
        "models": [],
        "judgeModels": [],
        "qualityMetrics": quality_metrics,
        "isRAG": False,
    }


def build_modeleval_json(examples):
    random.seed(SEED)

    # Balance the dataset: equal recommended / not recommended
    rec = [e for e in examples if e["expected_answer"] == "Recommended"]
    not_rec = [e for e in examples if e["expected_answer"] == "Not Recommended"]
    random.shuffle(rec)
    random.shuffle(not_rec)

    half = MAX_VARIABLE_SETS // 2  # 100 each

    # 2 evals × 200 = 400 total, balanced
    def take_balanced(offset):
        r = rec[offset * half:(offset + 1) * half]
        n = not_rec[offset * half:(offset + 1) * half]
        combined = r + n
        random.shuffle(combined)
        return combined

    accuracy_candidates = take_balanced(0)
    robustness_candidates = take_balanced(1)

    print(f"Accuracy eval: {len(accuracy_candidates)} test cases")
    print(f"Robustness eval: {len(robustness_candidates)} test cases")

    accuracy_eval = _build_evaluation(
        name="Accuracy — Text Classification (Clothing Reviews)",
        system_prompt=ACCURACY_SYSTEM_PROMPT,
        user_prompts=ACCURACY_USER_PROMPTS,
        candidates=accuracy_candidates,
        quality_metrics=["correctness"],
        perturb=False,
    )

    robustness_eval = _build_evaluation(
        name="Robustness — Text Classification (Clothing Reviews)",
        system_prompt=ROBUSTNESS_SYSTEM_PROMPT,
        user_prompts=ROBUSTNESS_USER_PROMPTS,
        candidates=robustness_candidates,
        quality_metrics=["correctness"],
        perturb=True,
    )

    output = {
        "dataset": {
            "id": "womens-clothing-reviews",
            "name": "Women's Clothing Reviews — Text Classification",
            "description": (
                "Evaluates a model's ability to classify customer sentiment from "
                "product reviews. Based on the Women's E-Commerce Clothing Reviews "
                "dataset (23K reviews). The task is binary classification: predict "
                "whether the customer recommends the product. Includes two evaluations: "
                "(1) Accuracy — classification on clean reviews, "
                "(2) Robustness — classification under noisy/perturbed input."
            ),
            "source": "https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews",
            "citation": "Women's E-Commerce Clothing Reviews, Kaggle (CC0 Public Domain)",
            "taskType": "Text classification",
            "metrics": ["Correctness"],
        },
        "project": {
            "customerName": "Women's Clothing Reviews — Text Classification",
            "description": (
                "Built-in benchmark: Evaluates text classification (sentiment) "
                "using the Women's E-Commerce Clothing Reviews dataset. "
                "Contains two evaluations — Accuracy and Robustness."
            ),
        },
        "evaluations": [accuracy_eval, robustness_eval],
    }
    return output


def main():
    print("=" * 60)
    print("Women's Clothing Reviews → ModelEval Transform")
    print("=" * 60)

    examples = load_raw_examples()
    if not examples:
        print("ERROR: No valid examples found.")
        sys.exit(1)

    output = build_modeleval_json(examples)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "clothing_modeleval.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2, ensure_ascii=False)

    print(f"\nOutput written to {out_path}")
    for i, ev in enumerate(output["evaluations"]):
        print(f"  Eval {i+1}: {ev['useCaseName']}")
        print(f"    Variable sets: {len(ev['promptVariableSets'])}")
        print(f"    Quality metrics: {ev['qualityMetrics']}")

    # Preview
    preview = {**output}
    preview["evaluations"] = []
    for ev in output["evaluations"]:
        pev = {**ev}
        pev["promptVariableSets"] = ev["promptVariableSets"][:3]
        pev["expectedAnswers"] = ev["expectedAnswers"][:3]
        preview["evaluations"].append(pev)

    preview_path = os.path.join(OUTPUT_DIR, "clothing_preview.json")
    with open(preview_path, "w", encoding="utf-8") as fh:
        json.dump(preview, fh, indent=2, ensure_ascii=False)
    print(f"\n  Preview: {preview_path}")


if __name__ == "__main__":
    main()
