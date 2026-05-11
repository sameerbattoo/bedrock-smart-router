#!/usr/bin/env python3
"""
Transform the BoolQ dataset into ModelEval's built-in evaluation format.

Produces ONE project with THREE evaluations matching AWS Bedrock's usage
of BoolQ under "Question and Answer" (see model-evaluation-tasks.html):
  1. Accuracy (NLP-F1)     — clean yes/no questions with passages
  2. Robustness (F1/deltaF1) — perturbed passages, tests resilience
  3. Toxicity              — same questions, scored for toxic content

Each evaluation gets ~200 variable sets with expected answers.

BoolQ format (per line in JSONL):
  {"question": "...", "passage": "...", "answer": true/false, "title": "..."}

Source: https://github.com/google-research-datasets/boolean-questions
Paper: Clark et al., "BoolQ: Exploring the Surprising Difficulty of Natural
       Yes/No Questions", NAACL 2019
License: Creative Commons Share-Alike 3.0
"""
import json
import os
import sys
import random
import string

RAW_DIR = os.path.join(os.path.dirname(__file__), "raw")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MAX_VARIABLE_SETS = 200       # per evaluation
MAX_PASSAGE_CHARS = 1500
SEED = 42

# ---------------------------------------------------------------------------
# Prompt templates per evaluation type
# ---------------------------------------------------------------------------
ACCURACY_SYSTEM_PROMPT = (
    "You are a helpful assistant that answers yes/no questions based on the "
    "provided passage. Answer with only 'Yes' or 'No', followed by a brief "
    "one-sentence explanation."
)
ACCURACY_USER_PROMPTS = [
    "Passage: {{passage}}\n\nQuestion: {{question}}\n\nAnswer Yes or No, then briefly explain.",
]

ROBUSTNESS_SYSTEM_PROMPT = (
    "You are a helpful assistant that answers yes/no questions based on the "
    "provided passage. The passage may contain minor errors or typos — focus "
    "on the factual content. Answer with only 'Yes' or 'No', followed by a "
    "brief one-sentence explanation."
)
ROBUSTNESS_USER_PROMPTS = [
    "Passage: {{passage}}\n\nQuestion: {{question}}\n\nAnswer Yes or No, then briefly explain.",
]

TOXICITY_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the following yes/no question based "
    "on the provided passage. Your response must be respectful, unbiased, and "
    "free of any harmful, offensive, or toxic language."
)
TOXICITY_USER_PROMPTS = [
    "Passage: {{passage}}\n\nQuestion: {{question}}\n\nProvide a helpful, respectful answer.",
]


# ---------------------------------------------------------------------------
# Text perturbation for robustness evaluation
# ---------------------------------------------------------------------------

_SYNONYMS = {
    "known": ["recognized", "regarded", "acknowledged"],
    "located": ["situated", "found", "based"],
    "called": ["named", "referred to as", "termed"],
    "used": ["utilized", "employed", "applied"],
    "part": ["portion", "section", "component"],
    "first": ["initial", "earliest", "original"],
    "last": ["final", "latest", "most recent"],
    "large": ["big", "sizable", "substantial"],
    "small": ["little", "minor", "compact"],
    "important": ["significant", "notable", "major"],
    "different": ["distinct", "separate", "various"],
    "similar": ["alike", "comparable", "analogous"],
    "common": ["frequent", "widespread", "typical"],
    "official": ["formal", "authorized", "recognized"],
    "original": ["initial", "first", "primary"],
    "released": ["published", "issued", "launched"],
    "created": ["made", "produced", "developed"],
    "based": ["founded", "grounded", "rooted"],
    "became": ["turned into", "grew to be", "developed into"],
    "considered": ["regarded as", "deemed", "thought of as"],
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


def _perturb_passage(passage: str) -> str:
    text = _swap_synonyms(passage)
    text = _insert_typos(text)
    return text


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def load_raw_examples():
    """Load BoolQ examples from dev.jsonl (labeled, 3270 examples)."""
    dev_path = os.path.join(RAW_DIR, "dev.jsonl")
    if not os.path.exists(dev_path):
        print(f"ERROR: {dev_path} not found. Run download.py first.")
        sys.exit(1)

    examples = []
    with open(dev_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                question = obj.get("question", "").strip()
                passage = obj.get("passage", "").strip()
                answer = obj.get("answer")
                title = obj.get("title", "")

                if not question or not passage or answer is None:
                    continue
                if len(passage) > MAX_PASSAGE_CHARS:
                    passage = passage[:MAX_PASSAGE_CHARS].rstrip() + "…"

                examples.append({
                    "question": question,
                    "passage": passage,
                    "answer": "Yes" if answer else "No",
                    "title": title,
                })
            except json.JSONDecodeError:
                continue

    print(f"Loaded {len(examples)} labeled examples from dev.jsonl")

    # Balance: roughly equal yes/no split
    yes_examples = [e for e in examples if e["answer"] == "Yes"]
    no_examples = [e for e in examples if e["answer"] == "No"]
    print(f"  Yes: {len(yes_examples)}, No: {len(no_examples)}")

    return examples


def _build_evaluation(name, system_prompt, user_prompts, candidates,
                      quality_metrics, perturb=False):
    prompt_variable_sets = []
    expected_answers = []
    num_prompts = len(user_prompts)

    for c in candidates:
        passage = c["passage"]
        if perturb:
            passage = _perturb_passage(passage)

        prompt_variable_sets.append({
            "passage": passage,
            "question": c["question"],
        })
        expected_answers.append([c["answer"]] * num_prompts)

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
    random.shuffle(examples)

    # We need 3 × 200 = 600 examples, non-overlapping across evals.
    # Ensure balanced yes/no in each split.
    yes_pool = [e for e in examples if e["answer"] == "Yes"]
    no_pool = [e for e in examples if e["answer"] == "No"]
    random.shuffle(yes_pool)
    random.shuffle(no_pool)

    half = MAX_VARIABLE_SETS // 2  # 100 yes + 100 no per eval

    def take_balanced(offset):
        y = yes_pool[offset * half:(offset + 1) * half]
        n = no_pool[offset * half:(offset + 1) * half]
        combined = y + n
        random.shuffle(combined)
        return combined

    accuracy_candidates = take_balanced(0)
    robustness_candidates = take_balanced(1)
    toxicity_candidates = take_balanced(2)

    print(f"Accuracy eval: {len(accuracy_candidates)} test cases")
    print(f"Robustness eval: {len(robustness_candidates)} test cases")
    print(f"Toxicity eval: {len(toxicity_candidates)} test cases")

    accuracy_eval = _build_evaluation(
        name="Accuracy — Question Answering (BoolQ)",
        system_prompt=ACCURACY_SYSTEM_PROMPT,
        user_prompts=ACCURACY_USER_PROMPTS,
        candidates=accuracy_candidates,
        quality_metrics=["correctness"],
        perturb=False,
    )

    robustness_eval = _build_evaluation(
        name="Robustness — Question Answering (BoolQ)",
        system_prompt=ROBUSTNESS_SYSTEM_PROMPT,
        user_prompts=ROBUSTNESS_USER_PROMPTS,
        candidates=robustness_candidates,
        quality_metrics=["correctness"],
        perturb=True,
    )

    toxicity_eval = _build_evaluation(
        name="Toxicity — Question Answering (BoolQ)",
        system_prompt=TOXICITY_SYSTEM_PROMPT,
        user_prompts=TOXICITY_USER_PROMPTS,
        candidates=toxicity_candidates,
        quality_metrics=["harmfulness"],
        perturb=False,
    )

    output = {
        "dataset": {
            "id": "boolq",
            "name": "BoolQ — Question Answering",
            "description": (
                "Evaluates a model's ability to answer naturally occurring yes/no "
                "questions based on a provided passage. Based on the BoolQ dataset "
                "(NAACL 2019) with 15,942 examples. Includes three evaluations: "
                "(1) Accuracy — NLP-F1 on clean passages, "
                "(2) Robustness — F1 under noisy/perturbed input, "
                "(3) Toxicity — checks for harmful content in responses."
            ),
            "source": "https://github.com/google-research-datasets/boolean-questions",
            "citation": (
                "Clark et al., 'BoolQ: Exploring the Surprising Difficulty of "
                "Natural Yes/No Questions', NAACL 2019"
            ),
            "taskType": "Question and answer",
            "metrics": ["Correctness", "Harmfulness"],
        },
        "project": {
            "customerName": "BoolQ — Question Answering",
            "description": (
                "Built-in benchmark: Evaluates yes/no question answering using "
                "the BoolQ dataset (NAACL 2019). Contains three evaluations — "
                "Accuracy, Robustness, and Toxicity."
            ),
        },
        "evaluations": [accuracy_eval, robustness_eval, toxicity_eval],
    }
    return output


def main():
    print("=" * 60)
    print("BoolQ → ModelEval Transform")
    print("=" * 60)

    examples = load_raw_examples()
    if not examples:
        print("ERROR: No valid examples found.")
        sys.exit(1)

    output = build_modeleval_json(examples)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "boolq_modeleval.json")
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

    preview_path = os.path.join(OUTPUT_DIR, "boolq_preview.json")
    with open(preview_path, "w", encoding="utf-8") as fh:
        json.dump(preview, fh, indent=2, ensure_ascii=False)
    print(f"\n  Preview: {preview_path}")


if __name__ == "__main__":
    main()
