#!/usr/bin/env python3
"""
Transform the WikiText-2 dataset into ModelEval's built-in evaluation format.

Produces ONE project with ONE evaluation matching AWS Bedrock's usage
of WikiText2 under "General text generation" (see model-evaluation-tasks.html):
  1. Robustness (Word Error Rate) — tests whether models produce coherent,
     fluent continuations of Wikipedia passages under perturbation

Each evaluation gets ~200 variable sets. The passage is split: the first
portion becomes the prompt, the remainder becomes the expected continuation.

WikiText-2 format (per line in JSONL):
  {"text": "..."}

Source: https://huggingface.co/datasets/Salesforce/wikitext
Paper: Merity et al., "Pointer Sentinel Mixture Models", 2016
License: Creative Commons Attribution-ShareAlike (CC BY-SA 4.0)
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
MAX_VARIABLE_SETS = 200
MIN_PASSAGE_LEN = 150   # Characters — need enough to split into prompt + expected
MAX_PASSAGE_LEN = 1500
PROMPT_RATIO = 0.5       # Use first half as prompt, second half as expected
SEED = 42

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------
ROBUSTNESS_SYSTEM_PROMPT = (
    "You are a helpful assistant. Continue the following Wikipedia passage "
    "naturally and coherently. The text may contain minor errors or typos — "
    "focus on producing a fluent, factually consistent continuation."
)
ROBUSTNESS_USER_PROMPTS = [
    "Continue the following text:\n\n{{passage_start}}",
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
    "released": ["published", "issued", "launched"],
    "created": ["made", "produced", "developed"],
    "became": ["turned into", "grew to be", "developed into"],
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
                replacement = replacement[0].upper() + replacement[1:]
            trailing = ""
            while w and w[-1] in ".,;:!?()\"'":
                trailing = w[-1] + trailing
                w = w[:-1]
            words[i] = replacement + trailing
    return " ".join(words)


def _perturb_text(text: str) -> str:
    text = _swap_synonyms(text)
    text = _insert_typos(text)
    return text


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def load_raw_examples():
    """Load WikiText-2 passages from wikitext2.jsonl."""
    path = os.path.join(RAW_DIR, "wikitext2.jsonl")
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
                text = obj.get("text", "").strip()
                if len(text) < MIN_PASSAGE_LEN:
                    continue
                if len(text) > MAX_PASSAGE_LEN:
                    text = text[:MAX_PASSAGE_LEN].rstrip()
                examples.append({"text": text})
            except json.JSONDecodeError:
                continue

    print(f"Loaded {len(examples)} passages from wikitext2.jsonl")
    return examples


def _split_passage(text):
    """Split passage into prompt (first half) and expected continuation."""
    # Find a sentence boundary near the midpoint
    mid = int(len(text) * PROMPT_RATIO)
    # Look for sentence end (. ! ?) near midpoint
    best = mid
    for offset in range(0, min(100, mid)):
        for pos in [mid + offset, mid - offset]:
            if 0 < pos < len(text) - 20 and text[pos] in ".!?" and text[pos + 1] == " ":
                best = pos + 1
                break
        else:
            continue
        break

    prompt_part = text[:best].strip()
    expected_part = text[best:].strip()
    return prompt_part, expected_part


def _build_evaluation(name, system_prompt, user_prompts, candidates, quality_metrics):
    prompt_variable_sets = []
    expected_answers = []
    num_prompts = len(user_prompts)

    for c in candidates:
        prompt_part, expected_part = _split_passage(c["text"])
        perturbed = _perturb_text(prompt_part)

        prompt_variable_sets.append({
            "passage_start": perturbed,
        })
        expected_answers.append([expected_part] * num_prompts)

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

    candidates = examples[:MAX_VARIABLE_SETS]
    print(f"Robustness eval: {len(candidates)} test cases")

    robustness_eval = _build_evaluation(
        name="Robustness — General Text Generation (WikiText-2)",
        system_prompt=ROBUSTNESS_SYSTEM_PROMPT,
        user_prompts=ROBUSTNESS_USER_PROMPTS,
        candidates=candidates,
        quality_metrics=["correctness"],
    )

    output = {
        "dataset": {
            "id": "wikitext2",
            "name": "WikiText-2 — Language Modeling",
            "description": (
                "Evaluates robustness of text generation using Wikipedia passages "
                "from verified Good and Featured articles. Based on the WikiText-2 "
                "dataset (Merity et al., 2016) with 2M+ tokens. Tests whether "
                "models produce coherent continuations under perturbed input."
            ),
            "source": "https://huggingface.co/datasets/Salesforce/wikitext",
            "citation": (
                "Merity et al., 'Pointer Sentinel Mixture Models', 2016"
            ),
            "taskType": "General text generation",
            "metrics": ["Correctness"],
        },
        "project": {
            "customerName": "WikiText-2 — Language Modeling",
            "description": (
                "Built-in benchmark: Evaluates robustness of text generation "
                "using the WikiText-2 dataset (Merity et al., 2016). "
                "Contains one evaluation — Robustness."
            ),
        },
        "evaluations": [robustness_eval],
    }
    return output


def main():
    print("=" * 60)
    print("WikiText-2 → ModelEval Transform")
    print("=" * 60)

    examples = load_raw_examples()
    if not examples:
        print("ERROR: No valid examples found.")
        sys.exit(1)

    output = build_modeleval_json(examples)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "wikitext2_modeleval.json")
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

    preview_path = os.path.join(OUTPUT_DIR, "wikitext2_preview.json")
    with open(preview_path, "w", encoding="utf-8") as fh:
        json.dump(preview, fh, indent=2, ensure_ascii=False)
    print(f"\n  Preview: {preview_path}")


if __name__ == "__main__":
    main()
