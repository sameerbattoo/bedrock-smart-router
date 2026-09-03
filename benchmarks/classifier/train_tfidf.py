# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Train the TF-IDF + LogisticRegression prompt-complexity classifier.

Reads all training data from ``benchmarks/classifier/datasets/`` (populated by
``download/download_all.py``) plus the in-repo synthetic prompts under
``datasets/generated/``, then trains a 4-class classifier
(simple / moderate / complex / reasoning) and exports it for pure-numpy
inference to ``bedrock_smart_router/data/ml_classifier.json``.

Dataset files in datasets/ use one of two schemas:
  - Labeled          : [{"text": str, "label": "simple"|"moderate"|"complex"}]
                       (DevQuasar uses "simple"/"complex"; "complex" is split
                        into moderate/complex by keyword heuristics below.)
  - cross_difficulty : [{"text": str, "label": <float z-score>}] — binned into
                       moderate/complex/reasoning by score.

Usage:
    # 1. Download datasets (one-time / to refresh):
    python benchmarks/classifier/download/download_all.py
    # 2. Train:
    python benchmarks/classifier/train_tfidf.py

Output: benchmarks/classifier/tfidf_model/ + bedrock_smart_router/data/ml_classifier.json
Set CLASSIFIER_OUTPUT=ml_classifier_new.json to write a test file instead.
"""
import json
import os
import pickle
import random
from collections import Counter
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline

random.seed(42)

# ── Paths ───────────────────────────────────────────────────────────
CLASSIFIER_DIR = Path(__file__).parent
DATASETS_DIR = CLASSIFIER_DIR / "datasets"
GENERATED_DIR = DATASETS_DIR / "generated"
OUTPUT_DIR = CLASSIFIER_DIR / "tfidf_model"
_OUTPUT_NAME = os.environ.get("CLASSIFIER_OUTPUT", "ml_classifier.json")
PACKAGE_DATA = CLASSIFIER_DIR.parent.parent / "bedrock_smart_router" / "data" / _OUTPUT_NAME
OUTPUT_DIR.mkdir(exist_ok=True)

texts: list[str] = []
labels: list[str] = []

# DevQuasar's "complex" bucket really spans moderate→complex. Split it:
# design/build/optimize-type prompts (or long prompts) → complex, else moderate.
_COMPLEX_INDICATORS = [
    "design", "implement", "architect", "build a system", "optimize",
    "distributed", "scalab", "algorithm", "data structure", "prove",
    "derive", "formal", "trade-off", "microservice",
]


def _split_devquasar_complex(text: str) -> str:
    t = text.lower()
    if any(kw in t for kw in _COMPLEX_INDICATORS) or len(text) > 300:
        return "complex"
    return "moderate"


def _bin_cross_difficulty(score: float) -> str:
    # Reasoning benchmarks: easy → moderate overall, hardest → reasoning.
    if score < -0.5:
        return "moderate"
    if score < 1.0:
        return "complex"
    return "reasoning"


# ═══════════════════════════════════════════════════════════════════
# 1. Downloaded datasets (datasets/*.json)
# ═══════════════════════════════════════════════════════════════════
if not DATASETS_DIR.exists() or not any(DATASETS_DIR.glob("*.json")):
    raise SystemExit(
        "No datasets found in benchmarks/classifier/datasets/.\n"
        "Run: python benchmarks/classifier/download/download_all.py"
    )

per_source: dict[str, int] = {}
for path in sorted(DATASETS_DIR.glob("*.json")):
    with open(path) as f:
        rows = json.load(f)
    name = path.stem
    added = 0
    is_cross = name.startswith("cross_difficulty_")
    is_devquasar = name == "devquasar_router"
    for d in rows:
        text = d.get("text")
        if not text:
            continue
        if is_cross:
            try:
                label = _bin_cross_difficulty(float(d["label"]))
            except (KeyError, ValueError, TypeError):
                continue
        elif is_devquasar:
            raw = d.get("label")
            label = "simple" if raw == "simple" else _split_devquasar_complex(text)
        else:
            # Labeled file — label is already the final 3-class value.
            label = d.get("label")
            if label not in ("simple", "moderate", "complex", "reasoning"):
                continue
        texts.append(text)
        labels.append(label)
        added += 1
    per_source[name] = added
    print(f"  {name}: {added} samples")
print(f"1. Downloaded datasets: {sum(per_source.values())} samples "
      f"from {len(per_source)} sources")

# ═══════════════════════════════════════════════════════════════════
# 2. In-repo synthetic prompts (datasets/generated/*.json)
# ═══════════════════════════════════════════════════════════════════
DIFFICULTY_MAP = {"simple": "simple", "medium": "moderate",
                  "complex": "complex", "hard": "complex"}
gen_count = 0
for gen_file in sorted(GENERATED_DIR.glob("*.json")):
    with open(gen_file) as f:
        gen_data = json.load(f)
    for item in gen_data:
        parts = []
        if item.get("system_prompt"):
            parts.append(item["system_prompt"])
        if item.get("context"):
            parts.append(item["context"])
        if item.get("user_prompt"):
            parts.append(item["user_prompt"])
        elif item.get("text"):
            parts.append(item["text"])
        full_text = "\n\n".join(parts)
        if not full_text.strip():
            continue
        label = DIFFICULTY_MAP.get(item.get("difficulty", "medium"), "moderate")
        texts.append(full_text)
        labels.append(label)
        # Also add the user prompt alone (helps generalization).
        if item.get("user_prompt") and item.get("system_prompt"):
            texts.append(item["user_prompt"])
            labels.append(label)
        gen_count += 1
print(f"2. Generated (synthetic) data: {gen_count} samples")

# ═══════════════════════════════════════════════════════════════════
# 3. Synthetic reasoning exemplars (in-script; strengthens the small
#    "reasoning" class with unambiguous proof/derivation prompts).
# ═══════════════════════════════════════════════════════════════════
reasoning_prompts = [
    "Prove by induction that the sum of first n squares equals n(n+1)(2n+1)/6",
    "Prove that there are infinitely many prime numbers using Euclid's proof",
    "Derive the closed-form solution for the Fibonacci sequence using generating functions",
    "Prove the Cauchy-Schwarz inequality for inner product spaces",
    "Show that P ≠ NP implies one-way functions exist. Discuss the implications.",
    "Prove that every continuous function on [a,b] is Riemann integrable",
    "Derive the Black-Scholes equation from first principles using Ito's lemma",
    "Prove the fundamental theorem of calculus using the epsilon-delta definition",
    "Show that the halting problem is undecidable using diagonalization",
    "Prove that every finite group of order p^2 is abelian",
    "Think step by step: A farmer has 17 sheep. All but 9 die. How many are left?",
    "Let's think through this carefully. If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?",
    "Reason through this problem: Three people check into a hotel room that costs $30...",
    "Think carefully about this logic puzzle: You have 12 balls, one is heavier or lighter...",
    "Prove by contradiction that sqrt(2) is irrational.",
    "Reason about the Monty Hall problem using Bayes' theorem with full derivation.",
    "Analyze step by step: Given a directed graph with negative edge weights, prove Bellman-Ford correctness.",
    "Prove the FLP impossibility result step by step using the bivalency argument.",
    "Derive the Euler-Lagrange equation from the calculus of variations",
    "Prove Gödel's first incompleteness theorem for Peano arithmetic",
] * 5
texts.extend(reasoning_prompts)
labels.extend(["reasoning"] * len(reasoning_prompts))
print(f"3. Synthetic reasoning: {len(reasoning_prompts)} samples")

# ═══════════════════════════════════════════════════════════════════
# 4. System-prompt augmentation (add a system prompt / tool context to
#    ~10% of samples so the model is robust to that framing).
# ═══════════════════════════════════════════════════════════════════
SYSTEM_PROMPTS = [
    "You are a helpful assistant.",
    "You are a senior Python developer. Write clean, production-ready code.",
    "You are an AWS solutions architect. Design for scale and reliability.",
    "You are a data scientist. Analyze data and provide insights.",
    "You are a security expert. Identify vulnerabilities and suggest fixes.",
    "You are a DevOps engineer. Focus on automation and infrastructure.",
    "You are a mathematics professor. Show all steps rigorously.",
    "You are a distributed systems engineer. Design for fault tolerance.",
    "You are a technical writer. Be clear and concise.",
    "You are an ML engineer. Focus on model performance and efficiency.",
    "You are a database administrator. Optimize queries and schema design.",
    "You are a frontend developer. Write accessible, performant UI code.",
    "You are a cloud architect specializing in serverless applications.",
    "You are a principal engineer reviewing system designs.",
    "You are an algorithms researcher. Prove correctness formally.",
]
TOOL_CONTEXTS = [
    "[Tools available: query_database, generate_chart]",
    "[Tools available: search_docs, file_write, calculator]",
    "[Tools available: python_repl, file_read, file_write]",
    "[Tools available: web_search, http_request]",
    "[Tools available: use_aws, shell, file_write]",
]
n_augment = len(texts) // 10
indices = random.sample(range(len(texts)), min(n_augment, len(texts)))
aug_count = 0
for idx in indices:
    sys_prompt = random.choice(SYSTEM_PROMPTS)
    tool_ctx = random.choice(TOOL_CONTEXTS) if random.random() < 0.3 else ""
    augmented = (f"{sys_prompt}\n\n{tool_ctx}\n\n{texts[idx]}" if tool_ctx
                 else f"{sys_prompt}\n\n{texts[idx]}")
    texts.append(augmented)
    labels.append(labels[idx])
    aug_count += 1
print(f"4. System-prompt augmentation: {aug_count} samples")

# ═══════════════════════════════════════════════════════════════════
# Train
# ═══════════════════════════════════════════════════════════════════
print(f"\nTotal samples: {len(texts)}")
print(f"Label distribution: {dict(Counter(labels))}")

X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.15, random_state=42, stratify=labels
)
print(f"Train: {len(X_train)}, Test: {len(X_test)}")

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=25000,
        ngram_range=(1, 3),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        strip_accents="unicode",
    )),
    ("clf", LogisticRegression(
        C=5.0,
        max_iter=1000,
        solver="lbfgs",
        class_weight="balanced",
        random_state=42,
    )),
])

print("\nCross-validation (5-fold)...")
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="accuracy")
print(f"CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
print(f"\nTest set results:\n{classification_report(y_test, y_pred)}")

# ── Save pickle (for local use) ─────────────────────────────────────
model_path = OUTPUT_DIR / "classifier.pkl"
with open(model_path, "wb") as f:
    pickle.dump(pipeline, f)
print(f"✅ Pickle: {model_path} ({model_path.stat().st_size / 1024:.1f} KB)")

# ── Save JSON for pure-numpy runtime inference ──────────────────────
tfidf = pipeline.named_steps["tfidf"]
clf = pipeline.named_steps["clf"]
model_data = {
    "vocabulary": {k: int(v) for k, v in tfidf.vocabulary_.items()},
    "idf": tfidf.idf_.tolist(),
    "coefficients": clf.coef_.tolist(),
    "intercept": clf.intercept_.tolist(),
    "classes": clf.classes_.tolist(),
    "tfidf_params": {
        "max_features": 25000,
        "ngram_range": [1, 3],
        "sublinear_tf": True,
    },
}
json_path = OUTPUT_DIR / "classifier_data.json"
with open(json_path, "w") as f:
    json.dump(model_data, f)
print(f"✅ JSON: {json_path} ({json_path.stat().st_size / 1024:.1f} KB)")

import shutil
shutil.copy(json_path, PACKAGE_DATA)
print(f"✅ Copied to: {PACKAGE_DATA}")

# ── Quick sanity predictions ────────────────────────────────────────
test_prompts = [
    ("What is AWS S3?", "simple"),
    ("Write a Python decorator with retry logic", "moderate"),
    ("Design a distributed fraud detection system at 1M TPS", "complex"),
    ("Prove by induction that sum of cubes equals (n(n+1)/2)^2", "reasoning"),
    ("You are a senior architect.\n\nDesign a microservices platform with service mesh", "complex"),
]
print("\n--- Quick predictions ---")
for prompt, expected in test_prompts:
    pred = pipeline.predict([prompt])[0]
    confidence = max(pipeline.predict_proba([prompt])[0])
    match = "✓" if pred == expected else "✗"
    print(f"  {match} [{pred:>9}] ({confidence:.2f}) expected={expected} | {prompt[:60]}")
