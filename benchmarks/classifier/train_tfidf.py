# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Train a TF-IDF + LogisticRegression classifier for prompt complexity.

Incorporates:
- Original training_data.json (3.5K samples)
- Generated data with system+user prompts (295 samples)
- DevQuasar LLM router dataset (15K samples)
- Deita complexity dataset (52K samples)
- ShareGPT sample with numeric difficulty (5K samples)
- Synthetic system prompt augmentation
- Synthetic reasoning examples

Usage: python benchmarks/classifier/train_tfidf.py
Output: benchmarks/classifier/tfidf_model/ + bedrock_smart_router/data/ml_classifier.json
"""
import json
import os
import pickle
import random
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline

random.seed(42)

# Paths
DATA_PATH = Path(__file__).parent / "training_data.json"
GENERATED_DIR = Path(__file__).parent.parent / "data" / "generated"
INDUSTRY_DIR = Path(__file__).parent.parent / "data" / "industry_standard"
OUTPUT_DIR = Path(__file__).parent / "tfidf_model"
PACKAGE_DATA = Path(__file__).parent.parent.parent / "bedrock_smart_router" / "data" / "ml_classifier.json"
OUTPUT_DIR.mkdir(exist_ok=True)

texts: list[str] = []
labels: list[str] = []

# ═══════════════════════════════════════════════════════════════
# 1. Original training data (3.5K)
# ═══════════════════════════════════════════════════════════════
LABEL_MAP = {"simple": "simple", "medium": "moderate", "complex": "complex"}

with open(DATA_PATH) as f:
    data = json.load(f)
for d in data:
    texts.append(d["text"])
    labels.append(LABEL_MAP.get(d["label"], d["label"]))
print(f"1. Original training data: {len(data)} samples")

# ═══════════════════════════════════════════════════════════════
# 2. Generated data with system+user prompts
# ═══════════════════════════════════════════════════════════════
DIFFICULTY_MAP = {"simple": "simple", "medium": "moderate", "complex": "complex", "hard": "complex"}
gen_count = 0
for gen_file in sorted(GENERATED_DIR.glob("*.json")):
    with open(gen_file) as f:
        gen_data = json.load(f)
    for item in gen_data:
        text_parts = []
        if item.get("system_prompt"):
            text_parts.append(item["system_prompt"])
        if item.get("context"):
            text_parts.append(item["context"])
        if item.get("user_prompt"):
            text_parts.append(item["user_prompt"])
        elif item.get("text"):
            text_parts.append(item["text"])
        full_text = "\n\n".join(text_parts)
        if not full_text.strip():
            continue
        difficulty = item.get("difficulty", "medium")
        label = DIFFICULTY_MAP.get(difficulty, "moderate")
        texts.append(full_text)
        labels.append(label)
        # Also add user_prompt alone
        if item.get("user_prompt") and item.get("system_prompt"):
            texts.append(item["user_prompt"])
            labels.append(label)
        gen_count += 1
print(f"2. Generated data: {gen_count} samples")

# ═══════════════════════════════════════════════════════════════
# 3. DevQuasar LLM Router dataset (15K, binary → mapped to 4-class)
#    DevQuasar's "complex" is really our moderate+complex mix.
#    Use heuristics to split: explain/discuss → moderate, design/implement → complex
# ═══════════════════════════════════════════════════════════════
devquasar_path = INDUSTRY_DIR / "devquasar_router.json"
if devquasar_path.exists():
    with open(devquasar_path) as f:
        dq_data = json.load(f)
    complex_indicators = [
        "design", "implement", "architect", "build a system", "optimize",
        "distributed", "scalab", "algorithm", "data structure", "prove",
        "derive", "formal", "trade-off", "microservice",
    ]
    for d in dq_data:
        text = d["text"]
        if d["label"] == "simple":
            texts.append(text)
            labels.append("simple")
        else:
            # Split DevQuasar's "complex" into moderate vs complex
            text_lower = text.lower()
            if any(kw in text_lower for kw in complex_indicators) or len(text) > 300:
                texts.append(text)
                labels.append("complex")
            else:
                texts.append(text)
                labels.append("moderate")
    print(f"3. DevQuasar router: {len(dq_data)} samples (remapped binary → 3-class)")

# ═══════════════════════════════════════════════════════════════
# 4. Deita complexity dataset — SKIPPED (z-scores don't align with our taxonomy)
# ═══════════════════════════════════════════════════════════════
print("4. Deita complexity: skipped (label mismatch with our taxonomy)")

# ═══════════════════════════════════════════════════════════════
# 5. ShareGPT sample (5K, numeric z-scores → mapped)
# ═══════════════════════════════════════════════════════════════
sharegpt_path = INDUSTRY_DIR / "sharegpt_sample.json"
if sharegpt_path.exists():
    with open(sharegpt_path) as f:
        sg_data = json.load(f)
    sg_count = 0
    for d in sg_data:
        try:
            score = float(d["label"])
        except (ValueError, TypeError):
            continue
        if score < -0.5:
            label = "simple"
        elif score < 0.5:
            label = "moderate"
        elif score < 1.5:
            label = "complex"
        else:
            label = "reasoning"
        texts.append(d["text"])
        labels.append(label)
        sg_count += 1
    print(f"5. ShareGPT sample: {sg_count} samples")

# ═══════════════════════════════════════════════════════════════
# 5b. Cross-difficulty datasets (BBH, GSM8K, MATH, IFEval)
# ═══════════════════════════════════════════════════════════════
cross_diff_count = 0
for name in ["cross_difficulty_bbh", "cross_difficulty_gsm8k", "cross_difficulty_math", "cross_difficulty_ifeval"]:
    path = INDUSTRY_DIR / f"{name}.json"
    if not path.exists():
        continue
    with open(path) as f:
        cd_data = json.load(f)
    for d in cd_data:
        try:
            score = float(d["label"])
        except (ValueError, TypeError):
            continue
        # Reasoning benchmarks: easy = moderate overall, hard = reasoning
        if score < -0.5:
            label = "moderate"
        elif score < 1.0:
            label = "complex"
        else:
            label = "reasoning"
        texts.append(d["text"])
        labels.append(label)
        cross_diff_count += 1
print(f"5b. Cross-difficulty (BBH/GSM8K/MATH/IFEval): {cross_diff_count} samples")

# ═══════════════════════════════════════════════════════════════
# 5c. Claude Opus reasoning dataset (3K, with system+user prompts)
# ═══════════════════════════════════════════════════════════════
claude_path = INDUSTRY_DIR / "claude_reasoning.json"
if claude_path.exists():
    with open(claude_path) as f:
        claude_data = json.load(f)
    claude_count = 0
    for d in claude_data:
        msgs = d.get("messages", [])
        # Only use math category as reasoning (coding is moderate/complex, not reasoning)
        if d.get("category") != "math":
            continue
        sys_prompt = ""
        user_prompt = ""
        for m in msgs:
            if m.get("role") == "system":
                sys_prompt = m.get("content", "")
            elif m.get("role") == "user":
                user_prompt = m.get("content", "")
                break
        if not user_prompt:
            continue
        full_text = f"{sys_prompt}\n\n{user_prompt}" if sys_prompt else user_prompt
        texts.append(full_text)
        labels.append("reasoning")
        texts.append(user_prompt)
        labels.append("reasoning")
        claude_count += 1
    print(f"5c. Claude reasoning (math only, system+user): {claude_count} samples")

# ═══════════════════════════════════════════════════════════════
# 5d. Easy2Hard-Bench (NeurIPS 2024) — difficulty-scored prompts
# ═══════════════════════════════════════════════════════════════
easy2hard_path = INDUSTRY_DIR / "easy2hard_bench.json"
if easy2hard_path.exists():
    with open(easy2hard_path) as f:
        e2h_data = json.load(f)
    for d in e2h_data:
        texts.append(d["text"])
        labels.append(d["label"])
    print(f"5d. Easy2Hard-Bench: {len(e2h_data)} samples")

# ═══════════════════════════════════════════════════════════════
# 5e. LeetCode — coding problems with Easy/Medium/Hard labels
# ═══════════════════════════════════════════════════════════════
leetcode_path = INDUSTRY_DIR / "leetcode.json"
if leetcode_path.exists():
    with open(leetcode_path) as f:
        lc_data = json.load(f)
    for d in lc_data:
        texts.append(d["text"])
        labels.append(d["label"])
    print(f"5e. LeetCode: {len(lc_data)} samples")

# ═══════════════════════════════════════════════════════════════
# 5f. MT-Bench — multi-turn evaluation prompts (complex)
# ═══════════════════════════════════════════════════════════════
mt_bench_path = INDUSTRY_DIR / "mt_bench.json"
if mt_bench_path.exists():
    with open(mt_bench_path) as f:
        mt_data = json.load(f)
    for d in mt_data:
        texts.append(d["text"])
        labels.append(d["label"])
    print(f"5f. MT-Bench: {len(mt_data)} samples")

# ═══════════════════════════════════════════════════════════════
# 5g. IFEval — instruction-following with constraints
# ═══════════════════════════════════════════════════════════════
ifeval_path = INDUSTRY_DIR / "ifeval.json"
if ifeval_path.exists():
    with open(ifeval_path) as f:
        if_data = json.load(f)
    for d in if_data:
        texts.append(d["text"])
        labels.append(d["label"])
    print(f"5g. IFEval: {len(if_data)} samples")

# ═══════════════════════════════════════════════════════════════
# 5g2. Curated moderate prompts (targeted boundary samples)
# ═══════════════════════════════════════════════════════════════
curated_mod_path = INDUSTRY_DIR / "curated_moderate.json"
if curated_mod_path.exists():
    with open(curated_mod_path) as f:
        cm_data = json.load(f)
    # Repeat 5x to give these high-quality boundary samples more weight
    for _ in range(5):
        for d in cm_data:
            texts.append(d["text"])
            labels.append(d["label"])
    print(f"5g2. Curated moderate: {len(cm_data)} × 5 = {len(cm_data) * 5} samples")

# ═══════════════════════════════════════════════════════════════
# 5h. OpenOrca subset — system + user prompts
# ═══════════════════════════════════════════════════════════════
orca_path = INDUSTRY_DIR / "openorca_subset.json"
if orca_path.exists():
    with open(orca_path) as f:
        orca_data = json.load(f)
    for d in orca_data:
        texts.append(d["text"])
        labels.append(d["label"])
    print(f"5h. OpenOrca subset: {len(orca_data)} samples")

# ═══════════════════════════════════════════════════════════════
# 5i. Big Bench Hard — tasks that are hard for LLMs
# ═══════════════════════════════════════════════════════════════
bbh_path = INDUSTRY_DIR / "bbh.json"
if bbh_path.exists():
    with open(bbh_path) as f:
        bbh_data = json.load(f)
    for d in bbh_data:
        texts.append(d["text"])
        labels.append(d["label"])
    print(f"5i. Big Bench Hard: {len(bbh_data)} samples")

# ═══════════════════════════════════════════════════════════════
# 5d-5f (old): Additional datasets — DISABLED (diluted accuracy from 80% to 67%)
# The Alpaca, EricLu, and Claude full datasets have label definitions
# that conflict with our taxonomy. Keeping the cleaner 35K dataset.
# ═══════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════
# 6. Synthetic reasoning examples
# ═══════════════════════════════════════════════════════════════
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
] * 5  # 100 samples
texts.extend(reasoning_prompts)
labels.extend(["reasoning"] * len(reasoning_prompts))
print(f"6. Synthetic reasoning: {len(reasoning_prompts)} samples")

# ═══════════════════════════════════════════════════════════════
# 7. System prompt augmentation (add system prompts to ~10% of samples)
# ═══════════════════════════════════════════════════════════════
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

# Augment ~10% of existing samples with system prompts
n_augment = len(texts) // 10
indices = random.sample(range(len(texts)), min(n_augment, len(texts)))
aug_count = 0
for idx in indices:
    sys_prompt = random.choice(SYSTEM_PROMPTS)
    tool_ctx = random.choice(TOOL_CONTEXTS) if random.random() < 0.3 else ""
    augmented = f"{sys_prompt}\n\n{tool_ctx}\n\n{texts[idx]}" if tool_ctx else f"{sys_prompt}\n\n{texts[idx]}"
    texts.append(augmented)
    labels.append(labels[idx])
    aug_count += 1
print(f"7. System prompt augmentation: {aug_count} samples")

# ═══════════════════════════════════════════════════════════════
# Summary & Train
# ═══════════════════════════════════════════════════════════════
from collections import Counter
print(f"\nTotal samples: {len(texts)}")
print(f"Label distribution: {dict(Counter(labels))}")

# ═══════════════════════════════════════════════════════════════
# 8. Balance: Use class_weight='balanced' in LogisticRegression
#    instead of data-level undersampling, which loses useful signal.
# ═══════════════════════════════════════════════════════════════
print(f"\nTotal samples: {len(texts)}")
print(f"Label distribution: {dict(Counter(labels))}")

# Split
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.15, random_state=42, stratify=labels
)
print(f"Train: {len(X_train)}, Test: {len(X_test)}")

# Build pipeline
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

# Cross-validation
print("\nCross-validation (5-fold)...")
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="accuracy")
print(f"CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Train
pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pipeline.predict(X_test)
print(f"\nTest set results:")
print(classification_report(y_test, y_pred))

# Save pickle
model_path = OUTPUT_DIR / "classifier.pkl"
with open(model_path, "wb") as f:
    pickle.dump(pipeline, f)
print(f"✅ Pickle: {model_path} ({model_path.stat().st_size / 1024:.1f} KB)")

# Save JSON for pure-numpy inference
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

# Copy to package data
import shutil
shutil.copy(json_path, PACKAGE_DATA)
print(f"✅ Copied to: {PACKAGE_DATA}")

# Quick test
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
    probs = pipeline.predict_proba([prompt])[0]
    confidence = max(probs)
    match = "✓" if pred == expected else "✗"
    print(f"  {match} [{pred:>9}] ({confidence:.2f}) expected={expected} | {prompt[:60]}")
