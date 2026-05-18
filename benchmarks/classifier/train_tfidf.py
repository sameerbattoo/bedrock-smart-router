"""Train a TF-IDF + LogisticRegression classifier for prompt complexity.

Produces a lightweight model (~1-3MB) that classifies prompts into:
simple, moderate (medium), complex, reasoning

Usage: python benchmarks/classifier/train_tfidf.py
Output: benchmarks/classifier/tfidf_model/
"""
import json
import os
import pickle
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline

# Load training data
DATA_PATH = Path(__file__).parent / "training_data.json"
GENERATED_DIR = Path(__file__).parent.parent / "data" / "generated"
OUTPUT_DIR = Path(__file__).parent / "tfidf_model"
OUTPUT_DIR.mkdir(exist_ok=True)

with open(DATA_PATH) as f:
    data = json.load(f)

print(f"Loaded {len(data)} samples from training_data.json")

# Map labels: medium → moderate (to match router's terminology)
LABEL_MAP = {"simple": "simple", "medium": "moderate", "complex": "complex"}

texts = [d["text"] for d in data]
labels = [LABEL_MAP.get(d["label"], d["label"]) for d in data]

# Load generated samples (from benchmarks/data/generated/)
DIFFICULTY_MAP = {"simple": "simple", "medium": "moderate", "complex": "complex", "hard": "complex"}
gen_count = 0
for gen_file in sorted(GENERATED_DIR.glob("*.json")):
    with open(gen_file) as f:
        gen_data = json.load(f)
    for item in gen_data:
        # Combine system_prompt + user_prompt as the full text
        text_parts = []
        if item.get("system_prompt"):
            text_parts.append(item["system_prompt"])
        if item.get("user_prompt"):
            text_parts.append(item["user_prompt"])
        elif item.get("text"):
            text_parts.append(item["text"])
        if item.get("context"):
            text_parts.append(item["context"])
        full_text = "\n\n".join(text_parts)
        if not full_text.strip():
            continue
        difficulty = item.get("difficulty", "medium")
        label = DIFFICULTY_MAP.get(difficulty, "moderate")
        texts.append(full_text)
        labels.append(label)
        gen_count += 1

print(f"Loaded {gen_count} samples from generated data")

# Add synthetic reasoning examples (the dataset doesn't have them)
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
    "Derive the Euler-Lagrange equation from the calculus of variations",
    "Prove the central limit theorem for i.i.d. random variables",
    "Show that NP-complete problems are closed under polynomial-time reductions",
    "Prove Gödel's first incompleteness theorem for Peano arithmetic",
    "Derive the wave equation from Maxwell's equations in free space",
    "Prove that the set of real numbers is uncountable using Cantor's diagonal argument",
    "Show step by step why the integral of e^(-x^2) from -inf to inf equals sqrt(pi)",
    "Prove that every vector space has a basis using Zorn's lemma",
    "Derive the Navier-Stokes equations from conservation of momentum",
    "Prove the Banach fixed-point theorem and discuss its applications",
    "Think step by step: A farmer has 17 sheep. All but 9 die. How many are left? Explain your reasoning carefully.",
    "Let's think through this carefully. If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets? Show all reasoning.",
    "Reason through this problem: Three people check into a hotel room that costs $30. They each pay $10. The manager realizes the room is only $25 and gives $5 to the bellboy to return. The bellboy keeps $2 and gives $1 back to each person. Now each person paid $9 (total $27) plus the bellboy has $2 = $29. Where is the missing dollar? Explain step by step.",
    "Think carefully about this logic puzzle: You have 12 balls, one is heavier or lighter. Using a balance scale exactly 3 times, identify the odd ball and whether it's heavier or lighter. Show your complete reasoning.",
    "Analyze step by step: In a game, you can either take $1 million guaranteed, or flip a coin for $5 million. Using expected utility theory with a concave utility function, derive the conditions under which a rational agent would choose the guaranteed amount.",
    "Prove by contradiction that sqrt(2) is irrational. Then extend the proof to show sqrt(p) is irrational for any prime p.",
    "Reason through the Monty Hall problem: You pick door 1, Monty opens door 3 (showing a goat). Should you switch? Prove your answer using Bayes' theorem with full derivation.",
    "Think step by step about this optimization: A company needs to minimize shipping costs across 5 warehouses and 8 stores. Formulate as a linear program, write the dual, and prove strong duality holds.",
    "Carefully analyze: Given a directed graph with negative edge weights (but no negative cycles), prove that the Bellman-Ford algorithm correctly finds shortest paths. Show the loop invariant.",
    "Reason about this distributed systems problem: Prove that in an asynchronous system with crash failures, consensus is impossible (FLP impossibility). Walk through the bivalency argument step by step.",
] * 3  # Repeat to get ~90 samples

texts.extend(reasoning_prompts)
labels.extend(["reasoning"] * len(reasoning_prompts))

print(f"Total samples after adding reasoning: {len(texts)}")
from collections import Counter
print(f"Label distribution: {dict(Counter(labels))}")

# Split
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

# Build pipeline
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=15000,
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

# Train on full training set
pipeline.fit(X_train, y_train)

# Evaluate on test set
y_pred = pipeline.predict(X_test)
print(f"\nTest set results:")
print(classification_report(y_test, y_pred))

# Save model
model_path = OUTPUT_DIR / "classifier.pkl"
with open(model_path, "wb") as f:
    pickle.dump(pipeline, f)

model_size = model_path.stat().st_size
print(f"\n✅ Model saved to: {model_path}")
print(f"   Size: {model_size / 1024:.1f} KB ({model_size / 1024 / 1024:.2f} MB)")

# Also save as JSON-friendly format for portability
# Export the vocabulary and weights
tfidf = pipeline.named_steps["tfidf"]
clf = pipeline.named_steps["clf"]

model_data = {
    "vocabulary": {k: int(v) for k, v in tfidf.vocabulary_.items()},
    "idf": tfidf.idf_.tolist(),
    "coefficients": clf.coef_.tolist(),
    "intercept": clf.intercept_.tolist(),
    "classes": clf.classes_.tolist(),
    "tfidf_params": {
        "max_features": 15000,
        "ngram_range": [1, 3],
        "sublinear_tf": True,
    },
}

json_path = OUTPUT_DIR / "classifier_data.json"
with open(json_path, "w") as f:
    json.dump(model_data, f)

json_size = json_path.stat().st_size
print(f"   JSON export: {json_size / 1024:.1f} KB ({json_size / 1024 / 1024:.2f} MB)")

# Quick test
test_prompts = [
    "What is AWS S3?",
    "Write a Python decorator with retry logic and exponential backoff",
    "Design a distributed system for real-time fraud detection at 1M TPS",
    "Prove by induction that the sum of first n cubes equals (n(n+1)/2)^2",
]
print("\n--- Quick predictions ---")
for prompt in test_prompts:
    pred = pipeline.predict([prompt])[0]
    probs = pipeline.predict_proba([prompt])[0]
    confidence = max(probs)
    print(f"  [{pred:>9}] ({confidence:.2f}) {prompt[:70]}")
