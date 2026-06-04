"""Train a Hybrid TF-IDF + Heuristic Features classifier for prompt complexity.

This combines:
1. TF-IDF n-gram features (vocabulary/keyword signal)
2. 15 hand-engineered heuristic features from HeuristicClassifier (structural signal)

The heuristic features capture what TF-IDF misses:
- Text length (log-scaled)
- Code presence and language keywords
- Reasoning marker density
- Technical keyword density
- Simple indicator presence
- Structural complexity (tables, code blocks, lists)
- Tool use signals
- Domain specificity (AWS, math, data analysis)
- Multi-step patterns
- Question complexity patterns
- Creative/open-ended signals
- Output format constraints
- Constraint density
- Context references

Usage: .venv/bin/python benchmarks/classifier/train_hybrid.py
Output: benchmarks/classifier/hybrid_model/ + bedrock_smart_router/data/ml_classifier.json
"""
import json
import math
import os
import pickle
import random
import re
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

random.seed(42)

# ═══════════════════════════════════════════════════════════════
# Paths
# ═══════════════════════════════════════════════════════════════
DATA_PATH = Path(__file__).parent / "training_data.json"
GENERATED_DIR = Path(__file__).parent.parent / "data" / "generated"
INDUSTRY_DIR = Path(__file__).parent.parent / "data" / "industry_standard"
OUTPUT_DIR = Path(__file__).parent / "hybrid_model"
PACKAGE_DATA = Path(__file__).parent.parent.parent / "bedrock_smart_router" / "data" / "ml_classifier.json"
OUTPUT_DIR.mkdir(exist_ok=True)


# ═══════════════════════════════════════════════════════════════
# Heuristic Feature Extraction (from heuristic_classifier.py)
# Standalone — no import dependency on the package itself
# ═══════════════════════════════════════════════════════════════

REASONING_MARKERS = {
    "step by step", "step-by-step", "analyze", "analyse", "analysis",
    "evaluate", "compare and contrast",
    "prove", "derive", "reason through", "think through", "work through",
    "explain why", "explain how", "trade-off", "tradeoff", "pros and cons",
    "critically", "systematically", "deduce", "infer", "hypothesize",
    "build a", "design a", "architect", "implement a", "construct",
    "optimize", "refactor", "for each", "for every",
    "calculate the", "compute the", "determine the",
    "showing", "demonstrating", "comprehensive",
}

CODE_MARKERS = {
    "```", "def ", "class ", "function ", "import ", "const ", "let ", "var ",
    "return ", "if __name__", "async def", "lambda ", "=>", "public static",
    "private ", "protected ", "#include", "package ", "func ", "fn ",
    "write a function", "write a program", "implement a", "code that",
    "write code", "write a script", "write a class", "write a method",
}

CODE_LANG_KEYWORDS = {
    "python", "javascript", "typescript", "java", "rust", "golang", "go ",
    "c++", "c#", "ruby", "swift", "kotlin", "scala", "sql", "html", "css",
    "react", "angular", "vue", "django", "flask", "fastapi", "spring",
    "terraform", "dockerfile", "yaml", "json schema",
}

SIMPLE_INDICATORS = {
    "hello", "hi ", "hey ", "thanks", "thank you", "yes", "no", "ok",
    "what is", "what's", "define ", "who is", "when was", "where is",
    "how old", "how many", "how much", "translate",
}

MULTI_STEP_PATTERNS = {
    "first,", "first ", "then,", "then ", "next,", "next ", "finally,",
    "step 1", "step 2", "1.", "2.", "3.", "after that", "followed by",
    "once you", "before you", "make sure to",
}

TOOL_USE_SIGNALS = {
    "function call", "tool_use", "tool use", "json schema", "structured output",
    "json output", "return json", "api call", "execute", "run the",
    "call the function", "invoke",
}

AWS_SIGNALS = {
    "aws", "amazon web services", "s3", "ec2", "lambda", "dynamodb",
    "cloudformation", "cdk", "iam", "vpc", "ecs", "eks", "sagemaker",
    "bedrock", "cloudwatch", "sns", "sqs", "api gateway", "route 53",
    "rds", "aurora", "redshift", "kinesis", "step functions",
}

MATH_SIGNALS = {
    "equation", "formula", "calculate", "compute", "integral", "derivative",
    "probability", "optimize", "minimize", "maximize", "proof", "theorem",
    "algorithm", "complexity", "big-o", "matrix", "vector", "linear algebra",
}

DATA_ANALYSIS_SIGNALS = {
    "cohort", "retention", "funnel", "segmentation", "rfm",
    "churn", "lifetime value", "clv", "ltv",
    "window function", "partition by", "ntile", "percentile",
    "dense_rank", "cte", "subquery", "nested query",
    "month-over-month", "year-over-year", "forecast", "trend",
    "running total", "moving average", "cumulative",
}

CREATIVE_SIGNALS = {
    "write a story", "write a poem", "imagine", "creative", "brainstorm",
    "come up with", "invent", "fiction", "narrative", "compose",
    "design a", "create a", "generate ideas",
}

COMPLEX_QUESTION_PATTERNS = {
    "how would", "how can i", "how do i", "how to implement",
    "what are the tradeoffs", "what are the pros", "what approach",
    "design a", "build a", "create a system", "architect",
    "optimize", "debug", "troubleshoot", "refactor",
    "compare", "evaluate", "analyze the",
}

SIMPLE_QUESTION_PATTERNS = {
    "what is", "what's", "who is", "when was", "where is",
    "how old", "how many", "how much", "define ",
    "what does", "is it", "can you",
}

OUTPUT_FORMAT_SIGNALS = {
    "return as json", "return json", "output as json", "json format",
    "format as", "output format", "in the format", "formatted as",
    "as a table", "as a list", "as bullet points", "as markdown",
    "```json", "```yaml", "```xml", "```csv",
    "structured output", "json schema", "output schema",
}

CONSTRAINT_SIGNALS = {
    "must be", "must not", "must include", "must have",
    "should be", "should not", "should include",
    "no more than", "no less than", "no longer than",
    "at least", "at most", "exactly", "precisely",
    "without using", "only use", "do not use", "don't use",
    "limited to", "restricted to", "ensure that", "make sure",
    "maximum", "minimum",
}

CONTEXT_REFERENCE_SIGNALS = {
    "the above", "the following", "the below",
    "given the", "based on the", "according to the",
    "this document", "this text", "this article", "this paper",
    "the provided", "the attached", "the given",
    "extract from", "summarize the", "analyze the",
}

_TABLE_PATTERN = re.compile(r'[\|\+][-=+|]+[\|\+]|(\w{1,50}\s*[,\t]\s*){3,}')
_CSV_DATA = re.compile(r'^[^,\n]+(?:,[^,\n]+){2,}$', re.MULTILINE)
_PARAGRAPH_BREAK = re.compile(r'\n\s*\n')
_NUMBERED_LIST = re.compile(r'^\s*\d+[\.\)]\s', re.MULTILINE)
_CODE_BLOCK = re.compile(r'```[\s\S]*?```|^    \S', re.MULTILINE)


def _count_matches(text_lower: str, keywords: set) -> int:
    count = 0
    for kw in keywords:
        if len(kw) <= 3:
            idx = text_lower.find(kw)
            while idx != -1:
                before_ok = (idx == 0 or not text_lower[idx - 1].isalnum())
                after_ok = (idx + len(kw) >= len(text_lower) or not text_lower[idx + len(kw)].isalnum())
                if before_ok and after_ok:
                    count += 1
                    break
                idx = text_lower.find(kw, idx + 1)
        elif kw in text_lower:
            count += 1
    return count


def extract_heuristic_features(text: str) -> np.ndarray:
    """Extract 15 heuristic features from text. Returns array of shape (15,)."""
    text_lower = text.lower()
    text_len = len(text_lower)

    # 1. Token count (log-scaled)
    if text_len <= 20:
        f_token = 0.0
    else:
        f_token = min(1.0, max(0.0,
            (math.log(text_len) - math.log(20)) / (math.log(3000) - math.log(20))
        ))

    # 2. Code presence
    code_hits = _count_matches(text_lower, CODE_MARKERS)
    lang_hits = _count_matches(text_lower, CODE_LANG_KEYWORDS)
    f_code = min(1.0, (code_hits + lang_hits) * 0.35)

    # 3. Reasoning markers
    reasoning_hits = _count_matches(text_lower, REASONING_MARKERS)
    f_reasoning = min(1.0, reasoning_hits * 0.35)

    # 4. Technical depth (density)
    total_tech = code_hits + lang_hits + reasoning_hits
    if text_len > 0:
        density = total_tech / max(1, text_len / 200)
        f_tech = min(1.0, density * 0.5)
    else:
        f_tech = 0.0

    # 5. Simple indicators (inverted — more simple indicators = lower score)
    simple_hits = _count_matches(text_lower, SIMPLE_INDICATORS)
    if text_len < 100 and simple_hits >= 1:
        f_simple = 0.0
    elif simple_hits >= 2:
        f_simple = 0.05
    elif simple_hits == 1:
        f_simple = 0.2
    else:
        f_simple = 0.5

    # 6. Structural complexity
    struct_signals = 0
    if _TABLE_PATTERN.search(text):
        struct_signals += 2
    if _CSV_DATA.search(text):
        struct_signals += 2
    num_paragraphs = len(_PARAGRAPH_BREAK.findall(text))
    if num_paragraphs >= 3:
        struct_signals += 1
    if num_paragraphs >= 6:
        struct_signals += 1
    if len(_NUMBERED_LIST.findall(text)) >= 3:
        struct_signals += 1
    if _CODE_BLOCK.search(text):
        struct_signals += 2
    f_struct = min(1.0, struct_signals * 0.2)

    # 7. Tool use signals
    tool_hits = _count_matches(text_lower, TOOL_USE_SIGNALS)
    f_tool = min(1.0, tool_hits * 0.4)

    # 8. Domain specificity (AWS + math + data)
    aws_hits = _count_matches(text_lower, AWS_SIGNALS)
    math_hits = _count_matches(text_lower, MATH_SIGNALS)
    data_hits = _count_matches(text_lower, DATA_ANALYSIS_SIGNALS)
    f_domain = min(1.0, (aws_hits + math_hits + data_hits) * 0.25)

    # 9. Multi-step patterns
    multi_hits = _count_matches(text_lower, MULTI_STEP_PATTERNS)
    f_multi = min(1.0, multi_hits * 0.25)

    # 10. Question complexity
    complex_q = _count_matches(text_lower, COMPLEX_QUESTION_PATTERNS)
    simple_q = _count_matches(text_lower, SIMPLE_QUESTION_PATTERNS)
    if complex_q > 0 and simple_q == 0:
        f_question = min(1.0, complex_q * 0.4)
    elif simple_q > 0 and complex_q == 0:
        f_question = 0.0
    else:
        f_question = min(1.0, max(0, complex_q - simple_q) * 0.3)

    # 11. Creative/open-ended
    creative_hits = _count_matches(text_lower, CREATIVE_SIGNALS)
    f_creative = min(1.0, creative_hits * 0.35)

    # 12. Output format constraints
    format_hits = _count_matches(text_lower, OUTPUT_FORMAT_SIGNALS)
    f_format = min(1.0, format_hits * 0.4)

    # 13. Constraint density
    constraint_hits = _count_matches(text_lower, CONSTRAINT_SIGNALS)
    f_constraint = min(1.0, constraint_hits * 0.2)

    # 14. Context references
    context_hits = _count_matches(text_lower, CONTEXT_REFERENCE_SIGNALS)
    f_context = min(1.0, context_hits * 0.2 + (0.2 if text_len > 500 else 0.0)) if context_hits > 0 else 0.0

    # 15. Raw reasoning count (unnormalized — helps separate reasoning from complex)
    f_reasoning_raw = min(1.0, reasoning_hits / 6.0)

    return np.array([
        f_token, f_code, f_reasoning, f_tech, f_simple,
        f_struct, f_tool, f_domain, f_multi, f_question,
        f_creative, f_format, f_constraint, f_context, f_reasoning_raw,
    ], dtype=np.float64)


# ═══════════════════════════════════════════════════════════════
# Load Training Data (same as train_tfidf.py)
# ═══════════════════════════════════════════════════════════════

texts: list[str] = []
labels: list[str] = []

# 1. Original training data
LABEL_MAP = {"simple": "simple", "medium": "moderate", "complex": "complex"}
with open(DATA_PATH) as f:
    data = json.load(f)
for d in data:
    texts.append(d["text"])
    labels.append(LABEL_MAP.get(d["label"], d["label"]))
print(f"1. Original training data: {len(data)} samples")

# 2. Generated data
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
        if item.get("user_prompt") and item.get("system_prompt"):
            texts.append(item["user_prompt"])
            labels.append(label)
        gen_count += 1
print(f"2. Generated data: {gen_count} samples")

# 3. DevQuasar (remapped)
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
            text_lower = text.lower()
            if any(kw in text_lower for kw in complex_indicators) or len(text) > 300:
                texts.append(text)
                labels.append("complex")
            else:
                texts.append(text)
                labels.append("moderate")
    print(f"3. DevQuasar router: {len(dq_data)} samples (remapped)")

# 4. ShareGPT
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
    print(f"4. ShareGPT: {sg_count} samples")

# 5. Cross-difficulty
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
        if score < -0.5:
            label = "moderate"
        elif score < 1.0:
            label = "complex"
        else:
            label = "reasoning"
        texts.append(d["text"])
        labels.append(label)
        cross_diff_count += 1
print(f"5. Cross-difficulty: {cross_diff_count} samples")

# 6. Claude reasoning
claude_path = INDUSTRY_DIR / "claude_reasoning.json"
if claude_path.exists():
    with open(claude_path) as f:
        claude_data = json.load(f)
    claude_count = 0
    for d in claude_data:
        msgs = d.get("messages", [])
        if d.get("category") != "math":
            continue
        user_prompt = ""
        for m in msgs:
            if m.get("role") == "user":
                user_prompt = m.get("content", "")
                break
        if not user_prompt:
            continue
        texts.append(user_prompt)
        labels.append("reasoning")
        claude_count += 1
    print(f"6. Claude reasoning: {claude_count} samples")

# 7. New datasets
for name, filename in [("Easy2Hard", "easy2hard_bench.json"), ("LeetCode", "leetcode.json"),
                       ("MT-Bench", "mt_bench.json"), ("IFEval", "ifeval.json"),
                       ("OpenOrca", "openorca_subset.json"), ("BBH", "bbh.json")]:
    path = INDUSTRY_DIR / filename
    if path.exists():
        with open(path) as f:
            ds = json.load(f)
        for d in ds:
            texts.append(d["text"])
            labels.append(d["label"])
        print(f"7. {name}: {len(ds)} samples")

# 8. Synthetic reasoning
reasoning_prompts = [
    "Prove by induction that the sum of first n squares equals n(n+1)(2n+1)/6",
    "Prove that there are infinitely many prime numbers using Euclid's proof",
    "Derive the closed-form solution for the Fibonacci sequence using generating functions",
    "Show that the halting problem is undecidable using diagonalization",
    "Prove by contradiction that sqrt(2) is irrational.",
    "Prove the fundamental theorem of calculus using the epsilon-delta definition",
    "Derive the Black-Scholes equation from first principles using Ito's lemma",
    "Prove Gödel's first incompleteness theorem for Peano arithmetic",
    "Show that P ≠ NP implies one-way functions exist.",
    "Prove that every continuous function on [a,b] is Riemann integrable",
] * 10
texts.extend(reasoning_prompts)
labels.extend(["reasoning"] * len(reasoning_prompts))
print(f"8. Synthetic reasoning: {len(reasoning_prompts)} samples")

# 9. System prompt augmentation
SYSTEM_PROMPTS = [
    "You are a helpful assistant.",
    "You are a senior Python developer. Write clean, production-ready code.",
    "You are an AWS solutions architect. Design for scale and reliability.",
    "You are a mathematics professor. Show all steps rigorously.",
    "You are a distributed systems engineer. Design for fault tolerance.",
]
TOOL_CONTEXTS = [
    "[Tools available: query_database, generate_chart]",
    "[Tools available: python_repl, file_read, file_write]",
]
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
print(f"9. System prompt augmentation: {aug_count} samples")

from collections import Counter
print(f"\nTotal samples: {len(texts)}")
print(f"Label distribution: {dict(Counter(labels))}")


# ═══════════════════════════════════════════════════════════════
# Extract Heuristic Features for ALL samples
# ═══════════════════════════════════════════════════════════════
print("\nExtracting heuristic features for all samples...")
heuristic_features = np.array([extract_heuristic_features(t) for t in texts])
print(f"  Heuristic features shape: {heuristic_features.shape}")


# ═══════════════════════════════════════════════════════════════
# Train / Test Split
# ═══════════════════════════════════════════════════════════════
X_train_text, X_test_text, y_train, y_test, X_train_heur, X_test_heur = train_test_split(
    texts, labels, heuristic_features, test_size=0.15, random_state=42, stratify=labels
)
print(f"Train: {len(X_train_text)}, Test: {len(X_test_text)}")


# ═══════════════════════════════════════════════════════════════
# Build TF-IDF features
# ═══════════════════════════════════════════════════════════════
print("\nBuilding TF-IDF features...")
tfidf = TfidfVectorizer(
    max_features=25000,
    ngram_range=(1, 3),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True,
    strip_accents="unicode",
)
X_train_tfidf = tfidf.fit_transform(X_train_text)
X_test_tfidf = tfidf.transform(X_test_text)
print(f"  TF-IDF shape: {X_train_tfidf.shape}")


# ═══════════════════════════════════════════════════════════════
# Combine: TF-IDF (sparse) + Heuristic (dense) → Hybrid features
# ═══════════════════════════════════════════════════════════════
print("\nCombining TF-IDF + Heuristic features...")

# Scale heuristic features to match TF-IDF magnitude
scaler = StandardScaler()
X_train_heur_scaled = scaler.fit_transform(X_train_heur)
X_test_heur_scaled = scaler.transform(X_test_heur)

# Convert to sparse and concatenate
X_train_combined = hstack([X_train_tfidf, csr_matrix(X_train_heur_scaled)])
X_test_combined = hstack([X_test_tfidf, csr_matrix(X_test_heur_scaled)])
print(f"  Combined shape: {X_train_combined.shape}")


# ═══════════════════════════════════════════════════════════════
# Train Hybrid Classifier
# ═══════════════════════════════════════════════════════════════
print("\nTraining hybrid classifier...")
clf = LogisticRegression(
    C=5.0,
    max_iter=1000,
    solver="lbfgs",
    class_weight="balanced",
    random_state=42,
)

# Cross-validation
print("  Cross-validation (5-fold)...")
cv_scores = cross_val_score(clf, X_train_combined, y_train, cv=5, scoring="accuracy")
print(f"  CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Train final model
clf.fit(X_train_combined, y_train)

# Evaluate
y_pred = clf.predict(X_test_combined)
print(f"\n  Test set results:")
print(classification_report(y_test, y_pred))


# ═══════════════════════════════════════════════════════════════
# Also train TF-IDF only for comparison
# ═══════════════════════════════════════════════════════════════
print("\n--- TF-IDF ONLY (for comparison) ---")
clf_tfidf_only = LogisticRegression(
    C=5.0, max_iter=1000, solver="lbfgs", class_weight="balanced", random_state=42,
)
cv_tfidf = cross_val_score(clf_tfidf_only, X_train_tfidf, y_train, cv=5, scoring="accuracy")
print(f"  CV Accuracy (TF-IDF only): {cv_tfidf.mean():.4f} ± {cv_tfidf.std():.4f}")
clf_tfidf_only.fit(X_train_tfidf, y_train)
y_pred_tfidf = clf_tfidf_only.predict(X_test_tfidf)
print(classification_report(y_test, y_pred_tfidf))

print(f"\n  IMPROVEMENT: Hybrid CV {cv_scores.mean():.4f} vs TF-IDF CV {cv_tfidf.mean():.4f} = {(cv_scores.mean() - cv_tfidf.mean())*100:+.2f}%")


# ═══════════════════════════════════════════════════════════════
# Save Model (JSON format for numpy-only inference)
# ═══════════════════════════════════════════════════════════════
print("\nSaving hybrid model...")

model_data = {
    "model_type": "hybrid_tfidf_heuristic",
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
    "heuristic_params": {
        "n_features": 15,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
    },
}

json_path = OUTPUT_DIR / "classifier_data.json"
with open(json_path, "w") as f:
    json.dump(model_data, f)
print(f"✅ JSON: {json_path} ({json_path.stat().st_size / 1024:.1f} KB)")

# Also save as pickle for convenience
pickle_path = OUTPUT_DIR / "classifier.pkl"
with open(pickle_path, "wb") as f:
    pickle.dump({"tfidf": tfidf, "scaler": scaler, "clf": clf}, f)
print(f"✅ Pickle: {pickle_path} ({pickle_path.stat().st_size / 1024:.1f} KB)")


# ═══════════════════════════════════════════════════════════════
# Real-world routing test suite
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("REAL-WORLD ROUTING TEST (30 prompts)")
print("=" * 60)

test_prompts = [
    # SIMPLE
    ("What is Python?", "simple"),
    ("How do I print hello world in JavaScript?", "simple"),
    ("What is the capital of France?", "simple"),
    ("Translate hello to Spanish", "simple"),
    ("What time is it in Tokyo?", "simple"),
    ("List 5 popular programming languages", "simple"),
    ("What does HTTP stand for?", "simple"),
    ("How do I install numpy?", "simple"),
    # MODERATE
    ("Explain the difference between SQL and NoSQL databases", "moderate"),
    ("Write a Python function to reverse a linked list", "moderate"),
    ("Summarize the main principles of object-oriented programming", "moderate"),
    ("Compare React and Vue.js for building web applications", "moderate"),
    ("Write a bash script to find all files larger than 100MB", "moderate"),
    ("Explain how OAuth 2.0 authentication works", "moderate"),
    ("Write a Python decorator with retry logic and exponential backoff", "moderate"),
    ("How does garbage collection work in Java?", "moderate"),
    # COMPLEX
    ("Design a distributed caching system that handles 10M requests per second with cross-region replication", "complex"),
    ("Implement a B-tree data structure with insert, delete, and range query operations", "complex"),
    ("Design a real-time fraud detection pipeline for financial transactions at scale", "complex"),
    ("Architect a multi-tenant SaaS platform with tenant isolation, billing, and auto-scaling", "complex"),
    ("Write a lock-free concurrent hash map implementation in C++", "complex"),
    ("Design the database schema and API for a social media feed with ranking algorithm", "complex"),
    ("Implement a distributed consensus protocol similar to Raft", "complex"),
    ("Design a CI/CD pipeline with canary deployments, automated rollback, and chaos testing", "complex"),
    # REASONING
    ("Prove by mathematical induction that 1+2+3+...+n = n(n+1)/2", "reasoning"),
    ("Prove that the halting problem is undecidable", "reasoning"),
    ("Derive the time complexity of mergesort using the master theorem", "reasoning"),
    ("Prove that P != NP implies one-way functions exist", "reasoning"),
    ("Show using contradiction that there is no largest prime number", "reasoning"),
    ("Prove the correctness of Dijkstras algorithm using loop invariants", "reasoning"),
]

correct_hybrid = 0
correct_tfidf = 0
results = {"simple": [0, 0, 0], "moderate": [0, 0, 0], "complex": [0, 0, 0], "reasoning": [0, 0, 0]}

for prompt, expected in test_prompts:
    # Hybrid prediction
    tfidf_vec = tfidf.transform([prompt])
    heur_vec = scaler.transform([extract_heuristic_features(prompt)])
    combined = hstack([tfidf_vec, csr_matrix(heur_vec)])
    hybrid_pred = clf.predict(combined)[0]
    hybrid_probs = clf.predict_proba(combined)[0]
    hybrid_conf = max(hybrid_probs)

    # TF-IDF only prediction
    tfidf_pred = clf_tfidf_only.predict(tfidf_vec)[0]

    h_ok = hybrid_pred == expected
    t_ok = tfidf_pred == expected
    if h_ok:
        correct_hybrid += 1
    if t_ok:
        correct_tfidf += 1

    results[expected][2] += 1  # total
    if h_ok:
        results[expected][0] += 1  # hybrid correct
    if t_ok:
        results[expected][1] += 1  # tfidf correct

    mark = "✓" if h_ok else "✗"
    tfidf_mark = "✓" if t_ok else "✗"
    print(f"  {mark} Hybrid:[{hybrid_pred:>9}]({hybrid_conf:.2f})  {tfidf_mark} TF-IDF:[{tfidf_pred:>9}]  exp={expected:>9} | {prompt[:55]}")

print(f"\n{'='*60}")
print(f"FINAL COMPARISON")
print(f"{'='*60}")
print(f"  Hybrid:   {correct_hybrid}/{len(test_prompts)} = {correct_hybrid/len(test_prompts)*100:.1f}%")
print(f"  TF-IDF:   {correct_tfidf}/{len(test_prompts)} = {correct_tfidf/len(test_prompts)*100:.1f}%")
print(f"  Delta:    {correct_hybrid - correct_tfidf:+d} ({(correct_hybrid - correct_tfidf)/len(test_prompts)*100:+.1f}%)")
print()
print(f"  Per-class breakdown:")
print(f"  {'Class':>12}  {'Hybrid':>8}  {'TF-IDF':>8}")
for cls in ["simple", "moderate", "complex", "reasoning"]:
    h, t, total = results[cls]
    print(f"  {cls:>12}  {h}/{total} ({h/total*100:3.0f}%)  {t}/{total} ({t/total*100:3.0f}%)")
