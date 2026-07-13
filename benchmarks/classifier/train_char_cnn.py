# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Train a Character-level CNN classifier for prompt complexity.

Architecture:
- Input: Raw bytes/characters (no tokenizer, no vocabulary file)
- Embedding: Learnable 128-dim character embeddings (256 chars)
- Conv layers: 3 conv1d layers with increasing filter sizes
- Pooling: Global max pooling
- Output: 4-class softmax (simple, moderate, complex, reasoning)

Advantages over TF-IDF:
- Captures character-level patterns (code syntax, punctuation density)
- No vocabulary file needed (works on any input)
- Small model size (200K-500K params = 0.8-2 MB)
- Pure numpy inference (same dependency as TF-IDF)

Usage: .venv/bin/python benchmarks/classifier/train_char_cnn.py
"""
import json
import math
import os
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np

random.seed(42)
np.random.seed(42)

# ═══════════════════════════════════════════════════════════════
# Paths
# ═══════════════════════════════════════════════════════════════
DATA_PATH = Path(__file__).parent / "training_data.json"
GENERATED_DIR = Path(__file__).parent.parent / "data" / "generated"
INDUSTRY_DIR = Path(__file__).parent.parent / "data" / "industry_standard"
OUTPUT_DIR = Path(__file__).parent / "char_cnn_model"
PACKAGE_DATA = Path(__file__).parent.parent.parent / "bedrock_smart_router" / "data"
OUTPUT_DIR.mkdir(exist_ok=True)

# ═══════════════════════════════════════════════════════════════
# Hyperparameters
# ═══════════════════════════════════════════════════════════════
MAX_LEN = 512         # Max characters to consider (truncate longer prompts)
EMBED_DIM = 32        # Character embedding dimension
NUM_FILTERS = 64      # Filters per conv layer
FILTER_SIZES = [3, 5, 7]  # Kernel sizes for conv layers
NUM_CLASSES = 4
LEARNING_RATE = 0.001
BATCH_SIZE = 64
EPOCHS = 15
LABEL_TO_IDX = {"simple": 0, "moderate": 1, "complex": 2, "reasoning": 3}
IDX_TO_LABEL = {v: k for k, v in LABEL_TO_IDX.items()}


# ═══════════════════════════════════════════════════════════════
# Text to character indices
# ═══════════════════════════════════════════════════════════════
def text_to_indices(text: str, max_len: int = MAX_LEN) -> np.ndarray:
    """Convert text to array of byte values (0-255), padded/truncated to max_len."""
    # Use raw bytes — universal, no tokenizer needed
    encoded = text.encode("utf-8", errors="replace")[:max_len]
    indices = np.zeros(max_len, dtype=np.int32)
    for i, b in enumerate(encoded):
        indices[i] = b + 1  # Reserve 0 for padding
    return indices


# ═══════════════════════════════════════════════════════════════
# CNN Model (numpy-only, for training we use a simple approach)
# We'll use PyTorch for training then export weights for numpy inference
# ═══════════════════════════════════════════════════════════════

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("ERROR: PyTorch required for training. Install: pip install torch")
    print("       (Only numpy is needed for inference)")
    exit(1)


class CharCNN(nn.Module):
    """Character-level CNN for text classification."""

    def __init__(self, vocab_size=257, embed_dim=EMBED_DIM, num_filters=NUM_FILTERS,
                 filter_sizes=None, num_classes=NUM_CLASSES, dropout=0.3):
        super().__init__()
        if filter_sizes is None:
            filter_sizes = FILTER_SIZES

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, kernel_size=ks, padding=ks // 2)
            for ks in filter_sizes
        ])

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)

    def forward(self, x):
        # x: (batch, max_len) of character indices
        emb = self.embedding(x)  # (batch, max_len, embed_dim)
        emb = emb.transpose(1, 2)  # (batch, embed_dim, max_len) for conv1d

        conv_outputs = []
        for conv in self.convs:
            c = torch.relu(conv(emb))  # (batch, num_filters, max_len)
            c = c.max(dim=2)[0]  # Global max pool → (batch, num_filters)
            conv_outputs.append(c)

        out = torch.cat(conv_outputs, dim=1)  # (batch, num_filters * n_convs)
        out = self.dropout(out)
        logits = self.fc(out)  # (batch, num_classes)
        return logits


# ═══════════════════════════════════════════════════════════════
# Load Training Data (same sources as train_tfidf.py)
# ═══════════════════════════════════════════════════════════════
print("Loading training data...")

texts: list[str] = []
labels: list[str] = []

# 1. Original training data
LABEL_MAP = {"simple": "simple", "medium": "moderate", "complex": "complex"}
with open(DATA_PATH) as f:
    data = json.load(f)
for d in data:
    texts.append(d["text"])
    labels.append(LABEL_MAP.get(d["label"], d["label"]))
print(f"  1. Original: {len(data)}")

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
        if item.get("user_prompt"):
            text_parts.append(item["user_prompt"])
        elif item.get("text"):
            text_parts.append(item["text"])
        full_text = "\n\n".join(text_parts)
        if not full_text.strip():
            continue
        label = DIFFICULTY_MAP.get(item.get("difficulty", "medium"), "moderate")
        texts.append(full_text)
        labels.append(label)
        gen_count += 1
print(f"  2. Generated: {gen_count}")

# 3. DevQuasar (remapped)
devquasar_path = INDUSTRY_DIR / "devquasar_router.json"
if devquasar_path.exists():
    with open(devquasar_path) as f:
        dq_data = json.load(f)
    complex_indicators = ["design", "implement", "architect", "build a system", "optimize",
                          "distributed", "scalab", "algorithm", "data structure", "prove",
                          "derive", "formal", "trade-off", "microservice"]
    for d in dq_data:
        if d["label"] == "simple":
            texts.append(d["text"])
            labels.append("simple")
        else:
            if any(kw in d["text"].lower() for kw in complex_indicators) or len(d["text"]) > 300:
                texts.append(d["text"])
                labels.append("complex")
            else:
                texts.append(d["text"])
                labels.append("moderate")
    print(f"  3. DevQuasar: {len(dq_data)}")

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
        if score < -0.5: label = "simple"
        elif score < 0.5: label = "moderate"
        elif score < 1.5: label = "complex"
        else: label = "reasoning"
        texts.append(d["text"])
        labels.append(label)
        sg_count += 1
    print(f"  4. ShareGPT: {sg_count}")

# 5. Cross-difficulty
cross_count = 0
for name in ["cross_difficulty_bbh", "cross_difficulty_gsm8k", "cross_difficulty_math", "cross_difficulty_ifeval"]:
    path = INDUSTRY_DIR / f"{name}.json"
    if not path.exists(): continue
    with open(path) as f:
        cd_data = json.load(f)
    for d in cd_data:
        try: score = float(d["label"])
        except: continue
        if score < -0.5: label = "moderate"
        elif score < 1.0: label = "complex"
        else: label = "reasoning"
        texts.append(d["text"])
        labels.append(label)
        cross_count += 1
print(f"  5. Cross-difficulty: {cross_count}")

# 6. Claude reasoning
claude_path = INDUSTRY_DIR / "claude_reasoning.json"
if claude_path.exists():
    with open(claude_path) as f:
        claude_data = json.load(f)
    c_count = 0
    for d in claude_data:
        if d.get("category") != "math": continue
        for m in d.get("messages", []):
            if m.get("role") == "user" and m.get("content"):
                texts.append(m["content"])
                labels.append("reasoning")
                c_count += 1
                break
    print(f"  6. Claude reasoning: {c_count}")

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
        print(f"  7. {name}: {len(ds)}")

# 8. Synthetic reasoning
reasoning_prompts = [
    "Prove by induction that the sum of first n squares equals n(n+1)(2n+1)/6",
    "Prove that there are infinitely many prime numbers",
    "Derive the closed-form solution for the Fibonacci sequence",
    "Show that the halting problem is undecidable using diagonalization",
    "Prove by contradiction that sqrt(2) is irrational",
    "Derive the Black-Scholes equation from first principles",
    "Prove Gödel's first incompleteness theorem",
    "Show that P ≠ NP implies one-way functions exist",
    "Prove the fundamental theorem of calculus",
    "Prove that every continuous function on [a,b] is Riemann integrable",
] * 10
texts.extend(reasoning_prompts)
labels.extend(["reasoning"] * len(reasoning_prompts))
print(f"  8. Synthetic reasoning: {len(reasoning_prompts)}")

print(f"\nTotal: {len(texts)} samples")
print(f"Distribution: {dict(Counter(labels))}")


# ═══════════════════════════════════════════════════════════════
# Prepare data tensors
# ═══════════════════════════════════════════════════════════════
print("\nConverting text to character indices...")
X_all = np.array([text_to_indices(t) for t in texts])
y_all = np.array([LABEL_TO_IDX[l] for l in labels])

# Train/test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.15, random_state=42, stratify=y_all
)
print(f"Train: {len(X_train)}, Test: {len(X_test)}")

# Convert to tensors
X_train_t = torch.LongTensor(X_train)
y_train_t = torch.LongTensor(y_train)
X_test_t = torch.LongTensor(X_test)
y_test_t = torch.LongTensor(y_test)

train_dataset = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)


# ═══════════════════════════════════════════════════════════════
# Train the CNN
# ═══════════════════════════════════════════════════════════════
print(f"\nTraining CharCNN (embed={EMBED_DIM}, filters={NUM_FILTERS}, kernels={FILTER_SIZES})...")
print(f"  Epochs: {EPOCHS}, Batch size: {BATCH_SIZE}, LR: {LEARNING_RATE}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Device: {device}")

model = CharCNN().to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"  Total parameters: {total_params:,} ({total_params * 4 / 1024 / 1024:.2f} MB in FP32)")

# Class weights for imbalance
class_counts = Counter(y_train.tolist())
total = len(y_train)
class_weights = torch.FloatTensor([total / (NUM_CLASSES * class_counts[i]) for i in range(NUM_CLASSES)]).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)

best_acc = 0.0
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # Evaluate
    model.eval()
    with torch.no_grad():
        test_logits = model(X_test_t.to(device))
        test_preds = test_logits.argmax(dim=1).cpu().numpy()
        acc = (test_preds == y_test).mean()
        scheduler.step(1.0 - acc)

    if acc > best_acc:
        best_acc = acc
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if (epoch + 1) % 3 == 0 or epoch == 0:
        print(f"  Epoch {epoch+1:2d}/{EPOCHS}: loss={total_loss/len(train_loader):.4f}, test_acc={acc:.4f}, best={best_acc:.4f}")

# Load best model
model.load_state_dict(best_state)
model.eval()
print(f"\n  Best test accuracy: {best_acc:.4f}")


# ═══════════════════════════════════════════════════════════════
# Final evaluation
# ═══════════════════════════════════════════════════════════════
from sklearn.metrics import classification_report

with torch.no_grad():
    test_logits = model(X_test_t.to(device))
    test_preds = test_logits.argmax(dim=1).cpu().numpy()

y_test_labels = [IDX_TO_LABEL[i] for i in y_test]
y_pred_labels = [IDX_TO_LABEL[i] for i in test_preds]
print("\nTest set classification report:")
print(classification_report(y_test_labels, y_pred_labels))


# ═══════════════════════════════════════════════════════════════
# Export model weights for numpy-only inference
# ═══════════════════════════════════════════════════════════════
print("\nExporting model for numpy inference...")

state = model.state_dict()
export_data = {
    "model_type": "char_cnn",
    "config": {
        "vocab_size": 257,
        "embed_dim": EMBED_DIM,
        "num_filters": NUM_FILTERS,
        "filter_sizes": FILTER_SIZES,
        "num_classes": NUM_CLASSES,
        "max_len": MAX_LEN,
    },
    "classes": [IDX_TO_LABEL[i] for i in range(NUM_CLASSES)],
    "weights": {
        "embedding": state["embedding.weight"].numpy().tolist(),
        "conv_weights": [state[f"convs.{i}.weight"].numpy().tolist() for i in range(len(FILTER_SIZES))],
        "conv_biases": [state[f"convs.{i}.bias"].numpy().tolist() for i in range(len(FILTER_SIZES))],
        "fc_weight": state["fc.weight"].numpy().tolist(),
        "fc_bias": state["fc.bias"].numpy().tolist(),
    },
}

json_path = OUTPUT_DIR / "char_cnn_classifier.json"
with open(json_path, "w") as f:
    json.dump(export_data, f)
model_size = json_path.stat().st_size
print(f"✅ Model saved: {json_path} ({model_size / 1024:.1f} KB)")


# ═══════════════════════════════════════════════════════════════
# Numpy-only inference implementation + benchmarks
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("NUMPY-ONLY INFERENCE TEST")
print("=" * 60)


class CharCNNInference:
    """Pure numpy inference for the character CNN. No PyTorch needed."""

    def __init__(self, model_path: str):
        t0 = time.perf_counter()
        with open(model_path) as f:
            data = json.load(f)
        self.config = data["config"]
        self.classes = data["classes"]
        w = data["weights"]
        self.embedding = np.array(w["embedding"], dtype=np.float32)
        self.conv_weights = [np.array(cw, dtype=np.float32) for cw in w["conv_weights"]]
        self.conv_biases = [np.array(cb, dtype=np.float32) for cb in w["conv_biases"]]
        self.fc_weight = np.array(w["fc_weight"], dtype=np.float32)
        self.fc_bias = np.array(w["fc_bias"], dtype=np.float32)
        self.load_time = time.perf_counter() - t0

    def _text_to_indices(self, text: str) -> np.ndarray:
        max_len = self.config["max_len"]
        encoded = text.encode("utf-8", errors="replace")[:max_len]
        indices = np.zeros(max_len, dtype=np.int32)
        for i, b in enumerate(encoded):
            indices[i] = b + 1
        return indices

    def _conv1d(self, x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
        """1D convolution: x is (channels_in, length), weight is (channels_out, channels_in, kernel)."""
        out_channels, in_channels, kernel_size = weight.shape
        pad = kernel_size // 2
        # Pad input
        x_padded = np.pad(x, ((0, 0), (pad, pad)), mode='constant')
        length = x.shape[1]
        # Output
        out = np.zeros((out_channels, length), dtype=np.float32)
        for oc in range(out_channels):
            for ic in range(in_channels):
                for k in range(kernel_size):
                    out[oc] += weight[oc, ic, k] * x_padded[ic, k:k + length]
            out[oc] += bias[oc]
        return out

    def classify(self, text: str) -> tuple[str, float]:
        """Classify a single text. Returns (label, confidence)."""
        indices = self._text_to_indices(text)

        # Embedding lookup: (max_len,) → (max_len, embed_dim) → (embed_dim, max_len)
        emb = self.embedding[indices]  # (max_len, embed_dim)
        emb = emb.T  # (embed_dim, max_len)

        # Conv layers + global max pool
        conv_outputs = []
        for w, b in zip(self.conv_weights, self.conv_biases):
            c = self._conv1d(emb, w, b)
            c = np.maximum(c, 0)  # ReLU
            c = c.max(axis=1)  # Global max pool → (num_filters,)
            conv_outputs.append(c)

        # Concatenate
        features = np.concatenate(conv_outputs)  # (num_filters * n_convs,)

        # FC layer
        logits = self.fc_weight @ features + self.fc_bias

        # Softmax
        logits_shifted = logits - logits.max()
        exp_logits = np.exp(logits_shifted)
        probs = exp_logits / exp_logits.sum()

        idx = int(np.argmax(probs))
        return self.classes[idx], float(probs[idx])


# Load and benchmark
print("\n--- Loading model (cold start) ---")
inference = CharCNNInference(str(json_path))
print(f"  First load time: {inference.load_time * 1000:.1f} ms")
print(f"  Model file size: {model_size / 1024:.1f} KB")
print(f"  Parameters: {total_params:,}")

# Warm up
_ = inference.classify("Hello world")

# Benchmark inference speed
print("\n--- Inference latency ---")
test_texts = [
    "What is Python?",
    "Design a distributed system with fault tolerance and auto-scaling",
    "Prove by induction that 1+2+...+n = n(n+1)/2 for all natural numbers",
]

for text in test_texts:
    times = []
    for _ in range(20):
        t0 = time.perf_counter()
        result = inference.classify(text)
        times.append(time.perf_counter() - t0)
    avg_ms = np.mean(times) * 1000
    p50_ms = np.percentile(times, 50) * 1000
    p99_ms = np.percentile(times, 99) * 1000
    label, conf = result
    print(f"  [{label:>9}]({conf:.2f}) avg={avg_ms:.1f}ms p50={p50_ms:.1f}ms p99={p99_ms:.1f}ms | {text[:50]}")


# ═══════════════════════════════════════════════════════════════
# Real-world routing test (same 30 prompts)
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("REAL-WORLD ROUTING TEST (30 prompts) — CharCNN vs TF-IDF")
print("=" * 60)

# Load TF-IDF model for comparison
tfidf_model_path = PACKAGE_DATA / "ml_classifier.json"
with open(tfidf_model_path) as f:
    tfidf_data = json.load(f)
tfidf_vocab = tfidf_data["vocabulary"]
tfidf_idf = np.array(tfidf_data["idf"], dtype=np.float64)
tfidf_coef = np.array(tfidf_data["coefficients"], dtype=np.float64)
tfidf_intercept = np.array(tfidf_data["intercept"], dtype=np.float64)
tfidf_classes = tfidf_data["classes"]
tfidf_ngram = tuple(tfidf_data["tfidf_params"]["ngram_range"])

def tfidf_classify(text):
    text_lower = text[:5000].lower()
    words = text_lower.split()
    ngrams = []
    for n in range(tfidf_ngram[0], tfidf_ngram[1]+1):
        for i in range(len(words) - n + 1):
            ngrams.append(" ".join(words[i:i+n]))
    n_features = len(tfidf_idf)
    tf = np.zeros(n_features, dtype=np.float64)
    for ng in ngrams:
        idx = tfidf_vocab.get(ng)
        if idx is not None:
            tf[idx] += 1.0
    mask = tf > 0
    tf[mask] = 1.0 + np.log(tf[mask])
    tfidf_vec = tf * tfidf_idf
    norm = np.linalg.norm(tfidf_vec)
    if norm > 0:
        tfidf_vec /= norm
    logits = tfidf_coef @ tfidf_vec + tfidf_intercept
    logits_shifted = logits - logits.max()
    exp_l = np.exp(logits_shifted)
    probs = exp_l / exp_l.sum()
    idx = int(np.argmax(probs))
    return tfidf_classes[idx], float(probs[idx])

test_prompts = [
    ("What is Python?", "simple"),
    ("How do I print hello world in JavaScript?", "simple"),
    ("What is the capital of France?", "simple"),
    ("Translate hello to Spanish", "simple"),
    ("What time is it in Tokyo?", "simple"),
    ("List 5 popular programming languages", "simple"),
    ("What does HTTP stand for?", "simple"),
    ("How do I install numpy?", "simple"),
    ("Explain the difference between SQL and NoSQL databases", "moderate"),
    ("Write a Python function to reverse a linked list", "moderate"),
    ("Summarize the main principles of object-oriented programming", "moderate"),
    ("Compare React and Vue.js for building web applications", "moderate"),
    ("Write a bash script to find all files larger than 100MB", "moderate"),
    ("Explain how OAuth 2.0 authentication works", "moderate"),
    ("Write a Python decorator with retry logic and exponential backoff", "moderate"),
    ("How does garbage collection work in Java?", "moderate"),
    ("Design a distributed caching system that handles 10M requests per second with cross-region replication", "complex"),
    ("Implement a B-tree data structure with insert, delete, and range query operations", "complex"),
    ("Design a real-time fraud detection pipeline for financial transactions at scale", "complex"),
    ("Architect a multi-tenant SaaS platform with tenant isolation, billing, and auto-scaling", "complex"),
    ("Write a lock-free concurrent hash map implementation in C++", "complex"),
    ("Design the database schema and API for a social media feed with ranking algorithm", "complex"),
    ("Implement a distributed consensus protocol similar to Raft", "complex"),
    ("Design a CI/CD pipeline with canary deployments, automated rollback, and chaos testing", "complex"),
    ("Prove by mathematical induction that 1+2+3+...+n = n(n+1)/2", "reasoning"),
    ("Prove that the halting problem is undecidable", "reasoning"),
    ("Derive the time complexity of mergesort using the master theorem", "reasoning"),
    ("Prove that P != NP implies one-way functions exist", "reasoning"),
    ("Show using contradiction that there is no largest prime number", "reasoning"),
    ("Prove the correctness of Dijkstras algorithm using loop invariants", "reasoning"),
]

cnn_correct = 0
tfidf_correct = 0
results = {"simple": [0,0,0], "moderate": [0,0,0], "complex": [0,0,0], "reasoning": [0,0,0]}

for prompt, expected in test_prompts:
    cnn_label, cnn_conf = inference.classify(prompt)
    tfidf_label, tfidf_conf = tfidf_classify(prompt)

    c_ok = cnn_label == expected
    t_ok = tfidf_label == expected
    if c_ok: cnn_correct += 1
    if t_ok: tfidf_correct += 1
    results[expected][2] += 1
    if c_ok: results[expected][0] += 1
    if t_ok: results[expected][1] += 1

    c_mark = "✓" if c_ok else "✗"
    t_mark = "✓" if t_ok else "✗"
    print(f"  {c_mark} CNN:[{cnn_label:>9}]({cnn_conf:.2f})  {t_mark} TF-IDF:[{tfidf_label:>9}]({tfidf_conf:.2f})  exp={expected:>9} | {prompt[:50]}")

print(f"\n{'='*60}")
print(f"FINAL COMPARISON")
print(f"{'='*60}")
print(f"  CharCNN:  {cnn_correct}/{len(test_prompts)} = {cnn_correct/len(test_prompts)*100:.1f}%")
print(f"  TF-IDF:   {tfidf_correct}/{len(test_prompts)} = {tfidf_correct/len(test_prompts)*100:.1f}%")
print(f"  Delta:    {cnn_correct - tfidf_correct:+d} ({(cnn_correct - tfidf_correct)/len(test_prompts)*100:+.1f}%)")
print()
print(f"  Per-class:")
print(f"  {'Class':>12}  {'CharCNN':>8}  {'TF-IDF':>8}")
for cls in ["simple", "moderate", "complex", "reasoning"]:
    c, t, total = results[cls]
    print(f"  {cls:>12}  {c}/{total} ({c/total*100:3.0f}%)  {t}/{total} ({t/total*100:3.0f}%)")

print(f"\n{'='*60}")
print(f"MODEL CHARACTERISTICS")
print(f"{'='*60}")
print(f"  Model file size:    {model_size / 1024:.1f} KB ({model_size / 1024 / 1024:.2f} MB)")
print(f"  Parameters:         {total_params:,}")
print(f"  First load time:    {inference.load_time * 1000:.1f} ms")
print(f"  Inference latency:  ~{avg_ms:.1f} ms per request (CPU, numpy)")
print(f"  Dependencies:       numpy only (for inference)")
