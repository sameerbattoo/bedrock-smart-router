#!/usr/bin/env python3
"""Re-label training data using NVIDIA's prompt-task-and-complexity-classifier.

NVIDIA's model produces a continuous complexity score (0-1) across 6 dimensions:
- Creativity, Reasoning, Contextual Knowledge, Domain Knowledge, Constraints, Few-shots

We use their `prompt_complexity_score` (weighted composite) to assign our 4-tier labels:
- simple:    score < 0.15
- moderate:  0.15 <= score < 0.35
- complex:   0.35 <= score < 0.55
- reasoning: score >= 0.55

This gives us gold-standard labels from a production-quality classifier (DeBERTa backbone,
human-annotated training data, 98%+ accuracy on their eval set).

Usage: .venv/bin/python benchmarks/classifier/relabel_with_nvidia.py
Output: benchmarks/data/industry_standard/nvidia_relabeled.json
"""
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

try:
    from transformers import AutoModel, AutoTokenizer
    from huggingface_hub import PyTorchModelHubMixin
except ImportError:
    print("ERROR: pip install transformers huggingface_hub")
    sys.exit(1)

# ═══════════════════════════════════════════════════════════════
# NVIDIA Model (from their HuggingFace model card)
# ═══════════════════════════════════════════════════════════════

class MeanPooling(nn.Module):
    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        return sum_embeddings / sum_mask


class MulticlassHead(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x):
        return self.fc(x)


class CustomModel(nn.Module, PyTorchModelHubMixin):
    def __init__(self, target_sizes, task_type_map, weights_map, divisor_map):
        super().__init__()
        self.backbone = AutoModel.from_pretrained("microsoft/DeBERTa-v3-base")
        self.target_sizes = target_sizes.values()
        self.task_type_map = task_type_map
        self.weights_map = weights_map
        self.divisor_map = divisor_map
        self.heads = [MulticlassHead(self.backbone.config.hidden_size, sz) for sz in self.target_sizes]
        for i, head in enumerate(self.heads):
            self.add_module(f"head_{i}", head)
        self.pool = MeanPooling()

    def compute_results(self, preds, target):
        if target == "task_type":
            softmax_probs = torch.softmax(preds, dim=1)
            top1_indices = torch.topk(preds, k=1, dim=1).indices
            top1 = top1_indices.detach().cpu().tolist()
            return [self.task_type_map[str(idx[0])] for idx in top1]
        else:
            preds = torch.softmax(preds, dim=1)
            weights = np.array(self.weights_map[target])
            weighted_sum = np.sum(np.array(preds.detach().cpu()) * weights, axis=1)
            scores = weighted_sum / self.divisor_map[target]
            return [round(float(v), 4) for v in scores]

    def forward(self, batch):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        mean_pooled = self.pool(last_hidden_state, attention_mask)
        logits = [self.heads[k](mean_pooled) for k in range(len(list(self.target_sizes)))]

        result = {}
        # Creativity
        result["creativity"] = self.compute_results(logits[1], "creativity_scope")
        # Reasoning
        result["reasoning"] = self.compute_results(logits[2], "reasoning")
        # Contextual knowledge
        result["contextual_knowledge"] = self.compute_results(logits[3], "contextual_knowledge")
        # Few shots
        result["few_shots"] = self.compute_results(logits[4], "number_of_few_shots")
        # Domain knowledge
        result["domain_knowledge"] = self.compute_results(logits[5], "domain_knowledge")
        # Constraints
        result["constraints"] = self.compute_results(logits[7], "constraint_ct")
        # Task type
        result["task_type"] = self.compute_results(logits[0], "task_type")

        # Composite score
        result["complexity_score"] = [
            round(
                0.35 * c + 0.25 * r + 0.15 * con + 0.15 * dk + 0.05 * ck + 0.05 * fs, 5
            )
            for c, r, con, dk, ck, fs in zip(
                result["creativity"], result["reasoning"], result["constraints"],
                result["domain_knowledge"], result["contextual_knowledge"], result["few_shots"],
            )
        ]
        return result


# ═══════════════════════════════════════════════════════════════
# Map NVIDIA score to our taxonomy
# ═══════════════════════════════════════════════════════════════
def nvidia_score_to_label(score: float, reasoning_score: float = 0.0) -> str:
    """Map NVIDIA's composite complexity score to our 4-tier labels."""
    # If reasoning dimension is very high, classify as reasoning regardless
    if reasoning_score >= 0.6:
        return "reasoning"
    if score < 0.15:
        return "simple"
    elif score < 0.35:
        return "moderate"
    elif score < 0.55:
        return "complex"
    else:
        return "reasoning"


# ═══════════════════════════════════════════════════════════════
# Load and process training data
# ═══════════════════════════════════════════════════════════════
DATA_PATH = Path(__file__).parent / "training_data.json"
GENERATED_DIR = Path(__file__).parent.parent / "data" / "generated"
INDUSTRY_DIR = Path(__file__).parent.parent / "data" / "industry_standard"
DIFFICULTY_MAP = {"simple": "simple", "medium": "moderate", "complex": "complex", "hard": "complex"}

print("Loading all training texts...")
all_texts: list[str] = []
all_sources: list[str] = []

# 1. Original
with open(DATA_PATH) as f:
    data = json.load(f)
for d in data:
    all_texts.append(d["text"][:512])  # NVIDIA model has 512 token limit
    all_sources.append("original")
print(f"  Original: {len(data)}")

# 2. DevQuasar
dq_path = INDUSTRY_DIR / "devquasar_router.json"
if dq_path.exists():
    with open(dq_path) as f:
        dq = json.load(f)
    for d in dq:
        all_texts.append(d["text"][:512])
        all_sources.append("devquasar")
    print(f"  DevQuasar: {len(dq)}")

# 3. Easy2Hard
e2h_path = INDUSTRY_DIR / "easy2hard_bench.json"
if e2h_path.exists():
    with open(e2h_path) as f:
        e2h = json.load(f)
    for d in e2h:
        all_texts.append(d["text"][:512])
        all_sources.append("easy2hard")
    print(f"  Easy2Hard: {len(e2h)}")

# 4. LeetCode
lc_path = INDUSTRY_DIR / "leetcode.json"
if lc_path.exists():
    with open(lc_path) as f:
        lc = json.load(f)
    for d in lc:
        all_texts.append(d["text"][:512])
        all_sources.append("leetcode")
    print(f"  LeetCode: {len(lc)}")

# 5. OpenOrca
orca_path = INDUSTRY_DIR / "openorca_subset.json"
if orca_path.exists():
    with open(orca_path) as f:
        orca = json.load(f)
    for d in orca:
        all_texts.append(d["text"][:512])
        all_sources.append("openorca")
    print(f"  OpenOrca: {len(orca)}")

# 6. IFEval
if_path = INDUSTRY_DIR / "ifeval.json"
if if_path.exists():
    with open(if_path) as f:
        ifdata = json.load(f)
    for d in ifdata:
        all_texts.append(d["text"][:512])
        all_sources.append("ifeval")
    print(f"  IFEval: {len(ifdata)}")

# 7. BBH
bbh_path = INDUSTRY_DIR / "bbh.json"
if bbh_path.exists():
    with open(bbh_path) as f:
        bbh = json.load(f)
    for d in bbh:
        all_texts.append(d["text"][:512])
        all_sources.append("bbh")
    print(f"  BBH: {len(bbh)}")

# 8. ShareGPT
sg_path = INDUSTRY_DIR / "sharegpt_sample.json"
if sg_path.exists():
    with open(sg_path) as f:
        sg = json.load(f)
    for d in sg:
        all_texts.append(d["text"][:512])
        all_sources.append("sharegpt")
    print(f"  ShareGPT: {len(sg)}")

print(f"\nTotal texts to label: {len(all_texts)}")

# IMPORTANT: DeBERTa on CPU is slow (~5-10s per batch of 32).
# Only label a strategically chosen subset for practical runtime.
# Focus on the datasets with questionable labels (DevQuasar moderate/complex boundary + OpenOrca)
# Skip datasets that already have clean labels (LeetCode has human labels, BBH is clearly complex)
MAX_SAMPLES = 5000  # Enough to make a difference, fast enough to finish in ~15 min
if len(all_texts) > MAX_SAMPLES:
    print(f"  Sampling {MAX_SAMPLES} from {len(all_texts)} (prioritizing noisy sources)...")
    # Prioritize DevQuasar (most samples, noisiest labels) and OpenOrca
    priority_indices = [i for i, s in enumerate(all_sources) if s in ("devquasar", "openorca", "sharegpt")]
    other_indices = [i for i, s in enumerate(all_sources) if s not in ("devquasar", "openorca", "sharegpt")]
    random.shuffle(priority_indices)
    random.shuffle(other_indices)
    # Take 3000 from priority, 2000 from others
    selected = priority_indices[:3000] + other_indices[:2000]
    random.shuffle(selected)
    all_texts = [all_texts[i] for i in selected]
    all_sources = [all_sources[i] for i in selected]
    print(f"  Selected {len(all_texts)} samples for relabeling")


# ═══════════════════════════════════════════════════════════════
# Load NVIDIA model
# ═══════════════════════════════════════════════════════════════
print("\nLoading NVIDIA prompt-task-and-complexity-classifier...")
print("  (This downloads ~700MB on first run)")

MODEL_ID = "nvidia/prompt-task-and-complexity-classifier"

# Load config manually (not a standard HF config)
from huggingface_hub import hf_hub_download
config_path = hf_hub_download(MODEL_ID, "config.json")
with open(config_path) as f:
    config_data = json.load(f)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Load model using PyTorchModelHubMixin
model = CustomModel(
    target_sizes=config_data["target_sizes"],
    task_type_map=config_data["task_type_map"],
    weights_map=config_data["weights_map"],
    divisor_map=config_data["divisor_map"],
).from_pretrained(MODEL_ID)
model.eval()
print("  ✅ Model loaded")


# ═══════════════════════════════════════════════════════════════
# Batch inference
# ═══════════════════════════════════════════════════════════════
BATCH_SIZE = 32
results = []
total = len(all_texts)

print(f"\nRunning NVIDIA classifier on {total} samples (batch_size={BATCH_SIZE})...")
t0 = time.time()

for i in range(0, total, BATCH_SIZE):
    batch_texts = all_texts[i:i + BATCH_SIZE]
    # Prepend "Prompt: " as per NVIDIA's model card
    batch_prompts = [f"Prompt: {t}" for t in batch_texts]

    encoded = tokenizer(
        batch_prompts,
        return_tensors="pt",
        add_special_tokens=True,
        max_length=512,
        padding="max_length",
        truncation=True,
    )

    with torch.no_grad():
        output = model(encoded)

    for j in range(len(batch_texts)):
        score = output["complexity_score"][j]
        reasoning = output["reasoning"][j]
        label = nvidia_score_to_label(score, reasoning)
        results.append({
            "text": all_texts[i + j],
            "label": label,
            "nvidia_score": score,
            "nvidia_reasoning": reasoning,
            "nvidia_creativity": output["creativity"][j],
            "nvidia_domain": output["domain_knowledge"][j],
            "nvidia_constraints": output["constraints"][j],
            "task_type": output["task_type"][j],
            "source": all_sources[i + j],
        })

    if (i + BATCH_SIZE) % (BATCH_SIZE * 10) == 0 or i + BATCH_SIZE >= total:
        elapsed = time.time() - t0
        pct = min(100, (i + BATCH_SIZE) / total * 100)
        rate = (i + BATCH_SIZE) / elapsed
        print(f"  {pct:5.1f}% ({i + BATCH_SIZE}/{total}) — {rate:.0f} samples/sec — {elapsed:.0f}s elapsed")

elapsed = time.time() - t0
print(f"\n  Done in {elapsed:.1f}s ({total/elapsed:.0f} samples/sec)")


# ═══════════════════════════════════════════════════════════════
# Save results
# ═══════════════════════════════════════════════════════════════
output_path = INDUSTRY_DIR / "nvidia_relabeled.json"

# Save only text + label (compact format for training)
training_data = [{"text": r["text"], "label": r["label"], "source": r["source"]} for r in results]
with open(output_path, "w") as f:
    json.dump(training_data, f, indent=2)
print(f"\n✅ Saved {len(training_data)} relabeled samples to {output_path}")

# Also save full results with scores for analysis
full_path = INDUSTRY_DIR / "nvidia_relabeled_full.json"
with open(full_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"✅ Saved full results (with scores) to {full_path}")

# Print distribution
from collections import Counter
dist = Counter(r["label"] for r in results)
print(f"\nLabel distribution (NVIDIA):")
for label in ["simple", "moderate", "complex", "reasoning"]:
    print(f"  {label:>9}: {dist.get(label, 0):>6} ({dist.get(label, 0)/len(results)*100:.1f}%)")

# Compare with original labels for overlap analysis
print(f"\nSize: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
