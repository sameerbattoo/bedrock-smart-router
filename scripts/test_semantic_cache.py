"""Test script: Semantic cache similarity scoring for Text2SQL queries.

Tests various query pairs to find the right similarity threshold.
Run: python scripts/test_semantic_cache.py
"""
import os
import sys
os.environ["BYPASS_TOOL_CONSENT"] = "true"
os.environ["MPLBACKEND"] = "Agg"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'demo', 'backend'))

import json
import time
import boto3
import numpy as np

REGION = "us-west-2"
EMBEDDING_MODEL = "amazon.titan-embed-text-v2:0"

bedrock = boto3.client("bedrock-runtime", region_name=REGION)


def get_embedding(text: str) -> np.ndarray:
    """Get embedding vector for a text."""
    response = bedrock.invoke_model(
        modelId=EMBEDDING_MODEL,
        body=json.dumps({"inputText": text}),
    )
    result = json.loads(response["body"].read())
    return np.array(result["embedding"])


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# Test pairs: (query1, query2, should_match)
TEST_PAIRS = [
    # Should match (same intent, different wording)
    ("Show monthly order trends for 2025", "Visualize the order trends for the year 2025", True),
    ("Top 5 products by revenue", "What are the best selling products by total revenue?", True),
    ("Show me sales by category", "Display total sales for each product category", True),
    ("How many orders per month in 2025?", "Monthly order count for 2025", True),
    ("Which customers have the most orders?", "Top customers by order count", True),
    ("Average order value by category", "What is the mean order amount per category?", True),
    
    # Should NOT match (different intent/variables)
    ("Show monthly order trends for 2025", "Show monthly order trends for 2024", False),
    ("Top 5 products by revenue", "Top 5 customers by revenue", False),
    ("Sales by category", "Orders by shipping method", False),
    ("Monthly orders for 2025", "Daily orders for January 2025", False),
    ("Products with low stock", "Products with highest revenue", False),
]

print("=" * 80)
print("Semantic Cache Similarity Test")
print(f"Embedding model: {EMBEDDING_MODEL}")
print("=" * 80)
print()

# First, test with auto_extract (intent extraction)
print("Testing RAW embedding similarity (no intent extraction):")
print("-" * 80)
print(f"{'Query 1':<45} {'Query 2':<45} {'Sim':>6} {'Match':>6} {'OK':>4}")
print("-" * 80)

results = []
for q1, q2, should_match in TEST_PAIRS:
    e1 = get_embedding(q1)
    e2 = get_embedding(q2)
    sim = cosine_similarity(e1, e2)
    results.append((q1, q2, sim, should_match))
    
    status = "✓" if (sim >= 0.92 and should_match) or (sim < 0.92 and not should_match) else "✗"
    print(f"{q1[:44]:<45} {q2[:44]:<45} {sim:.4f} {'YES' if should_match else 'NO':>6} {status:>4}")

print()
print("=" * 80)
print("THRESHOLD ANALYSIS")
print("=" * 80)

for threshold in [0.80, 0.82, 0.85, 0.87, 0.88, 0.89, 0.90, 0.91, 0.92, 0.93, 0.95]:
    tp = sum(1 for _, _, sim, match in results if sim >= threshold and match)
    fp = sum(1 for _, _, sim, match in results if sim >= threshold and not match)
    fn = sum(1 for _, _, sim, match in results if sim < threshold and match)
    tn = sum(1 for _, _, sim, match in results if sim < threshold and not match)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"  Threshold {threshold:.2f}: TP={tp} FP={fp} FN={fn} TN={tn} | Precision={precision:.2f} Recall={recall:.2f} F1={f1:.2f}")

print()
print("Current threshold: 0.92")
print()

# Find optimal
best_f1 = 0
best_threshold = 0.92
for threshold in [t/100 for t in range(75, 98)]:
    tp = sum(1 for _, _, sim, match in results if sim >= threshold and match)
    fp = sum(1 for _, _, sim, match in results if sim >= threshold and not match)
    fn = sum(1 for _, _, sim, match in results if sim < threshold and match)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print(f"RECOMMENDED threshold: {best_threshold:.2f} (F1={best_f1:.2f})")
