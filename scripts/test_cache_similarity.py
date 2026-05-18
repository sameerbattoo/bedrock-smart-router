"""Test: Semantic similarity for the 4 demo questions with 3 variants each.

Run: python scripts/test_cache_similarity.py
"""
import json
import boto3
import numpy as np

REGION = "us-west-2"
EMBEDDING_MODEL = "amazon.titan-embed-text-v2:0"
THRESHOLD = 0.80

bedrock = boto3.client("bedrock-runtime", region_name=REGION)


def get_embedding(text: str) -> np.ndarray:
    response = bedrock.invoke_model(
        modelId=EMBEDDING_MODEL,
        body=json.dumps({"inputText": text}),
    )
    return np.array(json.loads(response["body"].read())["embedding"])


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# Original questions and their semantic variants
QUESTIONS = {
    "Show me the month-over-month growth rate of orders for each category in 2025": [
        "What is the percentage change in orders per category each month in 2025?",
        "Display the monthly order growth by product category for 2025",
        "How did order volumes change month to month across categories in 2025?",
    ],
    "Show monthly order trends for 2025": [
        "Visualize the order trends for the year 2025",
        "How many orders were placed each month in 2025?",
        "Display 2025 monthly order counts",
    ],
    "For customers who placed orders in both Q1 and Q2 of 2025, what was their average order value change between quarters?": [
        "Compare average order values between Q1 and Q2 2025 for repeat customers",
        "How did spending change from Q1 to Q2 2025 for customers active in both quarters?",
        "What's the AOV difference between first and second quarter 2025 for returning buyers?",
    ],
    "Show me a chart of sales by category": [
        "Visualize revenue breakdown by product category",
        "Display a graph of total sales per category",
        "Chart the sales distribution across categories",
    ],
}

print("=" * 90)
print(f"Semantic Cache Similarity Test — Threshold: {THRESHOLD}")
print("=" * 90)

all_pass = 0
all_fail = 0

for original, variants in QUESTIONS.items():
    print(f"\n{'─' * 90}")
    print(f"ORIGINAL: {original[:80]}")
    print(f"{'─' * 90}")
    
    orig_emb = get_embedding(original)
    
    for variant in variants:
        var_emb = get_embedding(variant)
        sim = cosine_sim(orig_emb, var_emb)
        hit = sim >= THRESHOLD
        status = "✅ HIT " if hit else "❌ MISS"
        if hit:
            all_pass += 1
        else:
            all_fail += 1
        print(f"  {status} ({sim:.4f}) → {variant[:70]}")

print(f"\n{'=' * 90}")
print(f"RESULTS: {all_pass} hits, {all_fail} misses out of {all_pass + all_fail} pairs")
print(f"Hit rate at threshold {THRESHOLD}: {all_pass/(all_pass+all_fail)*100:.0f}%")
print(f"{'=' * 90}")
