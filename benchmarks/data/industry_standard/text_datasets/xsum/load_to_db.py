#!/usr/bin/env python3
"""
Load the transformed XSum dataset into the Datasets DynamoDB table.

Usage:
    python load_to_db.py [--table-prefix ModelEval] [--region us-west-2]
"""
import json
import os
import sys
import argparse

try:
    import boto3
except ImportError:
    print("ERROR: boto3 is required.  pip install boto3")
    sys.exit(1)

OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "output", "xsum_modeleval.json")
META_SK = "__meta__"


def load(table_prefix="ModelEval", region="us-west-2"):
    if not os.path.exists(OUTPUT_FILE):
        print(f"ERROR: {OUTPUT_FILE} not found. Run transform.py first.")
        sys.exit(1)

    with open(OUTPUT_FILE, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    dataset_meta = data["dataset"]
    evaluations = data["evaluations"]

    dynamodb = boto3.resource("dynamodb", region_name=region)
    table_name = os.environ.get("DATASETS_TABLE", f"{table_prefix}-Datasets")
    table = dynamodb.Table(table_name)

    dataset_id = dataset_meta["id"]

    meta_item = {
        "datasetId": dataset_id,
        "evaluationKey": META_SK,
        "name": dataset_meta["name"],
        "description": dataset_meta["description"],
        "source": dataset_meta["source"],
        "citation": dataset_meta.get("citation", ""),
        "taskType": dataset_meta["taskType"],
        "metrics": dataset_meta.get("metrics", []),
        "evaluationCount": len(evaluations),
        "icon": "file-text",
        "tags": ["text-summarization", "news", "accuracy", "robustness", "toxicity"],
        "whatItTests": (
            "BBC news articles where the model must produce a single-sentence "
            "summary. Open-access alternative to Gigaword."
        ),
        "whyItMatters": (
            "Tests abstractive summarization — distilling long text into concise, "
            "accurate summaries."
        ),
        "evalMetrics": [
            {"name": "Correctness", "desc": "Summary captures the key facts"},
            {"name": "Completeness", "desc": "All important points covered"},
            {"name": "Toxicity", "desc": "Summaries free of harmful content"},
        ],
    }
    table.put_item(Item=meta_item)
    print(f"Wrote dataset metadata: {dataset_id} / {META_SK}")

    eval_keys = []
    for ev in evaluations:
        name = ev["useCaseName"].lower()
        if "accuracy" in name:
            eval_key = "accuracy"
        elif "robustness" in name:
            eval_key = "robustness"
        elif "toxicity" in name:
            eval_key = "toxicity"
        else:
            eval_key = name.replace(" ", "-")[:30]

        eval_keys.append(eval_key)

        eval_item = {
            "datasetId": dataset_id,
            "evaluationKey": eval_key,
            "useCaseName": ev["useCaseName"],
            "systemPrompt": ev["systemPrompt"],
            "userPrompts": ev["userPrompts"],
            "promptVariableSets": ev["promptVariableSets"],
            "expectedAnswers": ev["expectedAnswers"],
            "qualityMetrics": ev["qualityMetrics"],
            "isRAG": ev.get("isRAG", False),
        }
        table.put_item(Item=eval_item)
        print(f"  Wrote evaluation: {dataset_id} / {eval_key}")
        print(f"    Use case: {ev['useCaseName']}")
        print(f"    Variable sets: {len(ev['promptVariableSets'])}")

    print(f"\nDone. Dataset '{dataset_id}' loaded with {len(eval_keys)} evaluation(s).")
    return dataset_id, eval_keys


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load XSum dataset into ModelEval Datasets table")
    parser.add_argument("--table-prefix", default="ModelEval", help="DynamoDB table name prefix")
    parser.add_argument("--region", default="us-west-2", help="AWS region")
    args = parser.parse_args()
    load(table_prefix=args.table_prefix, region=args.region)
