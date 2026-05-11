# Built-in Datasets for Automatic Model Evaluation

This directory contains scripts to download, transform, and load industry-standard
benchmark datasets into the `ModelEval-Datasets` DynamoDB table.

These datasets appear as a "Built-in Benchmarks" section on the homepage. Users
can preview them and adopt them into their own projects, choosing which models
and judges to use. The prompts, variable sets, expected answers, and metrics
are read-only (enforced in the frontend via `sourceDatasetId` check).

## Table Schema

```
PK: datasetId     (e.g. "trex")
SK: evaluationKey  ("__meta__" for dataset metadata, "accuracy" / "robustness" for evals)
```

## Supported Datasets

| Dataset | Task | Evaluations | Script |
|---------|------|-------------|--------|
| T-REx   | General text generation | Accuracy, Robustness | `trex/` |
| BoolQ   | Question and answer | Accuracy, Robustness, Toxicity | `boolq/` |
| NaturalQuestions | Question and answer (RAG) | Accuracy, Robustness, Toxicity | `natural_questions/` |
| Women's Clothing Reviews | Text classification | Accuracy, Robustness | `womens_clothing/` |
| XSum    | Text summarization | Accuracy, Robustness, Toxicity | `xsum/` |
| RealToxicityPrompts | General text generation | Toxicity, Toxicity Challenging | `real_toxicity_prompts/` |
| BOLD    | General text generation | Robustness, Toxicity | `bold/` |
| TriviaQA | Question and answer | Accuracy, Robustness, Toxicity | `triviaqa/` |
| WikiText-2 | General text generation | Robustness | `wikitext2/` |

## Usage

```bash
# 1. Download the T-REx sample dataset
python backend/datasets/trex/download.py

# 2. Transform to ModelEval format
python backend/datasets/trex/transform.py

# 3. Load into DynamoDB (requires AWS credentials)
python backend/datasets/trex/load_to_db.py --region us-west-2
```

## Adding a New Dataset

1. Create a new directory: `backend/datasets/<name>/`
2. Add `download.py` — fetches the raw data
3. Add `transform.py` — converts to our JSON format with `dataset`, `project`, `evaluations` keys
4. Add `load_to_db.py` — writes to the Datasets table (can copy from trex/load_to_db.py)
5. Run the three scripts in order
