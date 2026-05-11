# Complexity Classifier Training

Trains a sentence-embedding-based complexity classifier to replace/augment the keyword heuristic in the router's `request_analyzer.py`.

## Approach

1. **Prepare training data** from our 295 custom prompts + industry standard datasets
2. **Encode prompts** using a pre-trained sentence transformer (all-MiniLM-L6-v2, 22MB)
3. **Train a classifier** (logistic regression or small MLP) on the embeddings
4. **Evaluate** accuracy vs the keyword-based approach
5. **Export** the trained model for use in the router

## Data Sources

| Source | Prompts | Labels |
|--------|---------|--------|
| Custom prompts (benchmarks/prompts/) | 295 | simple/medium/complex (hand-labeled) |
| Industry datasets (ind_standard_datasets/) | ~200 | Labeled by task complexity |
| Existing datasets (datasets/) | ~4000+ | Labeled by task type mapping |
| **Total** | ~4500+ | 3-class classification |

## Usage

```bash
# 1. Prepare training data
python benchmarks/complexity_classifier/prepare_data.py

# 2. Train the classifier
python benchmarks/complexity_classifier/train.py

# 3. Evaluate
python benchmarks/complexity_classifier/evaluate.py

# 4. Export for router integration
python benchmarks/complexity_classifier/export_model.py
```

## Requirements

```
pip install sentence-transformers scikit-learn torch
```
