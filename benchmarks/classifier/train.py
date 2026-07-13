# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

#!/usr/bin/env python3
"""Train a sentence-embedding-based complexity classifier.

Uses all-MiniLM-L6-v2 (22MB) to encode prompts, then trains a
logistic regression classifier on top.

Requirements:
    pip install sentence-transformers scikit-learn

Usage:
    python benchmarks/complexity_classifier/train.py
"""
import json
import os
import sys
import pickle
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.preprocessing import LabelEncoder
except ImportError:
    print("ERROR: Required libraries not installed.")
    print("Run: pip install sentence-transformers scikit-learn")
    sys.exit(1)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "training_data.json")
MODEL_DIR = os.path.join(BASE_DIR, "trained_model")
os.makedirs(MODEL_DIR, exist_ok=True)

# Configuration
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # 22MB, fast, good quality
MAX_TEXT_LENGTH = 1024  # Truncate very long prompts for embedding
TEST_SPLIT = 0.2
RANDOM_STATE = 42


def main():
    print("=" * 60)
    print("Training Complexity Classifier")
    print("=" * 60)

    # Load training data
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: {DATA_PATH} not found. Run prepare_data.py first.")
        sys.exit(1)

    with open(DATA_PATH) as f:
        data = json.load(f)

    print(f"\nLoaded {len(data)} training samples")

    # Extract texts and labels
    texts = [item["text"][:MAX_TEXT_LENGTH] for item in data]
    labels = [item["label"] for item in data]

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(labels)
    print(f"Classes: {list(le.classes_)}")
    print(f"Distribution: {dict(zip(le.classes_, np.bincount(y)))}")

    # Load sentence transformer
    print(f"\nLoading embedding model: {EMBEDDING_MODEL}...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print(f"  Embedding dimension: {model.get_sentence_embedding_dimension()}")

    # Encode all texts
    print(f"\nEncoding {len(texts)} texts...")
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)
    print(f"  Embeddings shape: {embeddings.shape}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, y, test_size=TEST_SPLIT, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")

    # Train logistic regression
    print("\nTraining logistic regression classifier...")
    clf = LogisticRegression(
        max_iter=1000,
        C=1.0,
        class_weight="balanced",  # Handle class imbalance
        random_state=RANDOM_STATE,
    )
    clf.fit(X_train, y_train)

    # Evaluate
    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)

    # Test set accuracy
    y_pred = clf.predict(X_test)
    accuracy = (y_pred == y_test).mean()
    print(f"\nTest accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")

    # Classification report
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Confusion matrix
    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"  {'':>10} {'simple':>8} {'medium':>8} {'complex':>8}")
    for i, row_label in enumerate(le.classes_):
        print(f"  {row_label:>10} {cm[i][0]:>8} {cm[i][1]:>8} {cm[i][2]:>8}")

    # Cross-validation
    print(f"\n5-fold cross-validation:")
    cv_scores = cross_val_score(clf, embeddings, y, cv=5, scoring="accuracy")
    print(f"  Scores: {cv_scores}")
    print(f"  Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

    # Compare with keyword-based approach
    print("\n" + "=" * 60)
    print("COMPARISON WITH KEYWORD ANALYZER")
    print("=" * 60)

    sys.path.insert(0, os.path.dirname(os.path.dirname(BENCHMARKS_DIR := os.path.dirname(BASE_DIR))))
    try:
        from bedrock_smart_router.request_analyzer import RequestAnalyzer
        analyzer = RequestAnalyzer()

        keyword_correct = 0
        embedding_correct = 0
        total = len(data)

        # Map analyzer complexity to our labels
        complexity_to_label = {
            "simple": "simple",
            "moderate": "medium",
            "complex": "complex",
            "reasoning": "complex",
        }

        for i, item in enumerate(data):
            true_label = item["label"]

            # Keyword analyzer prediction
            messages = [{"role": "user", "content": [{"text": item["text"][:2000]}]}]
            result = analyzer.analyze(messages)
            keyword_pred = complexity_to_label.get(result.complexity.value, "medium")
            if keyword_pred == true_label:
                keyword_correct += 1

        # Embedding classifier prediction (already computed on test set)
        # Use full dataset for fair comparison
        y_pred_all = clf.predict(embeddings)
        embedding_correct = (y_pred_all == y).sum()

        print(f"\n  Keyword analyzer accuracy: {keyword_correct}/{total} ({keyword_correct/total*100:.1f}%)")
        print(f"  Embedding classifier accuracy: {embedding_correct}/{total} ({embedding_correct/total*100:.1f}%)")
        print(f"  Improvement: +{(embedding_correct - keyword_correct)/total*100:.1f}%")

    except Exception as e:
        print(f"  Could not compare with keyword analyzer: {e}")

    # Save model
    print("\n" + "=" * 60)
    print("SAVING MODEL")
    print("=" * 60)

    # Save classifier
    clf_path = os.path.join(MODEL_DIR, "classifier.pkl")
    with open(clf_path, "wb") as f:
        pickle.dump(clf, f)

    # Save label encoder
    le_path = os.path.join(MODEL_DIR, "label_encoder.pkl")
    with open(le_path, "wb") as f:
        pickle.dump(le, f)

    # Save config
    config = {
        "embedding_model": EMBEDDING_MODEL,
        "max_text_length": MAX_TEXT_LENGTH,
        "classes": list(le.classes_),
        "accuracy": float(accuracy),
        "cv_mean": float(cv_scores.mean()),
        "training_samples": len(data),
        "embedding_dim": int(embeddings.shape[1]),
    }
    config_path = os.path.join(MODEL_DIR, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n  Classifier: {clf_path}")
    print(f"  Label encoder: {le_path}")
    print(f"  Config: {config_path}")
    print(f"\n  Note: The sentence-transformers model ({EMBEDDING_MODEL}) is")
    print(f"  downloaded automatically on first use (~22MB).")
    print(f"\nDone!")


if __name__ == "__main__":
    main()
