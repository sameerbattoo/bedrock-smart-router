#!/usr/bin/env python3
"""Use the trained complexity classifier to predict prompt complexity.

Usage:
    python benchmarks/complexity_classifier/predict.py "Write a SQL query to find top customers"
    python benchmarks/complexity_classifier/predict.py --file prompts.txt
"""
import argparse
import json
import os
import pickle
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "trained_model")


def load_model():
    """Load the trained classifier and embedding model."""
    config_path = os.path.join(MODEL_DIR, "config.json")
    clf_path = os.path.join(MODEL_DIR, "classifier.pkl")
    le_path = os.path.join(MODEL_DIR, "label_encoder.pkl")

    if not all(os.path.exists(p) for p in [config_path, clf_path, le_path]):
        print("ERROR: Model not found. Run train.py first.")
        sys.exit(1)

    with open(config_path) as f:
        config = json.load(f)

    with open(clf_path, "rb") as f:
        clf = pickle.load(f)

    with open(le_path, "rb") as f:
        le = pickle.load(f)

    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer(config["embedding_model"])

    return clf, le, embedder, config


def predict(text, clf, le, embedder, config):
    """Predict complexity for a single text."""
    text = text[:config["max_text_length"]]
    embedding = embedder.encode([text])
    pred_idx = clf.predict(embedding)[0]
    proba = clf.predict_proba(embedding)[0]
    label = le.inverse_transform([pred_idx])[0]

    return {
        "label": label,
        "confidence": float(max(proba)),
        "probabilities": {
            le.inverse_transform([i])[0]: float(p)
            for i, p in enumerate(proba)
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Predict prompt complexity")
    parser.add_argument("text", nargs="?", help="Prompt text to classify")
    parser.add_argument("--file", type=str, help="File with prompts (one per line)")
    args = parser.parse_args()

    clf, le, embedder, config = load_model()
    print(f"Model loaded (accuracy: {config['accuracy']:.1%}, trained on {config['training_samples']} samples)")

    if args.file:
        with open(args.file) as f:
            texts = [line.strip() for line in f if line.strip()]
        for text in texts:
            result = predict(text, clf, le, embedder, config)
            print(f"  [{result['label']:>7}] ({result['confidence']:.2f}) {text[:80]}...")
    elif args.text:
        result = predict(args.text, clf, le, embedder, config)
        print(f"\n  Prediction: {result['label']}")
        print(f"  Confidence: {result['confidence']:.2%}")
        print(f"  Probabilities:")
        for label, prob in sorted(result["probabilities"].items(), key=lambda x: -x[1]):
            bar = "#" * int(prob * 30)
            print(f"    {label:>8}: {prob:.3f} {bar}")
    else:
        # Interactive mode
        print("\nInteractive mode (type a prompt, press Enter):")
        while True:
            try:
                text = input("\n> ").strip()
                if not text:
                    continue
                result = predict(text, clf, le, embedder, config)
                print(f"  → {result['label']} (confidence: {result['confidence']:.2%})")
            except (EOFError, KeyboardInterrupt):
                print("\nBye!")
                break


if __name__ == "__main__":
    main()
