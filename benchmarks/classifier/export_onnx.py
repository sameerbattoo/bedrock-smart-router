#!/usr/bin/env python3
"""Export the trained sentence-transformer model to ONNX format.

This creates:
  - model.onnx (the MiniLM encoder in ONNX format, ~22MB)
  - tokenizer.json (HuggingFace fast tokenizer config)
  - classifier_weights.json (logistic regression weights as JSON, no pickle needed)
  - config.json (metadata)

These files can be shipped with the package for inference using only
onnxruntime + tokenizers (no PyTorch/sentence-transformers needed at runtime).

Requirements (for export only):
    pip install sentence-transformers onnx onnxruntime
"""
import json
import os
import pickle
import sys

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    import torch
    import onnxruntime as ort
except ImportError:
    print("ERROR: pip install sentence-transformers onnx onnxruntime")
    sys.exit(1)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINED_DIR = os.path.join(BASE_DIR, "trained_model")
EXPORT_DIR = os.path.join(BASE_DIR, "onnx_model")
os.makedirs(EXPORT_DIR, exist_ok=True)

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
MAX_LENGTH = 256  # Max token length for the model


def export_model():
    print("=" * 60)
    print("Exporting complexity classifier to ONNX")
    print("=" * 60)

    # Load the sentence transformer
    print("\n1. Loading sentence-transformers model...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    tokenizer = model.tokenizer

    # Save tokenizer (this is what we need at runtime)
    print("\n2. Saving tokenizer...")
    tokenizer.save_pretrained(EXPORT_DIR)
    print(f"   Saved tokenizer to {EXPORT_DIR}/")

    # Export the transformer to ONNX
    print("\n3. Exporting transformer to ONNX...")
    # Get the underlying transformer model and move to CPU
    transformer = model[0].auto_model.cpu()

    # Create dummy input
    dummy_input = tokenizer(
        "This is a test sentence for export",
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
    )

    onnx_path = os.path.join(EXPORT_DIR, "model.onnx")

    torch.onnx.export(
        transformer,
        (dummy_input["input_ids"], dummy_input["attention_mask"]),
        onnx_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "last_hidden_state": {0: "batch_size", 1: "sequence_length"},
        },
        opset_version=14,
    )

    onnx_size = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"   Exported to {onnx_path} ({onnx_size:.1f} MB)")

    # Export classifier weights as JSON (no pickle dependency at runtime)
    print("\n4. Exporting classifier weights...")
    clf_path = os.path.join(TRAINED_DIR, "classifier.pkl")
    le_path = os.path.join(TRAINED_DIR, "label_encoder.pkl")

    with open(clf_path, "rb") as f:
        clf = pickle.load(f)
    with open(le_path, "rb") as f:
        le = pickle.load(f)

    classifier_data = {
        "weights": clf.coef_.tolist(),
        "bias": clf.intercept_.tolist(),
        "classes": le.classes_.tolist(),
    }

    clf_json_path = os.path.join(EXPORT_DIR, "classifier_weights.json")
    with open(clf_json_path, "w") as f:
        json.dump(classifier_data, f)
    print(f"   Saved classifier weights to {clf_json_path}")

    # Save config
    print("\n5. Saving config...")
    config = {
        "embedding_model": EMBEDDING_MODEL,
        "max_length": MAX_LENGTH,
        "embedding_dim": 384,
        "classes": le.classes_.tolist(),
        "pooling": "mean",  # MiniLM uses mean pooling
        "normalize": True,
    }
    config_path = os.path.join(EXPORT_DIR, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"   Saved config to {config_path}")

    # Verify the ONNX model works
    print("\n6. Verifying ONNX inference...")
    session = ort.InferenceSession(onnx_path)

    # Tokenize a test input
    test_text = "Build a cohort analysis showing retention by signup month"
    inputs = tokenizer(
        test_text,
        return_tensors="np",
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
    )

    # Run ONNX inference
    onnx_output = session.run(
        None,
        {
            "input_ids": inputs["input_ids"].astype(np.int64),
            "attention_mask": inputs["attention_mask"].astype(np.int64),
        },
    )

    # Mean pooling (same as sentence-transformers)
    token_embeddings = onnx_output[0]  # (1, seq_len, 384)
    attention_mask = inputs["attention_mask"]
    mask_expanded = np.expand_dims(attention_mask, -1)  # (1, seq_len, 1)
    sum_embeddings = np.sum(token_embeddings * mask_expanded, axis=1)
    sum_mask = np.sum(mask_expanded, axis=1)
    embedding = sum_embeddings / sum_mask  # (1, 384)

    # Normalize
    norm = np.linalg.norm(embedding, axis=1, keepdims=True)
    embedding = embedding / norm

    # Classify
    weights = np.array(classifier_data["weights"])
    bias = np.array(classifier_data["bias"])
    logits = embedding @ weights.T + bias
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    pred_idx = np.argmax(probs, axis=1)[0]
    pred_label = classifier_data["classes"][pred_idx]

    print(f"   Test: '{test_text[:50]}...'")
    print(f"   Prediction: {pred_label} (confidence: {probs[0][pred_idx]:.2%})")
    print(f"   Probabilities: {dict(zip(classifier_data['classes'], [f'{p:.3f}' for p in probs[0]]))}")

    # Compare with original sentence-transformers output
    original_embedding = model.encode([test_text])
    original_logits = original_embedding @ clf.coef_.T + clf.intercept_
    original_pred = le.inverse_transform(clf.predict(original_embedding))[0]
    print(f"   Original prediction: {original_pred}")
    print(f"   Match: {'✓' if pred_label == original_pred else '✗'}")

    # Summary
    print("\n" + "=" * 60)
    print("EXPORT COMPLETE")
    print("=" * 60)
    print(f"\n  Output directory: {EXPORT_DIR}/")
    print(f"  Files:")
    for f in sorted(os.listdir(EXPORT_DIR)):
        size = os.path.getsize(os.path.join(EXPORT_DIR, f))
        if size > 1024 * 1024:
            print(f"    {f:<30} {size/1024/1024:.1f} MB")
        else:
            print(f"    {f:<30} {size/1024:.1f} KB")

    total_size = sum(os.path.getsize(os.path.join(EXPORT_DIR, f)) for f in os.listdir(EXPORT_DIR))
    print(f"\n  Total size: {total_size/1024/1024:.1f} MB")
    print(f"\n  Runtime dependencies: onnxruntime (~40MB) + tokenizers (~7MB)")
    print(f"  No PyTorch, no sentence-transformers needed at inference time.")


if __name__ == "__main__":
    export_model()
