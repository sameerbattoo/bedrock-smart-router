"""ML-based prompt complexity classifier using TF-IDF + Logistic Regression.

This module provides an optional ML classifier that uses a pre-trained TF-IDF
vectorizer and Logistic Regression model for prompt complexity classification.
It uses pure numpy inference at runtime — no sklearn dependency required.

Inherits from ``ComplexityClassifier``, gaining system prompt floor logic,
message extraction helpers, and the ``classify_request()`` pipeline for free.
Adds an ML-specific low-confidence guard that defaults to "simple" when the
model predicts complex/reasoning with < 50% confidence.

The classifier categorizes prompts into four complexity levels:
- simple: Basic questions, greetings, factual lookups
- moderate: Multi-step tasks, summarization, moderate reasoning
- complex: Architecture design, code generation, deep analysis
- reasoning: Multi-step logical reasoning, proofs, complex problem-solving

Usage
-----
Install with the ml extra:

    pip install bedrock-smart-router[ml]

Then use the classifier:

    from bedrock_smart_router.ml_classifier import MLComplexityClassifier

    classifier = MLComplexityClassifier()
    label, confidence = classifier.classify("Design a distributed system...")
    # ('complex', 0.87)

    # Batch classification
    results = classifier.classify_batch([
        "What is Python?",
        "Implement a B-tree with rebalancing",
    ])
    # [('simple', 0.92), ('complex', 0.78)]

The model data is lazily loaded on first use to avoid startup overhead.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Optional

from bedrock_smart_router.complexity_classifier import (
    ComplexityClassifier,
    COMPLEXITY_ORDER,
    LEVEL_TO_LABEL,
    DEFAULT_FLOOR_CONFIDENCE_THRESHOLD,
    DEFAULT_FLOOR_DAMPENING,
)

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


class MLComplexityClassifier(ComplexityClassifier):
    """ML-based complexity classifier using TF-IDF + Logistic Regression.

    Uses pure numpy for inference — no sklearn required at runtime.
    Model data is lazily loaded from a JSON file on first use.

    Parameters
    ----------
    model_path : str or Path, optional
        Path to the classifier_data.json model file.
        Defaults to the bundled model in the package data directory.
    floor_confidence_threshold : float, optional
        Minimum confidence the system prompt floor must have to be applied.
        Default 0.7 — prevents low-confidence floor from overriding user message.
    floor_dampening : float, optional
        Confidence dampening when floor overrides user message classification.
        Default 0.8 — signals the result was floor-influenced.
    """

    # Minimum confidence to trust the ML prediction.
    # Below this threshold, the model is essentially guessing (near-uniform distribution).
    # Falls back to "simple" to avoid over-classifying ambiguous/short inputs.
    MIN_CONFIDENCE_THRESHOLD = 0.50

    def __init__(
        self,
        model_path: Optional[str | Path] = None,
        floor_confidence_threshold: float = DEFAULT_FLOOR_CONFIDENCE_THRESHOLD,
        floor_dampening: float = DEFAULT_FLOOR_DAMPENING,
    ) -> None:
        if not HAS_NUMPY:
            raise ImportError(
                "numpy is required for the ML classifier. "
                "Install it with: pip install bedrock-smart-router[ml]"
            )

        super().__init__(
            floor_confidence_threshold=floor_confidence_threshold,
            floor_dampening=floor_dampening,
        )

        if model_path is None:
            model_path = Path(__file__).parent / "data" / "ml_classifier.json"
        self._model_path = Path(model_path)

        # Lazy-loaded model components
        self._vocabulary: Optional[dict[str, int]] = None
        self._idf: Optional[np.ndarray] = None
        self._coefficients: Optional[np.ndarray] = None
        self._intercept: Optional[np.ndarray] = None
        self._classes: Optional[list[str]] = None
        self._ngram_range: Optional[tuple[int, int]] = None
        self._loaded = False

    def _load_model(self) -> None:
        """Load model data from JSON file (called lazily on first use)."""
        if self._loaded:
            return

        with open(self._model_path, "r") as f:
            data = json.load(f)

        self._vocabulary = data["vocabulary"]
        self._idf = np.array(data["idf"], dtype=np.float64)
        self._coefficients = np.array(data["coefficients"], dtype=np.float64)
        self._intercept = np.array(data["intercept"], dtype=np.float64)
        self._classes = data["classes"]

        tfidf_params = data.get("tfidf_params", {})
        ngram_range = tfidf_params.get("ngram_range", [1, 3])
        self._ngram_range = (ngram_range[0], ngram_range[1])

        self._loaded = True

    @property
    def classes(self) -> list[str]:
        """Return the list of class labels."""
        self._load_model()
        return self._classes  # type: ignore

    def _extract_ngrams(self, text: str) -> list[str]:
        """Extract word-level character n-grams (1-gram to 3-gram tokens).

        Splits on whitespace and generates n-grams of words matching
        the training configuration. Truncates to first 5000 chars for
        performance (longer text adds no classification value).
        """
        # Truncate for performance — classifier was trained on short prompts
        text = text[:5000].lower()
        words = text.split()
        ngrams: list[str] = []
        min_n, max_n = self._ngram_range  # type: ignore

        for n in range(min_n, max_n + 1):
            for i in range(len(words) - n + 1):
                ngram = " ".join(words[i:i + n])
                ngrams.append(ngram)

        return ngrams

    def _vectorize(self, text: str) -> np.ndarray:
        """Transform text into a TF-IDF feature vector using pure numpy.

        Steps:
        1. Extract n-grams from the text
        2. Look up each n-gram in the vocabulary to get its index
        3. Compute term frequencies with sublinear TF: 1 + log(tf) if tf > 0
        4. Multiply by IDF values
        5. L2-normalize the resulting vector
        """
        vocab = self._vocabulary  # type: ignore
        idf = self._idf  # type: ignore
        n_features = len(idf)

        # Extract n-grams and compute term frequencies
        ngrams = self._extract_ngrams(text)
        tf = np.zeros(n_features, dtype=np.float64)

        for ngram in ngrams:
            idx = vocab.get(ngram)
            if idx is not None:
                tf[idx] += 1.0

        # Sublinear TF: 1 + log(tf) for tf > 0
        mask = tf > 0
        tf[mask] = 1.0 + np.log(tf[mask])

        # TF-IDF: multiply by IDF
        tfidf = tf * idf

        # L2 normalization
        norm = np.linalg.norm(tfidf)
        if norm > 0:
            tfidf /= norm

        return tfidf

    def _predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Compute class probabilities using logistic regression.

        Computes: softmax(coefficients @ features + intercept)
        """
        coef = self._coefficients  # type: ignore
        intercept = self._intercept  # type: ignore

        # Linear scores: (n_classes,)
        logits = coef @ features + intercept

        # Softmax for numerical stability
        logits_shifted = logits - np.max(logits)
        exp_logits = np.exp(logits_shifted)
        probabilities = exp_logits / np.sum(exp_logits)

        return probabilities

    def classify(self, text: str) -> tuple[str, float]:
        """Classify a single text prompt.

        Parameters
        ----------
        text : str
            The prompt text to classify.

        Returns
        -------
        tuple[str, float]
            A tuple of (label, confidence) where label is one of
            'simple', 'moderate', 'complex', 'reasoning' and confidence
            is the probability score (0.0 to 1.0).
        """
        self._load_model()

        features = self._vectorize(text)
        probabilities = self._predict_proba(features)

        predicted_idx = int(np.argmax(probabilities))
        label = self._classes[predicted_idx]  # type: ignore
        confidence = float(probabilities[predicted_idx])

        return label, confidence

    def classify_batch(self, texts: list[str]) -> list[tuple[str, float]]:
        """Classify a batch of text prompts.

        Parameters
        ----------
        texts : list[str]
            List of prompt texts to classify.

        Returns
        -------
        list[tuple[str, float]]
            List of (label, confidence) tuples for each input text.
        """
        self._load_model()
        return [self.classify(text) for text in texts]

    def predict_proba_all(self, text: str) -> dict[str, float]:
        """Return probabilities for all classes.

        Parameters
        ----------
        text : str
            The prompt text to classify.

        Returns
        -------
        dict[str, float]
            Dictionary mapping class labels to their probabilities.
        """
        self._load_model()

        features = self._vectorize(text)
        probabilities = self._predict_proba(features)

        return {
            label: float(prob)
            for label, prob in zip(self._classes, probabilities)  # type: ignore
        }

    def classify_request(
        self,
        messages: list[dict],
        system: list[dict] | None = None,
        tool_config: dict | None = None,
    ) -> tuple[str, float]:
        """Classify a full Bedrock Converse request.

        Extends the base class pipeline with an ML-specific low-confidence
        guard: if the model predicts complex/reasoning with < 50% confidence,
        it defaults to "simple" to avoid over-classifying ambiguous inputs.
        """
        # 1. Extract and classify the LAST USER MESSAGE (primary signal)
        last_user_text = self.extract_last_user_text(messages)
        if last_user_text:
            user_label, user_conf = self.classify(last_user_text)
        else:
            # No text in user messages — fall back to full context
            full_text = self.assemble_full_context(messages, system, tool_config)
            return self.classify(full_text)

        # 1b. ML-specific: Low-confidence guard
        # If the model predicts complex/reasoning but with low confidence
        # (near-uniform distribution), default to "simple".
        if user_conf < self.MIN_CONFIDENCE_THRESHOLD and user_label in ("complex", "reasoning"):
            user_label = "simple"

        # 2. Apply system prompt floor (shared logic from base class)
        return self._apply_floor(user_label, user_conf, system, tool_config)

    @staticmethod
    def _extract_last_user_text(messages: list[dict]) -> str:
        """Legacy alias — use extract_last_user_text() from base class."""
        return ComplexityClassifier.extract_last_user_text(messages)

    @staticmethod
    def _assemble_floor_context(
        system: list[dict] | None = None,
        tool_config: dict | None = None,
    ) -> str:
        """Legacy alias — use assemble_floor_context() from base class."""
        return ComplexityClassifier.assemble_floor_context(system, tool_config)

    def _assemble_context(
        self,
        messages: list[dict],
        system: list[dict] | None = None,
        tool_config: dict | None = None,
    ) -> str:
        """Legacy alias — use assemble_full_context() from base class."""
        return self.assemble_full_context(messages, system, tool_config)
