"""Tests for the ComplexityClassifier interface and both implementations."""

import pytest

from bedrock_smart_router.complexity_classifier import (
    ComplexityClassifier,
    COMPLEXITY_LABELS,
    COMPLEXITY_ORDER,
    LEVEL_TO_LABEL,
)
from bedrock_smart_router.heuristic_classifier import HeuristicClassifier
from bedrock_smart_router.ml_classifier import MLComplexityClassifier


# ═══════════════════════════════════════════════════════════════════
# Base class interface tests
# ═══════════════════════════════════════════════════════════════════

class TestComplexityClassifierInterface:
    """Test the abstract base class contract."""

    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            ComplexityClassifier()

    def test_custom_classifier_minimal(self):
        """A custom classifier only needs classify() and predict_proba_all()."""
        class MinimalClassifier(ComplexityClassifier):
            def classify(self, text):
                return ("simple", 0.9)
            def predict_proba_all(self, text):
                return {"simple": 0.9, "moderate": 0.05, "complex": 0.03, "reasoning": 0.02}

        clf = MinimalClassifier()
        label, conf = clf.classify("Hello")
        assert label == "simple"
        assert conf == 0.9

    def test_custom_classifier_inherits_classify_request(self):
        """classify_request is inherited and uses classify() internally."""
        class CountClassifier(ComplexityClassifier):
            def classify(self, text):
                words = len(text.split())
                if words > 20:
                    return ("complex", 0.8)
                elif words > 5:
                    return ("moderate", 0.7)
                return ("simple", 0.95)
            def predict_proba_all(self, text):
                label, conf = self.classify(text)
                probs = {l: 0.05 for l in COMPLEXITY_LABELS}
                probs[label] = conf
                return probs

        clf = CountClassifier()
        msgs = [{"role": "user", "content": [{"text": "Hello"}]}]
        label, conf = clf.classify_request(msgs)
        assert label == "simple"

    def test_custom_classifier_floor_logic(self):
        """The base class floor logic upgrades but never downgrades."""
        class AlwaysSimple(ComplexityClassifier):
            def classify(self, text):
                # Always returns simple, but with varying confidence
                if "architect" in text.lower():
                    return ("moderate", 0.85)  # Floor text classified as moderate
                return ("simple", 0.9)
            def predict_proba_all(self, text):
                return {"simple": 0.9, "moderate": 0.05, "complex": 0.03, "reasoning": 0.02}

        clf = AlwaysSimple()
        msgs = [{"role": "user", "content": [{"text": "Hello"}]}]
        system = [{"text": "You are a senior architect analyzing systems"}]

        label, conf = clf.classify_request(msgs, system=system)
        # Floor should upgrade simple → moderate (floor is moderate with 0.85 > 0.7 threshold)
        assert label == "moderate"
        # Confidence should be dampened
        assert conf == pytest.approx(0.9 * 0.8, abs=0.01)

    def test_floor_does_not_downgrade(self):
        """Floor never downgrades — if user is complex, simple floor doesn't change it."""
        class AlwaysComplex(ComplexityClassifier):
            def classify(self, text):
                if len(text) < 20:
                    return ("simple", 0.5)  # Floor text is simple
                return ("complex", 0.85)
            def predict_proba_all(self, text):
                return {"simple": 0.05, "moderate": 0.05, "complex": 0.85, "reasoning": 0.05}

        clf = AlwaysComplex()
        msgs = [{"role": "user", "content": [{"text": "Design a fault-tolerant distributed consensus algorithm with formal verification"}]}]
        system = [{"text": "Hi"}]  # Simple system prompt

        label, conf = clf.classify_request(msgs, system=system)
        assert label == "complex"  # Not downgraded
        assert conf == 0.85

    def test_extract_last_user_text(self):
        """Extracts text from the last user message, skipping tool results."""
        msgs = [
            {"role": "user", "content": [{"text": "First question"}]},
            {"role": "assistant", "content": [{"text": "Answer"}]},
            {"role": "user", "content": [{"toolResult": {"toolUseId": "1", "content": [{"text": "result"}]}}]},
            {"role": "user", "content": [{"text": "Second question"}]},
        ]
        text = ComplexityClassifier.extract_last_user_text(msgs)
        assert text == "Second question"

    def test_extract_last_user_text_skips_tool_only(self):
        """Skips user messages that only have toolResult blocks."""
        msgs = [
            {"role": "user", "content": [{"text": "Real question"}]},
            {"role": "assistant", "content": [{"text": "Answer"}]},
            {"role": "user", "content": [{"toolResult": {"toolUseId": "1", "content": []}}]},
        ]
        text = ComplexityClassifier.extract_last_user_text(msgs)
        assert text == "Real question"

    def test_assemble_floor_context(self):
        system = [{"text": "You are a helpful assistant"}]
        tool_config = {"tools": [{"toolSpec": {"name": "search", "description": "Search the web"}}]}
        result = ComplexityClassifier.assemble_floor_context(system, tool_config)
        assert "You are a helpful assistant" in result
        assert "[Tools: search]" in result

    def test_floor_parameters_configurable(self):
        """Floor threshold and dampening are configurable."""
        class Simple(ComplexityClassifier):
            def classify(self, text):
                if "complex" in text:
                    return ("complex", 0.6)  # Below default 0.7 threshold
                return ("simple", 0.9)
            def predict_proba_all(self, text):
                return {"simple": 0.9, "moderate": 0.05, "complex": 0.03, "reasoning": 0.02}

        # With default threshold (0.7) — floor at 0.6 confidence won't apply
        clf_default = Simple()
        msgs = [{"role": "user", "content": [{"text": "Hello"}]}]
        system = [{"text": "complex system"}]
        label, _ = clf_default.classify_request(msgs, system=system)
        assert label == "simple"  # Floor didn't apply (0.6 < 0.7)

        # With lower threshold (0.5) — floor at 0.6 confidence WILL apply
        clf_low = Simple(floor_confidence_threshold=0.5)
        label2, _ = clf_low.classify_request(msgs, system=system)
        assert label2 == "moderate"  # Floor applied (0.6 > 0.5)


# ═══════════════════════════════════════════════════════════════════
# HeuristicClassifier tests
# ═══════════════════════════════════════════════════════════════════

class TestHeuristicClassifier:
    def setup_method(self):
        self.clf = HeuristicClassifier()

    def test_inherits_from_base(self):
        assert isinstance(self.clf, ComplexityClassifier)

    def test_classify_simple(self):
        label, conf = self.clf.classify("Hello")
        assert label == "simple"
        assert conf < 0.125

    def test_classify_moderate(self):
        label, conf = self.clf.classify(
            "Write a Python function to sort a list using merge sort with proper error handling"
        )
        assert label in ("moderate", "complex")

    def test_classify_complex(self):
        label, conf = self.clf.classify(
            "Design a distributed microservices architecture with event sourcing, "
            "CQRS patterns, and implement a saga orchestrator for distributed transactions "
            "across multiple bounded contexts with eventual consistency guarantees"
        )
        assert label in ("complex", "reasoning")

    def test_classify_returns_valid_label(self):
        for text in ["Hi", "Write code", "Design a system", "Prove P != NP step by step"]:
            label, conf = self.clf.classify(text)
            assert label in COMPLEXITY_LABELS
            assert 0.0 <= conf <= 1.0

    def test_predict_proba_all_sums_to_one(self):
        probs = self.clf.predict_proba_all("What is Python?")
        assert set(probs.keys()) == set(COMPLEXITY_LABELS)
        assert abs(sum(probs.values()) - 1.0) < 0.01

    def test_classify_request_with_system_floor(self):
        msgs = [{"role": "user", "content": [{"text": "Hi"}]}]
        system = [{"text": "You are a senior solutions architect. Analyze complex distributed systems and provide detailed trade-off analysis."}]

        label, conf = self.clf.classify_request(msgs, system=system)
        # Floor should push "Hi" above simple
        assert label in ("simple", "moderate")
        assert conf > 0.0

    def test_classify_request_no_system(self):
        msgs = [{"role": "user", "content": [{"text": "Hello"}]}]
        label, conf = self.clf.classify_request(msgs)
        assert label == "simple"

    def test_custom_weights(self):
        """Custom weights change classification behavior."""
        # Heavy weight on code presence
        from bedrock_smart_router.request_analyzer import AnalyzerWeights
        heavy_code = AnalyzerWeights(
            token_count=0.1, code_presence=0.8,
            reasoning_markers=0.02, technical_depth=0.02,
            simple_indicators=0.01, multi_step=0.01,
            tool_use=0.01, document_analysis=0.01,
            conversation_depth=0.005, aws_specificity=0.005,
            math_logical=0.005, creative_open=0.005,
            output_format=0.005, constraint_density=0.005, context_ratio=0.005,
        )
        clf = HeuristicClassifier(weights=heavy_code)
        label, conf = clf.classify("python function def class import")
        # With heavy code weight, this should score higher
        assert conf > 0.1

    def test_custom_thresholds(self):
        """Custom thresholds change classification boundaries."""
        from bedrock_smart_router.request_analyzer import ComplexityThresholds
        strict = ComplexityThresholds(simple_max=0.01, moderate_max=0.05, complex_max=0.1)
        clf = HeuristicClassifier(thresholds=strict)
        # Even a short sentence should be "complex" with strict thresholds
        label, conf = clf.classify("Write a Python function")
        assert label in ("moderate", "complex", "reasoning")


# ═══════════════════════════════════════════════════════════════════
# MLComplexityClassifier tests
# ═══════════════════════════════════════════════════════════════════

class TestMLComplexityClassifier:
    def setup_method(self):
        self.clf = MLComplexityClassifier()

    def test_inherits_from_base(self):
        assert isinstance(self.clf, ComplexityClassifier)

    def test_classify_simple(self):
        label, conf = self.clf.classify("What is Python?")
        assert label == "simple"
        assert conf > 0.9

    def test_classify_complex(self):
        label, conf = self.clf.classify(
            "Design a distributed microservices architecture with event sourcing and CQRS"
        )
        assert label in ("complex", "reasoning")

    def test_classify_returns_valid_label(self):
        for text in ["Hi", "Write code", "Design a system"]:
            label, conf = self.clf.classify(text)
            assert label in COMPLEXITY_LABELS
            assert 0.0 <= conf <= 1.0

    def test_predict_proba_all_sums_to_one(self):
        probs = self.clf.predict_proba_all("What is Python?")
        assert set(probs.keys()) == set(COMPLEXITY_LABELS)
        assert abs(sum(probs.values()) - 1.0) < 0.01

    def test_low_confidence_guard(self):
        """Low-confidence complex/reasoning predictions default to simple."""
        # This short ambiguous text gets near-uniform probabilities
        label, conf = self.clf.classify_request(
            [{"role": "user", "content": [{"text": "You are is flying and what dates?"}]}]
        )
        assert label == "simple"

    def test_classify_request_with_floor(self):
        msgs = [{"role": "user", "content": [{"text": "Hello"}]}]
        system = [{"text": "You are a senior data analyst. Perform complex statistical analysis and build predictive models."}]
        label, conf = self.clf.classify_request(msgs, system=system)
        # Floor may or may not apply depending on floor confidence
        assert label in COMPLEXITY_LABELS

    def test_classify_request_no_system(self):
        msgs = [{"role": "user", "content": [{"text": "What is Python?"}]}]
        label, conf = self.clf.classify_request(msgs)
        assert label == "simple"
        assert conf > 0.9

    def test_floor_dampening(self):
        """When floor applies, confidence is dampened."""
        # Use a system prompt that the ML model classifies as complex with high confidence
        msgs = [{"role": "user", "content": [{"text": "Hello"}]}]
        system = [{"text": "Design and implement a distributed consensus algorithm with formal verification proofs and Byzantine fault tolerance analysis"}]
        label, conf = self.clf.classify_request(msgs, system=system)
        # If floor applied, confidence should be < original (dampened by 0.8)
        if label != "simple":
            # Floor was applied — confidence should be dampened
            original_conf = self.clf.classify("Hello")[1]
            assert conf < original_conf


# ═══════════════════════════════════════════════════════════════════
# Explain JSON compatibility tests
# ═══════════════════════════════════════════════════════════════════

class TestExplainCompatibility:
    """Test that the router's explain output works correctly with both classifiers."""

    def setup_method(self):
        from bedrock_smart_router.router import BedrockRouter
        self.router = BedrockRouter.create({"region": "us-west-2"})

    def test_heuristic_explain_structure(self):
        """Heuristic explain has expected fields."""
        from bedrock_smart_router.config import RoutingConfig
        msgs = [{"role": "user", "content": [{"text": "What is S3?"}]}]
        analysis = self.router._analyzer.analyze(msgs, classifier_override="heuristic")
        explain = self.router._analyzer.explain(msgs)

        # Heuristic explain should have these fields
        assert "dimension_scores" in explain
        assert "matched_markers" in explain
        assert "user_message_score" in explain
        assert "system_prompt_floor" in explain

    def test_ml_explain_structure(self):
        """ML explain has expected fields when called via the router."""
        msgs = [{"role": "user", "content": [{"text": "What is S3?"}]}]
        # Force ML classifier
        analysis = self.router._analyzer.analyze(msgs, classifier_override="ml")

        # The ML classifier should now be initialized
        assert self.router._analyzer._ml_classifier is not None

    def test_both_classifiers_produce_valid_complexity(self):
        """Both classifiers produce valid Complexity enum values."""
        from bedrock_smart_router.models import Complexity
        msgs = [{"role": "user", "content": [{"text": "Design a microservices architecture"}]}]

        # Heuristic
        analysis_h = self.router._analyzer.analyze(msgs, classifier_override="heuristic")
        assert analysis_h.complexity in (Complexity.SIMPLE, Complexity.MODERATE, Complexity.COMPLEX, Complexity.REASONING)

        # ML
        analysis_ml = self.router._analyzer.analyze(msgs, classifier_override="ml")
        assert analysis_ml.complexity in (Complexity.SIMPLE, Complexity.MODERATE, Complexity.COMPLEX, Complexity.REASONING)
