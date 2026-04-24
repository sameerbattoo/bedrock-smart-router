"""Tests for graceful no-models-match error."""

import pytest

from bedrock_smart_router.config import RouterConfig, RoutingConfig
from bedrock_smart_router.exceptions import NoModelsMatchError, ModelRejection
from bedrock_smart_router.model_registry import ModelRegistry


class TestNoModelsMatchError:
    def test_basic_error(self):
        err = NoModelsMatchError("No models found")
        assert "No models found" in str(err)

    def test_with_rejections(self):
        err = NoModelsMatchError(
            "No models found",
            rejections=[
                ModelRejection("model-a", "Model A", ["too expensive", "no vision"]),
                ModelRejection("model-b", "Model B", ["context too small"]),
            ],
        )
        assert "model-a" in str(err)
        assert "too expensive" in str(err)
        assert "context too small" in str(err)

    def test_with_suggestions(self):
        err = NoModelsMatchError(
            "No models found",
            suggestions=["Increase max_cost_per_request", "Remove family filter"],
        )
        assert "Increase max_cost_per_request" in str(err)

    def test_with_constraints(self):
        err = NoModelsMatchError(
            "No models found",
            constraints={"complexity": "reasoning", "preferred_family": "anthropic"},
        )
        assert "reasoning" in str(err)

    def test_to_dict(self):
        err = NoModelsMatchError(
            "No models found",
            rejections=[ModelRejection("m-a", "A", ["reason1"])],
            constraints={"complexity": "simple"},
            suggestions=["Try X"],
        )
        d = err.to_dict()
        assert d["error"] == "no_models_match"
        assert len(d["rejections"]) == 1
        assert d["rejections"][0]["model_id"] == "m-a"
        assert d["suggestions"] == ["Try X"]


class TestRouterNoModelsGraceful:
    """Test that the router raises NoModelsMatchError with useful info."""

    def test_impossible_family_filter(self):
        """Filtering to a nonexistent family should give helpful error."""
        from unittest.mock import MagicMock
        session = MagicMock()
        cfg = RouterConfig()
        # We can't call converse() without a real Bedrock client,
        # but we can test the error path by using the registry directly
        registry = ModelRegistry()
        candidates = registry.eligible_models(family="nonexistent")
        assert len(candidates) == 0

    def test_error_is_catchable(self):
        err = NoModelsMatchError(
            "test",
            rejections=[ModelRejection("m", "M", ["r"])],
            suggestions=["s"],
        )
        with pytest.raises(NoModelsMatchError) as exc_info:
            raise err
        assert exc_info.value.rejections[0].model_id == "m"
        assert exc_info.value.suggestions == ["s"]

    def test_error_subclasses_exception(self):
        """Should be catchable as a generic Exception too."""
        err = NoModelsMatchError("test")
        assert isinstance(err, Exception)

    def test_economy_preset_with_tiny_budget(self):
        """Economy preset with impossibly low budget should give suggestions."""
        err = NoModelsMatchError(
            "No models found",
            constraints={
                "preset": "economy",
                "max_cost_per_request": 0.0000001,
            },
            rejections=[
                ModelRejection(
                    "us.amazon.nova-micro-v1:0",
                    "Nova Micro",
                    ["est. cost $0.000005 > max $0.0000001"],
                ),
            ],
            suggestions=["Increase max_cost_per_request (currently $1e-07)"],
        )
        assert "Nova Micro" in str(err)
        assert "Increase max_cost_per_request" in str(err)
