"""Tests for A/B testing."""

from bedrock_smart_router.ab_testing import ABTestConfig, ABTestManager, ABVariant


def _config(sticky: bool = True) -> ABTestConfig:
    return ABTestConfig(
        name="test-experiment",
        variants=[
            ABVariant(name="control", model="model-a", weight=0.5),
            ABVariant(name="treatment", model="model-b", weight=0.5),
        ],
        sticky=sticky,
        enabled=True,
    )


class TestABTestManager:
    def test_inactive_when_disabled(self):
        mgr = ABTestManager(ABTestConfig(enabled=False))
        assert not mgr.is_active
        assert mgr.assign() is None

    def test_inactive_with_one_variant(self):
        cfg = ABTestConfig(
            enabled=True,
            variants=[ABVariant(name="only", model="m", weight=1.0)],
        )
        mgr = ABTestManager(cfg)
        assert not mgr.is_active

    def test_assigns_variant(self):
        mgr = ABTestManager(_config(sticky=False))
        result = mgr.assign()
        assert result is not None
        assert result.variant_name in ("control", "treatment")
        assert result.test_name == "test-experiment"

    def test_sticky_same_user_same_variant(self):
        mgr = ABTestManager(_config(sticky=True))
        results = [mgr.assign(user_id="user-123") for _ in range(20)]
        # All should be the same variant for the same user
        names = {r.variant_name for r in results}
        assert len(names) == 1

    def test_sticky_different_users_can_differ(self):
        mgr = ABTestManager(_config(sticky=True))
        variants = set()
        for i in range(100):
            r = mgr.assign(user_id=f"user-{i}")
            variants.add(r.variant_name)
        # With 100 users and 50/50 split, both variants should appear
        assert len(variants) == 2

    def test_weighted_distribution(self):
        cfg = ABTestConfig(
            name="weighted",
            variants=[
                ABVariant(name="heavy", model="m-a", weight=0.9),
                ABVariant(name="light", model="m-b", weight=0.1),
            ],
            sticky=False,
            enabled=True,
        )
        mgr = ABTestManager(cfg)
        counts = {"heavy": 0, "light": 0}
        for _ in range(1000):
            r = mgr.assign()
            counts[r.variant_name] += 1
        # Heavy should get ~90% of traffic
        assert counts["heavy"] > 700

    def test_stats(self):
        mgr = ABTestManager(_config(sticky=False))
        for _ in range(10):
            mgr.assign()
        stats = mgr.stats
        assert stats["test_name"] == "test-experiment"
        assert stats["total_requests"] == 10
        assert "control" in stats["variant_counts"] or "treatment" in stats["variant_counts"]
