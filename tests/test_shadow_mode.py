"""Tests for shadow mode."""

import time

from bedrock_smart_router.shadow_mode import ShadowConfig, ShadowManager


class TestShadowManager:
    def test_inactive_when_disabled(self):
        mgr = ShadowManager(ShadowConfig(enabled=False))
        assert not mgr.is_active
        assert not mgr.should_shadow()

    def test_inactive_without_model(self):
        mgr = ShadowManager(ShadowConfig(enabled=True, shadow_model=""))
        assert not mgr.is_active

    def test_sample_rate_respected(self):
        mgr = ShadowManager(ShadowConfig(enabled=True, shadow_model="m", sample_rate=1.0))
        assert mgr.is_active
        # With 100% sample rate, should always shadow
        assert all(mgr.should_shadow() for _ in range(10))

    def test_zero_sample_rate(self):
        mgr = ShadowManager(ShadowConfig(enabled=True, shadow_model="m", sample_rate=0.0))
        assert not any(mgr.should_shadow() for _ in range(10))

    def test_mirror_success(self):
        calls = []

        def mock_invoke(**kwargs):
            calls.append(kwargs)
            return {"output": {"message": {"content": [{"text": "ok"}]}}}

        mgr = ShadowManager(
            ShadowConfig(enabled=True, shadow_model="shadow-m"),
            invoke_fn=mock_invoke,
        )
        mgr.mirror(
            primary_model="primary-m",
            messages=[{"role": "user", "content": [{"text": "Hi"}]}],
        )
        time.sleep(0.2)  # Wait for background thread
        assert len(calls) == 1
        assert calls[0]["modelId"] == "shadow-m"
        assert len(mgr.results) == 1
        assert mgr.results[0].success

    def test_mirror_failure_logged(self):
        def mock_invoke(**kwargs):
            raise RuntimeError("Shadow model down")

        mgr = ShadowManager(
            ShadowConfig(enabled=True, shadow_model="shadow-m"),
            invoke_fn=mock_invoke,
        )
        mgr.mirror(
            primary_model="primary-m",
            messages=[{"role": "user", "content": [{"text": "Hi"}]}],
        )
        time.sleep(0.2)
        assert len(mgr.results) == 1
        assert not mgr.results[0].success
        assert "Shadow model down" in mgr.results[0].error

    def test_stats(self):
        def mock_invoke(**kwargs):
            return {}

        mgr = ShadowManager(
            ShadowConfig(enabled=True, shadow_model="shadow-m"),
            invoke_fn=mock_invoke,
        )
        mgr.mirror(primary_model="p", messages=[{"role": "user", "content": [{"text": "a"}]}])
        mgr.mirror(primary_model="p", messages=[{"role": "user", "content": [{"text": "b"}]}])
        time.sleep(0.3)
        stats = mgr.stats
        assert stats["total"] == 2
        assert stats["success_rate"] == 1.0

    def test_no_invoke_fn_warns(self):
        mgr = ShadowManager(ShadowConfig(enabled=True, shadow_model="m"))
        # Should not crash, just warn
        mgr.mirror(primary_model="p", messages=[])
