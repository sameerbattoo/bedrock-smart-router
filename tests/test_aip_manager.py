# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Application Inference Profile manager."""

from unittest.mock import MagicMock

import pytest

from bedrock_smart_router.aip_manager import AIPConfig, AIPManager


class TestAIPManager:
    def setup_method(self):
        self.mock_session = MagicMock()
        self.mock_client = MagicMock()
        # Default: list_inference_profiles returns empty (no existing profiles)
        self.mock_client.list_inference_profiles.return_value = {
            "inferenceProfileSummaries": [],
        }
        # Default: STS mock for account ID
        mock_sts = MagicMock()
        mock_sts.get_caller_identity.return_value = {"Account": "123456789012"}
        self.mock_session.client.side_effect = lambda svc, **kw: (
            mock_sts if svc == "sts" else self.mock_client
        )

    def test_disabled_returns_raw_model_id(self):
        mgr = AIPManager(
            config=AIPConfig(enabled=False),
            boto_session=self.mock_session,
        )
        result = mgr.get_model_id_for_tenant(
            "us.amazon.nova-micro-v1:0",
            {"tenant": "acme"},
        )
        assert result == "us.amazon.nova-micro-v1:0"

    def test_no_tags_returns_raw_model_id(self):
        mgr = AIPManager(
            config=AIPConfig(enabled=True),
            boto_session=self.mock_session,
        )
        result = mgr.get_model_id_for_tenant(
            "us.amazon.nova-micro-v1:0",
            {},
        )
        assert result == "us.amazon.nova-micro-v1:0"

    def test_non_cris_model_returns_raw(self):
        """Models without CRIS prefix (us., global., etc.) can't have AIPs."""
        mgr = AIPManager(
            config=AIPConfig(enabled=True, auto_create=True),
            boto_session=self.mock_session,
        )
        result = mgr.get_model_id_for_tenant(
            "amazon.nova-micro-v1:0",
            {"tenant": "acme"},
        )
        assert result == "amazon.nova-micro-v1:0"

    def test_creates_profile_on_first_call(self):
        self.mock_client.create_inference_profile.return_value = {
            "inferenceProfileArn": "arn:aws:bedrock:us-west-2:123:inference-profile/bsr-acme-nova"
        }

        mgr = AIPManager(
            config=AIPConfig(enabled=True, auto_create=True),
            boto_session=self.mock_session,
            region="us-west-2",
        )
        result = mgr.get_model_id_for_tenant(
            "us.amazon.nova-micro-v1:0",
            {"tenant": "acme"},
        )
        assert result == "arn:aws:bedrock:us-west-2:123:inference-profile/bsr-acme-nova"

        # Verify the call args — should use full ARN
        call_kwargs = self.mock_client.create_inference_profile.call_args[1]
        expected_arn = "arn:aws:bedrock:us-west-2:123456789012:inference-profile/us.amazon.nova-micro-v1:0"
        assert call_kwargs["modelSource"] == {"copyFrom": expected_arn}
        assert any(t["key"] == "tenant" and t["value"] == "acme" for t in call_kwargs["tags"])

    def test_caches_profile_on_second_call(self):
        self.mock_client.create_inference_profile.return_value = {
            "inferenceProfileArn": "arn:aws:bedrock:us-west-2:123:inference-profile/bsr-acme"
        }
        mgr = AIPManager(
            config=AIPConfig(enabled=True, auto_create=True),
            boto_session=self.mock_session,
        )
        tags = {"tenant": "acme"}
        r1 = mgr.get_model_id_for_tenant("us.amazon.nova-micro-v1:0", tags)
        r2 = mgr.get_model_id_for_tenant("us.amazon.nova-micro-v1:0", tags)

        assert r1 == r2
        # Only one API call — second was served from cache
        assert self.mock_client.create_inference_profile.call_count == 1

    def test_different_tenants_get_different_profiles(self):
        call_count = 0

        def mock_create(**kwargs):
            nonlocal call_count
            call_count += 1
            return {"inferenceProfileArn": f"arn:profile-{call_count}"}

        self.mock_client.create_inference_profile.side_effect = mock_create
        mgr = AIPManager(
            config=AIPConfig(enabled=True, auto_create=True),
            boto_session=self.mock_session,
        )
        r1 = mgr.get_model_id_for_tenant("us.amazon.nova-micro-v1:0", {"tenant": "acme"})
        r2 = mgr.get_model_id_for_tenant("us.amazon.nova-micro-v1:0", {"tenant": "globex"})

        assert r1 != r2
        assert call_count == 2

    def test_different_models_same_tenant_get_different_profiles(self):
        call_count = 0

        def mock_create(**kwargs):
            nonlocal call_count
            call_count += 1
            return {"inferenceProfileArn": f"arn:profile-{call_count}"}

        self.mock_client.create_inference_profile.side_effect = mock_create
        mgr = AIPManager(
            config=AIPConfig(enabled=True, auto_create=True),
            boto_session=self.mock_session,
        )
        r1 = mgr.get_model_id_for_tenant("us.amazon.nova-micro-v1:0", {"tenant": "acme"})
        r2 = mgr.get_model_id_for_tenant("us.amazon.nova-pro-v1:0", {"tenant": "acme"})

        assert r1 != r2
        assert call_count == 2

    def test_auto_create_false_returns_raw(self):
        mgr = AIPManager(
            config=AIPConfig(enabled=True, auto_create=False),
            boto_session=self.mock_session,
        )
        result = mgr.get_model_id_for_tenant(
            "us.amazon.nova-micro-v1:0",
            {"tenant": "acme"},
        )
        assert result == "us.amazon.nova-micro-v1:0"
        self.mock_client.create_inference_profile.assert_not_called()

    def test_api_failure_falls_back_to_raw(self):
        self.mock_client.create_inference_profile.side_effect = Exception("Access denied")
        mgr = AIPManager(
            config=AIPConfig(enabled=True, auto_create=True),
            boto_session=self.mock_session,
        )
        result = mgr.get_model_id_for_tenant(
            "us.amazon.nova-micro-v1:0",
            {"tenant": "acme"},
        )
        # Should not crash — falls back to raw model ID
        assert result == "us.amazon.nova-micro-v1:0"

    def test_invalidate_cache(self):
        self.mock_client.create_inference_profile.return_value = {
            "inferenceProfileArn": "arn:profile-1"
        }
        mgr = AIPManager(
            config=AIPConfig(enabled=True, auto_create=True),
            boto_session=self.mock_session,
        )
        mgr.get_model_id_for_tenant("us.amazon.nova-micro-v1:0", {"tenant": "acme"})
        assert len(mgr.cached_profiles) == 1
