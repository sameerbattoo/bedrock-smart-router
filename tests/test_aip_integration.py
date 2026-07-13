# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration test — create real Application Inference Profiles.

Run with:
    INTEGRATION_TEST=1 .venv/bin/python -m pytest tests/test_aip_integration.py -v -s

Creates real AIPs in your account, invokes Bedrock through them, and
cleans up by deleting the profiles after the test.

Requires:
    bedrock:CreateInferenceProfile
    bedrock:GetInferenceProfile
    bedrock:DeleteInferenceProfile
    bedrock:ListInferenceProfiles
    bedrock:TagResource
    bedrock:InvokeModel
"""

from __future__ import annotations

import os
import uuid

import boto3
import pytest

from bedrock_smart_router.aip_manager import AIPConfig, AIPManager

SKIP_REASON = "Set INTEGRATION_TEST=1 to run against real AWS"
REGION = "us-west-2"


@pytest.fixture
def aip_manager():
    """Create an AIPManager and track created profiles for cleanup."""
    session = boto3.Session(region_name=REGION)
    short_id = uuid.uuid4().hex[:6]
    mgr = AIPManager(
        config=AIPConfig(
            enabled=True,
            auto_create=True,
            tag_keys=["tenant", "team"],
            profile_name_prefix=f"bsr-test-{short_id}",
        ),
        boto_session=session,
        region=REGION,
    )
    created_arns: list[str] = []

    yield mgr, created_arns, session

    # Cleanup: delete all created profiles
    client = session.client("bedrock", region_name=REGION)
    for arn in created_arns:
        try:
            # Extract profile ID from ARN
            profile_id = arn.split("/")[-1] if "/" in arn else arn
            client.delete_inference_profile(inferenceProfileIdentifier=profile_id)
            print(f"\n  Deleted AIP: {profile_id}")
        except Exception as exc:
            print(f"\n  Warning: could not delete AIP {arn}: {exc}")


@pytest.mark.skipif(
    os.environ.get("INTEGRATION_TEST") != "1",
    reason=SKIP_REASON,
)
class TestAIPIntegration:

    def test_create_aip_for_nova_micro(self, aip_manager):
        """Create an AIP for Nova Micro with tenant tags."""
        mgr, created_arns, session = aip_manager

        arn = mgr.get_model_id_for_tenant(
            "amazon.nova-micro-v1:0",
            {"tenant": "acme-corp", "team": "engineering"},
        )

        assert arn.startswith("arn:aws:bedrock:")
        assert "inference-profile" in arn
        created_arns.append(arn)
        print(f"\n  Created AIP: {arn}")

        # Verify it exists via GetInferenceProfile
        client = session.client("bedrock", region_name=REGION)
        profile_id = arn.split("/")[-1]
        resp = client.get_inference_profile(inferenceProfileIdentifier=profile_id)
        assert resp["inferenceProfileName"].startswith("bsr-test-")
        print(f"  Profile name: {resp['inferenceProfileName']}")
        print(f"  Status: {resp.get('status', 'N/A')}")

    def test_create_aip_for_sonnet(self, aip_manager):
        """Create an AIP for Sonnet 4.6 with tenant tags."""
        mgr, created_arns, session = aip_manager

        arn = mgr.get_model_id_for_tenant(
            "anthropic.claude-sonnet-4-6",
            {"tenant": "globex", "team": "research"},
        )

        assert arn.startswith("arn:aws:bedrock:")
        created_arns.append(arn)
        print(f"\n  Created Sonnet AIP: {arn}")

    def test_cached_on_second_call(self, aip_manager):
        """Second call with same tenant+model should return cached AIP."""
        mgr, created_arns, _ = aip_manager
        tags = {"tenant": "acme", "team": "data"}

        arn1 = mgr.get_model_id_for_tenant("amazon.nova-micro-v1:0", tags)
        created_arns.append(arn1)

        arn2 = mgr.get_model_id_for_tenant("amazon.nova-micro-v1:0", tags)

        assert arn1 == arn2
        assert len(mgr.cached_profiles) == 1
        print(f"\n  Cached AIP: {arn1}")

    def test_invoke_through_aip(self, aip_manager):
        """Create an AIP and invoke Bedrock through it."""
        mgr, created_arns, session = aip_manager

        arn = mgr.get_model_id_for_tenant(
            "amazon.nova-micro-v1:0",
            {"tenant": "test-invoke", "team": "qa"},
        )
        created_arns.append(arn)

        # Invoke Bedrock using the AIP ARN
        client = session.client("bedrock-runtime", region_name=REGION)
        response = client.converse(
            modelId=arn,
            messages=[{"role": "user", "content": [{"text": "Say hi in one word."}]}],
        )

        text = response["output"]["message"]["content"][0]["text"]
        assert len(text) > 0
        print(f"\n  Invoked via AIP: {arn}")
        print(f"  Response: {text[:100]}")

    def test_different_tenants_different_arns(self, aip_manager):
        """Different tenants should get different AIPs."""
        mgr, created_arns, _ = aip_manager

        arn1 = mgr.get_model_id_for_tenant(
            "amazon.nova-micro-v1:0",
            {"tenant": "tenant-a"},
        )
        arn2 = mgr.get_model_id_for_tenant(
            "amazon.nova-micro-v1:0",
            {"tenant": "tenant-b"},
        )

        created_arns.extend([arn1, arn2])
        assert arn1 != arn2
        print(f"\n  Tenant A: {arn1}")
        print(f"  Tenant B: {arn2}")
