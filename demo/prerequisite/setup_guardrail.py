#!/usr/bin/env python3
"""Create a Bedrock Guardrail for the Smart Router demo.

Idempotent: skips creation if a guardrail named 'bsr-demo-guardrail' already exists.
Saves the guardrail config to .guardrail_config.json for use by the demo backend.
"""

import json
import sys
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

REGION = "us-west-2"
GUARDRAIL_NAME = "bsr-demo-guardrail"
CONFIG_PATH = Path(__file__).parent / ".guardrail_config.json"


def find_existing_guardrail(client) -> dict | None:
    """Check if a guardrail with our name already exists."""
    try:
        paginator = client.get_paginator("list_guardrails")
        for page in paginator.paginate():
            for guardrail in page.get("guardrails", []):
                if guardrail["name"] == GUARDRAIL_NAME:
                    return guardrail
    except ClientError as e:
        print(f"  Warning: Could not list guardrails: {e}")
        return None
    return None


def create_guardrail(client) -> dict:
    """Create the demo guardrail and return the response."""
    response = client.create_guardrail(
        name=GUARDRAIL_NAME,
        description="Demo guardrail for Smart Router - PII detection + content filtering",
        sensitiveInformationPolicyConfig={
            "piiEntitiesConfig": [
                {"type": "US_SOCIAL_SECURITY_NUMBER", "action": "ANONYMIZE"},
                {"type": "EMAIL", "action": "ANONYMIZE"},
                {"type": "PHONE", "action": "ANONYMIZE"},
                {"type": "CREDIT_DEBIT_CARD_NUMBER", "action": "ANONYMIZE"},
                {"type": "NAME", "action": "ANONYMIZE"},
                {"type": "ADDRESS", "action": "ANONYMIZE"},
            ]
        },
        contentPolicyConfig={
            "filtersConfig": [
                {"type": "HATE", "inputStrength": "HIGH", "outputStrength": "HIGH"},
                {"type": "INSULTS", "inputStrength": "HIGH", "outputStrength": "HIGH"},
                {"type": "SEXUAL", "inputStrength": "HIGH", "outputStrength": "HIGH"},
                {"type": "VIOLENCE", "inputStrength": "HIGH", "outputStrength": "HIGH"},
            ]
        },
        topicPolicyConfig={
            "topicsConfig": [
                {
                    "name": "investment_advice",
                    "definition": "Providing specific investment recommendations, stock tips, or financial planning advice",
                    "examples": [
                        "Should I buy Tesla stock?",
                        "What's a good investment for retirement?",
                    ],
                    "type": "DENY",
                },
                {
                    "name": "medical_diagnosis",
                    "definition": "Providing medical diagnoses or prescribing medications",
                    "examples": [
                        "What medication should I take for headaches?",
                        "Do I have diabetes?",
                    ],
                    "type": "DENY",
                },
            ]
        },
        blockedInputMessaging="I'm sorry, I can't process this request due to content safety policies.",
        blockedOutputsMessaging="I'm sorry, I can't provide this response due to content safety policies.",
    )
    return response


def save_config(guardrail_id: str, version: str) -> None:
    """Save guardrail config to JSON file."""
    config = {
        "guardrail_id": guardrail_id,
        "guardrail_version": version,
        "guardrail_name": GUARDRAIL_NAME,
    }
    CONFIG_PATH.write_text(json.dumps(config, indent=2) + "\n")
    print(f"  Config saved to {CONFIG_PATH}")


def main() -> bool:
    """Set up the Bedrock Guardrail. Returns True on success."""
    print("Setting up Bedrock Guardrail...")

    try:
        client = boto3.Session(region_name=REGION).client("bedrock")
    except Exception as e:
        print(f"  Error: Could not create Bedrock client: {e}")
        return False

    # Check if guardrail already exists
    existing = find_existing_guardrail(client)
    if existing:
        guardrail_id = existing["id"]
        version = existing.get("version", "1")
        print(f"  Guardrail already exists: {GUARDRAIL_NAME} (id={guardrail_id})")
        save_config(guardrail_id, version)
        return True

    # Create new guardrail
    try:
        print(f"  Creating guardrail: {GUARDRAIL_NAME}...")
        response = create_guardrail(client)
        guardrail_id = response["guardrailId"]
        print(f"  Guardrail created (id={guardrail_id})")

        # Create version
        print("  Creating guardrail version...")
        version_response = client.create_guardrail_version(
            guardrailIdentifier=guardrail_id,
            description="Initial version for Smart Router demo",
        )
        version = version_response["version"]
        print(f"  Version {version} created")

        save_config(guardrail_id, version)
        return True

    except ClientError as e:
        error_code = e.response["Error"]["Code"]
        if error_code == "AccessDeniedException":
            print(f"  Error: Missing permissions to create Bedrock Guardrails.")
            print(f"  Ensure your IAM role has bedrock:CreateGuardrail permission.")
        else:
            print(f"  Error creating guardrail: {e}")
        return False
    except Exception as e:
        print(f"  Unexpected error: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
