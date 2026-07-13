# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bedrock Guardrails integration.

Provides pre-route and post-route guardrail checks using the
``bedrock-runtime:ApplyGuardrail`` API.

Pre-route:  Screen the user input BEFORE model selection.  If blocked,
            reject or sanitize the request.
Post-route: Screen the model output AFTER invocation.  If blocked,
            retry with a different model or return sanitized output.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


class GuardrailBlockedError(Exception):
    """Raised when a guardrail blocks a request and action is 'reject'.

    Attributes:
        action: The action taken ("reject").
        assessments: Full guardrail assessment details (topics, PII, filters detected).
        output_text: The blocked message text from the guardrail.
        latency_ms: Time taken for the guardrail check.
        guardrail_id: The guardrail identifier that blocked.
        guardrail_version: The guardrail version that blocked.
    """

    def __init__(
        self,
        message: str,
        action: str,
        assessments: list[Any] | None = None,
        output_text: str | None = None,
        latency_ms: float | None = None,
        guardrail_id: str | None = None,
        guardrail_version: str | None = None,
    ):
        super().__init__(message)
        self.action = action
        self.assessments = assessments or []
        self.output_text = output_text
        self.latency_ms = latency_ms
        self.guardrail_id = guardrail_id
        self.guardrail_version = guardrail_version


@dataclass
class GuardrailCheckConfig:
    """Configuration for a single guardrail check (pre or post)."""

    guardrail_id: str
    guardrail_version: str = "DRAFT"
    action_on_block: str = "reject"  # "reject" | "sanitize"


@dataclass
class GuardrailsConfig:
    """Top-level guardrails configuration."""

    pre_route: GuardrailCheckConfig | None = None
    post_route: GuardrailCheckConfig | None = None


@dataclass
class GuardrailResult:
    """Outcome of a guardrail check."""

    action: str  # "NONE" | "GUARDRAIL_INTERVENED"
    blocked: bool
    output_text: str | None = None  # Sanitized text if action_on_block="sanitize"
    assessments: list[Any] | None = None
    latency_ms: float | None = None  # Time taken for the guardrail API call


class GuardrailsManager:
    """Manages pre-route and post-route guardrail checks."""

    def __init__(
        self,
        config: GuardrailsConfig | None = None,
        boto_session: Any | None = None,
        region: str = "us-west-2",
    ) -> None:
        self.config = config or GuardrailsConfig()
        self._region = region
        self._session = boto_session
        self._client: Any | None = None

    def _get_client(self) -> Any:
        if self._client is None:
            if self._session is None:
                import boto3
                self._session = boto3.Session(region_name=self._region)
            self._client = self._session.client(
                "bedrock-runtime", region_name=self._region
            )
        return self._client

    @property
    def has_pre_route(self) -> bool:
        return self.config.pre_route is not None

    @property
    def has_post_route(self) -> bool:
        return self.config.post_route is not None

    def check_input(
        self, messages: list[dict[str, Any]]
    ) -> GuardrailResult:
        """Run the pre-route guardrail on user input.

        Extracts text from messages and calls ApplyGuardrail with
        source="INPUT".
        """
        cfg = self.config.pre_route
        if cfg is None:
            return GuardrailResult(action="NONE", blocked=False)

        text_parts = self._extract_text(messages)
        if not text_parts:
            return GuardrailResult(action="NONE", blocked=False)

        return self._apply(cfg, text_parts, source="INPUT")

    def check_output(self, output_text: str) -> GuardrailResult:
        """Run the post-route guardrail on model output.

        Calls ApplyGuardrail with source="OUTPUT".
        """
        cfg = self.config.post_route
        if cfg is None:
            return GuardrailResult(action="NONE", blocked=False)

        if not output_text.strip():
            return GuardrailResult(action="NONE", blocked=False)

        return self._apply(cfg, [output_text], source="OUTPUT")

    def _apply(
        self,
        cfg: GuardrailCheckConfig,
        texts: list[str],
        source: str,
    ) -> GuardrailResult:
        """Call the ApplyGuardrail API."""
        import time as _time
        client = self._get_client()

        content = [{"text": {"text": t}} for t in texts]

        t0 = _time.perf_counter()
        try:
            resp = client.apply_guardrail(
                guardrailIdentifier=cfg.guardrail_id,
                guardrailVersion=cfg.guardrail_version,
                source=source,
                content=content,
            )
        except Exception as exc:
            logger.error("ApplyGuardrail failed: %s", exc)
            # Fail open — don't block the request if guardrails are down
            return GuardrailResult(action="NONE", blocked=False)
        latency_ms = (_time.perf_counter() - t0) * 1000

        action = resp.get("action", "NONE")
        blocked = action == "GUARDRAIL_INTERVENED"
        assessments = resp.get("assessments", [])

        # Extract sanitized/blocked output text
        output_text = None
        outputs = resp.get("outputs", [])
        if outputs:
            output_text = outputs[0].get("text")

        result = GuardrailResult(
            action=action,
            blocked=blocked,
            output_text=output_text,
            assessments=assessments,
            latency_ms=round(latency_ms, 1),
        )

        if blocked:
            logger.warning(
                "Guardrail %s blocked %s content (action_on_block=%s, latency=%.1fms)",
                cfg.guardrail_id,
                source,
                cfg.action_on_block,
                latency_ms,
            )
            if cfg.action_on_block == "reject":
                raise GuardrailBlockedError(
                    f"Guardrail {cfg.guardrail_id} blocked the {source.lower()} content",
                    action=cfg.action_on_block,
                    assessments=assessments,
                    output_text=output_text,
                    latency_ms=round(latency_ms, 1),
                    guardrail_id=cfg.guardrail_id,
                    guardrail_version=cfg.guardrail_version,
                )

        return result

    @staticmethod
    def _extract_text(messages: list[dict[str, Any]]) -> list[str]:
        """Extract text content from Bedrock Converse-format messages."""
        texts: list[str] = []
        for msg in messages:
            content = msg.get("content", [])
            if isinstance(content, str):
                texts.append(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and "text" in block:
                        texts.append(block["text"])
        return [t for t in texts if t.strip()]
