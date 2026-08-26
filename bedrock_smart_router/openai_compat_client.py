# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""HTTP client for Bedrock's OpenAI-compatible APIs (Chat Completions & Responses).

Talks to either Bedrock endpoint that exposes the OpenAI-compatible surface:

- ``bedrock-runtime`` — ``https://bedrock-runtime.{region}.amazonaws.com``
  (preferred; Chat Completions at ``/openai/v1/chat/completions``,
  Responses at ``/openai/v1/responses``)
- ``bedrock-mantle``  — ``https://bedrock-mantle.{region}.api.aws``
  (transitionary; Chat Completions at ``/v1/chat/completions``,
  Responses at ``/v1/responses`` or ``/openai/v1/responses`` per model)

The endpoint and path for a given model+API come from the model catalog's
``api_support`` map, so this client is endpoint-agnostic: the caller supplies
the ``endpoint`` and ``path`` and this client handles auth, retries, and
streaming identically for both hosts.

Supports two authentication methods:
- AWS SigV4 signing (default — uses boto3 credential chain)
- Bedrock API key (Bearer token — set via api_key parameter)

Usage::

    from bedrock_smart_router.openai_compat_client import OpenAICompatClient

    # SigV4 auth (default), runtime endpoint
    client = OpenAICompatClient(region="us-west-2", endpoint="bedrock-runtime")

    # Chat Completions on runtime
    response = client.chat_completions(
        model="openai.gpt-oss-120b-1:0",
        messages=[{"role": "user", "content": "Hello"}],
        path="/openai/v1/chat/completions",
    )

    # Responses on mantle (per-model path)
    response = client.responses(
        model="openai.gpt-oss-120b",
        input="Hello",
        endpoint="bedrock-mantle",
        path="/v1/responses",
    )
"""

from __future__ import annotations

import json
import logging
import random
import time
from typing import Any, Generator

import boto3
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest

try:
    import requests
except ImportError:
    requests = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# Throttle/retry status codes
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
_MAX_RETRIES = 3
_BACKOFF_BASE = 0.5

ENDPOINT_RUNTIME = "bedrock-runtime"
ENDPOINT_MANTLE = "bedrock-mantle"

# Default paths per endpoint (used when the caller does not pass an explicit path)
_DEFAULT_CHAT_PATH = {
    ENDPOINT_RUNTIME: "/openai/v1/chat/completions",
    ENDPOINT_MANTLE: "/v1/chat/completions",
}
_DEFAULT_RESPONSES_PATH = {
    ENDPOINT_RUNTIME: "/openai/v1/responses",
    ENDPOINT_MANTLE: "/v1/responses",
}


def _base_url_for(endpoint: str, region: str) -> str:
    """Return the base host URL for a given endpoint + region."""
    if endpoint == ENDPOINT_RUNTIME:
        return f"https://bedrock-runtime.{region}.amazonaws.com"
    if endpoint == ENDPOINT_MANTLE:
        return f"https://bedrock-mantle.{region}.api.aws"
    raise ValueError(f"Unknown endpoint: {endpoint!r}")


class OpenAICompatError(Exception):
    """Error from a Bedrock OpenAI-compatible endpoint."""

    def __init__(self, status_code: int, message: str, error_type: str = ""):
        self.status_code = status_code
        self.error_type = error_type
        super().__init__(f"OpenAI-compat {status_code}: {message}")


class OpenAICompatThrottleError(OpenAICompatError):
    """Request was throttled (429)."""
    pass


class OpenAICompatClient:
    """HTTP client for Bedrock's OpenAI-compatible APIs (runtime or mantle).

    Handles SigV4 signing or Bearer token auth, retries on throttle/5xx,
    and provides both synchronous and streaming interfaces. The target
    endpoint (bedrock-runtime vs bedrock-mantle) is chosen per instance
    (via ``endpoint``) and can be overridden per call.

    Parameters
    ----------
    region : str
        AWS region.
    endpoint : str
        Default endpoint: "bedrock-runtime" or "bedrock-mantle".
    api_key : str, optional
        Bedrock API key. If provided, uses Bearer token auth instead of SigV4.
    session : boto3.Session, optional
        Custom boto3 session for credentials. Defaults to a new session.
    timeout : float
        Request timeout in seconds.
    max_retries : int
        Maximum retries on throttle/5xx errors.
    """

    def __init__(
        self,
        region: str = "us-west-2",
        endpoint: str = ENDPOINT_MANTLE,
        api_key: str | None = None,
        session: Any = None,
        timeout: float = 60.0,
        max_retries: int = _MAX_RETRIES,
    ) -> None:
        if requests is None:
            raise ImportError(
                "The 'requests' package is required for OpenAICompatClient. "
                "Install it with: pip install requests"
            )

        self._region = region
        self._endpoint = endpoint
        self._api_key = api_key
        self._timeout = timeout
        self._max_retries = max_retries

        # SigV4 credentials (only needed if no api_key)
        if not api_key:
            self._session = session or boto3.Session(region_name=region)
        else:
            self._session = None

    @property
    def region(self) -> str:
        return self._region

    @property
    def endpoint(self) -> str:
        return self._endpoint

    # ── Chat Completions API ────────────────────────────────────────

    def chat_completions(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        path: str | None = None,
        endpoint: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        stop: list[str] | None = None,
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Send a Chat Completions request (synchronous).

        Returns the full response dict in OpenAI Chat Completions format.
        """
        ep = endpoint or self._endpoint
        req_path = path or _DEFAULT_CHAT_PATH[ep]
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if temperature is not None:
            payload["temperature"] = temperature
        if top_p is not None:
            payload["top_p"] = top_p
        if stop is not None:
            payload["stop"] = stop
        if tools is not None:
            payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        payload.update(kwargs)

        return self._request("POST", req_path, payload, endpoint=ep)

    def chat_completions_stream(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        path: str | None = None,
        endpoint: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        stop: list[str] | None = None,
        tools: list[dict] | None = None,
        **kwargs: Any,
    ) -> Generator[dict[str, Any], None, None]:
        """Send a streaming Chat Completions request.

        Yields SSE event dicts (OpenAI streaming chunk format).
        """
        ep = endpoint or self._endpoint
        req_path = path or _DEFAULT_CHAT_PATH[ep]
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if temperature is not None:
            payload["temperature"] = temperature
        if top_p is not None:
            payload["top_p"] = top_p
        if stop is not None:
            payload["stop"] = stop
        if tools is not None:
            payload["tools"] = tools
        payload.update(kwargs)

        yield from self._request_stream("POST", req_path, payload, endpoint=ep)

    # ── Responses API ───────────────────────────────────────────────

    def responses(
        self,
        model: str,
        input: str | list[dict[str, Any]],
        *,
        path: str | None = None,
        endpoint: str | None = None,
        store: bool = True,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
        tools: list[dict] | None = None,
        previous_response_id: str | None = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Send a Responses API request (synchronous).

        Args:
            model: Model ID on the target endpoint.
            input: Prompt text or structured input.
            path: URL path. Defaults per endpoint; some models override
                (e.g. mantle serves certain models at /openai/v1/responses).
            endpoint: Override the instance endpoint for this call.
            store: Whether to store for stateful continuation (default True).
            stream: Whether to stream the response.

        Returns the full response dict in OpenAI Responses format.
        """
        ep = endpoint or self._endpoint
        req_path = path or _DEFAULT_RESPONSES_PATH[ep]
        payload: dict[str, Any] = {
            "model": model,
            "input": input,
            "store": store,
        }
        if max_output_tokens is not None:
            payload["max_output_tokens"] = max_output_tokens
        if temperature is not None:
            payload["temperature"] = temperature
        if tools is not None:
            payload["tools"] = tools
        if previous_response_id is not None:
            payload["previous_response_id"] = previous_response_id
        if stream:
            payload["stream"] = True
        payload.update(kwargs)

        return self._request("POST", req_path, payload, endpoint=ep)

    def responses_stream(
        self,
        model: str,
        input: str | list[dict[str, Any]],
        *,
        path: str | None = None,
        endpoint: str | None = None,
        store: bool = True,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
        tools: list[dict] | None = None,
        previous_response_id: str | None = None,
        **kwargs: Any,
    ) -> Generator[dict[str, Any], None, None]:
        """Send a streaming Responses API request.

        Yields SSE event dicts (OpenAI Responses streaming format).
        """
        ep = endpoint or self._endpoint
        req_path = path or _DEFAULT_RESPONSES_PATH[ep]
        payload: dict[str, Any] = {
            "model": model,
            "input": input,
            "store": store,
            "stream": True,
        }
        if max_output_tokens is not None:
            payload["max_output_tokens"] = max_output_tokens
        if temperature is not None:
            payload["temperature"] = temperature
        if tools is not None:
            payload["tools"] = tools
        if previous_response_id is not None:
            payload["previous_response_id"] = previous_response_id
        payload.update(kwargs)

        yield from self._request_stream("POST", req_path, payload, endpoint=ep)

    # ── Internal HTTP methods ───────────────────────────────────────

    def _request(self, method: str, path: str, payload: dict,
                 *, endpoint: str | None = None) -> dict[str, Any]:
        """Make an authenticated request with retries."""
        ep = endpoint or self._endpoint
        url = f"{_base_url_for(ep, self._region)}{path}"
        body = json.dumps(payload)

        for attempt in range(self._max_retries + 1):
            headers = self._build_headers(method, url, body)

            try:
                resp = requests.request(
                    method, url, data=body, headers=headers, timeout=self._timeout,
                )
            except requests.exceptions.Timeout:
                if attempt < self._max_retries:
                    self._backoff(attempt)
                    continue
                raise OpenAICompatError(408, "Request timed out")
            except requests.exceptions.ConnectionError as e:
                if attempt < self._max_retries:
                    self._backoff(attempt)
                    continue
                raise OpenAICompatError(0, f"Connection failed: {e}")

            if resp.status_code == 200:
                return resp.json()

            # Parse error
            try:
                err_body = resp.json()
                err_msg = err_body.get("error", {}).get("message", resp.text[:200])
                err_type = err_body.get("error", {}).get("type", "")
            except (json.JSONDecodeError, ValueError):
                err_msg = resp.text[:200]
                err_type = ""

            # Retry on throttle or server errors
            if resp.status_code in _RETRYABLE_STATUS_CODES and attempt < self._max_retries:
                logger.warning(
                    "%s %s %s returned %d (attempt %d/%d): %s",
                    ep, method, path, resp.status_code, attempt + 1, self._max_retries + 1, err_msg[:100],
                )
                self._backoff(attempt)
                continue

            # Non-retryable error
            if resp.status_code == 429:
                raise OpenAICompatThrottleError(429, err_msg, err_type)
            raise OpenAICompatError(resp.status_code, err_msg, err_type)

        # Should not reach here, but just in case
        raise OpenAICompatError(500, "Max retries exceeded")

    def _request_stream(
        self, method: str, path: str, payload: dict,
        *, endpoint: str | None = None,
    ) -> Generator[dict[str, Any], None, None]:
        """Make a streaming authenticated request. Yields parsed SSE events."""
        ep = endpoint or self._endpoint
        url = f"{_base_url_for(ep, self._region)}{path}"
        body = json.dumps(payload)
        headers = self._build_headers(method, url, body)

        resp = requests.request(
            method, url, data=body, headers=headers,
            timeout=self._timeout, stream=True,
        )

        if resp.status_code != 200:
            try:
                err_body = resp.json()
                err_msg = err_body.get("error", {}).get("message", resp.text[:200])
            except (json.JSONDecodeError, ValueError):
                err_msg = resp.text[:200]
            raise OpenAICompatError(resp.status_code, err_msg)

        # Parse SSE stream
        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                continue
            if line.startswith("data: "):
                data_str = line[6:]
                if data_str.strip() == "[DONE]":
                    break
                try:
                    yield json.loads(data_str)
                except json.JSONDecodeError:
                    continue

    def _build_headers(self, method: str, url: str, body: str) -> dict[str, str]:
        """Build authenticated headers (SigV4 or Bearer token)."""
        if self._api_key:
            return {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            }

        # SigV4 signing — resolve credentials fresh each time to handle rotation.
        # Service name is "bedrock" for both runtime and mantle endpoints.
        credentials = self._session.get_credentials().get_frozen_credentials()
        request = AWSRequest(
            method=method, url=url, data=body,
            headers={"Content-Type": "application/json"},
        )
        SigV4Auth(credentials, "bedrock", self._region).add_auth(request)
        return dict(request.headers)

    def _backoff(self, attempt: int) -> None:
        """Exponential backoff with jitter."""
        delay = _BACKOFF_BASE * (2 ** attempt) + random.uniform(0, 0.5)
        time.sleep(delay)
