"""HTTP client for the bedrock-mantle endpoint (Chat Completions & Responses API).

Supports two authentication methods:
- AWS SigV4 signing (default — uses boto3 credential chain)
- Bedrock API key (Bearer token — set via api_key parameter)

Usage::

    from bedrock_smart_router.mantle_client import MantleClient

    # SigV4 auth (default)
    client = MantleClient(region="us-west-2")

    # API key auth
    client = MantleClient(region="us-west-2", api_key="brk_xxxx...")

    # Chat Completions
    response = client.chat_completions(
        model="openai.gpt-oss-120b",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=100,
    )

    # Streaming
    for event in client.chat_completions_stream(
        model="openai.gpt-oss-120b",
        messages=[{"role": "user", "content": "Tell me a story"}],
    ):
        print(event)
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

# Mantle-specific throttle/retry status codes
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
_MAX_RETRIES = 3
_BACKOFF_BASE = 0.5


class MantleError(Exception):
    """Error from the bedrock-mantle endpoint."""

    def __init__(self, status_code: int, message: str, error_type: str = ""):
        self.status_code = status_code
        self.error_type = error_type
        super().__init__(f"Mantle {status_code}: {message}")


class MantleThrottleError(MantleError):
    """Request was throttled (429)."""
    pass


class MantleClient:
    """HTTP client for the bedrock-mantle endpoint.

    Handles SigV4 signing or Bearer token auth, retries on throttle/5xx,
    and provides both synchronous and streaming interfaces.

    Parameters
    ----------
    region : str
        AWS region for the Mantle endpoint.
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
        api_key: str | None = None,
        session: Any = None,
        timeout: float = 60.0,
        max_retries: int = _MAX_RETRIES,
    ) -> None:
        if requests is None:
            raise ImportError(
                "The 'requests' package is required for MantleClient. "
                "Install it with: pip install requests"
            )

        self._region = region
        self._api_key = api_key
        self._timeout = timeout
        self._max_retries = max_retries
        self._base_url = f"https://bedrock-mantle.{region}.api.aws"

        # SigV4 credentials (only needed if no api_key)
        if not api_key:
            self._session = session or boto3.Session(region_name=region)
        else:
            self._session = None

    @property
    def region(self) -> str:
        return self._region

    # ── Chat Completions API ────────────────────────────────────────

    def chat_completions(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
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

        return self._request("POST", "/v1/chat/completions", payload)

    def chat_completions_stream(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
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

        yield from self._request_stream("POST", "/v1/chat/completions", payload)

    # ── Responses API (for future use with GPT-5.4/5.5) ────────────

    def responses(
        self,
        model: str,
        input: str | list[dict[str, Any]],
        *,
        path: str = "/v1/responses",
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
            model: Model ID on Mantle.
            input: Prompt text or structured input.
            path: URL path (some models use /openai/v1/responses).
            store: Whether to store for stateful continuation (default True).
            stream: Whether to stream the response.

        Returns the full response dict in OpenAI Responses format.
        """
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

        return self._request("POST", path, payload)

    def responses_stream(
        self,
        model: str,
        input: str | list[dict[str, Any]],
        *,
        path: str = "/v1/responses",
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

        yield from self._request_stream("POST", path, payload)

    # ── Internal HTTP methods ───────────────────────────────────────

    def _request(self, method: str, path: str, payload: dict) -> dict[str, Any]:
        """Make an authenticated request with retries."""
        url = f"{self._base_url}{path}"
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
                raise MantleError(408, "Request timed out")
            except requests.exceptions.ConnectionError as e:
                if attempt < self._max_retries:
                    self._backoff(attempt)
                    continue
                raise MantleError(0, f"Connection failed: {e}")

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
                    "Mantle %s %s returned %d (attempt %d/%d): %s",
                    method, path, resp.status_code, attempt + 1, self._max_retries + 1, err_msg[:100],
                )
                self._backoff(attempt)
                continue

            # Non-retryable error
            if resp.status_code == 429:
                raise MantleThrottleError(429, err_msg, err_type)
            raise MantleError(resp.status_code, err_msg, err_type)

        # Should not reach here, but just in case
        raise MantleError(500, "Max retries exceeded")

    def _request_stream(
        self, method: str, path: str, payload: dict,
    ) -> Generator[dict[str, Any], None, None]:
        """Make a streaming authenticated request. Yields parsed SSE events."""
        url = f"{self._base_url}{path}"
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
            raise MantleError(resp.status_code, err_msg)

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

        # SigV4 signing — resolve credentials fresh each time to handle rotation
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
