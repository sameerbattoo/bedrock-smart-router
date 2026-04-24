"""Strands Agents SDK integration — use BedrockRouter as a Strands Model provider.

Implements the ``strands.models.Model`` interface so that a Strands
``Agent`` can transparently benefit from smart routing, fallbacks,
circuit breakers, cost/latency/quality strategies, CRIS, inference
tiers, caching, guardrails, and all other router features.

Usage::

    from strands import Agent
    from bedrock_smart_router.strands_model import SmartRouterModel

    model = SmartRouterModel(router_config={"region": "us-west-2"})
    agent = Agent(model=model)
    response = agent("Explain quantum computing")

    # Inspect routing decision
    print(model.last_routing_decision)

Requires the ``strands-agents`` package::

    pip install strands-agents
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, AsyncGenerator, Optional, TypeVar, TypedDict, cast

from pydantic import BaseModel
from typing_extensions import Unpack, override

from bedrock_smart_router.config import RouterConfig, RoutingConfig
from bedrock_smart_router.models import RoutingDecision
from bedrock_smart_router.router import BedrockRouter

logger = logging.getLogger(__name__)

try:
    from strands.models import Model
    from strands.event_loop import streaming
    from strands.tools import convert_pydantic_to_tool_spec
    from strands.types.content import ContentBlock, Messages, SystemContentBlock
    from strands.types.streaming import StreamEvent
    from strands.types.tools import ToolChoice, ToolSpec
except ImportError as _exc:
    raise ImportError(
        "strands-agents is required for SmartRouterModel. "
        "Install it with: pip install strands-agents"
    ) from _exc

# Strands raises these for specific error conditions the agent loop handles.
try:
    from strands.types.exceptions import (
        ContextWindowOverflowException,
        ModelThrottledException,
    )
except ImportError:
    # Older strands versions may not have these; fall back to plain exceptions.
    class ContextWindowOverflowException(Exception):  # type: ignore[no-redef]
        pass

    class ModelThrottledException(Exception):  # type: ignore[no-redef]
        pass

# Kwargs from Strands that should NOT be forwarded to the Bedrock API.
_STRANDS_ONLY_KWARGS = frozenset({
    "invocation_state",
    "model_state",
    "event_loop_metrics",
    "request_state",
    "callback_handler",
})

T = TypeVar("T", bound=BaseModel)


# Bedrock error messages that indicate context window overflow.
_CONTEXT_OVERFLOW_MESSAGES = (
    "too many tokens",
    "too long",
    "Input is too long",
    "expected maxLength",
    "Too many input tokens",
    "prompt is too long",
    "context window",
    "max_tokens",
)


class SmartRouterModel(Model):
    """Strands Model provider backed by BedrockRouter.

    Wraps :class:`~bedrock_smart_router.router.BedrockRouter` so that
    every call from a Strands ``Agent`` is intelligently routed across
    Bedrock models based on cost, latency, quality, and complexity.
    """

    class SmartRouterModelConfig(TypedDict, total=False):
        """Configuration for SmartRouterModel.

        Attributes:
            streaming: Enable streaming (default True).
            routing_preset: Named routing preset — ``"economy"``,
                ``"speed"``, ``"balanced"``, or ``"quality"``.
            routing_strategy: Explicit strategy name (overrides preset).
            preferred_model: Pin a specific Bedrock model ID.
            preferred_family: Prefer a model family (e.g. ``"anthropic"``).
            max_cost_per_request: Cost ceiling in dollars.
            exclude_models: Glob patterns of models to exclude.
            tags: Tags forwarded to the routing decision.
            metadata: Arbitrary metadata forwarded to the router.
        """

        streaming: bool
        routing_preset: Optional[str]
        routing_strategy: Optional[str]
        preferred_model: Optional[str]
        preferred_family: Optional[str]
        max_cost_per_request: Optional[float]
        exclude_models: Optional[list[str]]
        tags: Optional[list[str]]
        metadata: Optional[dict[str, Any]]

    def __init__(
        self,
        router_config: dict[str, Any] | RouterConfig | None = None,
        *,
        router: BedrockRouter | None = None,
        boto_session: Any | None = None,
        **model_config: Unpack[SmartRouterModelConfig],
    ) -> None:
        """Initialise the Strands model provider.

        Args:
            router_config: Configuration dict or ``RouterConfig`` for
                creating a new ``BedrockRouter``.  Ignored when
                *router* is provided.
            router: Pre-built ``BedrockRouter`` instance.  When given,
                *router_config* is ignored.
            boto_session: Optional boto3 session forwarded to the router.
            **model_config: Per-model configuration (see
                ``SmartRouterModelConfig``).
        """
        self.config: dict[str, Any] = dict(model_config)  # type: ignore[arg-type]
        self.config.setdefault("streaming", True)

        if router is not None:
            self._router = router
        else:
            self._router = BedrockRouter.create(
                router_config, boto_session=boto_session,
            )

        self._last_decision: RoutingDecision | None = None
        logger.debug("config=<%s> | SmartRouterModel initialised", self.config)

    # ── Public helpers ──────────────────────────────────────────

    @property
    def last_routing_decision(self) -> RoutingDecision | None:
        """The routing decision from the most recent call."""
        return self._last_decision

    @property
    def router(self) -> BedrockRouter:
        """The underlying ``BedrockRouter`` instance."""
        return self._router

    # ── Model interface ─────────────────────────────────────────

    @override
    def update_config(self, **model_config: Unpack[SmartRouterModelConfig]) -> None:  # type: ignore[override]
        """Update configuration at runtime.

        Strands tools can call this to change routing behaviour
        mid-conversation (e.g. switch to ``economy`` preset).
        """
        self.config.update(model_config)  # type: ignore[arg-type]

    @override
    def get_config(self) -> SmartRouterModelConfig:
        """Return the current configuration."""
        return self.config  # type: ignore[return-value]

    @override
    async def stream(
        self,
        messages: Messages,
        tool_specs: Optional[list[ToolSpec]] = None,
        system_prompt: Optional[str] = None,
        *,
        tool_choice: Optional[ToolChoice] = None,
        system_prompt_content: Optional[list[SystemContentBlock]] = None,
        invocation_state: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Stream a response through the smart router.

        This is the core method Strands calls on every agent loop
        iteration.  It converts Strands types to Bedrock Converse
        format, delegates to the router, and yields Bedrock stream
        events back — which are already valid Strands ``StreamEvent``s.

        Args:
            messages: Conversation history in Strands format.
            tool_specs: Available tools the model may call.
            system_prompt: System prompt string.
            tool_choice: Tool selection strategy (e.g. ``{"auto": {}}``).
            system_prompt_content: System prompt content blocks.
            invocation_state: Agent invocation state (unused by router).
            **kwargs: Additional keyword arguments.

        Yields:
            ``StreamEvent`` dicts consumed by the Strands agent loop.

        Raises:
            ContextWindowOverflowException: Input exceeds model context.
            ModelThrottledException: Bedrock is throttling requests.
        """
        loop = asyncio.get_event_loop()
        queue: asyncio.Queue[StreamEvent | None] = asyncio.Queue()

        def callback(event: StreamEvent | None = None) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, event)

        thread = asyncio.to_thread(
            self._stream_sync, callback, messages, tool_specs,
            system_prompt, system_prompt_content, tool_choice, **kwargs,
        )
        task = asyncio.create_task(thread)

        while True:
            event = await queue.get()
            if event is None:
                break
            yield event

        # Re-raise any exception from the background thread.
        await task

    @override
    async def structured_output(
        self,
        output_model: type[T],
        prompt: Messages,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[dict[str, T | Any], None]:
        """Get structured output from the model.

        Uses tool calling to extract a Pydantic model from the LLM
        response, following the same pattern as Strands' BedrockModel.

        Args:
            output_model: Pydantic model class for the expected output.
            prompt: Conversation messages.
            system_prompt: Optional system prompt.
            **kwargs: Additional keyword arguments.

        Yields:
            Stream events, with the final event containing the parsed output.
        """
        tool_spec = convert_pydantic_to_tool_spec(output_model)
        response = self.stream(
            messages=prompt,
            tool_specs=[tool_spec],
            system_prompt=system_prompt,
            tool_choice=cast(ToolChoice, {"any": {}}),
            **kwargs,
        )

        event: dict[str, Any] = {}
        async for event in streaming.process_stream(response):
            yield event

        stop_reason, messages, _, _ = event["stop"]
        if stop_reason != "tool_use":
            raise ValueError(
                f'Model returned stop_reason: {stop_reason} instead of "tool_use".'
            )

        content = messages["content"]
        for block in content:
            if block.get("toolUse") and block["toolUse"]["name"] == tool_spec["name"]:
                yield {"output": output_model(**block["toolUse"]["input"])}
                return

        raise ValueError(
            "No valid tool use or tool use input was found in the response."
        )

    # ── Synchronous core (runs in a thread) ─────────────────────

    def _stream_sync(
        self,
        callback: Any,
        messages: Messages,
        tool_specs: Optional[list[ToolSpec]],
        system_prompt: Optional[str],
        system_prompt_content: Optional[list[SystemContentBlock]] = None,
        tool_choice: Optional[ToolChoice] = None,
        **kwargs: Any,
    ) -> None:
        """Run the router call synchronously and push events via *callback*.

        Mirrors the pattern used by Strands' own ``BedrockModel``.
        """
        try:
            bedrock_messages = list(messages)

            # Build system blocks: prefer system_prompt_content, fall back to system_prompt string.
            if system_prompt_content:
                system = list(system_prompt_content)
            elif system_prompt:
                system = [{"text": system_prompt}]
            else:
                system = None

            tool_config = self._build_tool_config(tool_specs, tool_choice)
            inference_config = self._build_inference_config()
            routing = self._build_routing_config()

            streaming = self.config.get("streaming", True)

            if streaming:
                self._handle_streaming(
                    callback, bedrock_messages, system, tool_config,
                    inference_config, routing, kwargs,
                )
            else:
                self._handle_non_streaming(
                    callback, bedrock_messages, system, tool_config,
                    inference_config, routing, kwargs,
                )

        except Exception as exc:
            self._maybe_raise_strands_exception(exc)
            raise
        finally:
            callback(None)  # Signal end-of-stream.

    def _handle_streaming(
        self,
        callback: Any,
        messages: list[dict[str, Any]],
        system: list[dict[str, Any]] | None,
        tool_config: dict[str, Any] | None,
        inference_config: dict[str, Any] | None,
        routing: RoutingConfig,
        extra_kwargs: dict[str, Any],
    ) -> None:
        """Invoke ``router.converse_stream()`` and forward events."""
        # Filter out Strands-internal kwargs that Bedrock doesn't accept.
        clean_kwargs = {
            k: v for k, v in extra_kwargs.items()
            if k not in _STRANDS_ONLY_KWARGS
        }

        for event in self._router.converse_stream(
            messages=messages,
            system=system,
            tool_config=tool_config,
            inference_config=inference_config,
            routing=routing,
            **clean_kwargs,
        ):
            # The router appends a final {"routing_decision": ...} event.
            # Capture it but don't forward — Strands doesn't expect it.
            if "routing_decision" in event:
                self._last_decision = event["routing_decision"]
                continue
            callback(event)

    def _handle_non_streaming(
        self,
        callback: Any,
        messages: list[dict[str, Any]],
        system: list[dict[str, Any]] | None,
        tool_config: dict[str, Any] | None,
        inference_config: dict[str, Any] | None,
        routing: RoutingConfig,
        extra_kwargs: dict[str, Any],
    ) -> None:
        """Invoke ``router.converse()`` and convert to streaming events."""
        # Filter out Strands-internal kwargs that Bedrock doesn't accept.
        clean_kwargs = {
            k: v for k, v in extra_kwargs.items()
            if k not in _STRANDS_ONLY_KWARGS
        }

        response = self._router.converse(
            messages=messages,
            system=system,
            tool_config=tool_config,
            inference_config=inference_config,
            routing=routing,
            **clean_kwargs,
        )

        # Capture routing decision.
        self._last_decision = response.get("routing_decision")

        # Convert the single response to streaming events.
        for event in self._convert_response_to_stream(response):
            callback(event)

    # ── Format helpers ──────────────────────────────────────────

    @staticmethod
    def _build_tool_config(
        tool_specs: Optional[list[ToolSpec]],
        tool_choice: Optional[ToolChoice] = None,
    ) -> dict[str, Any] | None:
        """Convert Strands tool specs to Bedrock ``toolConfig``."""
        if not tool_specs:
            return None
        config: dict[str, Any] = {
            "tools": [{"toolSpec": spec} for spec in tool_specs],
        }
        if tool_choice is not None:
            config["toolChoice"] = tool_choice
        return config

    def _build_inference_config(self) -> dict[str, Any] | None:
        """Build ``inferenceConfig`` from model config."""
        cfg: dict[str, Any] = {}
        if "max_tokens" in self.config:
            cfg["maxTokens"] = self.config["max_tokens"]
        if "temperature" in self.config:
            cfg["temperature"] = self.config["temperature"]
        if "top_p" in self.config:
            cfg["topP"] = self.config["top_p"]
        if "stop_sequences" in self.config:
            cfg["stopSequences"] = self.config["stop_sequences"]
        return cfg or None

    def _build_routing_config(self) -> RoutingConfig:
        """Build a ``RoutingConfig`` from the current model config."""
        return RoutingConfig(
            preset=self.config.get("routing_preset"),
            strategy=self.config.get("routing_strategy"),
            preferred_model=self.config.get("preferred_model"),
            preferred_family=self.config.get("preferred_family"),
            max_cost_per_request=self.config.get("max_cost_per_request"),
            exclude_models=self.config.get("exclude_models"),
            tags=self.config.get("tags"),
            metadata=self.config.get("metadata"),
        )

    @staticmethod
    def _convert_response_to_stream(response: dict[str, Any]) -> list[StreamEvent]:
        """Convert a non-streaming Converse response to StreamEvent sequence.

        Follows the same conversion logic as Strands' own
        ``BedrockModel._convert_non_streaming_to_streaming``.
        """
        events: list[StreamEvent] = []
        output = response.get("output", {})
        message = output.get("message", {})

        # messageStart
        events.append({"messageStart": {"role": message.get("role", "assistant")}})

        # Content blocks
        for content in message.get("content", []):
            if "toolUse" in content:
                tool_use = content["toolUse"]
                events.append({
                    "contentBlockStart": {
                        "start": {
                            "toolUse": {
                                "toolUseId": tool_use["toolUseId"],
                                "name": tool_use["name"],
                            }
                        }
                    }
                })
                events.append({
                    "contentBlockDelta": {
                        "delta": {
                            "toolUse": {
                                "input": json.dumps(tool_use["input"]),
                            }
                        }
                    }
                })
                events.append({"contentBlockStop": {}})
            elif "text" in content:
                events.append({
                    "contentBlockDelta": {
                        "delta": {"text": content["text"]},
                    }
                })
                events.append({"contentBlockStop": {}})
            elif "reasoningContent" in content:
                reasoning = content["reasoningContent"]
                delta: dict[str, Any] = {}
                if "reasoningText" in reasoning:
                    rt = reasoning["reasoningText"]
                    delta["text"] = rt.get("text", "")
                    if rt.get("signature"):
                        delta["signature"] = rt["signature"]
                if "redactedContent" in reasoning:
                    delta["redactedContent"] = reasoning["redactedContent"]
                events.append({
                    "contentBlockDelta": {
                        "delta": {"reasoningContent": delta},
                    }
                })
                events.append({"contentBlockStop": {}})

        # messageStop
        events.append({
            "messageStop": {
                "stopReason": response.get("stopReason", "end_turn"),
            }
        })

        # metadata (usage + metrics)
        metadata_event: dict[str, Any] = {}
        if "usage" in response:
            metadata_event["usage"] = response["usage"]
        if "metrics" in response:
            metadata_event["metrics"] = response["metrics"]
        if metadata_event:
            events.append({"metadata": metadata_event})

        return events

    # ── Error mapping ───────────────────────────────────────────

    @staticmethod
    def _maybe_raise_strands_exception(exc: Exception) -> None:
        """Re-raise as a Strands-specific exception when appropriate."""
        from botocore.exceptions import ClientError

        if not isinstance(exc, ClientError):
            return

        error_code = exc.response.get("Error", {}).get("Code", "")
        error_msg = str(exc)

        if error_code in ("ThrottlingException", "throttlingException"):
            raise ModelThrottledException(error_msg) from exc

        if any(msg in error_msg for msg in _CONTEXT_OVERFLOW_MESSAGES):
            raise ContextWindowOverflowException(error_msg) from exc
