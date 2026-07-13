# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""AsyncBedrockRouter — async/await version of the router.

Wraps the synchronous BedrockRouter and runs Bedrock calls in a
thread pool executor so they don't block the event loop.  All routing
logic (analysis, strategy, caching) runs synchronously since it's
sub-millisecond, only the Bedrock API call is offloaded.

Usage::

    router = AsyncBedrockRouter.create({"region": "us-west-2"})
    response = await router.converse(
        messages=[{"role": "user", "content": [{"text": "Hello"}]}],
    )
"""

from __future__ import annotations

import asyncio
import functools
from typing import Any, Callable

from bedrock_smart_router.config import RouterConfig, RoutingConfig
from bedrock_smart_router.observability import RoutingEvent
from bedrock_smart_router.router import BedrockRouter


class AsyncBedrockRouter:
    """Async wrapper around BedrockRouter.

    Runs the synchronous ``converse()`` in a thread pool executor
    so it can be awaited without blocking the event loop.
    """

    def __init__(
        self,
        config: RouterConfig,
        boto_session: Any | None = None,
        callbacks: list[Callable[[RoutingEvent], None]] | None = None,
    ) -> None:
        self._sync_router = BedrockRouter(
            config=config,
            boto_session=boto_session,
            callbacks=callbacks,
        )

    @classmethod
    def create(
        cls,
        config: dict[str, Any] | RouterConfig | None = None,
        *,
        boto_session: Any | None = None,
        callbacks: list[Callable[[RoutingEvent], None]] | None = None,
    ) -> AsyncBedrockRouter:
        """Create an async router from a dict, RouterConfig, or defaults."""
        if config is None:
            resolved = RouterConfig()
        elif isinstance(config, dict):
            resolved = RouterConfig.from_dict(config)
        else:
            resolved = config
        return cls(resolved, boto_session=boto_session, callbacks=callbacks)

    async def converse(
        self,
        *,
        messages: list[dict[str, Any]],
        system: list[dict[str, Any]] | None = None,
        tool_config: dict[str, Any] | None = None,
        inference_config: dict[str, Any] | None = None,
        routing: RoutingConfig | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Async version of ``BedrockRouter.converse()``."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            functools.partial(
                self._sync_router.converse,
                messages=messages,
                system=system,
                tool_config=tool_config,
                inference_config=inference_config,
                routing=routing,
                **kwargs,
            ),
        )

    async def converse_stream(
        self,
        *,
        messages: list[dict[str, Any]],
        system: list[dict[str, Any]] | None = None,
        tool_config: dict[str, Any] | None = None,
        inference_config: dict[str, Any] | None = None,
        routing: RoutingConfig | None = None,
        **kwargs: Any,
    ):
        """Async generator wrapping ``BedrockRouter.converse_stream()``.

        Usage::

            async for event in router.converse_stream(messages=[...]):
                if "contentBlockDelta" in event:
                    print(event["contentBlockDelta"]["delta"]["text"], end="")
        """
        loop = asyncio.get_running_loop()
        # Run the sync generator in a thread and yield events
        import queue
        import threading

        q: queue.Queue = queue.Queue()
        sentinel = object()

        def _run():
            try:
                for event in self._sync_router.converse_stream(
                    messages=messages, system=system,
                    tool_config=tool_config, inference_config=inference_config,
                    routing=routing, **kwargs,
                ):
                    q.put(event)
            except Exception as exc:
                q.put(exc)
            finally:
                q.put(sentinel)

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()

        while True:
            item = await loop.run_in_executor(None, q.get)
            if item is sentinel:
                break
            if isinstance(item, Exception):
                raise item
            yield item

    # ── Delegate accessors to the sync router ───────────────────

    @property
    def config(self) -> RouterConfig:
        return self._sync_router.config

    @property
    def registry(self):
        return self._sync_router.registry

    @property
    def metrics(self):
        return self._sync_router.metrics

    @property
    def cache(self):
        return self._sync_router.cache

    @property
    def observability(self):
        return self._sync_router.observability

    def last_routing_decision(self):
        return self._sync_router.last_routing_decision()
