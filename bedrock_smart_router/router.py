"""BedrockRouter — the main entry point for the SDK.

Drop-in routing layer that sits between your application and Amazon
Bedrock.  Analyses each request, selects the optimal model, invokes
Bedrock, and handles fallbacks automatically.

Integrates all Phase 1–3 components: complexity analysis, strategy
engine, circuit breakers, fallbacks, retries, caching, metrics,
observability, CRIS profiles, inference tiers, prompt cache awareness,
guardrails, and AIP multi-tenant support.
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace as dataclass_replace
from typing import Any, Callable

import boto3

from bedrock_smart_router.ab_testing import ABTestManager
from bedrock_smart_router.aip_manager import AIPManager
from bedrock_smart_router.cache_layer import ResponseCache, build_cache
from bedrock_smart_router.canary import CanaryManager
from bedrock_smart_router.circuit_breaker import CircuitBreakerRegistry
from bedrock_smart_router.config import (
    MetricsConfig,
    RouterConfig,
    RoutingConfig,
    resolve_preset,
)
from bedrock_smart_router.context_validator import ContextValidator
from bedrock_smart_router.cris_manager import CRISManager
from bedrock_smart_router.exceptions import ModelRejection, NoModelsMatchError
from bedrock_smart_router.fallback_handler import FallbackHandler
from bedrock_smart_router.guardrails_integration import GuardrailsManager
from bedrock_smart_router.inference_tier import InferenceTierSelector
from bedrock_smart_router.metrics_store import (
    InMemoryMetricsStore,
    MetricsStore,
    RequestRecord,
)
from bedrock_smart_router.model_registry import (
    COMPLEXITY_MIN_TIER,
    COMPLEXITY_MAX_TIER,
    ModelRegistry,
)
from bedrock_smart_router.models import (
    BedrockModel,
    RoutingDecision,
)
from bedrock_smart_router.observability import ObservabilityManager, RoutingEvent
from bedrock_smart_router.otel_integration import OTelIntegration
from bedrock_smart_router.prompt_cache_advisor import PromptCacheAdvisor
from bedrock_smart_router.request_analyzer import RequestAnalyzer
from bedrock_smart_router.retry_handler import RetryConfig, RetryHandler
from bedrock_smart_router.shadow_mode import ShadowManager
from bedrock_smart_router.strategy_engine import resolve_strategy
from bedrock_smart_router.budget_strategy import BudgetExceededError, BudgetRule, BudgetTracker

logger = logging.getLogger(__name__)


def _build_metrics_store(
    cfg: MetricsConfig, region: str, session: Any | None = None,
) -> MetricsStore:
    if cfg.backend == "dynamodb":
        from bedrock_smart_router.dynamodb_metrics_store import DynamoDBMetricsStore
        return DynamoDBMetricsStore(
            table_name=cfg.table_name, ttl_hours=cfg.ttl_hours,
            boto_session=session, region=region,
            auto_create_table=cfg.auto_create_table,
        )
    return InMemoryMetricsStore(max_records_per_model=cfg.max_records_per_model)


class _CompletionsNamespace:
    """Implements router.chat.completions.create(...) — OpenAI SDK drop-in.

    Supports the same parameters as openai.chat.completions.create():
    messages, model, max_tokens, temperature, top_p, stop, tools,
    tool_choice, stream, n, response_format, etc.
    """

    def __init__(self, router: "BedrockRouter") -> None:
        self._router = router

    def create(
        self,
        *,
        messages: list[dict[str, Any]],
        model: str | None = None,
        max_tokens: int | None = None,
        max_completion_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        stop: list[str] | str | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict | None = None,
        parallel_tool_calls: bool | None = None,
        response_format: dict | None = None,
        stream: bool = False,
        stream_options: dict | None = None,
        n: int | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        logprobs: bool | None = None,
        top_logprobs: int | None = None,
        seed: int | None = None,
        user: str | None = None,
        metadata: dict | None = None,
        store: bool | None = None,
        reasoning_effort: str | None = None,
        service_tier: str | None = None,
        # Smart Router extras
        routing: Any | None = None,
        **kwargs: Any,
    ) -> "dict[str, Any] | _ChatCompletionStream":
        """Create a chat completion with smart routing.

        Drop-in for ``openai.chat.completions.create()``. The router selects
        the best model based on prompt complexity, or uses the specified model.

        Parameters match the OpenAI Chat Completions API specification.
        Additional ``routing`` parameter accepts a ``RoutingConfig`` for
        overriding strategy, weights, or preferred model.

        When ``stream=True``, returns a generator yielding OpenAI-style
        streaming chunks (``chat.completion.chunk`` objects).

        Returns a Chat Completions response dict (same schema as OpenAI).
        """
        # Resolve max_tokens (OpenAI deprecated max_tokens in favor of max_completion_tokens)
        effective_max_tokens = max_completion_tokens or max_tokens

        if stream:
            return self._router.chat_completions_stream(
                messages=messages,
                model=model,
                max_tokens=effective_max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
                tools=tools,
                routing=routing,
                **kwargs,
            )

        return self._router.chat_completions(
            messages=messages,
            model=model,
            max_tokens=effective_max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            tools=tools,
            routing=routing,
            **kwargs,
        )


class _ModelsNamespace:
    """Implements router.models.list() — lists available models."""

    def __init__(self, router: "BedrockRouter") -> None:
        self._router = router

    def list(self) -> "_DotDict":
        """List all models available through the router.

        Returns an object matching the OpenAI models.list() response format.
        Supports both attribute and dict access.
        """
        models = self._router._registry.all_models
        data = []
        for m in models:
            data.append({
                "id": m.model_id,
                "object": "model",
                "owned_by": m.family,
                "created": 0,
            })
        return _DotDict({"object": "list", "data": data})

    def retrieve(self, model_id: str) -> "_DotDict | None":
        """Retrieve details for a specific model."""
        m = self._router._registry.get(model_id)
        if not m:
            return None
        return _DotDict({
            "id": m.model_id,
            "object": "model",
            "owned_by": m.family,
            "created": 0,
            "capabilities": {
                "tool_use": m.capabilities.tool_use,
                "vision": m.capabilities.vision,
                "streaming": m.capabilities.streaming,
                "extended_thinking": m.capabilities.extended_thinking,
            },
            "api_support": m.api_support,
            "tier": m.tier.value,
            "pricing": {
                "input_per_1k": m.pricing.input_per_1k,
                "output_per_1k": m.pricing.output_per_1k,
            },
        })


class _ChatNamespace:
    """Namespace for router.chat.completions"""

    def __init__(self, router: "BedrockRouter") -> None:
        self.completions = _CompletionsNamespace(router)


class _DotDict(dict):
    """Dict subclass that supports attribute access (dot notation).

    Enables OpenAI SDK-style access: response.choices[0].message.content
    while remaining a regular dict: response["choices"][0]["message"]["content"]

    Matches OpenAI SDK behavior: accessing a missing attribute returns None
    (e.g., message.tool_calls is None when no tool calls are present).

    Also awaitable, so both sync and async patterns work:
        response = client.chat.completions.create(...)        # sync
        response = await client.chat.completions.create(...)  # async
    """

    def __getattr__(self, key: str) -> Any:
        # Dunder/private attributes should not fall through to dict lookup
        if key.startswith("_"):
            raise AttributeError(f"'DotDict' has no attribute '{key}'")
        value = self.get(key)  # returns None for missing keys (matches OpenAI SDK)
        if isinstance(value, dict) and not isinstance(value, _DotDict):
            return _DotDict(value)
        if isinstance(value, list):
            return [_DotDict(v) if isinstance(v, dict) else v for v in value]
        return value

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def __await__(self):
        return self._await_impl().__await__()

    async def _await_impl(self):
        return self


class _ChatCompletionStream:
    """Wrapper that makes a streaming generator behave like OpenAI's Stream object.

    Supports BOTH synchronous and asynchronous iteration patterns:

    Sync (OpenAI SDK sync client):
        stream = client.chat.completions.create(stream=True, ...)
        for chunk in stream:
            print(chunk.choices[0].delta.content or "", end="")

    Async (OpenAI SDK async client):
        stream = await client.chat.completions.create(stream=True, ...)
        async for chunk in stream:
            print(chunk.choices[0].delta.content or "", end="")
    """

    def __init__(self, generator):
        self._generator = generator
        self._items: list | None = None  # Lazily materialized for async access

    # Sync iteration
    def __iter__(self):
        return self._generator

    def __next__(self):
        return next(self._generator)

    # Async iteration (async for chunk in stream)
    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._generator)
        except StopIteration:
            raise StopAsyncIteration

    # Make it awaitable (stream = await client.chat.completions.create(stream=True))
    def __await__(self):
        return self._await_impl().__await__()

    async def _await_impl(self):
        return self

    # Context manager support
    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


class _ConverseStream:
    """Wrapper for converse_stream that supports both sync and async iteration.

    Sync:
        for event in router.converse_stream(messages=[...]):
            ...

    Async:
        stream = await router.converse_stream(messages=[...])
        async for event in stream:
            ...
    """

    def __init__(self, generator):
        self._generator = generator

    def __iter__(self):
        return self._generator

    def __next__(self):
        return next(self._generator)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._generator)
        except StopIteration:
            raise StopAsyncIteration

    def __await__(self):
        return self._await_impl().__await__()

    async def _await_impl(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


class BedrockRouter:
    """Intelligent routing layer for Amazon Bedrock."""

    def __init__(
        self,
        config: RouterConfig,
        boto_session: Any | None = None,
        boto_config: Any | None = None,
        callbacks: list[Callable[[RoutingEvent], None]] | None = None,
    ) -> None:
        self._config = config
        # Resolve region: explicit config > boto3 default region > fallback
        resolved_region = config.region
        if not resolved_region:
            resolved_region = boto3.Session().region_name or "us-east-1"
            config = dataclass_replace(config, region=resolved_region)
            self._config = config
        session = boto_session or boto3.Session(region_name=resolved_region)

        # Resolve botocore Config: explicit param > config dict > None
        resolved_boto_config = boto_config
        if resolved_boto_config is None and config.boto_config:
            from botocore.config import Config as BotocoreConfig
            resolved_boto_config = BotocoreConfig(**config.boto_config)

        # Phase 1: core
        self._registry = ModelRegistry(catalog_path=config.catalog_path)
        self._analyzer = RequestAnalyzer(classifier=config.classifier)
        self._context_validator = ContextValidator()
        self._circuit_breakers = CircuitBreakerRegistry(config.circuit_breaker)
        self._fallback_handler = FallbackHandler(self._registry, config.fallback)
        self._retry_handler = RetryHandler(config.retry)

        # If the user configured retries in boto_config, disable our native
        # RetryHandler to avoid multiplicative retry storms (boto retries
        # internally × our retries = excessive attempts).
        if resolved_boto_config is not None:
            has_user_retries = False
            # Check explicit Config object
            if hasattr(resolved_boto_config, 'retries') and resolved_boto_config.retries:
                has_user_retries = True
            # Check dict-based config
            if config.boto_config and "retries" in config.boto_config:
                has_user_retries = True
            if has_user_retries:
                logger.info(
                    "boto_config includes retries — disabling native RetryHandler "
                    "to avoid multiplicative retry storms. Boto3 will handle retries."
                )
                self._retry_handler = RetryHandler(RetryConfig(max_retries=0))

        # Phase 2: intelligence
        self._metrics_store = _build_metrics_store(config.metrics, config.region, session)
        self._cache = build_cache(config.cache)

        # CloudWatch metrics publisher (optional)
        cw_publisher = None
        if config.observability.cloudwatch_enabled:
            from bedrock_smart_router.cloudwatch_metrics import CloudWatchMetricsPublisher
            cw_publisher = CloudWatchMetricsPublisher(
                namespace=config.observability.cloudwatch_namespace,
                boto_session=session,
                region=config.region,
            )

        self._observability = ObservabilityManager(
            callbacks=callbacks or [],
            log_decisions=config.observability.log_decisions,
            cloudwatch_publisher=cw_publisher,
        )

        # Phase 3: Bedrock-native
        # Derive CRIS restrictions from excluded_models patterns
        cris_config = config.cris
        if config.excluded_models:
            import fnmatch
            allow_global = cris_config.allow_global
            blocked_prefixes: list[str] = []
            if any(fnmatch.fnmatch("global.test", pat) for pat in config.excluded_models):
                allow_global = False
                blocked_prefixes.append("global")
            if any(fnmatch.fnmatch("us.test", pat) for pat in config.excluded_models):
                blocked_prefixes.append("us")
            if any(fnmatch.fnmatch("eu.test", pat) for pat in config.excluded_models):
                blocked_prefixes.append("eu")
            if any(fnmatch.fnmatch("ap.test", pat) for pat in config.excluded_models):
                blocked_prefixes.append("ap")
            if blocked_prefixes or not allow_global:
                from bedrock_smart_router.cris_manager import CRISConfig
                cris_config = CRISConfig(
                    enabled=cris_config.enabled,
                    preferred_geography=cris_config.preferred_geography,
                    allow_global=allow_global,
                    blocked_prefixes=blocked_prefixes,
                )
        self._cris = CRISManager(cris_config)
        self._tier_selector = InferenceTierSelector(config.inference_tier)
        self._cache_advisor = PromptCacheAdvisor()
        self._guardrails = GuardrailsManager(
            config=config.guardrails, boto_session=session, region=config.region,
        )
        self._aip = AIPManager(
            config=config.aip, boto_session=session, region=config.region,
        )

        # Phase 4: Advanced deployment
        self._ab_test = ABTestManager(config.ab_test)
        self._canary = CanaryManager(config.canary)
        self._shadow = ShadowManager(
            config=config.shadow,
            invoke_fn=None,  # Set after bedrock client is created
            registry=self._registry,
            cris_manager=self._cris,
            region=config.region,
        )

        # Budget enforcement (optional — only active when rules are defined)
        self._budget_tracker: BudgetTracker | None = None
        self._budget_rules: dict[str, BudgetRule] = {}
        self._budget_scope_key = config.budget.scope_key
        self._budget_rule_key = config.budget.rule_key
        if config.budget.rules:
            from bedrock_smart_router.budget_store import build_budget_store
            store = build_budget_store(
                backend=config.budget.tracker_backend,
                sqlite_path=config.budget.sqlite_path,
                dynamodb_table=config.budget.dynamodb_table,
                dynamodb_region=config.region,
                dynamodb_ttl_seconds=config.budget.dynamodb_ttl_seconds,
                dynamodb_auto_create=config.budget.dynamodb_auto_create,
                boto_session=session,
            )
            self._budget_tracker = BudgetTracker(
                store=store,
                sync_interval=config.budget.sync_interval_seconds,
            )
            # Parse rules
            for rule_name, rule_data in config.budget.rules.items():
                if isinstance(rule_data, dict):
                    self._budget_rules[rule_name] = BudgetRule(
                        max_cost_per_request=rule_data.get("max_cost_per_request"),
                        max_hourly_spend=rule_data.get("max_hourly_spend"),
                        max_daily_spend=rule_data.get("max_daily_spend"),
                        on_exceeded=rule_data.get("on_exceeded", "downgrade"),
                        downgrade_to_tier=rule_data.get("downgrade_to_tier", "lite"),
                    )
                elif isinstance(rule_data, BudgetRule):
                    self._budget_rules[rule_name] = rule_data
            logger.info(
                "Budget enforcement enabled: %d rules, backend=%s, scope_key=%s",
                len(self._budget_rules), config.budget.tracker_backend, self._budget_scope_key,
            )

        # Bedrock client
        client_kwargs: dict[str, Any] = {}
        if resolved_boto_config is not None:
            client_kwargs["config"] = resolved_boto_config

        # Set Bedrock API key if provided (for both bedrock-runtime and mantle)
        if config.api_key:
            import os
            os.environ.setdefault("AWS_BEARER_TOKEN_BEDROCK", config.api_key)

        self._bedrock = session.client("bedrock-runtime", **client_kwargs)
        self._shadow._invoke_fn = self._bedrock.converse

        # Mantle client (Chat Completions / Responses API)
        self._mantle = None
        if config.enable_mantle:
            try:
                from bedrock_smart_router.mantle_client import MantleClient
                self._mantle = MantleClient(
                    region=config.region,
                    api_key=config.api_key,
                    session=session,
                    timeout=config.mantle_timeout,
                )
            except ImportError:
                logger.warning("MantleClient unavailable (missing 'requests' package) — Mantle-only models will be unreachable")
            except Exception as e:
                logger.warning("Failed to initialize MantleClient: %s", e)

        # OpenTelemetry (optional)
        self._otel = OTelIntegration(
            enabled=config.observability.otel_enabled,
            service_name=config.observability.otel_service_name,
        )

        self._last_decision_local = threading.local()

        # Bounded thread pool for background work (metrics, observability, OTEL)
        self._bg_executor = ThreadPoolExecutor(
            max_workers=4, thread_name_prefix="bsr-bg",
        )

        # OpenAI-compatible namespace: router.chat.completions.create(...)
        self.chat = _ChatNamespace(self)
        self.models = _ModelsNamespace(self)

    @classmethod
    def create(
        cls,
        config: dict[str, Any] | RouterConfig | None = None,
        *,
        boto_session: Any | None = None,
        boto_config: Any | None = None,
        callbacks: list[Callable[[RoutingEvent], None]] | None = None,
    ) -> BedrockRouter:
        """Create a router from a dict, RouterConfig, or defaults.

        Args:
            config: Router configuration as a dict, RouterConfig, or None for defaults.
            boto_session: Optional pre-configured boto3 Session.
            boto_config: Optional ``botocore.config.Config`` instance for the
                Bedrock client (timeouts, retries, etc.).  Takes precedence
                over ``boto_config`` in the config dict/YAML.
            callbacks: Optional list of observability callbacks.
        """
        if config is None:
            resolved = RouterConfig()
        elif isinstance(config, dict):
            resolved = RouterConfig.from_dict(config)
        else:
            resolved = config
        return cls(resolved, boto_session=boto_session, boto_config=boto_config, callbacks=callbacks)

    # ── Public API ──────────────────────────────────────────────

    def _build_decision(
        self,
        *,
        used_model: BedrockModel,
        resolved: dict[str, Any],
        analysis: Any,
        strategy_name: str,
        guardrail_checked: bool,
        t_start: float,
        t_routing_done: float,
        elapsed_ms: float,
        used_cris: str,
        used_tier: str,
        usage: dict[str, Any],
        stop_reason: str = "",
        bedrock_latency: float | None = None,
        actual_service_tier: str = "",
        perf_config: dict[str, Any] | None = None,
        guardrail_trace: dict[str, Any] | None = None,
        ttft_ms: float | None = None,
        api_backend: str = "converse",
    ) -> RoutingDecision:
        """Build a RoutingDecision from invocation results.

        Shared by converse() and converse_stream() post-invocation.
        """
        primary = resolved["primary"]
        fallback_chain = resolved["fallback_chain"]
        ab_variant = resolved["ab_variant"]
        is_canary = resolved["is_canary"]

        input_tokens = usage.get("inputTokens", analysis.estimated_input_tokens)
        output_tokens = usage.get("outputTokens", analysis.estimated_output_tokens)
        prompt_cache_read = usage.get("cacheReadInputTokens", 0)
        prompt_cache_write = usage.get("cacheWriteInputTokens", 0)
        total_tokens = usage.get("totalTokens", input_tokens + output_tokens)
        cache_details = usage.get("cacheDetails", [])

        actual_cost = used_model.pricing.estimate_cost(
            input_tokens, output_tokens,
            cache_read_tokens=prompt_cache_read,
            cache_write_tokens=prompt_cache_write,
        )

        return RoutingDecision(
            selected_model=used_model.model_id,
            strategy_used=strategy_name,
            complexity_detected=analysis.complexity.value,
            complexity_score=analysis.complexity_score,
            candidates_evaluated=resolved["candidates_evaluated"],
            candidate_scores=resolved.get("scores", {}),
            fallback_chain=[m.model_id for m in fallback_chain],
            estimated_cost=primary.pricing.estimate_cost(
                analysis.estimated_input_tokens,
                analysis.estimated_output_tokens,
            ),
            actual_cost=actual_cost,
            latency_ms=round(elapsed_ms, 1),
            ttft_ms=round(ttft_ms, 1) if ttft_ms is not None else None,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            fallback_used=(used_model.model_id != primary.model_id),
            fallback_model=(
                used_model.model_id if used_model.model_id != primary.model_id else None
            ),
            circuit_breaker_skipped=resolved.get("skipped", []),
            inference_tier=used_tier,
            cris_profile=used_cris,
            prompt_cache_savings=resolved.get("cache_savings", 0.0),
            prompt_cache_read_tokens=prompt_cache_read,
            prompt_cache_write_tokens=prompt_cache_write,
            guardrail_checked=guardrail_checked,
            stop_reason=stop_reason,
            bedrock_latency_ms=bedrock_latency,
            actual_service_tier=actual_service_tier,
            total_tokens=total_tokens,
            cache_details=cache_details,
            performance_config=perf_config or {},
            guardrail_trace=guardrail_trace or {},
            metadata={
                **({"ab_variant": ab_variant} if ab_variant else {}),
                **({"is_canary": is_canary} if is_canary else {}),
            },
            routing_decision_ms=round((t_routing_done - t_start) * 1000, 2),
            explanation=resolved.get("explanation"),
            api_backend=api_backend,
        )

    def _prepare_call_args(
        self,
        model: BedrockModel,
        messages: list[dict[str, Any]],
        system: list[dict[str, Any]] | None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None]:
        """Strip unsupported blocks (reasoning, cache points) for a target model."""
        call_messages = self._strip_reasoning_content(messages) if model.tier.value != "reasoning" else messages
        if not getattr(model.capabilities, "prompt_caching", False):
            call_messages = self._strip_cache_points_from_messages(call_messages)
            call_system = self._strip_cache_points(system) if system else system
        else:
            call_system = system
        return call_messages, call_system

    def _pre_invoke_pipeline(
        self,
        *,
        messages: list[dict[str, Any]],
        system: list[dict[str, Any]] | None,
        tool_config: dict[str, Any] | None,
        routing: RoutingConfig,
        requires_streaming_tool_use: bool = False,
        **kwargs: Any,
    ) -> tuple[list[dict[str, Any]], str, dict[str, float], float, bool, Any, dict[str, Any]]:
        """Shared pre-invocation: guardrail → budget → analyze → resolve model."""
        strategy_name = routing.strategy or self._config.strategy
        weights = routing.weights or self._config.weights
        t_start = time.monotonic()

        # Pre-route guardrail check
        guardrail_checked = False
        if self._guardrails.has_pre_route:
            gr_result = self._guardrails.check_input(messages)
            guardrail_checked = True
            # If sanitize mode returned cleaned text, swap it in
            if gr_result.output_text and not gr_result.blocked:
                # Check if text was modified (anonymized)
                original_texts = self._guardrails._extract_text(messages)
                if gr_result.output_text != "\n".join(original_texts):
                    messages = [dict(m) for m in messages]
                    for msg in reversed(messages):
                        if msg.get("role") == "user":
                            msg["content"] = [{"text": gr_result.output_text}]
                            break

        # Budget enforcement check (if rules are configured)
        if self._budget_tracker and self._budget_rules:
            metadata = routing.metadata or {}
            scope = metadata.get(self._budget_scope_key, "")
            rule_name = metadata.get(self._budget_rule_key, "default")
            rule = self._budget_rules.get(rule_name) or self._budget_rules.get("default")
            if scope and rule:
                exceeded = self._budget_tracker.check_budget(scope, rule)
                if exceeded:
                    if rule.on_exceeded == "reject":
                        raise BudgetExceededError(
                            f"Budget exceeded for '{scope}': {exceeded}"
                        )
                    else:
                        # Downgrade: switch to cost-optimized strategy
                        strategy_name = "cost-optimized"
                        logger.info(
                            "Budget exceeded for '%s' (%s) — downgrading to cost-optimized",
                            scope, exceeded,
                        )

        # Analyse the request
        analysis = self._analyzer.analyze(messages, system, tool_config,
                                          classifier_override=routing.classifier)

        # Resolve model
        resolved = self._resolve_model(
            analysis=analysis, routing=routing,
            strategy_name=strategy_name, weights=weights,
            messages=messages, system=system,
            requires_streaming_tool_use=requires_streaming_tool_use,
            requires_guardrail="guardrailConfig" in kwargs,
        )

        return messages, strategy_name, weights, t_start, guardrail_checked, analysis, resolved

    def _normalize_request_params(
        self,
        routing: RoutingConfig | None,
        inference_config: dict[str, Any] | None,
        tool_config: dict[str, Any] | None,
        kwargs: dict[str, Any],
    ) -> tuple[RoutingConfig, dict[str, Any] | None, dict[str, Any] | None]:
        """Normalize request parameters for boto3 drop-in compatibility.

        Handles:
        - modelId/model_id kwargs → routing.preferred_model
        - inferenceConfig/toolConfig camelCase kwargs → explicit params

        Returns (routing, inference_config, tool_config) with kwargs mutated in-place.
        """
        routing = resolve_preset(routing or RoutingConfig())

        boto3_model = kwargs.pop("modelId", None) or kwargs.pop("model_id", None)
        if boto3_model and not routing.preferred_model:
            routing = dataclass_replace(routing, preferred_model=boto3_model)
        if "inferenceConfig" in kwargs and inference_config is None:
            inference_config = kwargs.pop("inferenceConfig")
        if "toolConfig" in kwargs and tool_config is None:
            tool_config = kwargs.pop("toolConfig")

        return routing, inference_config, tool_config

    def converse(
        self,
        *,
        messages: list[dict[str, Any]],
        system: list[dict[str, Any]] | None = None,
        tool_config: dict[str, Any] | None = None,
        inference_config: dict[str, Any] | None = None,
        routing: RoutingConfig | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Route and invoke a Bedrock Converse call."""
        routing, inference_config, tool_config = self._normalize_request_params(
            routing, inference_config, tool_config, kwargs,
        )

        # ── Pre-invoke pipeline (guardrail → analyze → resolve) ─
        messages, strategy_name, weights, t_start, guardrail_checked, analysis, resolved = \
            self._pre_invoke_pipeline(
                messages=messages, system=system, tool_config=tool_config,
                routing=routing, **kwargs,
            )

        # ── Check response cache (converse-only, not streaming) ─
        cached = self._cache.get(messages, system, inference_config, routing_key=strategy_name)
        if cached is not None:
            decision = RoutingDecision(
                selected_model=cached.get("_cached_model", "unknown"),
                strategy_used=strategy_name,
                complexity_detected=analysis.complexity.value,
                complexity_score=analysis.complexity_score,
                candidates_evaluated=0,
                estimated_cost=0.0,
                actual_cost=0.0,
                cache_hit=True,
                guardrail_checked=guardrail_checked,
            )
            self._last_decision_local.value = decision
            cached["routing_decision"] = decision
            self._observability.emit(
                decision, cache_hit=True,
                duration_ms=(time.monotonic() - t_start) * 1000,
                tags=routing.tags, metadata=routing.metadata,
            )
            return cached

        primary = resolved["primary"]
        fallback_chain = resolved["fallback_chain"]

        # ── Step 5: Invoke with fallbacks ───────────────────────
        t_routing_done = time.monotonic()
        models_to_try = [primary] + fallback_chain
        last_error: Exception | None = None
        used_model: BedrockModel | None = None
        response: dict[str, Any] | None = None
        elapsed_ms: float = 0.0
        used_cris: str = resolved["cris_profile"]
        used_tier: str = resolved["inference_tier"]

        for i, model in enumerate(models_to_try):
            if i > 0 and not self._circuit_breakers.is_available(model.model_id):
                continue

            model_cris = self._cris.select_profile(model, self._config.region) if i > 0 else resolved["cris_profile"]
            model_tier = self._tier_selector.select_tier(model, analysis) if i > 0 else resolved["inference_tier"]
            invoke_model_id = self._aip.get_model_id_for_tenant(
                model_cris, routing.metadata or {},
            )

            try:
                t0 = time.monotonic()
                req_metadata = {}
                if routing.metadata:
                    req_metadata = {
                        k: str(v) for k, v in routing.metadata.items()
                        if isinstance(k, str) and len(str(v)) <= 256
                    }
                call_messages, call_system = self._prepare_call_args(model, messages, system)
                response = self._invoke_bedrock(
                    model_id=invoke_model_id,
                    messages=call_messages,
                    system=call_system,
                    tool_config=tool_config,
                    inference_config=inference_config,
                    service_tier=model_tier if model_tier != "standard" else None,
                    request_metadata=req_metadata or None,
                    **kwargs,
                )
                elapsed_ms = (time.monotonic() - t0) * 1000
                self._circuit_breakers.record_success(model.model_id)
                used_model = model
                used_cris = model_cris
                used_tier = model_tier
                break
            except Exception as exc:
                last_error = exc
                is_throttle = RetryHandler.is_throttle(exc)
                self._circuit_breakers.record_failure(
                    model.model_id, is_throttle=is_throttle,
                )
                logger.warning(
                    "Model %s failed (%s), trying fallback %d/%d. Error: %s",
                    model.model_id, RetryHandler.get_error_code(exc),
                    i + 1, len(models_to_try), str(exc)[:500],
                )

        if response is None or used_model is None:
            raise RuntimeError(
                f"All models in fallback chain failed. Last error: {last_error}"
            ) from last_error

        # ── Step 6: Post-route guardrail check ──────────────────
        if self._guardrails.has_post_route:
            output_text = self._extract_output_text(response)
            if output_text:
                self._guardrails.check_output(output_text)

        # ── Step 7: Record canary result ────────────────────────
        is_canary = resolved["is_canary"]
        if is_canary or self._canary.is_active:
            self._canary.record_result(
                is_canary=is_canary,
                latency_ms=elapsed_ms,
                success=True,
            )

        # ── Step 8: Shadow mode ─────────────────────────────────
        if self._shadow.should_shadow():
            self._shadow.mirror(
                primary_model=used_model.model_id,
                messages=messages,
                system=system,
                tool_config=tool_config,
                inference_config=inference_config,
            )

        # ── Step 9: Build routing decision ──────────────────────
        usage = response.get("usage", {})
        api_backend = response.pop("_api_backend", "converse")
        decision = self._build_decision(
            used_model=used_model,
            resolved=resolved,
            analysis=analysis,
            strategy_name=strategy_name,
            guardrail_checked=guardrail_checked,
            t_start=t_start,
            t_routing_done=t_routing_done,
            elapsed_ms=elapsed_ms,
            used_cris=used_cris,
            used_tier=used_tier,
            usage=usage,
            stop_reason=response.get("stopReason", ""),
            bedrock_latency=response.get("metrics", {}).get("latencyMs"),
            actual_service_tier=response.get("serviceTier", {}).get("type", ""),
            perf_config=response.get("performanceConfig", {}),
            guardrail_trace=response.get("trace", {}).get("guardrail", {}),
            api_backend=api_backend,
        )
        self._last_decision_local.value = decision
        response["routing_decision"] = decision

        # ── Step 10: Record metrics (background) ────────────────
        input_tokens = usage.get("inputTokens", analysis.estimated_input_tokens)
        output_tokens = usage.get("outputTokens", analysis.estimated_output_tokens)
        tenant_id = (routing.metadata or {}).get("tenant", "")
        self._record_async(
            RequestRecord(
                model_id=used_model.model_id,
                timestamp=time.monotonic(),
                latency_ms=elapsed_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost=decision.actual_cost or 0,
                success=True,
                strategy=strategy_name,
                complexity=analysis.complexity.value,
                tenant_id=tenant_id,
                inference_tier=used_tier,
                cris_profile=used_cris,
                fallback_used=(used_model.model_id != primary.model_id),
                cache_hit=False,
                prompt_cache_read_tokens=usage.get("cacheReadInputTokens", 0),
                prompt_cache_write_tokens=usage.get("cacheWriteInputTokens", 0),
            ),
            decision,
            duration_ms=(time.monotonic() - t_start) * 1000,
            tags=routing.tags,
            metadata=routing.metadata,
            input_tokens_for_cost=input_tokens,
            output_tokens_for_cost=output_tokens,
        )

        # ── Step 10b: Record budget spend ───────────────────────
        if self._budget_tracker and decision.actual_cost:
            self._record_budget_spend(routing, decision, used_model)

        # ── Step 11: Cache the response ─────────────────────────
        response["_cached_model"] = used_model.model_id
        self._cache.put(
            messages, response,
            model_id=used_model.model_id,
            system=system,
            inference_config=inference_config,
            routing_key=strategy_name,
        )

        return response

    def chat_completions(
        self,
        *,
        messages: list[dict[str, Any]],
        model: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        stop: list[str] | str | None = None,
        routing: RoutingConfig | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Route and invoke via Chat Completions API format.

        Accepts OpenAI Chat Completions format. The router selects the best model,
        then either:
        - Calls Mantle directly (if model supports chat_completions natively)
        - Translates to Converse and calls bedrock-runtime (if model is Converse-only)

        Returns a standard Chat Completions response dict with routing_decision attached.

        Parameters
        ----------
        messages : list[dict]
            Chat Completions messages (role + content).
        model : str, optional
            Preferred model ID (bypasses routing if set).
        tools : list[dict], optional
            OpenAI function tool definitions.
        max_tokens : int, optional
            Maximum output tokens.
        temperature : float, optional
            Sampling temperature.
        routing : RoutingConfig, optional
            Routing overrides (strategy, preset, etc.)
        """
        from bedrock_smart_router.format_translator import (
            chat_completions_to_converse,
            converse_response_to_chat_completions,
        )

        routing = resolve_preset(routing or RoutingConfig())
        if model and not routing.preferred_model:
            routing = dataclass_replace(routing, preferred_model=model)

        # Translate CC → Converse for analysis and routing
        converse_params = chat_completions_to_converse(
            messages=messages,
            tools=tools,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=[stop] if isinstance(stop, str) else stop,
        )

        # Auto-infer toolConfig when messages contain tool content but tools weren't passed.
        # This handles the case where users follow OpenAI SDK patterns (tools only on first call)
        # but Bedrock Converse requires toolConfig whenever toolUse/toolResult blocks are present.
        tool_config = converse_params.get("tool_config")
        if not tool_config:
            has_tool_content = any(
                msg.get("role") == "tool" or
                (msg.get("role") == "assistant" and msg.get("tool_calls"))
                for msg in messages
            )
            if has_tool_content:
                # Extract tool definitions from assistant's tool_calls in the history
                inferred_tools = []
                seen_names = set()
                for msg in messages:
                    for tc in msg.get("tool_calls", []):
                        fn = tc.get("function", {})
                        name = fn.get("name", "")
                        if name and name not in seen_names:
                            seen_names.add(name)
                            inferred_tools.append({
                                "toolSpec": {
                                    "name": name,
                                    "description": f"Tool: {name}",
                                    "inputSchema": {"json": {"type": "object"}},
                                }
                            })
                if inferred_tools:
                    tool_config = {"tools": inferred_tools}

        # Use the Converse path for routing (classification, model selection)
        converse_response = self.converse(
            messages=converse_params["messages"],
            system=converse_params.get("system"),
            tool_config=tool_config,
            inference_config=converse_params.get("inference_config"),
            routing=routing,
            **kwargs,
        )

        # Translate response back to Chat Completions format
        decision = converse_response.pop("routing_decision", None)
        cc_response = converse_response_to_chat_completions(
            converse_response,
            model=decision.selected_model if decision else "",
        )
        if decision:
            cc_response["routing_decision"] = decision

        return _DotDict(cc_response)

    def chat_completions_stream(
        self,
        *,
        messages: list[dict[str, Any]],
        model: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        stop: list[str] | str | None = None,
        routing: RoutingConfig | None = None,
        **kwargs: Any,
    ) -> "_ChatCompletionStream":
        """Route and invoke via Chat Completions streaming.

        Returns a ``_ChatCompletionStream`` that yields OpenAI-style streaming
        chunks (``chat.completion.chunk`` objects with ``delta`` instead of ``message``).

        If the selected model is Mantle-only, streams directly from Mantle's
        /v1/chat/completions endpoint. Otherwise, uses converse_stream() and
        translates Converse stream events into OpenAI streaming chunks.
        """
        from bedrock_smart_router.format_translator import chat_completions_to_converse

        routing = resolve_preset(routing or RoutingConfig())
        if model and not routing.preferred_model:
            routing = dataclass_replace(routing, preferred_model=model)

        # Translate messages for analysis/routing
        converse_params = chat_completions_to_converse(
            messages=messages,
            tools=tools,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=[stop] if isinstance(stop, str) else stop,
        )

        # Auto-infer toolConfig
        tool_config = converse_params.get("tool_config")
        if not tool_config:
            has_tool_content = any(
                msg.get("role") == "tool" or
                (msg.get("role") == "assistant" and msg.get("tool_calls"))
                for msg in messages
            )
            if has_tool_content:
                inferred_tools = []
                seen_names = set()
                for msg in messages:
                    for tc in msg.get("tool_calls", []):
                        fn = tc.get("function", {})
                        name = fn.get("name", "")
                        if name and name not in seen_names:
                            seen_names.add(name)
                            inferred_tools.append({
                                "toolSpec": {
                                    "name": name,
                                    "description": f"Tool: {name}",
                                    "inputSchema": {"json": {"type": "object"}},
                                }
                            })
                if inferred_tools:
                    tool_config = {"tools": inferred_tools}

        # Run the pre-invoke pipeline to select the model
        msgs, strategy_name, weights, t_start, guardrail_checked, analysis, resolved = \
            self._pre_invoke_pipeline(
                messages=converse_params["messages"],
                system=converse_params.get("system"),
                tool_config=tool_config,
                routing=routing,
                requires_streaming_tool_use=bool(tool_config),
                **kwargs,
            )

        primary = resolved["primary"]

        # Check if the selected model should go through Mantle streaming
        if "converse" not in primary.api_support and self._mantle:
            # Mantle streaming path
            return _ChatCompletionStream(
                self._stream_via_mantle(
                    model=primary,
                    messages=messages,
                    tools=tools,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stop=stop,
                    resolved=resolved,
                    analysis=analysis,
                    strategy_name=strategy_name,
                    guardrail_checked=guardrail_checked,
                    t_start=t_start,
                )
            )

        # Converse streaming path — translate converse_stream events to CC chunks
        return _ChatCompletionStream(
            self._stream_via_converse(
                messages=converse_params["messages"],
                system=converse_params.get("system"),
                tool_config=tool_config,
                inference_config=converse_params.get("inference_config"),
                routing=routing,
                **kwargs,
            )
        )

    def _stream_via_mantle(
        self,
        *,
        model: "BedrockModel",
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        max_tokens: int | None,
        temperature: float | None,
        top_p: float | None,
        stop: list[str] | str | None,
        resolved: dict[str, Any],
        analysis: Any,
        strategy_name: str,
        guardrail_checked: bool,
        t_start: float,
    ) -> "Generator[_DotDict, None, None]":
        """Stream from Mantle and yield OpenAI-style chunks."""
        import re
        import uuid
        from bedrock_smart_router.model_registry import base_model_id

        # Resolve Mantle model ID
        mantle_model_id = base_model_id(model.model_id)
        mantle_model_id = re.sub(r"-\d{8}-v\d+:\d+$", "", mantle_model_id)
        mantle_model_id = re.sub(r"-v\d+:\d+$", "", mantle_model_id)
        mantle_model_id = re.sub(r"-\d+:\d+$", "", mantle_model_id)

        stream_kwargs: dict[str, Any] = {}
        if max_tokens is not None:
            stream_kwargs["max_tokens"] = max_tokens
        if temperature is not None:
            stream_kwargs["temperature"] = temperature
        if top_p is not None:
            stream_kwargs["top_p"] = top_p
        if stop is not None:
            stream_kwargs["stop"] = [stop] if isinstance(stop, str) else stop
        if tools is not None:
            stream_kwargs["tools"] = tools

        t_routing_done = time.monotonic()

        # Stream from Mantle — chunks are already in OpenAI format
        for chunk in self._mantle.chat_completions_stream(
            model=mantle_model_id,
            messages=messages,
            **stream_kwargs,
        ):
            yield _DotDict(chunk)

        # After stream completes, yield a final routing_decision chunk
        elapsed_ms = (time.monotonic() - t_start) * 1000
        decision = RoutingDecision(
            selected_model=model.model_id,
            strategy_used=strategy_name,
            complexity_detected=analysis.complexity.value,
            complexity_score=analysis.complexity_score,
            candidates_evaluated=resolved["candidates_evaluated"],
            estimated_cost=model.pricing.estimate_cost(
                analysis.estimated_input_tokens,
                analysis.estimated_output_tokens,
            ),
            latency_ms=round(elapsed_ms, 1),
            guardrail_checked=guardrail_checked,
            api_backend="mantle",
            routing_decision_ms=round((t_routing_done - t_start) * 1000, 2),
        )
        self._last_decision_local.value = decision
        yield _DotDict({"routing_decision": decision})

    def _stream_via_converse(
        self,
        *,
        messages: list[dict[str, Any]],
        system: list[dict[str, Any]] | None = None,
        tool_config: dict[str, Any] | None = None,
        inference_config: dict[str, Any] | None = None,
        routing: RoutingConfig | None = None,
        **kwargs: Any,
    ) -> "Generator[_DotDict, None, None]":
        """Use converse_stream and translate events to OpenAI streaming chunks."""
        import uuid

        chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        model_name = ""
        index = 0

        for event in self.converse_stream(
            messages=messages,
            system=system,
            tool_config=tool_config,
            inference_config=inference_config,
            routing=routing,
            **kwargs,
        ):
            # Pass through the final routing_decision event
            if "routing_decision" in event:
                decision = event["routing_decision"]
                model_name = decision.selected_model
                yield _DotDict({"routing_decision": decision})
                continue

            # Translate Converse stream events → OpenAI streaming chunks
            if "contentBlockStart" in event:
                block = event["contentBlockStart"]
                start_block = block.get("start", {})
                if "toolUse" in start_block:
                    # Tool call start
                    tu = start_block["toolUse"]
                    yield _DotDict({
                        "id": chunk_id,
                        "object": "chat.completion.chunk",
                        "model": model_name,
                        "choices": [{
                            "index": 0,
                            "delta": {
                                "tool_calls": [{
                                    "index": index,
                                    "id": tu.get("toolUseId", ""),
                                    "type": "function",
                                    "function": {
                                        "name": tu.get("name", ""),
                                        "arguments": "",
                                    },
                                }],
                            },
                            "finish_reason": None,
                        }],
                    })

            elif "contentBlockDelta" in event:
                delta = event["contentBlockDelta"].get("delta", {})
                if "text" in delta:
                    yield _DotDict({
                        "id": chunk_id,
                        "object": "chat.completion.chunk",
                        "model": model_name,
                        "choices": [{
                            "index": 0,
                            "delta": {"content": delta["text"]},
                            "finish_reason": None,
                        }],
                    })
                elif "toolUse" in delta:
                    yield _DotDict({
                        "id": chunk_id,
                        "object": "chat.completion.chunk",
                        "model": model_name,
                        "choices": [{
                            "index": 0,
                            "delta": {
                                "tool_calls": [{
                                    "index": index,
                                    "function": {
                                        "arguments": delta["toolUse"].get("input", ""),
                                    },
                                }],
                            },
                            "finish_reason": None,
                        }],
                    })

            elif "contentBlockStop" in event:
                index += 1

            elif "messageStop" in event:
                stop_reason = event["messageStop"].get("stopReason", "end_turn")
                finish_map = {
                    "end_turn": "stop",
                    "max_tokens": "length",
                    "tool_use": "tool_calls",
                    "stop_sequence": "stop",
                }
                yield _DotDict({
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "model": model_name,
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "finish_reason": finish_map.get(stop_reason, "stop"),
                    }],
                })

            elif "metadata" in event:
                # Usage info — emit as final chunk with usage
                meta = event["metadata"]
                usage = meta.get("usage", {})
                if usage:
                    yield _DotDict({
                        "id": chunk_id,
                        "object": "chat.completion.chunk",
                        "model": model_name,
                        "choices": [],
                        "usage": {
                            "prompt_tokens": usage.get("inputTokens", 0),
                            "completion_tokens": usage.get("outputTokens", 0),
                            "total_tokens": usage.get("inputTokens", 0) + usage.get("outputTokens", 0),
                        },
                    })

    def converse_stream(
        self,
        *,
        messages: list[dict[str, Any]],
        system: list[dict[str, Any]] | None = None,
        tool_config: dict[str, Any] | None = None,
        inference_config: dict[str, Any] | None = None,
        routing: RoutingConfig | None = None,
        **kwargs: Any,
    ):
        """Route and invoke a Bedrock ConverseStream call.

        Yields stream events as they arrive from Bedrock.  The routing
        decision is attached to a final ``routing_decision`` event
        after the stream completes.

        Supports both sync and async iteration::

            # Sync
            for event in router.converse_stream(messages=[...]):
                if "contentBlockDelta" in event:
                    print(event["contentBlockDelta"]["delta"]["text"], end="")

            # Async
            stream = await router.converse_stream(messages=[...])
            async for event in stream:
                ...
        """
        return _ConverseStream(self._converse_stream_generator(
            messages=messages, system=system, tool_config=tool_config,
            inference_config=inference_config, routing=routing, **kwargs,
        ))

    def _converse_stream_generator(
        self,
        *,
        messages: list[dict[str, Any]],
        system: list[dict[str, Any]] | None = None,
        tool_config: dict[str, Any] | None = None,
        inference_config: dict[str, Any] | None = None,
        routing: RoutingConfig | None = None,
        **kwargs: Any,
    ):
        """Internal generator for converse_stream."""
        routing, inference_config, tool_config = self._normalize_request_params(
            routing, inference_config, tool_config, kwargs,
        )

        # ── Pre-invoke pipeline (guardrail → analyze → resolve) ─
        messages, strategy_name, weights, t_start, guardrail_checked, analysis, resolved = \
            self._pre_invoke_pipeline(
                messages=messages, system=system, tool_config=tool_config,
                routing=routing, requires_streaming_tool_use=bool(tool_config),
                **kwargs,
            )

        # Try invocation with fallbacks
        t_routing_done = time.monotonic()  # Routing decision complete
        models_to_try = [resolved["primary"]] + resolved["fallback_chain"]
        last_error: Exception | None = None
        used_model: BedrockModel | None = None
        stream = None
        used_cris = resolved["cris_profile"]
        used_tier = resolved["inference_tier"]

        for i, model in enumerate(models_to_try):
            if i > 0 and not self._circuit_breakers.is_available(model.model_id):
                continue

            model_cris = self._cris.select_profile(model, self._config.region) if i > 0 else resolved["cris_profile"]
            model_tier = self._tier_selector.select_tier(model, analysis) if i > 0 else resolved["inference_tier"]
            invoke_model_id = self._aip.get_model_id_for_tenant(
                model_cris, routing.metadata or {},
            )

            try:
                call_kwargs: dict[str, Any] = {
                    "modelId": invoke_model_id,
                }
                call_messages, call_system = self._prepare_call_args(model, messages, system)
                call_kwargs["messages"] = call_messages
                if call_system:
                    call_kwargs["system"] = call_system
                if tool_config:
                    call_kwargs["toolConfig"] = tool_config
                if inference_config:
                    call_kwargs["inferenceConfig"] = inference_config
                if model_tier and model_tier != "standard":
                    call_kwargs["performanceConfig"] = {"latency": "optimized"}
                if routing.metadata:
                    stream_req_meta = {
                        k: str(v) for k, v in routing.metadata.items()
                        if isinstance(k, str) and len(str(v)) <= 256
                    }
                    if stream_req_meta:
                        call_kwargs["requestMetadata"] = stream_req_meta
                call_kwargs.update(kwargs)

                logger.debug(
                    "Stream call: model=%s, keys=%s, msg_count=%d, has_tools=%s, has_system=%s, extra_kwargs=%s",
                    invoke_model_id, list(call_kwargs.keys()), len(messages),
                    bool(tool_config), bool(system), list(kwargs.keys()),
                )

                stream_resp = self._bedrock.converse_stream(**call_kwargs)
                stream = stream_resp.get("stream")
                self._circuit_breakers.record_success(model.model_id)
                used_model = model
                used_cris = model_cris
                used_tier = model_tier
                break
            except Exception as exc:
                last_error = exc
                is_throttle = RetryHandler.is_throttle(exc)
                self._circuit_breakers.record_failure(
                    model.model_id, is_throttle=is_throttle,
                )
                logger.warning(
                    "Stream: model %s (invoke_id=%s) failed (%s), trying fallback %d/%d. Error: %s. Extra kwargs keys: %s",
                    model.model_id, invoke_model_id, RetryHandler.get_error_code(exc),
                    i + 1, len(models_to_try), str(exc)[:500], list(kwargs.keys()),
                )

        if stream is None or used_model is None:
            raise RuntimeError(
                f"All models in fallback chain failed. Last error: {last_error}"
            ) from last_error

        # Yield stream events, tracking TTFT
        usage: dict[str, Any] = {}
        stream_metrics: dict[str, Any] = {}
        stream_stop_reason: str = ""
        stream_service_tier: str = ""
        stream_perf_config: dict[str, Any] = {}
        stream_guardrail_trace: dict[str, Any] = {}
        ttft_ms: float | None = None
        t_stream_start = time.monotonic()

        for event in stream:
            # Capture TTFT on first content delta
            if ttft_ms is None and "contentBlockDelta" in event:
                ttft_ms = (time.monotonic() - t_stream_start) * 1000
            # Capture stop reason from messageStop event
            if "messageStop" in event:
                stream_stop_reason = event["messageStop"].get("stopReason", "")
            # Capture usage, metrics, serviceTier from metadata event
            if "metadata" in event:
                meta = event["metadata"]
                usage = meta.get("usage", {})
                stream_metrics = meta.get("metrics", {})
                stream_service_tier = meta.get("serviceTier", {}).get("type", "")
                stream_perf_config = meta.get("performanceConfig", {})
                stream_guardrail_trace = meta.get("trace", {}).get("guardrail", {})
            yield event

        # Post-stream: build decision and record metrics
        elapsed_ms = (time.monotonic() - t_start) * 1000
        decision = self._build_decision(
            used_model=used_model,
            resolved=resolved,
            analysis=analysis,
            strategy_name=strategy_name,
            guardrail_checked=guardrail_checked,
            t_start=t_start,
            t_routing_done=t_routing_done,
            elapsed_ms=elapsed_ms,
            used_cris=used_cris,
            used_tier=used_tier,
            usage=usage,
            stop_reason=stream_stop_reason,
            bedrock_latency=stream_metrics.get("latencyMs"),
            actual_service_tier=stream_service_tier,
            perf_config=stream_perf_config,
            guardrail_trace=stream_guardrail_trace,
            ttft_ms=ttft_ms,
        )
        self._last_decision_local.value = decision

        input_tokens = usage.get("inputTokens", analysis.estimated_input_tokens)
        output_tokens = usage.get("outputTokens", analysis.estimated_output_tokens)
        stream_tenant = (routing.metadata or {}).get("tenant", "")
        self._record_async(
            RequestRecord(
                model_id=used_model.model_id,
                timestamp=time.monotonic(),
                latency_ms=elapsed_ms,
                ttft_ms=ttft_ms or 0.0,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost=decision.actual_cost or 0,
                success=True,
                strategy=strategy_name,
                complexity=analysis.complexity.value,
                tenant_id=stream_tenant,
                inference_tier=used_tier,
                cris_profile=used_cris,
                fallback_used=(used_model.model_id != resolved["primary"].model_id),
                cache_hit=False,
                prompt_cache_read_tokens=usage.get("cacheReadInputTokens", 0),
                prompt_cache_write_tokens=usage.get("cacheWriteInputTokens", 0),
            ),
            decision,
            duration_ms=elapsed_ms,
            tags=routing.tags,
            metadata=routing.metadata,
        )

        # Record budget spend (shared with converse)
        if self._budget_tracker and decision.actual_cost:
            self._record_budget_spend(routing, decision, used_model)

        # Yield final event with routing decision
        yield {"routing_decision": decision}

    # ── Helpers ──────────────────────────────────────────────────

    def _record_budget_spend(
        self, routing: RoutingConfig, decision: RoutingDecision, used_model: BedrockModel
    ) -> None:
        """Record spend in the budget tracker (shared by converse and converse_stream)."""
        metadata = routing.metadata or {}
        scope = metadata.get(self._budget_scope_key, "")
        if scope and decision.actual_cost:
            self._budget_tracker.record_spend(
                scope, decision.actual_cost, model_id=used_model.model_id,
            )

    def _resolve_model(
        self,
        *,
        analysis: Any,
        routing: RoutingConfig,
        strategy_name: str,
        weights: dict[str, float],
        messages: list[dict[str, Any]],
        system: list[dict[str, Any]] | None,
        requires_streaming_tool_use: bool = False,
        requires_guardrail: bool = False,
    ) -> dict[str, Any]:
        """Run the routing pipeline and return the selected model + metadata.

        Shared by ``converse()`` and ``converse_stream()``.
        """
        # For custom strategies (non-built-in), skip tier filtering so the
        # strategy's filter_candidates sees all models. Built-in strategies
        # rely on tier filtering for complexity-aware routing.
        builtin_strategies = {"cost-optimized", "latency-optimized", "balanced", "quality-optimized"}
        if strategy_name in builtin_strategies:
            min_tier = COMPLEXITY_MIN_TIER.get(analysis.complexity.value)
            max_tier = COMPLEXITY_MAX_TIER.get(analysis.complexity.value)
        else:
            min_tier = None
            max_tier = None

        # Tier expansion: if no candidates at the target tier range, progressively
        # expand upward (+1 tier at a time) until we find models or hit reasoning (max).
        # This ensures a response is returned rather than raising NoModelsMatchError
        # when all models in the target tier are unavailable due to region/capability
        # filtering. We only expand upward (never downward) to maintain quality.
        from bedrock_smart_router.models import Tier as _Tier
        _TIER_LIST = list(_Tier)

        candidates = self._get_filtered_candidates(
            min_tier=min_tier, max_tier=max_tier,
            analysis=analysis, routing=routing,
            requires_streaming_tool_use=requires_streaming_tool_use,
            requires_guardrail=requires_guardrail,
            messages=messages, system=system,
        )

        if not candidates and max_tier is not None and strategy_name in builtin_strategies:
            # Expand upward tier-by-tier until we find candidates
            current_max_idx = _TIER_LIST.index(max_tier)
            for expand_idx in range(current_max_idx + 1, len(_TIER_LIST)):
                expanded_max = _TIER_LIST[expand_idx]
                candidates = self._get_filtered_candidates(
                    min_tier=min_tier, max_tier=expanded_max,
                    analysis=analysis, routing=routing,
                    requires_streaming_tool_use=requires_streaming_tool_use,
                    requires_guardrail=requires_guardrail,
                    messages=messages, system=system,
                )
                if candidates:
                    logger.info(
                        "No models at tier %s–%s for complexity=%s; expanded to tier %s (%d candidates)",
                        min_tier.value if min_tier else "any", max_tier.value,
                        analysis.complexity.value, expanded_max.value, len(candidates),
                    )
                    break

        if not candidates:
            self._raise_no_models_error(analysis=analysis, routing=routing, messages=messages, system=system)

        available = [c for c in candidates if self._circuit_breakers.is_available(c.model_id)]
        skipped = [c.model_id for c in candidates if c not in available]
        if not available:
            available = candidates
            skipped = []

        strategy = resolve_strategy(strategy_name, weights=weights, metrics_store=self._metrics_store)

        # Pass config-level + per-request metadata to the strategy for custom dimensions
        strategy._metadata = {**(self._config.metadata or {}), **(routing.metadata or {})}

        # A/B / canary / preferred_model overrides
        ab_variant = None
        is_canary = False

        # preferred_model takes highest priority — user explicitly chose
        if routing.preferred_model:
            override = self._registry.get(routing.preferred_model)
            result = strategy.select(available, analysis)
            if override:
                # User explicitly chose this model — respect it even if
                # it wasn't in the eligible candidates (skip tier/capability
                # filtering).  Only fall back if the model doesn't exist
                # in the catalog at all.
                primary = override
                # Since preferred_model overrides the strategy's pick,
                # add the strategy's selected_model to the front of the
                # fallback chain so it's the first fallback option.
                if result.selected_model.model_id != override.model_id:
                    result.fallback_chain.insert(0, result.selected_model)
                if override not in available:
                    logger.info(
                        "preferred_model '%s' was not in eligible candidates "
                        "(filtered by complexity/capabilities), but using it "
                        "as explicitly requested",
                        routing.preferred_model,
                    )
            else:
                logger.warning(
                    "preferred_model '%s' not found in model catalog, "
                    "falling back to strategy selection",
                    routing.preferred_model,
                )
                primary = result.selected_model
        elif self._ab_test.is_active:
            user_id = (routing.metadata or {}).get("user_id")
            ab_result = self._ab_test.assign(user_id=user_id)
            if ab_result:
                ab_variant = ab_result.variant_name
                override = self._registry.get(ab_result.model_id)
                result = strategy.select(available, analysis)
                # A/B variant model is used regardless of tier filtering
                # (the operator explicitly configured this test)
                primary = override if override else result.selected_model
            else:
                result = strategy.select(available, analysis)
                primary = result.selected_model
        elif self._canary.is_active:
            canary_id, is_canary = self._canary.select_model()
            result = strategy.select(available, analysis)
            override = self._registry.get(canary_id)
            if is_canary and override:
                # Canary model is used regardless of tier filtering
                primary = override
            else:
                # Baseline: use the configured baseline model, not the strategy pick
                baseline_model = self._registry.get(self._canary.config.baseline_model)
                if baseline_model:
                    primary = baseline_model
                else:
                    primary = result.selected_model
                is_canary = False
        elif self._canary.is_promoted:
            # Canary was promoted — use the canary model as the new primary
            promoted_model = self._registry.get(self._canary.config.canary_model)
            result = strategy.select(available, analysis)
            if promoted_model:
                primary = promoted_model
            else:
                primary = result.selected_model
        else:
            result = strategy.select(available, analysis)
            primary = result.selected_model

        # Prompt cache boost (skip for quality/latency strategies where
        # overriding the strategy pick for cache savings is undesirable)
        cache_savings = 0.0
        cache_eligible_strategies = ("balanced", "cost-optimized")
        if (
            self._config.prompt_cache_boost
            and not routing.preferred_model
            and strategy_name in cache_eligible_strategies
            and (system or len(messages) > 2)
        ):
            benefit = self._cache_advisor.estimate(primary, messages, system)
            cache_savings = benefit.savings_per_request
            if not benefit.cache_eligible and cache_savings == 0:
                ranked = self._cache_advisor.rank_models_by_cache_benefit(available, messages, system)
                for alt, alt_b in ranked:
                    if not alt_b.cache_eligible:
                        continue
                    alt_s = result.scores.get(alt.model_id, {}).get("composite", 0)
                    pri_s = result.scores.get(primary.model_id, {}).get("composite", 0)
                    if alt_s >= pri_s * 0.90:
                        primary = alt
                        cache_savings = alt_b.savings_per_request
                        break

        # Build fallback chain from the PRIMARY model (not the strategy pick)
        fallback_chain = self._fallback_handler.build_chain(primary, result.fallback_chain)

        # Build explanation if requested
        explanation = None
        if routing.explain:
            # Get top candidates by composite score
            top_candidates = sorted(
                result.scores.items(),
                key=lambda x: x[1].get("composite", 0),
                reverse=True,
            )[:5]
            candidate_list = []
            for model_id, scores in top_candidates:
                m = self._registry.get(model_id)
                candidate_list.append({
                    "model": m.display_name if m else model_id,
                    "model_id": model_id,
                    "composite": round(scores.get("composite", 0), 4),
                    "cost": round(scores.get("cost", 0), 4),
                    "latency": round(scores.get("latency", 0), 4),
                    "quality": round(scores.get("quality", 0), 4),
                })

            # Get analysis explanation (matched markers, dimension scores)
            analysis_explanation = self._analyzer.explain(messages, system)

            # Calculate payload boost (same logic as in analyze())
            payload_bytes = analysis_explanation.get("multimodal_payload_bytes", 0)
            payload_boost = 0.0
            if payload_bytes > 5_000_000:
                payload_boost = 0.30
            elif payload_bytes > 1_000_000:
                payload_boost = 0.20
            elif payload_bytes > 100_000:
                payload_boost = 0.10
            elif payload_bytes > 0:
                payload_boost = 0.05

            # Build reason text
            reason_parts = [f"Selected {primary.display_name}"]
            if candidate_list:
                reason_parts.append(f"(composite score: {candidate_list[0]['composite']:.4f})")
            reason_parts.append(f"for {analysis.complexity.value} complexity.")
            if strategy_name == "quality-optimized":
                reason_parts.append(f"Quality baseline: {primary.quality_baseline:.1f}/60.")
            elif strategy_name == "cost-optimized":
                est_cost = primary.pricing.estimate_cost(
                    analysis.estimated_input_tokens, analysis.estimated_output_tokens
                )
                reason_parts.append(f"Estimated cost: ${est_cost:.6f}.")
            elif strategy_name == "balanced":
                reason_parts.append(f"Balanced across cost/latency/quality.")

            explanation = {
                "complexity": self._build_complexity_explanation(
                    analysis, analysis_explanation, payload_bytes, payload_boost,
                    min_tier, max_tier, messages, system, None,
                    classifier_override=routing.classifier,
                ),
                "strategy": {
                    "name": strategy_name,
                    "weights": weights if strategy_name == "balanced" else None,
                    "preferred_model": routing.preferred_model or None,
                },
                "top5_candidates": candidate_list,
                "candidates_evaluated": len(available),
                "reason": " ".join(reason_parts),
            }

        return {
            "primary": primary,
            "fallback_chain": fallback_chain,
            "cris_profile": self._cris.select_profile(primary, self._config.region),
            "inference_tier": self._tier_selector.select_tier(primary, analysis, max_cost_per_request=routing.max_cost_per_request),
            "candidates_evaluated": len(available),
            "skipped": skipped,
            "scores": result.scores,
            "cache_savings": cache_savings,
            "ab_variant": ab_variant,
            "is_canary": is_canary,
            "explanation": explanation,
        }

    def _record_async(
        self,
        record: RequestRecord,
        decision: RoutingDecision,
        *,
        duration_ms: float,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        input_tokens_for_cost: int = 0,
        output_tokens_for_cost: int = 0,
    ) -> None:
        """Fire metrics, observability, and OTEL recording in a background thread.

        This avoids blocking the response on DynamoDB writes, CloudWatch
        publishes, and OTEL exports — typically saving 10–30ms per request.
        """
        def _work() -> None:
            try:
                self._metrics_store.record(record)
            except Exception:
                logger.debug("Background metrics record failed", exc_info=True)
            try:
                # Compute most_expensive in background to avoid blocking response
                most_expensive_cost = 0.0
                if input_tokens_for_cost > 0:
                    try:
                        most_expensive_cost = max(
                            (m.pricing.estimate_cost(input_tokens_for_cost, output_tokens_for_cost)
                             for m in self._registry.eligible_models(
                                 prefer_global=self._config.cris.allow_global)),
                            default=0.0,
                        )
                    except Exception:
                        pass
                self._observability.emit(
                    decision,
                    duration_ms=duration_ms,
                    tags=tags,
                    metadata=metadata,
                    most_expensive_cost=most_expensive_cost,
                )
            except Exception:
                logger.debug("Background observability emit failed", exc_info=True)
            try:
                self._otel.record_request(
                    model=decision.selected_model,
                    strategy=decision.strategy_used,
                    complexity=decision.complexity_detected,
                    latency_ms=decision.latency_ms or 0,
                    cost=decision.actual_cost or 0,
                    cache_hit=decision.cache_hit,
                    fallback_used=decision.fallback_used,
                    ttft_ms=decision.ttft_ms,
                )
            except Exception:
                logger.debug("Background OTEL record failed", exc_info=True)

        self._bg_executor.submit(_work)

    def _invoke_bedrock(
        self,
        *,
        model_id: str,
        messages: list[dict[str, Any]],
        system: list[dict[str, Any]] | None = None,
        tool_config: dict[str, Any] | None = None,
        inference_config: dict[str, Any] | None = None,
        service_tier: str | None = None,
        request_metadata: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Invoke Bedrock Converse API with retries, or Mantle if model requires it."""
        # Check if this model needs to go through Mantle
        model = self._registry.get(model_id)
        if model and "converse" not in model.api_support and self._mantle:
            # Model is Mantle-only — translate and call Chat Completions
            resp = self._invoke_via_mantle(
                model_id=model_id,
                messages=messages,
                system=system,
                tool_config=tool_config,
                inference_config=inference_config,
            )
            resp["_api_backend"] = "mantle"
            return resp

        call_kwargs: dict[str, Any] = {
            "modelId": model_id,
            "messages": messages,
        }
        if system:
            call_kwargs["system"] = system
        if tool_config:
            call_kwargs["toolConfig"] = tool_config
        if inference_config:
            call_kwargs["inferenceConfig"] = inference_config
        if service_tier:
            call_kwargs["performanceConfig"] = {"latency": "optimized"}
        if request_metadata:
            call_kwargs["requestMetadata"] = request_metadata
        call_kwargs.update(kwargs)
        return self._retry_handler.execute(self._bedrock.converse, **call_kwargs)

    def _invoke_via_mantle(
        self,
        *,
        model_id: str,
        messages: list[dict[str, Any]],
        system: list[dict[str, Any]] | None = None,
        tool_config: dict[str, Any] | None = None,
        inference_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Invoke a Mantle-only model by translating Converse → Chat Completions.

        Converts the Converse-format request to Chat Completions format,
        calls the bedrock-mantle endpoint, and translates the response back.
        """
        from bedrock_smart_router.format_translator import (
            converse_to_chat_completions,
            chat_completions_response_to_converse,
        )

        # Translate Converse → Chat Completions body
        cc_body = converse_to_chat_completions(
            messages=messages,
            system=system,
            tool_config=tool_config,
            inference_config=inference_config,
        )

        # Resolve the Mantle model ID (strip geo prefix and version suffix)
        from bedrock_smart_router.model_registry import base_model_id
        mantle_model_id = base_model_id(model_id)
        # Strip version suffixes that Mantle doesn't use
        import re
        mantle_model_id = re.sub(r"-\d{8}-v\d+:\d+$", "", mantle_model_id)
        mantle_model_id = re.sub(r"-v\d+:\d+$", "", mantle_model_id)
        mantle_model_id = re.sub(r"-\d+:\d+$", "", mantle_model_id)

        # Call Mantle
        cc_response = self._mantle.chat_completions(
            model=mantle_model_id,
            **cc_body,
        )

        # Translate response back to Converse format
        return chat_completions_response_to_converse(cc_response)

    @staticmethod
    def _extract_output_text(response: dict[str, Any]) -> str:
        """Extract text from a Bedrock Converse response for guardrail check."""
        output = response.get("output", {})
        message = output.get("message", {})
        content = message.get("content", [])
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and "text" in block:
                parts.append(block["text"])
        return "\n".join(parts)

    @staticmethod
    def _strip_reasoning_content(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Strip reasoningContent blocks from messages.

        When the router switches from a reasoning-tier model to a non-reasoning
        model between agent loop iterations, the conversation history may contain
        reasoningContent blocks that the target model cannot process.
        This method removes them while preserving all other content.
        """
        cleaned = []
        for msg in messages:
            content = msg.get("content")
            if not isinstance(content, list):
                cleaned.append(msg)
                continue
            new_content = [
                block for block in content
                if not (isinstance(block, dict) and "reasoningContent" in block)
            ]
            if len(new_content) != len(content):
                # Content was modified — ensure non-empty
                if not new_content:
                    new_content = [{"text": " "}]
                cleaned.append({**msg, "content": new_content})
            else:
                cleaned.append(msg)
        return cleaned

    def _build_complexity_explanation(
        self,
        analysis: Any,
        analysis_explanation: dict[str, Any],
        payload_bytes: int,
        payload_boost: float,
        min_tier: Any,
        max_tier: Any,
        messages: list[dict[str, Any]],
        system: list[dict[str, Any]] | None,
        tool_config: dict[str, Any] | None,
        classifier_override: str | None = None,
    ) -> dict[str, Any]:
        """Build the complexity section of the explain dict.

        Returns ML-style explain when ML classifier is active,
        otherwise returns the full heuristic explain with dimension scores.
        """
        # Determine which classifier was used for this request
        use_ml = self._analyzer._ml_classifier is not None
        if classifier_override == "ml":
            use_ml = True
        elif classifier_override == "heuristic":
            use_ml = False

        if use_ml and self._analyzer._ml_classifier is not None:
            # ML classifier explain: show probabilities from the LAST USER MESSAGE
            # (matching what classify_request() actually uses for the decision)
            clf = self._analyzer._ml_classifier
            last_user_text = clf.extract_last_user_text(messages)
            probs = clf.predict_proba_all(last_user_text or "")

            # Determine user's classification from probabilities
            # Apply the same low-confidence guard as classify_request
            raw_user_label = max(probs, key=probs.get)
            user_confidence = probs.get(raw_user_label, 0.0)
            low_confidence_override = False
            if user_confidence < clf.MIN_CONFIDENCE_THRESHOLD and raw_user_label in ("complex", "reasoning"):
                user_label = "simple"
                low_confidence_override = True
            else:
                user_label = raw_user_label

            # Check if floor was actually applied by re-running classify_request
            # and comparing the returned label to the effective user label
            ml_label, ml_conf = clf.classify_request(messages, system=system, tool_config=tool_config)
            floor_applied = (ml_label != user_label)

            explain = {
                "classifier": "ml",
                "score": analysis.complexity_score,
                "classification": analysis.complexity.value,
                "probabilities": probs,
                "user_message_classification": user_label,
                "raw_prediction": raw_user_label if low_confidence_override else None,
                "low_confidence_override": low_confidence_override,
                "user_confidence": round(user_confidence, 4),
                "floor_applied": floor_applied,
                "floor_dampening": clf._floor_dampening,
                "tier_range": {
                    "min": min_tier.value if min_tier else "micro",
                    "max": max_tier.value if max_tier else "reasoning",
                },
                "model_version": "tfidf_v1_35k",
                "multimodal_payload": {
                    "bytes": payload_bytes,
                    "complexity_boost": payload_boost,
                } if payload_bytes > 0 else None,
                "tool_boost_applied": analysis.tool_boost_applied,
            }
            if floor_applied:
                explain["floor_reason"] = "System prompt complexity floor upgraded classification"
            return explain
        else:
            # Heuristic explain: full dimension scores + markers
            return {
                "classifier": "heuristic",
                "score": analysis.complexity_score,
                "score_before_boost": round(analysis.complexity_score - payload_boost, 4),
                "classification": analysis.complexity.value,
                "classification_thresholds": {
                    "simple": f"< {self._analyzer.thresholds.simple_max}",
                    "moderate": f"{self._analyzer.thresholds.simple_max} - {self._analyzer.thresholds.moderate_max}",
                    "complex": f"{self._analyzer.thresholds.moderate_max} - {self._analyzer.thresholds.complex_max}",
                    "reasoning": f">= {self._analyzer.thresholds.complex_max} OR reasoning_markers >= {self._analyzer.thresholds.reasoning_marker_count}",
                },
                "tier_range": {
                    "min": min_tier.value if min_tier else "micro",
                    "max": max_tier.value if max_tier else "reasoning",
                },
                "markers_hit": analysis_explanation.get("matched_markers", {}),
                "marker_counts": analysis_explanation.get("marker_counts", {}),
                "dimension_scores": analysis_explanation.get("dimension_scores", {}),
                "user_message_score": analysis_explanation.get("user_message_score"),
                "system_prompt_floor": analysis_explanation.get("system_prompt_floor"),
                "floor_applied": analysis_explanation.get("floor_applied", False),
                "system_floor_markers": analysis_explanation.get("system_floor_markers", {}),
                "multimodal_payload": {
                    "bytes": payload_bytes,
                    "complexity_boost": payload_boost,
                } if payload_bytes > 0 else None,
                "tool_boost_applied": analysis.tool_boost_applied,
            }

    @staticmethod
    def _strip_cache_points(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Strip cachePoint blocks from system prompt or message content.

        When the router selects a model that doesn't support prompt caching,
        cachePoint blocks cause AccessDeniedException.
        This method removes them while preserving all other content.
        """
        cleaned = [
            block for block in blocks
            if not (isinstance(block, dict) and "cachePoint" in block)
        ]
        return cleaned if cleaned else blocks  # Fall back to original if all were cache points

    @staticmethod
    def _strip_cache_points_from_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Strip cachePoint blocks from message content lists."""
        cleaned = []
        for msg in messages:
            content = msg.get("content")
            if not isinstance(content, list):
                cleaned.append(msg)
                continue
            new_content = [
                block for block in content
                if not (isinstance(block, dict) and "cachePoint" in block)
            ]
            if len(new_content) != len(content):
                if not new_content:
                    new_content = [{"text": " "}]
                cleaned.append({**msg, "content": new_content})
            else:
                cleaned.append(msg)
        return cleaned

    def _get_filtered_candidates(
        self,
        *,
        min_tier: Any | None,
        max_tier: Any | None,
        analysis: Any,
        routing: "RoutingConfig",
        requires_streaming_tool_use: bool,
        requires_guardrail: bool,
        messages: list[dict[str, Any]],
        system: list[dict[str, Any]] | None,
    ) -> list["BedrockModel"]:
        """Get eligible candidates with all filters applied.

        Shared logic for tier-based candidate retrieval, region filtering,
        guardrail compatibility, and context window validation.
        """
        candidates = self._registry.eligible_models(
            min_tier=min_tier,
            max_tier=max_tier,
            requires_vision=analysis.requires_vision,
            requires_document_support=analysis.requires_document_support,
            requires_tool_use=analysis.requires_tool_use,
            requires_streaming_tool_use=requires_streaming_tool_use,
            min_context=routing.min_context_window,
            exclude_patterns=routing.exclude_models or self._config.excluded_models or None,
            family=routing.preferred_family,
            prefer_global=self._config.cris.allow_global,
        )
        # Filter out models that don't support guardrails when guardrailConfig is passed
        if requires_guardrail:
            candidates = [c for c in candidates if c.guardrail_compatible]
        # Filter out Mantle-only models not available in the configured region
        if self._mantle:
            router_region = self._config.region
            candidates = [
                c for c in candidates
                if "converse" in c.api_support
                or any(r.get("name") == router_region for r in c.regions)
            ]
        candidates = self._context_validator.filter_by_context(candidates, messages, system)
        return candidates

    def _raise_no_models_error(
        self,
        analysis: Any,
        routing: RoutingConfig,
        messages: list[dict[str, Any]],
        system: list[dict[str, Any]] | None,
    ) -> None:
        """Build a detailed NoModelsMatchError with per-model rejection reasons."""
        from bedrock_smart_router.context_validator import _estimate_tokens_from_messages

        est_tokens = _estimate_tokens_from_messages(messages, system)
        min_tier = COMPLEXITY_MIN_TIER.get(analysis.complexity.value)
        rejections: list[ModelRejection] = []

        # Deduplicate: only show one entry per base model (skip global
        # variants — they have the same capabilities as the regional entry)
        seen_base: set[str] = set()
        for m in self._registry.all_models:
            bm = m.base_model_id
            if bm in seen_base:
                continue
            seen_base.add(bm)
            reasons: list[str] = []
            # Tier check
            if min_tier:
                from bedrock_smart_router.models import Tier
                tier_order = list(Tier)
                if tier_order.index(m.tier) < tier_order.index(min_tier):
                    reasons.append(
                        f"tier {m.tier.value} < required {min_tier.value} "
                        f"(complexity={analysis.complexity.value})"
                    )
            # Vision
            if analysis.requires_vision and not m.capabilities.vision:
                reasons.append("no vision support")
            # Document support
            if analysis.requires_document_support and not m.capabilities.document_support:
                reasons.append("no document support")
            # Tool use
            if analysis.requires_tool_use and not m.capabilities.tool_use:
                reasons.append("no tool_use support")
            # Context window
            if est_tokens > m.max_input_tokens * 0.95:
                reasons.append(
                    f"context too small ({m.max_input_tokens} tokens < "
                    f"~{est_tokens} estimated)"
                )
            # Family filter
            if routing.preferred_family and m.family != routing.preferred_family:
                reasons.append(f"family {m.family} != {routing.preferred_family}")
            # Exclude patterns
            excludes = routing.exclude_models or self._config.excluded_models
            if excludes:
                import fnmatch
                for pat in excludes:
                    if fnmatch.fnmatch(m.model_id, pat):
                        reasons.append(f"excluded by pattern '{pat}'")
                        break
            # Cost
            if routing.max_cost_per_request is not None:
                est_cost = m.pricing.estimate_cost(
                    analysis.estimated_input_tokens,
                    analysis.estimated_output_tokens,
                )
                if est_cost > routing.max_cost_per_request:
                    reasons.append(
                        f"est. cost ${est_cost:.6f} > max ${routing.max_cost_per_request:.6f}"
                    )

            if not reasons:
                reasons.append("passed filters but removed by context window validator")

            rejections.append(ModelRejection(
                model_id=m.model_id,
                display_name=m.display_name,
                reasons=reasons,
            ))

        # Build suggestions
        suggestions: list[str] = []
        if routing.preferred_family:
            suggestions.append(f"Remove preferred_family='{routing.preferred_family}' to consider all families")
        if routing.max_cost_per_request is not None:
            suggestions.append(f"Increase max_cost_per_request (currently ${routing.max_cost_per_request})")
        if routing.exclude_models:
            suggestions.append("Remove or relax exclude_models patterns")
        if analysis.complexity.value in ("complex", "reasoning"):
            suggestions.append("The request was classified as complex/reasoning — fewer models qualify")
        if est_tokens > 128_000:
            suggestions.append(f"Input is ~{est_tokens} tokens — only models with large context windows qualify")

        constraints = {
            "complexity": analysis.complexity.value,
            "estimated_tokens": est_tokens,
            "min_tier": min_tier.value if min_tier else None,
            "requires_vision": analysis.requires_vision,
            "requires_document_support": analysis.requires_document_support,
            "requires_tool_use": analysis.requires_tool_use,
            "preferred_family": routing.preferred_family,
            "max_cost_per_request": routing.max_cost_per_request,
            "exclude_models": routing.exclude_models,
            "preset": routing.preset,
        }

        raise NoModelsMatchError(
            "No eligible models found for this request.",
            rejections=rejections,
            constraints=constraints,
            suggestions=suggestions,
        )

    def last_routing_decision(self) -> RoutingDecision | None:
        return getattr(self._last_decision_local, "value", None)

    # ── Accessors ───────────────────────────────────────────────

    @property
    def config(self) -> RouterConfig:
        return self._config

    @property
    def registry(self) -> ModelRegistry:
        return self._registry

    @property
    def metrics(self) -> MetricsStore:
        return self._metrics_store

    @property
    def cache(self) -> ResponseCache:
        return self._cache

    @property
    def observability(self) -> ObservabilityManager:
        return self._observability

    @property
    def guardrails(self) -> GuardrailsManager:
        return self._guardrails

    @property
    def cris(self) -> CRISManager:
        return self._cris

    @property
    def tier_selector(self) -> InferenceTierSelector:
        return self._tier_selector

    @property
    def aip(self) -> AIPManager:
        return self._aip

    @property
    def ab_test(self) -> ABTestManager:
        return self._ab_test

    @property
    def canary(self) -> CanaryManager:
        return self._canary

    @property
    def shadow(self) -> ShadowManager:
        return self._shadow

    @property
    def otel(self) -> OTelIntegration:
        return self._otel
