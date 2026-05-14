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
        session = boto_session or boto3.Session(region_name=config.region)

        # Resolve botocore Config: explicit param > config dict > None
        resolved_boto_config = boto_config
        if resolved_boto_config is None and config.boto_config:
            from botocore.config import Config as BotocoreConfig
            resolved_boto_config = BotocoreConfig(**config.boto_config)

        # Phase 1: core
        self._registry = ModelRegistry(catalog_path=config.catalog_path)
        self._analyzer = RequestAnalyzer()
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
        self._cris = CRISManager(config.cris)
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
        )

        # Bedrock client
        client_kwargs: dict[str, Any] = {}
        if resolved_boto_config is not None:
            client_kwargs["config"] = resolved_boto_config
        self._bedrock = session.client("bedrock-runtime", **client_kwargs)
        self._shadow._invoke_fn = self._bedrock.converse

        # OpenTelemetry (optional)
        self._otel = OTelIntegration(
            enabled=config.observability.otel_enabled,
            service_name=config.observability.otel_service_name,
        )

        self._last_decision: RoutingDecision | None = None

        # Bounded thread pool for background work (metrics, observability, OTEL)
        self._bg_executor = ThreadPoolExecutor(
            max_workers=4, thread_name_prefix="bsr-bg",
        )

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
        routing = resolve_preset(routing or RoutingConfig())
        strategy_name = routing.strategy or self._config.strategy
        weights = routing.weights or self._config.weights
        t_start = time.monotonic()

        # ── Step 1: Pre-route guardrail check ───────────────────
        guardrail_checked = False
        if self._guardrails.has_pre_route:
            gr_result = self._guardrails.check_input(messages)
            guardrail_checked = True
            # If sanitize mode returned cleaned text, swap it in
            # Copy messages to avoid mutating the caller's data
            if gr_result.output_text and gr_result.blocked:
                messages = [dict(m) for m in messages]
                for msg in reversed(messages):
                    if msg.get("role") == "user":
                        msg["content"] = [{"text": gr_result.output_text}]
                        break

        # ── Step 2: Analyse the request ─────────────────────────
        analysis = self._analyzer.analyze(messages, system, tool_config)

        # ── Step 3: Check response cache ────────────────────────
        cached = self._cache.get(messages, system, inference_config)
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
            self._last_decision = decision
            cached["routing_decision"] = decision
            self._observability.emit(
                decision, cache_hit=True,
                duration_ms=(time.monotonic() - t_start) * 1000,
                tags=routing.tags, metadata=routing.metadata,
            )
            return cached

        # ── Step 4: Resolve model (shared with converse_stream) ─
        resolved = self._resolve_model(
            analysis=analysis, routing=routing,
            strategy_name=strategy_name, weights=weights,
            messages=messages, system=system,
        )
        primary = resolved["primary"]
        fallback_chain = resolved["fallback_chain"]
        skipped = resolved["skipped"]
        cache_savings = resolved["cache_savings"]
        ab_variant = resolved["ab_variant"]
        is_canary = resolved["is_canary"]

        # ── Step 5: Invoke with fallbacks ───────────────────────
        t_routing_done = time.monotonic()  # Routing decision complete
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

            model_cris = self._cris.select_profile(model) if i > 0 else resolved["cris_profile"]
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
                response = self._invoke_bedrock(
                    model_id=invoke_model_id,
                    messages=messages,
                    system=system,
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
                    "Model %s failed (%s), trying fallback %d/%d",
                    model.model_id, RetryHandler.get_error_code(exc),
                    i + 1, len(models_to_try),
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
        input_tokens = usage.get("inputTokens", analysis.estimated_input_tokens)
        output_tokens = usage.get("outputTokens", analysis.estimated_output_tokens)
        prompt_cache_read = usage.get("cacheReadInputTokens", 0)
        prompt_cache_write = usage.get("cacheWriteInputTokens", 0)
        bedrock_latency = response.get("metrics", {}).get("latencyMs")
        stop_reason = response.get("stopReason", "")
        actual_service_tier = response.get("serviceTier", {}).get("type", "")
        total_tokens = usage.get("totalTokens", input_tokens + output_tokens)
        cache_details = usage.get("cacheDetails", [])
        perf_config = response.get("performanceConfig", {})
        guardrail_trace = response.get("trace", {}).get("guardrail", {})
        actual_cost = used_model.pricing.estimate_cost(input_tokens, output_tokens)

        decision = RoutingDecision(
            selected_model=used_model.model_id,
            strategy_used=strategy_name,
            complexity_detected=analysis.complexity.value,
            complexity_score=analysis.complexity_score,
            candidates_evaluated=resolved["candidates_evaluated"],
            candidate_scores=resolved["scores"],
            fallback_chain=[m.model_id for m in fallback_chain],
            estimated_cost=primary.pricing.estimate_cost(
                analysis.estimated_input_tokens,
                analysis.estimated_output_tokens,
            ),
            actual_cost=actual_cost,
            latency_ms=round(elapsed_ms, 1),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            fallback_used=(used_model.model_id != primary.model_id),
            fallback_model=(
                used_model.model_id if used_model.model_id != primary.model_id else None
            ),
            circuit_breaker_skipped=skipped,
            inference_tier=used_tier,
            cris_profile=used_cris,
            prompt_cache_savings=cache_savings,
            prompt_cache_read_tokens=prompt_cache_read,
            prompt_cache_write_tokens=prompt_cache_write,
            guardrail_checked=guardrail_checked,
            stop_reason=stop_reason,
            bedrock_latency_ms=bedrock_latency,
            actual_service_tier=actual_service_tier,
            total_tokens=total_tokens,
            cache_details=cache_details,
            performance_config=perf_config,
            guardrail_trace=guardrail_trace,
            metadata={
                **({"ab_variant": ab_variant} if ab_variant else {}),
                **({"is_canary": is_canary} if is_canary else {}),
            },
            routing_decision_ms=round((t_routing_done - t_start) * 1000, 2),
            explanation=resolved.get("explanation"),
        )
        self._last_decision = decision
        response["routing_decision"] = decision

        # ── Step 10: Record metrics (background) ────────────────
        tenant_id = (routing.metadata or {}).get("tenant", "")
        self._record_async(
            RequestRecord(
                model_id=used_model.model_id,
                timestamp=time.monotonic(),
                latency_ms=elapsed_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost=actual_cost,
                success=True,
                strategy=strategy_name,
                complexity=analysis.complexity.value,
                tenant_id=tenant_id,
                inference_tier=used_tier,
                cris_profile=used_cris,
                fallback_used=(used_model.model_id != primary.model_id),
                cache_hit=False,
                prompt_cache_read_tokens=prompt_cache_read,
                prompt_cache_write_tokens=prompt_cache_write,
            ),
            decision,
            duration_ms=(time.monotonic() - t_start) * 1000,
            tags=routing.tags,
            metadata=routing.metadata,
            input_tokens_for_cost=input_tokens,
            output_tokens_for_cost=output_tokens,
        )

        # ── Step 11: Cache the response ─────────────────────────
        response["_cached_model"] = used_model.model_id
        self._cache.put(
            messages, response,
            model_id=used_model.model_id,
            system=system,
            inference_config=inference_config,
        )

        return response

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

        Usage::

            for event in router.converse_stream(messages=[...]):
                if "contentBlockDelta" in event:
                    print(event["contentBlockDelta"]["delta"]["text"], end="")
                elif "routing_decision" in event:
                    print(f"\\nModel: {event['routing_decision'].selected_model}")
        """
        routing = resolve_preset(routing or RoutingConfig())
        strategy_name = routing.strategy or self._config.strategy
        weights = routing.weights or self._config.weights
        t_start = time.monotonic()

        # Pre-route guardrail
        guardrail_checked = False
        if self._guardrails.has_pre_route:
            gr_result = self._guardrails.check_input(messages)
            guardrail_checked = True
            # If sanitize mode returned cleaned text, swap it in
            # Copy messages to avoid mutating the caller's data
            if gr_result.output_text and gr_result.blocked:
                messages = [dict(m) for m in messages]
                for msg in reversed(messages):
                    if msg.get("role") == "user":
                        msg["content"] = [{"text": gr_result.output_text}]
                        break

        # Routing pipeline (same as converse)
        analysis = self._analyzer.analyze(messages, system, tool_config)
        resolved = self._resolve_model(
            analysis=analysis, routing=routing,
            strategy_name=strategy_name, weights=weights,
            messages=messages, system=system,
            requires_streaming_tool_use=bool(tool_config),
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

            model_cris = self._cris.select_profile(model) if i > 0 else resolved["cris_profile"]
            model_tier = self._tier_selector.select_tier(model, analysis) if i > 0 else resolved["inference_tier"]
            invoke_model_id = self._aip.get_model_id_for_tenant(
                model_cris, routing.metadata or {},
            )

            try:
                call_kwargs: dict[str, Any] = {
                    "modelId": invoke_model_id,
                    "messages": messages,
                }
                if system:
                    call_kwargs["system"] = system
                if tool_config:
                    call_kwargs["toolConfig"] = tool_config
                if inference_config:
                    call_kwargs["inferenceConfig"] = inference_config
                if model_tier and model_tier != "standard":
                    call_kwargs["serviceTier"] = {"type": model_tier}
                if routing.metadata:
                    stream_req_meta = {
                        k: str(v) for k, v in routing.metadata.items()
                        if isinstance(k, str) and len(str(v)) <= 256
                    }
                    if stream_req_meta:
                        call_kwargs["requestMetadata"] = stream_req_meta
                call_kwargs.update(kwargs)

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
                    "Stream: model %s failed (%s), trying fallback %d/%d",
                    model.model_id, RetryHandler.get_error_code(exc),
                    i + 1, len(models_to_try),
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
        input_tokens = usage.get("inputTokens", analysis.estimated_input_tokens)
        output_tokens = usage.get("outputTokens", analysis.estimated_output_tokens)
        prompt_cache_read = usage.get("cacheReadInputTokens", 0)
        prompt_cache_write = usage.get("cacheWriteInputTokens", 0)
        total_tokens = usage.get("totalTokens", input_tokens + output_tokens)
        cache_details = usage.get("cacheDetails", [])
        bedrock_latency = stream_metrics.get("latencyMs")
        actual_cost = used_model.pricing.estimate_cost(input_tokens, output_tokens)

        decision = RoutingDecision(
            selected_model=used_model.model_id,
            strategy_used=strategy_name,
            complexity_detected=analysis.complexity.value,
            complexity_score=analysis.complexity_score,
            candidates_evaluated=resolved["candidates_evaluated"],
            fallback_chain=[m.model_id for m in resolved["fallback_chain"]],
            estimated_cost=resolved["primary"].pricing.estimate_cost(
                analysis.estimated_input_tokens, analysis.estimated_output_tokens,
            ),
            actual_cost=actual_cost,
            latency_ms=round(elapsed_ms, 1),
            ttft_ms=round(ttft_ms, 1) if ttft_ms is not None else None,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            fallback_used=(used_model.model_id != resolved["primary"].model_id),
            inference_tier=used_tier,
            cris_profile=used_cris,
            prompt_cache_read_tokens=prompt_cache_read,
            prompt_cache_write_tokens=prompt_cache_write,
            guardrail_checked=guardrail_checked,
            stop_reason=stream_stop_reason,
            bedrock_latency_ms=bedrock_latency,
            actual_service_tier=stream_service_tier,
            total_tokens=total_tokens,
            cache_details=cache_details,
            performance_config=stream_perf_config,
            guardrail_trace=stream_guardrail_trace,
            routing_decision_ms=round((t_routing_done - t_start) * 1000, 2),
            explanation=resolved.get("explanation"),
        )
        self._last_decision = decision

        stream_tenant = (routing.metadata or {}).get("tenant", "")
        self._record_async(
            RequestRecord(
                model_id=used_model.model_id,
                timestamp=time.monotonic(),
                latency_ms=elapsed_ms,
                ttft_ms=ttft_ms or 0.0,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost=actual_cost,
                success=True,
                strategy=strategy_name,
                complexity=analysis.complexity.value,
                tenant_id=stream_tenant,
                inference_tier=used_tier,
                cris_profile=used_cris,
                fallback_used=(used_model.model_id != resolved["primary"].model_id),
                cache_hit=False,
                prompt_cache_read_tokens=prompt_cache_read,
                prompt_cache_write_tokens=prompt_cache_write,
            ),
            decision,
            duration_ms=elapsed_ms,
            tags=routing.tags,
            metadata=routing.metadata,
        )

        # Yield final event with routing decision
        yield {"routing_decision": decision}

    # ── Helpers ──────────────────────────────────────────────────

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
    ) -> dict[str, Any]:
        """Run the routing pipeline and return the selected model + metadata.

        Shared by ``converse()`` and ``converse_stream()``.
        """
        min_tier = COMPLEXITY_MIN_TIER.get(analysis.complexity.value)
        max_tier = COMPLEXITY_MAX_TIER.get(analysis.complexity.value)
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
        candidates = self._context_validator.filter_by_context(candidates, messages, system)
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
                primary = override if (override and override in available) else result.selected_model
            else:
                result = strategy.select(available, analysis)
                primary = result.selected_model
        elif self._canary.is_active:
            canary_id, is_canary = self._canary.select_model()
            result = strategy.select(available, analysis)
            override = self._registry.get(canary_id)
            if is_canary and override and override in available:
                primary = override
            else:
                primary = result.selected_model
                is_canary = False
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
                "complexity": {
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
                    "multimodal_payload": {
                        "bytes": payload_bytes,
                        "complexity_boost": payload_boost,
                    } if payload_bytes > 0 else None,
                },
                "strategy": {
                    "name": strategy_name,
                    "weights": weights if strategy_name == "balanced" else None,
                },
                "top5_candidates": candidate_list,
                "candidates_evaluated": len(available),
                "reason": " ".join(reason_parts),
            }

        return {
            "primary": primary,
            "fallback_chain": fallback_chain,
            "cris_profile": self._cris.select_profile(primary),
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
        """Invoke Bedrock Converse API with retries."""
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
            call_kwargs["serviceTier"] = {"type": service_tier}
        if request_metadata:
            call_kwargs["requestMetadata"] = request_metadata
        call_kwargs.update(kwargs)
        return self._retry_handler.execute(self._bedrock.converse, **call_kwargs)

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
        return self._last_decision

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
