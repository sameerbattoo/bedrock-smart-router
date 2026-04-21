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
import time
from typing import Any, Callable

import boto3

from bedrock_smart_router.ab_testing import ABTestManager
from bedrock_smart_router.aip_manager import AIPManager
from bedrock_smart_router.cache_layer import ResponseCache
from bedrock_smart_router.canary import CanaryManager
from bedrock_smart_router.circuit_breaker import CircuitBreakerRegistry
from bedrock_smart_router.config import (
    MetricsConfig,
    RouterConfig,
    RoutingConfig,
)
from bedrock_smart_router.context_validator import ContextValidator
from bedrock_smart_router.cris_manager import CRISManager
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
    ModelRegistry,
)
from bedrock_smart_router.models import (
    BedrockModel,
    RoutingDecision,
)
from bedrock_smart_router.observability import ObservabilityManager, RoutingEvent
from bedrock_smart_router.prompt_cache_advisor import PromptCacheAdvisor
from bedrock_smart_router.request_analyzer import RequestAnalyzer
from bedrock_smart_router.retry_handler import RetryHandler
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
        callbacks: list[Callable[[RoutingEvent], None]] | None = None,
    ) -> None:
        self._config = config
        session = boto_session or boto3.Session(region_name=config.region)

        # Phase 1: core
        self._registry = ModelRegistry(catalog_path=config.catalog_path)
        self._analyzer = RequestAnalyzer()
        self._context_validator = ContextValidator()
        self._circuit_breakers = CircuitBreakerRegistry(config.circuit_breaker)
        self._fallback_handler = FallbackHandler(self._registry, config.fallback)
        self._retry_handler = RetryHandler(config.retry)

        # Phase 2: intelligence
        self._metrics_store = _build_metrics_store(config.metrics, config.region, session)
        self._cache = ResponseCache(config.cache)

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
        self._bedrock = session.client("bedrock-runtime")
        self._shadow._invoke_fn = self._bedrock.converse  # Wire shadow to bedrock
        self._last_decision: RoutingDecision | None = None

    @classmethod
    def create(
        cls,
        config: dict[str, Any] | RouterConfig | None = None,
        *,
        boto_session: Any | None = None,
        callbacks: list[Callable[[RoutingEvent], None]] | None = None,
    ) -> BedrockRouter:
        """Create a router from a dict, RouterConfig, or defaults."""
        if config is None:
            resolved = RouterConfig()
        elif isinstance(config, dict):
            resolved = RouterConfig.from_dict(config)
        else:
            resolved = config
        return cls(resolved, boto_session=boto_session, callbacks=callbacks)

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
        routing = routing or RoutingConfig()
        strategy_name = routing.strategy or self._config.strategy
        weights = routing.weights or self._config.weights
        t_start = time.monotonic()

        # ── Step 1: Pre-route guardrail check ───────────────────
        guardrail_checked = False
        if self._guardrails.has_pre_route:
            result = self._guardrails.check_input(messages)
            guardrail_checked = True
            # If sanitize mode returned cleaned text, we could swap
            # messages here — for now we just let the block/pass through

        # ── Step 2: Analyse the request ─────────────────────────
        analysis = self._analyzer.analyze(messages, system, tool_config)

        # ── Step 3: Determine eligible models ───────────────────
        min_tier = COMPLEXITY_MIN_TIER.get(analysis.complexity.value)
        candidates = self._registry.eligible_models(
            min_tier=min_tier,
            requires_vision=analysis.requires_vision,
            requires_tool_use=analysis.requires_tool_use,
            min_context=routing.min_context_window,
            exclude_patterns=routing.exclude_models or self._config.excluded_models or None,
            family=routing.preferred_family,
        )

        # ── Step 4: Filter by context window ────────────────────
        candidates = self._context_validator.filter_by_context(
            candidates, messages, system
        )
        if not candidates:
            raise RuntimeError(
                "No eligible models found for this request. "
                "Check your routing constraints and model registry."
            )

        # ── Step 5: Filter by circuit breaker ───────────────────
        available = [
            c for c in candidates
            if self._circuit_breakers.is_available(c.model_id)
        ]
        skipped = [c.model_id for c in candidates if c not in available]
        if not available:
            logger.warning("All candidates have open circuit breakers, using all anyway")
            available = candidates
            skipped = []

        # ── Step 6: Run strategy ────────────────────────────────
        strategy = resolve_strategy(
            strategy_name, weights=weights, metrics_store=self._metrics_store,
        )

        # A/B test and canary can override the strategy's model selection
        ab_variant = None
        is_canary = False

        if self._ab_test.is_active:
            user_id = (routing.metadata or {}).get("user_id")
            ab_result = self._ab_test.assign(user_id=user_id)
            if ab_result:
                ab_variant = ab_result.variant_name
                override_model = self._registry.get(ab_result.model_id)
                if override_model and override_model in available:
                    result = strategy.select(available, analysis)
                    primary = override_model  # Override after scoring
                else:
                    result = strategy.select(available, analysis)
                    primary = result.selected_model
            else:
                result = strategy.select(available, analysis)
                primary = result.selected_model
        elif self._canary.is_active:
            canary_model_id, is_canary = self._canary.select_model()
            result = strategy.select(available, analysis)
            override_model = self._registry.get(canary_model_id)
            if is_canary and override_model and override_model in available:
                primary = override_model
            else:
                primary = result.selected_model
                is_canary = False
        else:
            result = strategy.select(available, analysis)
            primary = result.selected_model

        # ── Step 6b: Prompt cache boost ─────────────────────────
        cache_savings = 0.0
        if self._config.prompt_cache_boost and (system or len(messages) > 2):
            benefit = self._cache_advisor.estimate(primary, messages, system)
            cache_savings = benefit.savings_per_request
            # If a cache-capable model wasn't selected but would save
            # significantly, check if any cache-capable candidate scores
            # close enough to swap
            if not benefit.cache_eligible and cache_savings == 0:
                ranked = self._cache_advisor.rank_models_by_cache_benefit(
                    available, messages, system,
                )
                for alt_model, alt_benefit in ranked:
                    if not alt_benefit.cache_eligible:
                        continue
                    alt_score = result.scores.get(alt_model.model_id, {}).get("composite", 0)
                    primary_score = result.scores.get(primary.model_id, {}).get("composite", 0)
                    # Swap if the cache-capable model is within 10% of primary score
                    if alt_score >= primary_score * 0.90:
                        logger.info(
                            "Swapping to %s for prompt cache savings ($%.6f/req)",
                            alt_model.model_id, alt_benefit.savings_per_request,
                        )
                        primary = alt_model
                        cache_savings = alt_benefit.savings_per_request
                        break

        # ── Step 6c: Select CRIS profile ────────────────────────
        cris_profile = self._cris.select_profile(primary)

        # ── Step 6d: Select inference tier ──────────────────────
        inference_tier = self._tier_selector.select_tier(
            primary, analysis,
            max_cost_per_request=routing.max_cost_per_request,
        )

        # ── Step 7: Check response cache ────────────────────────
        cached = self._cache.get(
            primary.model_id, messages, system, inference_config
        )
        if cached is not None:
            decision = RoutingDecision(
                selected_model=primary.model_id,
                strategy_used=strategy_name,
                complexity_detected=analysis.complexity.value,
                complexity_score=analysis.complexity_score,
                candidates_evaluated=len(available),
                candidate_scores=result.scores,
                fallback_chain=[],
                estimated_cost=primary.pricing.estimate_cost(
                    analysis.estimated_input_tokens,
                    analysis.estimated_output_tokens,
                ),
                actual_cost=0.0,
                cache_hit=True,
                inference_tier=inference_tier,
                cris_profile=cris_profile,
                prompt_cache_savings=cache_savings,
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

        # ── Step 8: Build fallback chain ────────────────────────
        fallback_chain = self._fallback_handler.build_chain(
            primary, result.fallback_chain
        )

        # ── Step 9: Invoke with fallbacks ───────────────────────
        models_to_try = [primary] + fallback_chain
        last_error: Exception | None = None
        used_model: BedrockModel | None = None
        response: dict[str, Any] | None = None
        elapsed_ms: float = 0.0
        used_cris: str = cris_profile
        used_tier: str = inference_tier

        for i, model in enumerate(models_to_try):
            if i > 0 and not self._circuit_breakers.is_available(model.model_id):
                continue

            # Resolve CRIS + tier for this specific model
            model_cris = self._cris.select_profile(model) if i > 0 else cris_profile
            model_tier = self._tier_selector.select_tier(model, analysis) if i > 0 else inference_tier

            # Resolve AIP if multi-tenant
            invoke_model_id = self._aip.get_model_id_for_tenant(
                model_cris, routing.metadata or {},
            )

            try:
                t0 = time.monotonic()
                response = self._invoke_bedrock(
                    model_id=invoke_model_id,
                    messages=messages,
                    system=system,
                    tool_config=tool_config,
                    inference_config=inference_config,
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

        # ── Step 10: Post-route guardrail check ─────────────────
        if self._guardrails.has_post_route:
            output_text = self._extract_output_text(response)
            if output_text:
                self._guardrails.check_output(output_text)

        # ── Step 10b: Record canary result ──────────────────────
        if is_canary or self._canary.is_active:
            self._canary.record_result(
                is_canary=is_canary,
                latency_ms=elapsed_ms,
                success=True,
            )

        # ── Step 10c: Shadow mode — mirror to secondary model ───
        if self._shadow.should_shadow():
            self._shadow.mirror(
                primary_model=used_model.model_id,
                messages=messages,
                system=system,
                tool_config=tool_config,
                inference_config=inference_config,
            )

        # ── Step 11: Build routing decision ─────────────────────
        usage = response.get("usage", {})
        input_tokens = usage.get("inputTokens", analysis.estimated_input_tokens)
        output_tokens = usage.get("outputTokens", analysis.estimated_output_tokens)
        actual_cost = used_model.pricing.estimate_cost(input_tokens, output_tokens)

        decision = RoutingDecision(
            selected_model=used_model.model_id,
            strategy_used=strategy_name,
            complexity_detected=analysis.complexity.value,
            complexity_score=analysis.complexity_score,
            candidates_evaluated=len(available),
            candidate_scores=result.scores,
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
            guardrail_checked=guardrail_checked,
            metadata={
                **({"ab_variant": ab_variant} if ab_variant else {}),
                **({"is_canary": is_canary} if is_canary else {}),
            },
        )
        self._last_decision = decision
        response["routing_decision"] = decision

        # ── Step 12: Record metrics ─────────────────────────────
        self._metrics_store.record(RequestRecord(
            model_id=used_model.model_id,
            timestamp=time.monotonic(),
            latency_ms=elapsed_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=actual_cost,
            success=True,
        ))

        # ── Step 13: Cache the response ─────────────────────────
        self._cache.put(
            used_model.model_id, messages, response,
            system, inference_config,
        )

        # ── Step 14: Emit observability event ───────────────────
        most_expensive = max(
            (m.pricing.estimate_cost(input_tokens, output_tokens) for m in available),
            default=0.0,
        )
        self._observability.emit(
            decision,
            duration_ms=(time.monotonic() - t_start) * 1000,
            tags=routing.tags, metadata=routing.metadata,
            most_expensive_cost=most_expensive,
        )

        return response

    # ── Helpers ──────────────────────────────────────────────────

    def _invoke_bedrock(
        self,
        *,
        model_id: str,
        messages: list[dict[str, Any]],
        system: list[dict[str, Any]] | None = None,
        tool_config: dict[str, Any] | None = None,
        inference_config: dict[str, Any] | None = None,
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
