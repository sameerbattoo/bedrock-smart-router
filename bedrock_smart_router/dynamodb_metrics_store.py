"""DynamoDB-backed metrics store.

Persists per-request records to a DynamoDB table with a TTL attribute
so old records are automatically cleaned up.  Supports shared state
across Lambda invocations and multiple instances.

Table schema::

    PK (partition key) : model_id  (S)
    SK (sort key)      : timestamp (N)  — epoch seconds
    TTL attribute      : expires_at (N) — epoch seconds

The table is auto-created on first use if ``auto_create_table=True``.

IAM permissions required
~~~~~~~~~~~~~~~~~~~~~~~~

The IAM role or user running the router needs the following permissions.
Two policy options are provided depending on whether you let the router
create the table automatically or provision it yourself.

**Option A — auto_create_table=True (default, recommended for dev)**

The router creates the table on first use and enables TTL.  Your role
needs::

    {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "BedrockRouterMetricsDynamoDB",
                "Effect": "Allow",
                "Action": [
                    "dynamodb:CreateTable",
                    "dynamodb:DescribeTable",
                    "dynamodb:UpdateTimeToLive",
                    "dynamodb:ListTables",
                    "dynamodb:PutItem",
                    "dynamodb:Query",
                    "dynamodb:Scan"
                ],
                "Resource": "arn:aws:dynamodb:*:*:table/BedrockSmartRouterMetrics"
            }
        ]
    }

Replace ``BedrockSmartRouterMetrics`` with your ``table_name`` if
customised.  Use ``arn:aws:dynamodb:<region>:<account>:table/<name>``
to scope to a specific region and account.

**Option B — auto_create_table=False (recommended for production)**

You create the table yourself (via CloudFormation, CDK, Terraform, or
console) and the router only reads and writes::

    {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "BedrockRouterMetricsDynamoDB",
                "Effect": "Allow",
                "Action": [
                    "dynamodb:PutItem",
                    "dynamodb:Query",
                    "dynamodb:Scan"
                ],
                "Resource": "arn:aws:dynamodb:*:*:table/BedrockSmartRouterMetrics"
            }
        ]
    }

CloudFormation snippet for pre-provisioning the table::

    BedrockRouterMetricsTable:
      Type: AWS::DynamoDB::Table
      Properties:
        TableName: BedrockSmartRouterMetrics
        BillingMode: PAY_PER_REQUEST
        KeySchema:
          - AttributeName: model_id
            KeyType: HASH
          - AttributeName: timestamp
            KeyType: RANGE
        AttributeDefinitions:
          - AttributeName: model_id
            AttributeType: S
          - AttributeName: timestamp
            AttributeType: N
        TimeToLiveSpecification:
          AttributeName: expires_at
          Enabled: true
"""

from __future__ import annotations

import logging
import time
from decimal import Decimal
from typing import Any

from bedrock_smart_router.metrics_store import (
    MetricsStore,
    ModelMetrics,
    RequestRecord,
    _percentile,
)

logger = logging.getLogger(__name__)

_DEFAULT_TABLE_NAME = "BedrockSmartRouterMetrics"
_DEFAULT_TTL_HOURS = 168  # 7 days


class DynamoDBMetricsStore(MetricsStore):
    """Persistent metrics store backed by Amazon DynamoDB.

    Each :meth:`record` call writes a single item.  :meth:`get_metrics`
    queries by ``model_id`` (partition key) with a timestamp range
    filter.

    Args:
        table_name: DynamoDB table name.
        ttl_hours: How long records are kept before DynamoDB TTL
            deletes them.
        boto_session: Optional ``boto3.Session``.  When *None* the
            default session is used.
        region: AWS region for the DynamoDB client.
        auto_create_table: Create the table on first use if it does
            not exist.
    """

    def __init__(
        self,
        table_name: str = _DEFAULT_TABLE_NAME,
        ttl_hours: int = _DEFAULT_TTL_HOURS,
        boto_session: Any | None = None,
        region: str = "us-west-2",
        auto_create_table: bool = True,
        metrics_cache_ttl: float = 60.0,
    ) -> None:
        self._table_name = table_name
        self._ttl_seconds = ttl_hours * 3600
        self._region = region
        self._auto_create = auto_create_table
        self._metrics_cache_ttl = metrics_cache_ttl

        import boto3
        session = boto_session or boto3.Session(region_name=region)
        self._dynamodb = session.resource("dynamodb", region_name=region)
        self._table: Any | None = None

        # Per-model metrics cache: model_id → (ModelMetrics, expiry_time)
        self._metrics_cache: dict[str, tuple[ModelMetrics, float]] = {}
        # Track model IDs seen via record() for get_all_metrics()
        self._known_model_ids: set[str] = set()

    # ── Table management ────────────────────────────────────────

    def _get_table(self) -> Any:
        """Lazy-initialise the DynamoDB Table resource."""
        if self._table is not None:
            return self._table

        if self._auto_create:
            self._ensure_table_exists()

        self._table = self._dynamodb.Table(self._table_name)
        return self._table

    def _ensure_table_exists(self) -> None:
        """Create the table if it does not already exist."""
        client = self._dynamodb.meta.client
        try:
            client.describe_table(TableName=self._table_name)
            return  # Table exists
        except client.exceptions.ResourceNotFoundException:
            pass  # Table doesn't exist — create it below

        logger.info("Creating DynamoDB table %s", self._table_name)
        client.create_table(
            TableName=self._table_name,
            KeySchema=[
                {"AttributeName": "model_id", "KeyType": "HASH"},
                {"AttributeName": "timestamp", "KeyType": "RANGE"},
            ],
            AttributeDefinitions=[
                {"AttributeName": "model_id", "AttributeType": "S"},
                {"AttributeName": "timestamp", "AttributeType": "N"},
            ],
            BillingMode="PAY_PER_REQUEST",
        )

        # Wait for table to become active
        waiter = client.get_waiter("table_exists")
        waiter.wait(
            TableName=self._table_name,
            WaiterConfig={"Delay": 1, "MaxAttempts": 30},
        )

        # Enable TTL
        try:
            client.update_time_to_live(
                TableName=self._table_name,
                TimeToLiveSpecification={
                    "Enabled": True,
                    "AttributeName": "expires_at",
                },
            )
        except Exception as exc:
            logger.warning("Could not enable TTL: %s", exc)

        logger.info("DynamoDB table %s created", self._table_name)

    # ── MetricsStore interface ──────────────────────────────────

    def record(self, rec: RequestRecord) -> None:
        """Write a single request record to DynamoDB."""
        self._known_model_ids.add(rec.model_id)
        table = self._get_table()
        now_epoch = Decimal(str(time.time()))
        expires = Decimal(str(time.time() + self._ttl_seconds))

        item: dict[str, Any] = {
            "model_id": rec.model_id,
            "timestamp": now_epoch,
            "expires_at": expires,
            "latency_ms": Decimal(str(rec.latency_ms)),
            "ttft_ms": Decimal(str(rec.ttft_ms)),
            "input_tokens": rec.input_tokens,
            "output_tokens": rec.output_tokens,
            "cost": Decimal(str(rec.cost)),
            "success": rec.success,
            "is_throttle": rec.is_throttle,
            # Routing context — enriches items for analytics/GSI queries
            "strategy": rec.strategy,
            "complexity": rec.complexity,
            "inference_tier": rec.inference_tier,
            "fallback_used": rec.fallback_used,
            "cache_hit": rec.cache_hit,
        }
        if rec.quality_score is not None:
            item["quality_score"] = Decimal(str(rec.quality_score))
        if rec.tenant_id:
            item["tenant_id"] = rec.tenant_id
        if rec.cris_profile:
            item["cris_profile"] = rec.cris_profile
        if rec.prompt_cache_read_tokens > 0:
            item["prompt_cache_read_tokens"] = rec.prompt_cache_read_tokens
        if rec.prompt_cache_write_tokens > 0:
            item["prompt_cache_write_tokens"] = rec.prompt_cache_write_tokens

        table.put_item(Item=item)

    def get_metrics(
        self, model_id: str, window_seconds: float = 3600.0
    ) -> ModelMetrics:
        """Query records for a model within the time window and aggregate.

        Results are cached per model for ``metrics_cache_ttl`` seconds
        (default 60s) to avoid repeated DynamoDB queries on every request.
        """
        now = time.time()
        cached = self._metrics_cache.get(model_id)
        if cached is not None:
            metrics, expiry = cached
            if now < expiry:
                return metrics

        records = self._query_records(model_id, window_seconds)
        metrics = self._aggregate(model_id, records, window_seconds)
        self._metrics_cache[model_id] = (metrics, now + self._metrics_cache_ttl)
        return metrics

    def get_all_metrics(
        self, window_seconds: float = 3600.0
    ) -> dict[str, ModelMetrics]:
        """Get aggregated metrics for all known models.

        Uses per-model DynamoDB Queries (partition key lookup) with
        in-memory caching instead of a full table scan.  Each model's
        metrics are cached for ``metrics_cache_ttl`` seconds.

        The method queries all model IDs that have been recorded via
        :meth:`record`.  To also include models that haven't been
        seen yet, callers can pass a list of model IDs to pre-populate.
        """
        # Collect all model IDs we know about from the cache + any
        # previously recorded models
        model_ids = set(self._metrics_cache.keys()) | self._known_model_ids
        result: dict[str, ModelMetrics] = {}
        for model_id in model_ids:
            metrics = self.get_metrics(model_id, window_seconds)
            if metrics.sample_count > 0:
                result[model_id] = metrics
        return result

    # ── Internal helpers ────────────────────────────────────────

    def _query_records(
        self, model_id: str, window_seconds: float
    ) -> list[RequestRecord]:
        """Query DynamoDB for records of a single model within the window."""
        from boto3.dynamodb.conditions import Key

        table = self._get_table()
        cutoff = Decimal(str(time.time() - window_seconds))

        items: list[dict[str, Any]] = []
        query_kwargs: dict[str, Any] = {
            "KeyConditionExpression": (
                Key("model_id").eq(model_id)
                & Key("timestamp").gte(cutoff)
            ),
        }

        while True:
            resp = table.query(**query_kwargs)
            items.extend(resp.get("Items", []))
            last_key = resp.get("LastEvaluatedKey")
            if not last_key:
                break
            query_kwargs["ExclusiveStartKey"] = last_key

        return [self._item_to_record(item) for item in items]

    @staticmethod
    def _item_to_record(item: dict[str, Any]) -> RequestRecord:
        """Convert a DynamoDB item back to a RequestRecord."""
        qs = item.get("quality_score")
        return RequestRecord(
            model_id=item["model_id"],
            timestamp=float(item["timestamp"]),
            latency_ms=float(item.get("latency_ms", 0)),
            ttft_ms=float(item.get("ttft_ms", 0)),
            input_tokens=int(item.get("input_tokens", 0)),
            output_tokens=int(item.get("output_tokens", 0)),
            cost=float(item.get("cost", 0)),
            quality_score=float(qs) if qs is not None else None,
            success=item.get("success", True),
            is_throttle=item.get("is_throttle", False),
            strategy=item.get("strategy", ""),
            complexity=item.get("complexity", ""),
            tenant_id=item.get("tenant_id", ""),
            inference_tier=item.get("inference_tier", ""),
            cris_profile=item.get("cris_profile", ""),
            fallback_used=item.get("fallback_used", False),
            cache_hit=item.get("cache_hit", False),
            prompt_cache_read_tokens=int(item.get("prompt_cache_read_tokens", 0)),
            prompt_cache_write_tokens=int(item.get("prompt_cache_write_tokens", 0)),
        )

    @staticmethod
    def _aggregate(
        model_id: str,
        records: list[RequestRecord],
        window: float,
    ) -> ModelMetrics:
        """Aggregate a list of records into ModelMetrics.

        Reuses the same aggregation logic as InMemoryMetricsStore.
        """
        if not records:
            return ModelMetrics(model_id=model_id, window_seconds=window)

        n = len(records)
        latencies = sorted(r.latency_ms for r in records if r.success)
        ttfts = [r.ttft_ms for r in records if r.success and r.ttft_ms > 0]
        costs = [r.cost for r in records]
        quality_scores = [
            r.quality_score for r in records if r.quality_score is not None
        ]
        errors = sum(1 for r in records if not r.success)
        throttles = sum(1 for r in records if r.is_throttle)

        return ModelMetrics(
            model_id=model_id,
            window_seconds=window,
            avg_latency_ms=sum(latencies) / max(1, len(latencies)),
            p50_latency_ms=_percentile(latencies, 50),
            p95_latency_ms=_percentile(latencies, 95),
            avg_ttft_ms=sum(ttfts) / max(1, len(ttfts)) if ttfts else 0.0,
            avg_cost_per_request=sum(costs) / n if costs else 0.0,
            error_rate=errors / n,
            throttle_rate=throttles / n,
            avg_quality_score=(
                sum(quality_scores) / len(quality_scores)
                if quality_scores
                else None
            ),
            sample_count=n,
        )
