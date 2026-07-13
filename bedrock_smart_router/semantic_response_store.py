# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pluggable response storage backends for the Semantic Cache.

The Semantic Cache separates two concerns:
1. **Vector Store** — stores embeddings + intent + variable hashes for
   similarity search (FAISS, Redis, OpenSearch, in-memory).
2. **Response Store** — stores the actual response payloads.

By default, responses are stored "inline" in the vector store payload.
For large responses (SQL results, charts, full LLM outputs), you can
configure an external response store to keep the vector store lean.

Built-in backends:
- ``InlineResponseStore`` — stores response in vector payload (default)
- ``FilesystemResponseStore`` — stores on local disk / EFS
- ``S3ResponseStore`` — stores in Amazon S3
- ``DynamoDBResponseStore`` — stores in Amazon DynamoDB

Custom backends: subclass ``ResponseStore`` and implement save/load/delete.

Usage
-----
::

    from bedrock_smart_router.semantic_cache import SemanticCache, SemanticCacheConfig
    from bedrock_smart_router.semantic_response_store import (
        FilesystemResponseStore,
        S3ResponseStore,
        DynamoDBResponseStore,
    )

    # Filesystem (dev/testing, Lambda /tmp, EFS)
    cache = SemanticCache(
        config=SemanticCacheConfig(vector_store_backend="faiss"),
        response_store=FilesystemResponseStore(path="/tmp/cache_responses"),
    )

    # S3 (production, large payloads, durability)
    cache = SemanticCache(
        config=SemanticCacheConfig(vector_store_backend="faiss"),
        response_store=S3ResponseStore(bucket="my-cache-bucket", prefix="responses/"),
    )

    # DynamoDB (serverless, low-latency, auto-expiry via TTL)
    cache = SemanticCache(
        config=SemanticCacheConfig(vector_store_backend="faiss"),
        response_store=DynamoDBResponseStore(table_name="cache-responses"),
    )

    # Custom
    class MyStore(ResponseStore):
        def save(self, key, response): ...
        def load(self, reference): ...
        def delete(self, reference): ...

    cache = SemanticCache(response_store=MyStore())
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ResponseStore(ABC):
    """Abstract base class for response storage backends.

    Subclass this to implement custom storage (e.g., DocumentDB, EFS, etc.).

    The contract:
    - ``save(key, response)`` → returns a reference string (used to retrieve later)
    - ``load(reference)`` → returns the response dict, or None if not found/expired
    - ``delete(reference)`` → removes the stored response
    """

    @abstractmethod
    def save(self, key: str, response: dict[str, Any]) -> str:
        """Save a response and return a reference string.

        Args:
            key: A unique key for this cache entry (derived from query hash).
            response: The response payload to store.

        Returns:
            A reference string that can be used to retrieve the response later.
            This reference is stored in the vector store payload.
        """
        ...

    @abstractmethod
    def load(self, reference: str) -> dict[str, Any] | None:
        """Load a response by its reference.

        Args:
            reference: The reference string returned by ``save()``.

        Returns:
            The response dict, or None if not found or expired.
        """
        ...

    @abstractmethod
    def delete(self, reference: str) -> None:
        """Delete a stored response.

        Args:
            reference: The reference string returned by ``save()``.
        """
        ...


class InlineResponseStore(ResponseStore):
    """Stores responses inline in the vector store payload (default behavior).

    No external storage — the response dict is stored directly in the
    vector store entry's payload. Simple and zero-config, but not suitable
    for large responses (>100KB) as it bloats the vector index.
    """

    def save(self, key: str, response: dict[str, Any]) -> str:
        """For inline storage, the 'reference' IS the response (JSON-encoded)."""
        return json.dumps(response, default=str)

    def load(self, reference: str) -> dict[str, Any] | None:
        """Decode the inline JSON response."""
        try:
            return json.loads(reference)
        except (json.JSONDecodeError, TypeError):
            return None

    def delete(self, reference: str) -> None:
        """No-op for inline storage (deleted when vector entry is removed)."""
        pass


class FilesystemResponseStore(ResponseStore):
    """Stores responses as JSON files on the local filesystem.

    Good for:
    - Development and testing
    - Lambda functions (using /tmp, up to 10GB)
    - ECS/EKS with EFS mounts (shared across containers)
    - Single-instance deployments

    Args:
        path: Directory to store response files. Created if it doesn't exist.
    """

    def __init__(self, path: str = "/tmp/semantic_cache_responses") -> None:
        self._dir = Path(path)
        self._dir.mkdir(parents=True, exist_ok=True)
        logger.info("FilesystemResponseStore initialized at %s", self._dir)

    def save(self, key: str, response: dict[str, Any]) -> str:
        """Save response as a JSON file. Returns the filename as reference."""
        filename = f"{key}.json"
        file_path = self._dir / filename
        try:
            file_path.write_text(json.dumps(response, default=str))
            logger.debug("Response saved: %s", filename)
        except Exception as exc:
            logger.warning("Failed to save response to %s: %s", file_path, exc)
        return filename

    def load(self, reference: str) -> dict[str, Any] | None:
        """Load response from a JSON file."""
        file_path = self._dir / reference
        if not file_path.exists():
            logger.debug("Response file not found: %s", reference)
            return None
        try:
            return json.loads(file_path.read_text())
        except Exception as exc:
            logger.warning("Failed to load response from %s: %s", reference, exc)
            return None

    def delete(self, reference: str) -> None:
        """Delete the response file."""
        file_path = self._dir / reference
        try:
            file_path.unlink(missing_ok=True)
        except Exception as exc:
            logger.debug("Failed to delete %s: %s", reference, exc)


class S3ResponseStore(ResponseStore):
    """Stores responses in Amazon S3.

    Good for:
    - Production workloads requiring durability
    - Large responses (up to 5TB per object)
    - Multi-region deployments (with S3 replication)
    - Auto-expiry via S3 lifecycle rules

    Args:
        bucket: S3 bucket name.
        prefix: Key prefix for all cached responses (e.g., "cache/responses/").
        region: AWS region for the S3 client.
        boto_session: Optional boto3 session (uses default credentials if None).

    Tip:
        Configure an S3 lifecycle rule on the prefix to auto-delete old entries:
        ``aws s3api put-bucket-lifecycle-configuration ...``
    """

    def __init__(
        self,
        bucket: str,
        prefix: str = "semantic_cache/",
        region: str = "us-west-2",
        boto_session: Any | None = None,
    ) -> None:
        self._bucket = bucket
        self._prefix = prefix.rstrip("/") + "/" if prefix else ""
        self._region = region
        self._session = boto_session
        self._client: Any | None = None
        logger.info("S3ResponseStore initialized: s3://%s/%s", bucket, self._prefix)

    def _get_client(self) -> Any:
        if self._client is None:
            if self._session is None:
                import boto3
                self._session = boto3.Session(region_name=self._region)
            self._client = self._session.client("s3", region_name=self._region)
        return self._client

    def save(self, key: str, response: dict[str, Any]) -> str:
        """Save response to S3. Returns the S3 key as reference."""
        s3_key = f"{self._prefix}{key}.json"
        try:
            self._get_client().put_object(
                Bucket=self._bucket,
                Key=s3_key,
                Body=json.dumps(response, default=str).encode("utf-8"),
                ContentType="application/json",
            )
            logger.debug("Response saved to s3://%s/%s", self._bucket, s3_key)
        except Exception as exc:
            logger.warning("Failed to save response to S3: %s", exc)
        return s3_key

    def load(self, reference: str) -> dict[str, Any] | None:
        """Load response from S3."""
        try:
            resp = self._get_client().get_object(
                Bucket=self._bucket, Key=reference,
            )
            body = resp["Body"].read().decode("utf-8")
            return json.loads(body)
        except self._get_client().exceptions.NoSuchKey:
            logger.debug("S3 response not found: %s", reference)
            return None
        except Exception as exc:
            logger.warning("Failed to load response from S3: %s", exc)
            return None

    def delete(self, reference: str) -> None:
        """Delete response from S3."""
        try:
            self._get_client().delete_object(
                Bucket=self._bucket, Key=reference,
            )
        except Exception as exc:
            logger.debug("Failed to delete S3 object %s: %s", reference, exc)


class DynamoDBResponseStore(ResponseStore):
    """Stores responses in Amazon DynamoDB.

    Good for:
    - Serverless production workloads
    - Low-latency key-value lookups (<10ms)
    - Auto-scaling with on-demand capacity
    - Built-in TTL for automatic expiry

    The table schema:
    - Partition key: ``cache_key`` (String)
    - Attributes: ``response`` (String, JSON), ``created_at`` (Number),
      ``ttl`` (Number, for DynamoDB TTL)

    Args:
        table_name: DynamoDB table name.
        region: AWS region.
        ttl_seconds: Time-to-live for entries (default 3600s = 1 hour).
            Set to 0 to disable TTL. Requires DynamoDB TTL enabled on
            the ``ttl`` attribute.
        boto_session: Optional boto3 session.

    Note:
        The table must exist. Create it with::

            aws dynamodb create-table \\
                --table-name cache-responses \\
                --attribute-definitions AttributeName=cache_key,AttributeType=S \\
                --key-schema AttributeName=cache_key,KeyType=HASH \\
                --billing-mode PAY_PER_REQUEST

            aws dynamodb update-time-to-live \\
                --table-name cache-responses \\
                --time-to-live-specification Enabled=true,AttributeName=ttl
    """

    def __init__(
        self,
        table_name: str,
        region: str = "us-west-2",
        ttl_seconds: int = 3600,
        boto_session: Any | None = None,
    ) -> None:
        self._table_name = table_name
        self._region = region
        self._ttl_seconds = ttl_seconds
        self._session = boto_session
        self._client: Any | None = None
        logger.info(
            "DynamoDBResponseStore initialized: table=%s, ttl=%ds",
            table_name, ttl_seconds,
        )

    def _get_client(self) -> Any:
        if self._client is None:
            if self._session is None:
                import boto3
                self._session = boto3.Session(region_name=self._region)
            self._client = self._session.client("dynamodb", region_name=self._region)
        return self._client

    def save(self, key: str, response: dict[str, Any]) -> str:
        """Save response to DynamoDB. Returns the cache_key as reference."""
        now = int(time.time())
        item: dict[str, Any] = {
            "cache_key": {"S": key},
            "response": {"S": json.dumps(response, default=str)},
            "created_at": {"N": str(now)},
        }
        if self._ttl_seconds > 0:
            item["ttl"] = {"N": str(now + self._ttl_seconds)}

        try:
            self._get_client().put_item(
                TableName=self._table_name, Item=item,
            )
            logger.debug("Response saved to DynamoDB: %s", key)
        except Exception as exc:
            logger.warning("Failed to save response to DynamoDB: %s", exc)
        return key

    def load(self, reference: str) -> dict[str, Any] | None:
        """Load response from DynamoDB."""
        try:
            resp = self._get_client().get_item(
                TableName=self._table_name,
                Key={"cache_key": {"S": reference}},
            )
            item = resp.get("Item")
            if not item:
                return None

            # Check TTL manually (DynamoDB TTL deletion is eventually consistent)
            if self._ttl_seconds > 0:
                ttl_val = int(item.get("ttl", {}).get("N", "0"))
                if ttl_val > 0 and time.time() > ttl_val:
                    logger.debug("DynamoDB entry expired: %s", reference)
                    return None

            response_json = item.get("response", {}).get("S", "")
            return json.loads(response_json)
        except Exception as exc:
            logger.warning("Failed to load response from DynamoDB: %s", exc)
            return None

    def delete(self, reference: str) -> None:
        """Delete response from DynamoDB."""
        try:
            self._get_client().delete_item(
                TableName=self._table_name,
                Key={"cache_key": {"S": reference}},
            )
        except Exception as exc:
            logger.debug("Failed to delete DynamoDB item %s: %s", reference, exc)


# ── Factory ─────────────────────────────────────────────────────────

def build_response_store(
    backend: str = "inline",
    *,
    path: str = "/tmp/semantic_cache_responses",
    s3_bucket: str = "",
    s3_prefix: str = "semantic_cache/",
    dynamodb_table: str = "",
    dynamodb_ttl_seconds: int = 3600,
    region: str = "us-west-2",
    boto_session: Any | None = None,
) -> ResponseStore:
    """Build a ResponseStore from configuration.

    Args:
        backend: One of "inline", "filesystem", "s3", "dynamodb".
        path: Directory path (for filesystem backend).
        s3_bucket: S3 bucket name (for s3 backend).
        s3_prefix: S3 key prefix (for s3 backend).
        dynamodb_table: DynamoDB table name (for dynamodb backend).
        dynamodb_ttl_seconds: TTL for DynamoDB entries.
        region: AWS region (for s3/dynamodb backends).
        boto_session: Optional boto3 session.

    Returns:
        A configured ResponseStore instance.
    """
    if backend == "inline":
        return InlineResponseStore()
    elif backend == "filesystem":
        return FilesystemResponseStore(path=path)
    elif backend == "s3":
        if not s3_bucket:
            raise ValueError("s3_bucket is required for S3 response store")
        return S3ResponseStore(
            bucket=s3_bucket, prefix=s3_prefix,
            region=region, boto_session=boto_session,
        )
    elif backend == "dynamodb":
        if not dynamodb_table:
            raise ValueError("dynamodb_table is required for DynamoDB response store")
        return DynamoDBResponseStore(
            table_name=dynamodb_table, region=region,
            ttl_seconds=dynamodb_ttl_seconds, boto_session=boto_session,
        )
    else:
        raise ValueError(
            f"Unknown response store backend: '{backend}'. "
            f"Available: inline, filesystem, s3, dynamodb"
        )
