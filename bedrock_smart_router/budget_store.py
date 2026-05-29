"""Pluggable persistent budget store backends.

The BudgetTracker uses an in-memory deque for fast budget checks on
every request.  When a persistent store is configured, spend records
are flushed to the store asynchronously and loaded on startup for
recovery after restarts.

Built-in backends:
  - ``sqlite`` — local file, zero dependencies, single-instance.
  - ``dynamodb`` — shared across instances, auto-TTL, serverless.

Custom backends: subclass ``BudgetStore`` and implement the 4 methods.
For example, a Postgres/RDS backend::

    class PostgresBudgetStore(BudgetStore):
        def __init__(self, connection_string: str):
            ...
        def write_batch(self, records): ...
        def get_spend(self, scope, window_seconds): ...
        def get_all_spend(self, window_seconds): ...
        def cleanup(self, older_than_seconds): ...

Usage::

    from bedrock_smart_router.budget_store import SQLiteBudgetStore

    store = SQLiteBudgetStore(path="/tmp/bsr_budget.db")
    tracker = BudgetTracker(store=store)

Or via config::

    router = BedrockRouter.create({
        "budget": {
            "tracker_backend": "sqlite",
            "sqlite_path": "/tmp/bsr_budget.db",
            "scope_key": "user_id",
            "rule_key": "tier",
            "rules": {
                "default": {"max_hourly_spend": 1.0, "on_exceeded": "downgrade"},
                "free": {"max_daily_spend": 0.10, "on_exceeded": "reject"},
            },
        },
    })
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class SpendRecord:
    """A single spend record to be persisted."""

    scope: str
    cost: float
    timestamp: float
    model_id: str = ""
    metadata: dict[str, str] | None = None


class BudgetStore(ABC):
    """Abstract interface for persistent budget storage.

    Implement this to add a custom backend (e.g., Postgres, Redis).
    The store is responsible for:
    - Writing spend records (batched, async from the hot path)
    - Querying total spend for a scope within a time window
    - Cleaning up old records beyond the retention window

    The BudgetTracker handles in-memory caching and sync scheduling.
    The store only needs to handle persistence.
    """

    @abstractmethod
    def write_batch(self, records: list[SpendRecord]) -> None:
        """Write a batch of spend records to the store.

        Called periodically by the sync thread (not on every request).
        Must be idempotent — duplicate writes should be safe.
        """
        ...

    @abstractmethod
    def get_spend(self, scope: str, window_seconds: float) -> float:
        """Get total spend for a scope within a rolling time window.

        Args:
            scope: The budget scope (e.g., user_id, tenant_id).
            window_seconds: How far back to look (e.g., 3600 for hourly).

        Returns:
            Total cost in the window.
        """
        ...

    @abstractmethod
    def get_all_spend(self, window_seconds: float) -> dict[str, float]:
        """Get total spend for ALL scopes within a time window.

        Used on startup to hydrate the in-memory tracker.

        Returns:
            Dict mapping scope → total cost in the window.
        """
        ...

    @abstractmethod
    def cleanup(self, older_than_seconds: float) -> int:
        """Delete records older than the given window.

        Called periodically to prevent unbounded storage growth.

        Returns:
            Number of records deleted.
        """
        ...


# ── SQLite Backend ──────────────────────────────────────────────────

class SQLiteBudgetStore(BudgetStore):
    """SQLite-backed budget store. Single-instance, zero dependencies.

    Auto-creates the table and indexes on first use. The database file
    is created at the specified path (default: /tmp/bsr_budget.db).

    Schema::

        CREATE TABLE budget_spend (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            scope TEXT NOT NULL,
            cost REAL NOT NULL,
            timestamp REAL NOT NULL,
            model_id TEXT DEFAULT '',
            metadata TEXT DEFAULT ''
        );
        CREATE INDEX idx_budget_scope_time ON budget_spend(scope, timestamp);
    """

    def __init__(self, path: str = "/tmp/bsr_budget.db") -> None:
        import sqlite3
        import threading

        self._path = path
        self._lock = threading.Lock()
        self._init_db()
        logger.info("SQLiteBudgetStore initialized: %s", path)

    def _get_conn(self):
        import sqlite3
        conn = sqlite3.connect(self._path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self) -> None:
        """Create table and indexes if they don't exist."""
        conn = self._get_conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS budget_spend (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scope TEXT NOT NULL,
                cost REAL NOT NULL,
                timestamp REAL NOT NULL,
                model_id TEXT DEFAULT '',
                metadata TEXT DEFAULT ''
            );
            CREATE INDEX IF NOT EXISTS idx_budget_scope_time
                ON budget_spend(scope, timestamp);
        """)
        conn.close()

    def write_batch(self, records: list[SpendRecord]) -> None:
        if not records:
            return
        with self._lock:
            conn = self._get_conn()
            conn.executemany(
                "INSERT INTO budget_spend (scope, cost, timestamp, model_id, metadata) "
                "VALUES (?, ?, ?, ?, ?)",
                [
                    (r.scope, r.cost, r.timestamp, r.model_id,
                     _serialize_metadata(r.metadata))
                    for r in records
                ],
            )
            conn.commit()
            conn.close()

    def get_spend(self, scope: str, window_seconds: float) -> float:
        cutoff = time.time() - window_seconds
        conn = self._get_conn()
        row = conn.execute(
            "SELECT COALESCE(SUM(cost), 0) as total "
            "FROM budget_spend WHERE scope = ? AND timestamp >= ?",
            (scope, cutoff),
        ).fetchone()
        conn.close()
        return row["total"] if row else 0.0

    def get_all_spend(self, window_seconds: float) -> dict[str, float]:
        cutoff = time.time() - window_seconds
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT scope, COALESCE(SUM(cost), 0) as total "
            "FROM budget_spend WHERE timestamp >= ? GROUP BY scope",
            (cutoff,),
        ).fetchall()
        conn.close()
        return {r["scope"]: r["total"] for r in rows}

    def cleanup(self, older_than_seconds: float) -> int:
        cutoff = time.time() - older_than_seconds
        with self._lock:
            conn = self._get_conn()
            cursor = conn.execute(
                "DELETE FROM budget_spend WHERE timestamp < ?", (cutoff,)
            )
            count = cursor.rowcount
            conn.commit()
            conn.close()
        if count > 0:
            logger.debug("SQLiteBudgetStore: cleaned up %d old records", count)
        return count


# ── DynamoDB Backend ────────────────────────────────────────────────

class DynamoDBBudgetStore(BudgetStore):
    """DynamoDB-backed budget store. Multi-instance, serverless, auto-TTL.

    Table schema:
      - Partition key: ``scope`` (String)
      - Sort key: ``timestamp`` (Number, epoch seconds)
      - Attributes: ``cost`` (Number), ``model_id`` (String), ``ttl`` (Number)

    The ``ttl`` attribute enables DynamoDB's built-in TTL for automatic
    cleanup (no need to call cleanup() manually).

    Args:
        table_name: DynamoDB table name.
        region: AWS region.
        ttl_seconds: Records expire after this many seconds (default 86400 = 24h).
        auto_create_table: If True, creates the table if it doesn't exist.
            Default False — requires the table to exist (see error message for
            the exact create command).
        boto_session: Optional boto3 session.
    """

    def __init__(
        self,
        table_name: str = "bsr-budget-tracking",
        region: str = "us-west-2",
        ttl_seconds: int = 86400,
        auto_create_table: bool = False,
        boto_session: Any | None = None,
    ) -> None:
        self._table_name = table_name
        self._region = region
        self._ttl_seconds = ttl_seconds
        self._session = boto_session
        self._client: Any | None = None

        # Validate or create table
        client = self._get_client()
        if not self._table_exists(client):
            if auto_create_table:
                self._create_table(client)
            else:
                raise BudgetStoreError(
                    f"DynamoDB table '{table_name}' not found. Create it with:\n\n"
                    f"  aws dynamodb create-table \\\n"
                    f"    --table-name {table_name} \\\n"
                    f"    --attribute-definitions \\\n"
                    f"      AttributeName=scope,AttributeType=S \\\n"
                    f"      AttributeName=timestamp,AttributeType=N \\\n"
                    f"    --key-schema \\\n"
                    f"      AttributeName=scope,KeyType=HASH \\\n"
                    f"      AttributeName=timestamp,KeyType=RANGE \\\n"
                    f"    --billing-mode PAY_PER_REQUEST\n\n"
                    f"  Then enable TTL:\n"
                    f"  aws dynamodb update-time-to-live \\\n"
                    f"    --table-name {table_name} \\\n"
                    f"    --time-to-live-specification Enabled=true,AttributeName=ttl\n\n"
                    f"  Or set auto_create_table: True in your budget config."
                )

        logger.info(
            "DynamoDBBudgetStore initialized: table=%s, ttl=%ds",
            table_name, ttl_seconds,
        )

    def _get_client(self) -> Any:
        if self._client is None:
            if self._session is None:
                import boto3
                self._session = boto3.Session(region_name=self._region)
            self._client = self._session.client("dynamodb", region_name=self._region)
        return self._client

    def _table_exists(self, client: Any) -> bool:
        try:
            client.describe_table(TableName=self._table_name)
            return True
        except client.exceptions.ResourceNotFoundException:
            return False
        except Exception:
            return True  # Assume exists if we can't check (permissions)

    def _create_table(self, client: Any) -> None:
        logger.info("Creating DynamoDB table: %s", self._table_name)
        client.create_table(
            TableName=self._table_name,
            AttributeDefinitions=[
                {"AttributeName": "scope", "AttributeType": "S"},
                {"AttributeName": "timestamp", "AttributeType": "N"},
            ],
            KeySchema=[
                {"AttributeName": "scope", "KeyType": "HASH"},
                {"AttributeName": "timestamp", "KeyType": "RANGE"},
            ],
            BillingMode="PAY_PER_REQUEST",
        )
        # Wait for table to become active
        waiter = client.get_waiter("table_exists")
        waiter.wait(TableName=self._table_name)
        # Enable TTL
        try:
            client.update_time_to_live(
                TableName=self._table_name,
                TimeToLiveSpecification={"Enabled": True, "AttributeName": "ttl"},
            )
        except Exception as e:
            logger.warning("Failed to enable TTL: %s", e)
        logger.info("DynamoDB table created: %s", self._table_name)

    def write_batch(self, records: list[SpendRecord]) -> None:
        if not records:
            return
        client = self._get_client()
        now = time.time()
        # DynamoDB BatchWriteItem supports max 25 items
        for i in range(0, len(records), 25):
            batch = records[i:i + 25]
            items = []
            for r in batch:
                item = {
                    "scope": {"S": r.scope},
                    "timestamp": {"N": str(r.timestamp)},
                    "cost": {"N": str(r.cost)},
                    "model_id": {"S": r.model_id},
                    "ttl": {"N": str(int(now + self._ttl_seconds))},
                }
                if r.metadata:
                    item["metadata"] = {"S": _serialize_metadata(r.metadata)}
                items.append({"PutRequest": {"Item": item}})
            try:
                client.batch_write_item(
                    RequestItems={self._table_name: items}
                )
            except Exception as e:
                logger.warning("DynamoDB batch write failed: %s", e)

    def get_spend(self, scope: str, window_seconds: float) -> float:
        cutoff = time.time() - window_seconds
        client = self._get_client()
        try:
            response = client.query(
                TableName=self._table_name,
                KeyConditionExpression="scope = :s AND #ts >= :cutoff",
                ExpressionAttributeNames={"#ts": "timestamp"},
                ExpressionAttributeValues={
                    ":s": {"S": scope},
                    ":cutoff": {"N": str(cutoff)},
                },
                ProjectionExpression="cost",
            )
            return sum(
                float(item["cost"]["N"])
                for item in response.get("Items", [])
            )
        except Exception as e:
            logger.warning("DynamoDB query failed: %s", e)
            return 0.0

    def get_all_spend(self, window_seconds: float) -> dict[str, float]:
        cutoff = time.time() - window_seconds
        client = self._get_client()
        try:
            # Scan with filter (not ideal for large tables, but budget
            # tables are typically small — one record per request per user)
            response = client.scan(
                TableName=self._table_name,
                FilterExpression="#ts >= :cutoff",
                ExpressionAttributeNames={"#ts": "timestamp"},
                ExpressionAttributeValues={":cutoff": {"N": str(cutoff)}},
                ProjectionExpression="scope, cost",
            )
            spend: dict[str, float] = {}
            for item in response.get("Items", []):
                scope = item["scope"]["S"]
                cost = float(item["cost"]["N"])
                spend[scope] = spend.get(scope, 0.0) + cost
            return spend
        except Exception as e:
            logger.warning("DynamoDB scan failed: %s", e)
            return {}

    def cleanup(self, older_than_seconds: float) -> int:
        # DynamoDB TTL handles cleanup automatically — no-op
        return 0


# ── Errors ──────────────────────────────────────────────────────────

class BudgetStoreError(Exception):
    """Raised when the budget store cannot be initialized."""


# ── Helpers ─────────────────────────────────────────────────────────

def _serialize_metadata(metadata: dict[str, str] | None) -> str:
    if not metadata:
        return ""
    import json
    return json.dumps(metadata, default=str)


# ── Factory ─────────────────────────────────────────────────────────

def build_budget_store(
    backend: str = "memory",
    *,
    sqlite_path: str = "/tmp/bsr_budget.db",
    dynamodb_table: str = "bsr-budget-tracking",
    dynamodb_region: str = "us-west-2",
    dynamodb_ttl_seconds: int = 86400,
    dynamodb_auto_create: bool = False,
    boto_session: Any | None = None,
) -> BudgetStore | None:
    """Build a BudgetStore from configuration.

    Args:
        backend: ``"memory"`` (no persistence), ``"sqlite"``, or ``"dynamodb"``.
        sqlite_path: Path for SQLite database file.
        dynamodb_table: DynamoDB table name.
        dynamodb_region: AWS region for DynamoDB.
        dynamodb_ttl_seconds: TTL for DynamoDB records.
        dynamodb_auto_create: Auto-create DynamoDB table if missing.
        boto_session: Optional boto3 session.

    Returns:
        A BudgetStore instance, or None for in-memory only.
    """
    if backend == "memory":
        return None

    if backend == "sqlite":
        return SQLiteBudgetStore(path=sqlite_path)

    if backend == "dynamodb":
        return DynamoDBBudgetStore(
            table_name=dynamodb_table,
            region=dynamodb_region,
            ttl_seconds=dynamodb_ttl_seconds,
            auto_create_table=dynamodb_auto_create,
            boto_session=boto_session,
        )

    raise ValueError(
        f"Unknown budget store backend: '{backend}'. "
        f"Available: memory, sqlite, dynamodb"
    )
