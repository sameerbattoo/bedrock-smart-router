# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenSearch Serverless vector store backend for semantic cache.

Uses Amazon OpenSearch Serverless (AOSS) with k-NN vector search.
Requires ``opensearch-py`` and ``requests-aws4auth``::

    pip install bedrock-smart-router[opensearch]

Authentication uses SigV4 via the caller's boto3 session / IAM role.

Usage::

    from bedrock_smart_router.opensearch_vector_store import OpenSearchVectorStore

    store = OpenSearchVectorStore(
        endpoint="https://abc123.us-west-2.aoss.amazonaws.com",
        index_name="bsr-semantic-cache",
        dimension=1024,
        region="us-west-2",
    )
"""

from __future__ import annotations

import copy
import logging
import time
from typing import Any

from bedrock_smart_router.vector_store import SearchResult, VectorStore

logger = logging.getLogger(__name__)

_INDEX_BODY_TEMPLATE = {
    "settings": {
        "index": {
            "knn": True,
        }
    },
    "mappings": {
        "properties": {
            "embedding": {
                "type": "knn_vector",
                "dimension": 1024,
                "method": {
                    "engine": "faiss",
                    "name": "hnsw",
                    "space_type": "cosinesimil",
                    "parameters": {"ef_construction": 256, "m": 48},
                },
            },
            "payload": {"type": "object", "enabled": True},
            "logical_id": {"type": "keyword"},
            "created_at": {"type": "float"},
        }
    },
}


class OpenSearchVectorStore(VectorStore):
    """OpenSearch Serverless vector store with k-NN search.

    Authenticates via SigV4 using the caller's AWS credentials.
    Auto-creates the index on first use if it doesn't exist.
    """

    def __init__(
        self,
        endpoint: str,
        index_name: str = "bsr-semantic-cache",
        dimension: int = 1024,
        region: str = "us-west-2",
        boto_session: Any | None = None,
    ) -> None:
        try:
            from opensearchpy import OpenSearch, RequestsHttpConnection
            from requests_aws4auth import AWS4Auth
        except ImportError as exc:
            raise ImportError(
                "opensearch-py and requests-aws4auth are required. "
                "Install with: pip install bedrock-smart-router[opensearch]"
            ) from exc

        self._index_name = index_name
        self._dimension = dimension
        # Map our logical IDs to OpenSearch auto-generated _ids
        self._id_map: dict[str, str] = {}

        # Build SigV4 auth
        if boto_session is None:
            import boto3
            boto_session = boto3.Session(region_name=region)
        credentials = boto_session.get_credentials().get_frozen_credentials()
        auth = AWS4Auth(
            credentials.access_key,
            credentials.secret_key,
            region,
            "aoss",
            session_token=credentials.token,
        )

        # Strip trailing slash from endpoint
        endpoint = endpoint.rstrip("/")

        self._client = OpenSearch(
            hosts=[{"host": endpoint.replace("https://", ""), "port": 443}],
            http_auth=auth,
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection,
            timeout=30,
        )

        self._ensure_index()

    def _ensure_index(self) -> None:
        """Create the index if it doesn't exist."""
        try:
            if not self._client.indices.exists(index=self._index_name):
                body = copy.deepcopy(_INDEX_BODY_TEMPLATE)
                body["mappings"]["properties"]["embedding"]["dimension"] = self._dimension
                self._client.indices.create(index=self._index_name, body=body)
                logger.info("Created OpenSearch index '%s'", self._index_name)
                # AOSS needs time to make a new index writable
                time.sleep(10)
            else:
                logger.debug("OpenSearch index '%s' already exists", self._index_name)
        except Exception as exc:
            logger.warning("Failed to ensure index '%s': %s", self._index_name, exc)

    def add(self, id: str, embedding: list[float], payload: dict) -> None:
        """Index a document with its embedding vector."""
        doc = {
            "embedding": embedding,
            "payload": payload,
            "logical_id": id,
            "created_at": time.time(),
        }
        # AOSS doesn't support client-specified document IDs
        try:
            resp = self._client.index(
                index=self._index_name,
                body=doc,
            )
            # Map our logical ID to the auto-generated _id
            aoss_id = resp.get("_id", "")
            self._id_map[id] = aoss_id
        except Exception as exc:
            logger.warning("OpenSearch add failed for '%s': %s", id, exc)
            raise

    def search(
        self,
        query: list[float],
        top_k: int = 5,
        threshold: float = 0.0,
    ) -> list[SearchResult]:
        """k-NN search for similar vectors."""
        body = {
            "size": top_k,
            "query": {
                "knn": {
                    "embedding": {
                        "vector": query,
                        "k": top_k,
                    }
                }
            },
        }

        try:
            resp = self._client.search(index=self._index_name, body=body)
        except Exception as exc:
            logger.warning("OpenSearch search failed: %s", exc)
            return []

        results: list[SearchResult] = []
        for hit in resp.get("hits", {}).get("hits", []):
            score = hit.get("_score", 0.0)
            if score < threshold:
                continue
            source = hit.get("_source", {})
            results.append(SearchResult(
                id=source.get("logical_id", hit["_id"]),
                score=score,
                payload=source.get("payload", {}),
            ))

        return results

    def delete(self, id: str) -> bool:
        """Delete a document by logical ID."""
        aoss_id = self._id_map.get(id)
        if not aoss_id:
            # Try searching by logical_id field
            try:
                resp = self._client.search(
                    index=self._index_name,
                    body={"query": {"term": {"logical_id": id}}},
                )
                hits = resp.get("hits", {}).get("hits", [])
                if not hits:
                    return False
                aoss_id = hits[0]["_id"]
            except Exception:
                return False
        try:
            self._client.delete(index=self._index_name, id=aoss_id)
            self._id_map.pop(id, None)
            return True
        except Exception:
            return False

    def clear(self) -> int:
        """Delete all documents by deleting and recreating the index.

        AOSS does not support _delete_by_query, so we drop the index
        and recreate it.  This takes ~10s for AOSS to propagate.
        """
        try:
            count = self.count()
            self._client.indices.delete(index=self._index_name)
            self._id_map.clear()
            time.sleep(10)
            self._ensure_index()
            return count
        except Exception as exc:
            logger.warning("OpenSearch clear failed: %s", exc)
            return 0

    def count(self) -> int:
        """Return the number of documents in the index."""
        try:
            resp = self._client.count(index=self._index_name)
            return resp.get("count", 0)
        except Exception:
            return 0
