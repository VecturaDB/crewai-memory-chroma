"""Chroma storage backend for the unified memory system of CrewAI.

It keeps the records of a ``Memory`` in a collection of a Chroma server (or of any server with the API of Chroma) through
the Python client of Chroma that CrewAI already installs. It does what ``LanceDBStorage``, the default backend of CrewAI,
does: the same fields, the same score (``1 / (1 + distance)`` with the squared L2 distance of the default space of Chroma),
the same checks of the dimension, the same caps on scans. As ``QdrantEdgeStorage`` does, scopes are matched by path
components (``/crew/research`` matches itself and ``/crew/research/...``, not ``/crew/research-2``), and the categories of a
search are matched by the server before the nearest records are taken.

No filter depends on a list in the metadata being absent: the scope paths are written for every record, and a search by
category also checks the JSON copy of the categories. ``reset()`` deletes the records before the collection.
"""

from __future__ import annotations

from datetime import datetime
import json
import logging
import re
from typing import Any

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection
from chromadb.errors import InvalidArgumentError, NotFoundError

from crewai.memory.storage.backend import EmbeddingDimensionMismatchError
from crewai.memory.types import MemoryRecord, ScopeInfo


_logger = logging.getLogger(__name__)

# Default embedding vector dimensionality, as in LanceDBStorage (OpenAI text-embedding-3-large). Used when the first write
# carries no embedding at all, and for the zero vectors of records without one.
DEFAULT_VECTOR_DIM = 3072

# Safety cap on the number of records read by a single scan, as in LanceDBStorage: scope info, categories, counts in a
# scope and deletes by category or metadata read at most this many.
_SCAN_ROWS_LIMIT = 50_000

# Records read by one request of a scan.
_SCAN_PAGE = 5_000

_DIMENSION_ERROR = re.compile(r"expecting embedding with dimension of (\d+), got (\d+)")


def _scope_ancestors(scope: str) -> list[str]:
    """All the scopes a record is in, as in QdrantEdgeStorage.

    For scope ``/crew/sales/agent``, returns ``["/", "/crew", "/crew/sales", "/crew/sales/agent"]``.
    """
    parts = scope.strip("/").split("/")
    ancestors: list[str] = ["/"]
    current = ""
    for part in parts:
        if part:
            current = f"{current}/{part}"
            ancestors.append(current)
    return ancestors


def _scope_where(scope_prefix: str | None) -> dict[str, Any] | None:
    """The filter of a scope and its subscopes, or None for the root (everything), as in QdrantEdgeStorage."""
    if scope_prefix is None or not scope_prefix.strip("/"):
        return None
    prefix = scope_prefix.rstrip("/")
    if not prefix.startswith("/"):
        prefix = "/" + prefix
    return {"scope_ancestors": {"$contains": prefix}}


def _parse_dt(val: Any) -> datetime:
    if val is None:
        return datetime.utcnow()
    if isinstance(val, datetime):
        return val
    return datetime.fromisoformat(str(val).replace("Z", "+00:00"))


class ChromaStorage:
    """Chroma-backed storage for the unified memory system."""

    def __init__(
        self,
        client: ClientAPI | None = None,
        collection_name: str = "memories",
        vector_dim: int | None = None,
    ) -> None:
        """Initialize Chroma storage.

        Args:
            client: A client of the Python package of Chroma, such as ``chromadb.HttpClient(host=..., port=...)``.
                  Defaults to ``chromadb.HttpClient()``, with the host and port of its own defaults.
            collection_name: Name of the collection for memory records.
            vector_dim: Dimensionality of the embedding vector. When ``None`` (default), the dimension is read from the
                  existing collection or taken from the first saved embedding.
        """
        self._client = client if client is not None else chromadb.HttpClient()
        self._collection_name = collection_name
        self._collection: Collection | None = None
        self._vector_dim = 0  # 0 = not yet known
        if self._open(create=vector_dim is not None) is not None:
            self._vector_dim = self._collection_dim() or vector_dim or 0

    def _open(self, create: bool) -> Collection | None:
        """Return the collection; create it when ``create``, otherwise None if it does not exist."""
        if self._collection is not None:
            return self._collection
        if create:
            self._collection = self._client.get_or_create_collection(self._collection_name, embedding_function=None)
        else:
            try:
                self._collection = self._client.get_collection(self._collection_name, embedding_function=None)
            except NotFoundError:
                return None
        return self._collection

    def _collection_dim(self) -> int:
        """The dimension of the collection, 0 while nothing was ever written to it."""
        assert self._collection is not None
        dim = getattr(getattr(self._collection, "_model", None), "dimension", None)
        if dim:
            return int(dim)
        rows = self._collection.get(limit=1, include=["embeddings"])
        embeddings = rows.get("embeddings")
        return len(embeddings[0]) if embeddings is not None and len(embeddings) > 0 else 0

    def _batch(self) -> int:
        return int(self._client.get_max_batch_size())

    def _record_to_row(self, record: MemoryRecord) -> tuple[list[float], dict[str, Any]]:
        metadata = {
            "scope": record.scope,
            "scope_ancestors": _scope_ancestors(record.scope),
            # The list for the filter of a search; Chroma takes no empty list, so a record without categories has None,
            # which an upsert takes as "remove" and an add leaves out. The categories of a record are read from the JSON.
            "categories": list(record.categories) or None,
            "categories_str": json.dumps(record.categories),
            "metadata_str": json.dumps(record.metadata),
            "importance": record.importance,
            "created_at": record.created_at.isoformat(),
            "last_accessed": record.last_accessed.isoformat(),
            "source": record.source or "",
            "private": record.private,
        }
        # As LanceDBStorage: a record without an embedding, or with one of another size, is kept with a vector of zeros.
        embedding = record.embedding or []
        vector = list(embedding) if len(embedding) == self._vector_dim else [0.0] * self._vector_dim
        return vector, metadata

    @staticmethod
    def _row_to_record(
        record_id: str, document: str | None, metadata: dict[str, Any], embedding: Any = None
    ) -> MemoryRecord:
        return MemoryRecord(
            id=str(record_id),
            content=str(document or ""),
            scope=str(metadata.get("scope", "/")),
            categories=json.loads(metadata["categories_str"]) if metadata.get("categories_str") else [],
            metadata=json.loads(metadata["metadata_str"]) if metadata.get("metadata_str") else {},
            importance=float(metadata.get("importance", 0.5)),
            created_at=_parse_dt(metadata.get("created_at")),
            last_accessed=_parse_dt(metadata.get("last_accessed")),
            embedding=[float(x) for x in embedding] if embedding is not None else None,
            source=metadata.get("source") or None,
            private=bool(metadata.get("private", False)),
        )

    def _write(self, op: str, ids: list[str], embeddings: list[list[float]], documents: list[str],
               metadatas: list[dict[str, Any]]) -> None:
        """Add or upsert in batches of the size the server takes, with a dimension refused as EmbeddingDimensionMismatchError."""
        assert self._collection is not None
        size = self._batch()
        for start in range(0, len(ids), size):
            end = start + size
            try:
                getattr(self._collection, op)(
                    ids=ids[start:end], embeddings=embeddings[start:end], documents=documents[start:end],
                    metadatas=metadatas[start:end],
                )
            except InvalidArgumentError as e:
                found = _DIMENSION_ERROR.search(str(e))
                if found is None:
                    raise
                raise EmbeddingDimensionMismatchError(int(found.group(1)), int(found.group(2))) from e

    def _scan(self, scope_prefix: str | None = None, capped: bool = True) -> list[tuple[str, dict[str, Any]]]:
        """The ids and metadata of the records in a scope: at most ``_SCAN_ROWS_LIMIT`` of them, or all if not ``capped``."""
        if self._open(create=False) is None:
            return []
        assert self._collection is not None
        limit = _SCAN_ROWS_LIMIT if capped else None
        where = _scope_where(scope_prefix)
        out: list[tuple[str, dict[str, Any]]] = []
        while limit is None or len(out) < limit:
            page = _SCAN_PAGE if limit is None else min(_SCAN_PAGE, limit - len(out))
            rows = self._collection.get(where=where, include=["metadatas"], limit=page, offset=len(out))
            out.extend(zip(rows["ids"], [m or {} for m in rows["metadatas"] or []]))
            if len(rows["ids"]) < page:
                break
        return out

    def _delete_ids(self, ids: list[str]) -> int:
        assert self._collection is not None
        size = self._batch()
        for start in range(0, len(ids), size):
            self._collection.delete(ids=ids[start : start + size])
        return len(ids)

    def save(self, records: list[MemoryRecord]) -> None:
        if not records:
            return
        # Auto-detect dimension from the first real embedding and validate the whole batch against it, as
        # LanceDBStorage does: a silent mismatch would otherwise be zero-filled below and corrupt search results.
        dim = None
        for r in records:
            if r.embedding and len(r.embedding) > 0:
                if dim is None:
                    dim = len(r.embedding)
                elif len(r.embedding) != dim:
                    raise EmbeddingDimensionMismatchError(dim, len(r.embedding))
        is_new_collection = self._open(create=False) is None
        if not is_new_collection and dim and self._vector_dim and dim != self._vector_dim:
            raise EmbeddingDimensionMismatchError(self._vector_dim, dim)
        self._open(create=True)
        if is_new_collection or not self._vector_dim:
            self._vector_dim = dim or self._vector_dim or DEFAULT_VECTOR_DIM
        rows = [self._record_to_row(rec) for rec in records]
        metadatas = [{k: v for k, v in m.items() if v is not None} for _, m in rows]
        self._write("add", [r.id for r in records], [v for v, _ in rows], [r.content for r in records], metadatas)

    def update(self, record: MemoryRecord) -> None:
        """Update a record by ID: the stored record is replaced, or added if there is none."""
        if (
            self._open(create=False) is not None
            and record.embedding
            and self._vector_dim
            and len(record.embedding) != self._vector_dim
        ):
            raise EmbeddingDimensionMismatchError(self._vector_dim, len(record.embedding))
        self._open(create=True)
        self._vector_dim = self._vector_dim or DEFAULT_VECTOR_DIM
        # Reject unserializable records before writing anything.
        vector, metadata = self._record_to_row(record)
        self._write("upsert", [record.id], [vector], [record.content], [metadata])

    def touch_records(self, record_ids: list[str]) -> None:
        """Update last_accessed to now for the given record IDs.

        Args:
            record_ids: IDs of records to touch.
        """
        if not record_ids or self._open(create=False) is None:
            return
        assert self._collection is not None
        now = datetime.utcnow().isoformat()
        size = self._batch()
        for start in range(0, len(record_ids), size):
            ids = [str(rid) for rid in record_ids[start : start + size]]
            self._collection.update(ids=ids, metadatas=[{"last_accessed": now} for _ in ids])

    def get_record(self, record_id: str) -> MemoryRecord | None:
        """Return a single record by ID, or None if not found."""
        if self._open(create=False) is None:
            return None
        assert self._collection is not None
        rows = self._collection.get(ids=[str(record_id)], include=["documents", "metadatas", "embeddings"])
        if not rows["ids"]:
            return None
        return self._row_to_record(rows["ids"][0], rows["documents"][0], rows["metadatas"][0] or {}, rows["embeddings"][0])

    def search(
        self,
        query_embedding: list[float],
        scope_prefix: str | None = None,
        categories: list[str] | None = None,
        metadata_filter: dict[str, Any] | None = None,
        limit: int = 10,
        min_score: float = 0.0,
    ) -> list[tuple[MemoryRecord, float]]:
        if self._open(create=False) is None:
            return []
        assert self._collection is not None
        if self._vector_dim and len(query_embedding) != self._vector_dim:
            raise EmbeddingDimensionMismatchError(self._vector_dim, len(query_embedding))
        where = _scope_where(scope_prefix)
        if categories:
            # Match categories on the server, so they are filtered before the nearest records are taken, as
            # QdrantEdgeStorage does. A record without categories is left out by its JSON copy, whatever list its
            # metadata shows.
            any_category = [{"categories": {"$contains": c}} for c in dict.fromkeys(categories)]
            where = {
                "$and": [
                    *([where] if where else []),
                    {"categories_str": {"$ne": "[]"}},
                    any_category[0] if len(any_category) == 1 else {"$or": any_category},
                ]
            }
        n_results = limit * 3 if metadata_filter else limit
        if n_results <= 0:
            return []
        try:
            results = self._collection.query(
                query_embeddings=[list(query_embedding)],
                n_results=n_results,
                where=where,
                include=["documents", "metadatas", "embeddings", "distances"],
            )
        except InvalidArgumentError as e:
            found = _DIMENSION_ERROR.search(str(e))
            if found is None:
                raise
            raise EmbeddingDimensionMismatchError(int(found.group(1)), int(found.group(2))) from e
        out: list[tuple[MemoryRecord, float]] = []
        for record_id, document, metadata, embedding, distance in zip(
            results["ids"][0], results["documents"][0], results["metadatas"][0], results["embeddings"][0],
            results["distances"][0],
        ):
            record = self._row_to_record(record_id, document, metadata or {}, embedding)
            if categories and not any(c in record.categories for c in categories):
                continue
            if metadata_filter and not all(record.metadata.get(k) == v for k, v in metadata_filter.items()):
                continue
            score = 1.0 / (1.0 + float(distance)) if distance is not None else 1.0
            if score >= min_score:
                out.append((record, score))
            if len(out) >= limit:
                break
        return out[:limit]

    def delete(
        self,
        scope_prefix: str | None = None,
        categories: list[str] | None = None,
        record_ids: list[str] | None = None,
        older_than: datetime | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> int:
        if self._open(create=False) is None:
            return 0
        assert self._collection is not None
        if record_ids and not (categories or metadata_filter):
            ids = list(dict.fromkeys(str(rid) for rid in record_ids))
            found = self._collection.get(ids=ids, include=[])["ids"]
            return self._delete_ids(list(found))
        if categories or metadata_filter:
            to_delete: list[str] = []
            for record_id, metadata in self._scan(scope_prefix):
                record = self._row_to_record(record_id, None, metadata)
                if categories and not any(c in record.categories for c in categories):
                    continue
                if metadata_filter and not all(record.metadata.get(k) == v for k, v in metadata_filter.items()):
                    continue
                if older_than and record.created_at >= older_than:
                    continue
                to_delete.append(record.id)
            return self._delete_ids(to_delete)
        rows = self._scan(scope_prefix, capped=False)
        if older_than is not None:
            # As LanceDBStorage, which compares the stored ISO strings.
            cutoff = older_than.isoformat()
            rows = [(rid, m) for rid, m in rows if str(m.get("created_at", "")) < cutoff]
        return self._delete_ids([rid for rid, _ in rows])

    def list_records(self, scope_prefix: str | None = None, limit: int = 200, offset: int = 0) -> list[MemoryRecord]:
        """List records in a scope, newest first.

        Args:
            scope_prefix: Optional scope path prefix to filter by.
            limit: Maximum number of records to return.
            offset: Number of records to skip (for pagination).

        Returns:
            List of MemoryRecord, ordered by created_at descending.
        """
        if self._open(create=False) is None or limit <= 0 or offset < 0:
            return []
        assert self._collection is not None
        rows = self._scan(scope_prefix, capped=False)
        # As LanceDBStorage, which orders by the stored ISO strings.
        rows.sort(key=lambda row: str(row[1].get("created_at", "")), reverse=True)
        page = [rid for rid, _ in rows[offset : offset + limit]]
        found: dict[str, MemoryRecord] = {}
        size = self._batch()
        for start in range(0, len(page), size):
            got = self._collection.get(ids=page[start : start + size], include=["documents", "metadatas", "embeddings"])
            for record_id, document, metadata, embedding in zip(
                got["ids"], got["documents"], got["metadatas"], got["embeddings"]
            ):
                found[record_id] = self._row_to_record(record_id, document, metadata or {}, embedding)
        return [found[rid] for rid in page if rid in found]

    def get_scope_info(self, scope: str) -> ScopeInfo:
        scope = scope.rstrip("/") or "/"
        prefix = scope if scope != "/" else ""
        if prefix and not prefix.startswith("/"):
            prefix = "/" + prefix
        rows = self._scan(prefix or None)
        if not rows:
            return ScopeInfo(
                path=scope or "/",
                record_count=0,
                categories=[],
                oldest_record=None,
                newest_record=None,
                child_scopes=[],
            )
        categories_set: set[str] = set()
        oldest: datetime | None = None
        newest: datetime | None = None
        child_prefix = (prefix + "/") if prefix else "/"
        children: set[str] = set()
        for _, metadata in rows:
            sc = str(metadata.get("scope", ""))
            if child_prefix and sc.startswith(child_prefix):
                rest = sc[len(child_prefix) :]
                first_component = rest.split("/", 1)[0]
                if first_component:
                    children.add(child_prefix + first_component)
            try:
                categories_set.update(json.loads(metadata.get("categories_str") or "[]"))
            except Exception:  # noqa: S110
                pass
            created = metadata.get("created_at")
            if created:
                dt = _parse_dt(created)
                if oldest is None or dt < oldest:
                    oldest = dt
                if newest is None or dt > newest:
                    newest = dt
        return ScopeInfo(
            path=scope or "/",
            record_count=len(rows),
            categories=sorted(categories_set),
            oldest_record=oldest,
            newest_record=newest,
            child_scopes=sorted(children),
        )

    def list_scopes(self, parent: str = "/") -> list[str]:
        parent = parent.rstrip("/") or ""
        prefix = (parent + "/") if parent else "/"
        children: set[str] = set()
        for _, metadata in self._scan(prefix if prefix != "/" else None):
            sc = str(metadata.get("scope", ""))
            if sc.startswith(prefix) and sc != (prefix.rstrip("/") or "/"):
                rest = sc[len(prefix) :]
                first_component = rest.split("/", 1)[0]
                if first_component:
                    children.add(prefix + first_component)
        return sorted(children)

    def list_categories(self, scope_prefix: str | None = None) -> dict[str, int]:
        counts: dict[str, int] = {}
        for _, metadata in self._scan(scope_prefix):
            try:
                parsed = json.loads(metadata.get("categories_str") or "[]")
            except Exception:  # noqa: S112
                continue
            for c in parsed:
                counts[c] = counts.get(c, 0) + 1
        return counts

    def count(self, scope_prefix: str | None = None) -> int:
        if self._open(create=False) is None:
            return 0
        assert self._collection is not None
        if scope_prefix is None or scope_prefix.strip("/") == "":
            return int(self._collection.count())
        return self.get_scope_info(scope_prefix).record_count

    def reset(self, scope_prefix: str | None = None) -> None:
        if scope_prefix is None or scope_prefix.strip("/") == "":
            # The records first, then the collection.
            if self._open(create=False) is not None:
                self._delete_ids([rid for rid, _ in self._scan(None, capped=False)])
            try:
                self._client.delete_collection(self._collection_name)
            except NotFoundError:
                pass
            self._collection = None
            return
        self.delete(scope_prefix=scope_prefix)

    async def asave(self, records: list[MemoryRecord]) -> None:
        self.save(records)

    async def asearch(
        self,
        query_embedding: list[float],
        scope_prefix: str | None = None,
        categories: list[str] | None = None,
        metadata_filter: dict[str, Any] | None = None,
        limit: int = 10,
        min_score: float = 0.0,
    ) -> list[tuple[MemoryRecord, float]]:
        return self.search(
            query_embedding,
            scope_prefix=scope_prefix,
            categories=categories,
            metadata_filter=metadata_filter,
            limit=limit,
            min_score=min_score,
        )

    async def adelete(
        self,
        scope_prefix: str | None = None,
        categories: list[str] | None = None,
        record_ids: list[str] | None = None,
        older_than: datetime | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> int:
        return self.delete(
            scope_prefix=scope_prefix,
            categories=categories,
            record_ids=record_ids,
            older_than=older_than,
            metadata_filter=metadata_filter,
        )
