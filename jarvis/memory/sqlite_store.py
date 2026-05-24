from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import RLock
from typing import Any

from jarvis.memory.models import (
    MemoryImportance,
    MemoryKind,
    MemoryPolicyClassification,
    MemoryQuery,
    MemoryRecord,
    MemoryRetention,
    MemoryRetrievalExplanation,
    MemoryRetrievalResult,
    MemoryScope,
    MemorySearchResult,
    MemorySensitivity,
    MemorySource,
    MemoryWriteRequest,
    utc_now,
)
from jarvis.memory.retrieval import MemoryRetrievalScorer
from jarvis.memory.store import MemoryStoreSnapshot
from jarvis.runtime.observability.structured_logger import get_logger

_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class SQLiteMemoryStoreConfig:
    """
    Configuration for SQLiteMemoryStore.

    WAL mode is enabled by default for safer concurrent read/write behavior.
    """

    path: Path
    name: str = "sqlite_memory_store"
    enable_wal: bool = True
    timeout_seconds: float = 5.0

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("name cannot be empty.")

        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero.")


class SQLiteMemoryStore:
    """
    Persistent SQLite implementation of the MemoryStore contract.

    Responsibilities:
    - persist typed MemoryRecord objects
    - retrieve ranked, explainable memory results
    - preserve strict memory contracts
    - expose diagnostics
    - keep storage replaceable behind MemoryStore

    Non-responsibilities:
    - no MemoryGateway policy
    - no embeddings
    - no vector search
    - no direct cognition access
    - no LLM calls
    """

    def __init__(
        self,
        *,
        config: SQLiteMemoryStoreConfig,
    ) -> None:
        self._config = config
        self._config.validate()

        self._path = self._config.path
        self._path.parent.mkdir(parents=True, exist_ok=True)

        self._lock = RLock()
        self._logger = get_logger("memory.sqlite_store")
        self._scorer = MemoryRetrievalScorer()

        self._write_count = 0
        self._retrieve_count = 0
        self._delete_count = 0
        self._clear_count = 0
        self._last_memory_id: str | None = None
        self._last_query_id: str | None = None
        self._last_error: str | None = None

        self._initialize()

    @property
    def name(self) -> str:
        return self._config.name

    @property
    def path(self) -> Path:
        return self._path

    def write(self, request: MemoryWriteRequest) -> MemoryRecord:
        """
        Convert a write request into a record and persist it.

        Policy approval should happen in MemoryGateway before this method.
        """

        return self.put(request.to_record())

    def put(self, record: MemoryRecord) -> MemoryRecord:
        """
        Persist an already-created memory record.
        """

        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO memory_records (
                    memory_id,
                    kind,
                    scope,
                    text,
                    source,
                    sensitivity,
                    importance,
                    retention,
                    confidence,
                    created_at,
                    updated_at,
                    expires_at,
                    tags_json,
                    metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(memory_id) DO UPDATE SET
                    kind=excluded.kind,
                    scope=excluded.scope,
                    text=excluded.text,
                    source=excluded.source,
                    sensitivity=excluded.sensitivity,
                    importance=excluded.importance,
                    retention=excluded.retention,
                    confidence=excluded.confidence,
                    created_at=excluded.created_at,
                    updated_at=excluded.updated_at,
                    expires_at=excluded.expires_at,
                    tags_json=excluded.tags_json,
                    metadata_json=excluded.metadata_json
                """,
                self._record_to_row(record),
            )
            connection.commit()

            self._write_count += 1
            self._last_memory_id = record.memory_id
            self._last_error = None

        self._logger.info(
            "sqlite_memory_record_stored",
            store=self.name,
            memory_id=record.memory_id,
            kind=record.kind.value,
            scope=record.scope.value,
            sensitivity=record.sensitivity.value,
            importance=record.importance.value,
        )

        return record

    def get(self, memory_id: str) -> MemoryRecord | None:
        """
        Get one memory record by id.
        """

        cleaned = memory_id.strip()

        if not cleaned:
            return None

        with self._lock, self._connect() as connection:
            row = connection.execute(
                """
                SELECT *
                FROM memory_records
                WHERE memory_id = ?
                """,
                (cleaned,),
            ).fetchone()

        if row is None:
            return None

        return self._row_to_record(row)

    def retrieve(self, query: MemoryQuery) -> MemoryRetrievalResult:
        """
        Retrieve ranked records matching the query.

        Every result preserves:
        - source
        - reason
        - confidence
        - timestamp
        - policy classification
        - score breakdown
        """

        with self._lock:
            self._retrieve_count += 1
            self._last_query_id = query.query_id
            self._last_error = None

            records = self.records()

        candidates = tuple(
            record for record in records if self._matches_query(record, query)
        )
        results = tuple(
            sorted(
                (
                    self._to_search_result(record=record, query=query)
                    for record in candidates
                ),
                key=lambda result: result.score,
                reverse=True,
            )[: query.max_results]
        )

        self._logger.info(
            "sqlite_memory_records_retrieved",
            store=self.name,
            query_id=query.query_id,
            query_text=query.text,
            result_count=len(results),
        )

        return MemoryRetrievalResult(
            query=query,
            results=results,
            metadata={
                "store": self.name,
                "path": str(self._path),
            },
        )

    def delete(self, memory_id: str) -> bool:
        """
        Delete one memory record by id.
        """

        cleaned = memory_id.strip()

        if not cleaned:
            return False

        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                """
                DELETE FROM memory_records
                WHERE memory_id = ?
                """,
                (cleaned,),
            )
            connection.commit()

            deleted = cursor.rowcount > 0

            if deleted:
                self._delete_count += 1
                self._last_memory_id = cleaned
                self._last_error = None

        if deleted:
            self._logger.info(
                "sqlite_memory_record_deleted",
                store=self.name,
                memory_id=cleaned,
            )

        return deleted

    def clear(self) -> None:
        """
        Delete all memory records from this SQLite store.
        """

        with self._lock, self._connect() as connection:
            connection.execute("DELETE FROM memory_records")
            connection.commit()

            self._clear_count += 1
            self._last_memory_id = None
            self._last_error = None

        self._logger.info("sqlite_memory_store_cleared", store=self.name)

    def records(self) -> tuple[MemoryRecord, ...]:
        """
        Return all stored records as immutable typed objects.
        """

        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT *
                FROM memory_records
                ORDER BY created_at ASC
                """
            ).fetchall()

        return tuple(self._row_to_record(row) for row in rows)

    def snapshot(self) -> MemoryStoreSnapshot:
        """
        Return store diagnostics.
        """

        records = self.records()
        active_count = sum(1 for record in records if not record.expired())
        expired_count = len(records) - active_count

        with self._lock:
            return MemoryStoreSnapshot(
                name=self.name,
                record_count=len(records),
                active_record_count=active_count,
                expired_record_count=expired_count,
                write_count=self._write_count,
                retrieve_count=self._retrieve_count,
                delete_count=self._delete_count,
                clear_count=self._clear_count,
                last_memory_id=self._last_memory_id,
                last_query_id=self._last_query_id,
                last_error=self._last_error,
            )

    def close(self) -> None:
        """
        Compatibility method for future pooled implementations.

        This store opens short-lived connections per operation, so close is a
        no-op for now.
        """

        self._logger.info("sqlite_memory_store_closed", store=self.name)

    def _initialize(self) -> None:
        with self._lock, self._connect() as connection:
            if self._config.enable_wal:
                connection.execute("PRAGMA journal_mode=WAL")

            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_schema_info (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_records (
                    memory_id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    scope TEXT NOT NULL,
                    text TEXT NOT NULL,
                    source TEXT NOT NULL,
                    sensitivity TEXT NOT NULL,
                    importance TEXT NOT NULL,
                    retention TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    expires_at TEXT,
                    tags_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_memory_records_kind
                ON memory_records(kind)
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_memory_records_scope
                ON memory_records(scope)
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_memory_records_updated_at
                ON memory_records(updated_at)
                """
            )
            connection.execute(
                """
                INSERT INTO memory_schema_info (key, value)
                VALUES ('schema_version', ?)
                ON CONFLICT(key) DO UPDATE SET value=excluded.value
                """,
                (str(_SCHEMA_VERSION),),
            )
            connection.commit()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self._path,
            timeout=self._config.timeout_seconds,
        )
        connection.row_factory = sqlite3.Row
        return connection

    def _matches_query(
        self,
        record: MemoryRecord,
        query: MemoryQuery,
    ) -> bool:
        if not query.include_expired and record.expired():
            return False

        if not query.include_sensitive and (
            record.sensitivity == MemorySensitivity.SENSITIVE
        ):
            return False

        if record.confidence < query.min_confidence:
            return False

        if query.kinds and record.kind not in query.kinds:
            return False

        if query.scopes and record.scope not in query.scopes:
            return False

        if query.tags and not set(query.tags).issubset(set(record.tags)):
            return False

        if query.text is None:
            return True

        breakdown = self._scorer.score(record=record, query=query)

        return breakdown.text_score > 0.0

    def _to_search_result(
        self,
        *,
        record: MemoryRecord,
        query: MemoryQuery,
    ) -> MemorySearchResult:
        breakdown = self._scorer.score(record=record, query=query)
        explanation = self._scorer.explain(
            record=record,
            query=query,
            breakdown=breakdown,
        )

        return MemorySearchResult(
            record=record,
            score=breakdown.final_score,
            explanation=explanation,
            metadata={
                "store": self.name,
                "path": str(self._path),
                "score_breakdown": explanation.metadata["score_breakdown"],
            },
        )

    @staticmethod
    def _record_to_row(record: MemoryRecord) -> tuple[Any, ...]:
        return (
            record.memory_id,
            record.kind.value,
            record.scope.value,
            record.text,
            record.source.value,
            record.sensitivity.value,
            record.importance.value,
            record.retention.value,
            record.confidence,
            record.created_at.isoformat(),
            record.updated_at.isoformat(),
            record.expires_at.isoformat() if record.expires_at else None,
            json.dumps(record.tags),
            json.dumps(record.metadata),
        )

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> MemoryRecord:
        expires_at_raw = row["expires_at"]
        tags_raw = row["tags_json"]
        metadata_raw = row["metadata_json"]

        return MemoryRecord(
            memory_id=row["memory_id"],
            kind=MemoryKind(row["kind"]),
            scope=MemoryScope(row["scope"]),
            text=row["text"],
            source=MemorySource(row["source"]),
            sensitivity=MemorySensitivity(row["sensitivity"]),
            importance=MemoryImportance(row["importance"]),
            retention=MemoryRetention(row["retention"]),
            confidence=float(row["confidence"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            expires_at=(
                datetime.fromisoformat(expires_at_raw)
                if expires_at_raw
                else None
            ),
            tags=tuple(json.loads(tags_raw)),
            metadata=dict(json.loads(metadata_raw)),
        )

    @staticmethod
    def _source_for_record(record: MemoryRecord) -> MemoryRetrievalExplanation:
        return MemoryRetrievalExplanation(
            source=record.source,
            reason="record loaded from sqlite memory store",
            confidence=record.confidence,
            retrieved_at=utc_now(),
            policy_classification=(
                MemoryPolicyClassification.RESTRICTED
                if record.sensitivity == MemorySensitivity.SENSITIVE
                else MemoryPolicyClassification.ALLOWED
            ),
            metadata={
                "memory_id": record.memory_id,
            },
        )