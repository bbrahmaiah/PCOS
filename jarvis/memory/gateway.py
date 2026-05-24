from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from threading import RLock
from typing import Protocol, runtime_checkable

from pydantic import Field, field_validator

from jarvis.memory.models import (
    MemoryModel,
    MemoryPolicyClassification,
    MemoryQuery,
    MemoryRecord,
    MemoryRetrievalResult,
    MemorySearchResult,
    MemoryWriteRequest,
    utc_now,
)
from jarvis.memory.store import MemoryStore, MemoryStoreSnapshot
from jarvis.memory.write_policy import (
    MemoryWritePolicy,
    MemoryWritePolicyConfig,
    MemoryWritePolicySnapshot,
)
from jarvis.runtime.observability.structured_logger import get_logger


class MemoryGatewayWriteResult(MemoryModel):
    """
    Result of one governed memory write request.

    This is returned by the gateway, not the raw store, so cognition can see
    whether the memory write was allowed, blocked, or transformed by policy.
    """

    request: MemoryWriteRequest
    record: MemoryRecord | None = None
    allowed: bool
    blocked: bool = False
    reason: str
    policy_classification: MemoryPolicyClassification
    decided_at: datetime = Field(default_factory=utc_now)

    @field_validator("reason")
    @classmethod
    def _reason_required(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("reason cannot be empty.")

        return cleaned


class MemoryGatewayRetrievalResult(MemoryModel):
    """
    Result of one governed memory retrieval request.

    This is the object cognition should receive from memory. It preserves the
    underlying MemoryRetrievalResult but adds gateway-level policy decision data.
    """

    query: MemoryQuery
    retrieval: MemoryRetrievalResult
    allowed: bool
    blocked: bool = False
    reason: str
    policy_classification: MemoryPolicyClassification
    decided_at: datetime = Field(default_factory=utc_now)

    @field_validator("reason")
    @classmethod
    def _reason_required(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("reason cannot be empty.")

        return cleaned

    @property
    def result_count(self) -> int:
        return self.retrieval.result_count

    @property
    def results(self) -> tuple[MemorySearchResult, ...]:
        return self.retrieval.results

    @property
    def records(self) -> tuple[MemoryRecord, ...]:
        return self.retrieval.records


@dataclass(frozen=True, slots=True)
class MemoryGatewayConfig:
    """
    Configuration for GovernedMemoryGateway.

    Defaults are conservative:
    - sensitive writes are blocked through MemoryWritePolicy
    - sensitive retrieval is filtered unless explicitly enabled
    - delete is allowed through gateway
    - clear is blocked by default
    """

    name: str = "memory_gateway"
    allow_sensitive_writes: bool = False
    allow_sensitive_retrieval: bool = False
    allow_delete: bool = True
    allow_clear: bool = False

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("name cannot be empty.")


@dataclass(frozen=True, slots=True)
class MemoryGatewaySnapshot:
    """
    Observable diagnostics for the Memory Gateway.
    """

    name: str
    write_count: int
    write_allowed_count: int
    write_blocked_count: int
    retrieve_count: int
    retrieve_allowed_count: int
    retrieve_blocked_count: int
    delete_count: int
    delete_blocked_count: int
    clear_count: int
    clear_blocked_count: int
    last_memory_id: str | None
    last_query_id: str | None
    last_error: str | None
    store_snapshot: MemoryStoreSnapshot
    write_policy_snapshot: MemoryWritePolicySnapshot


@runtime_checkable
class MemoryGateway(Protocol):
    """
    Only approved cognition-facing entry point into memory.

    Cognition Runtime should depend on this protocol, not MemoryStore.
    """

    @property
    def name(self) -> str:
        """Stable gateway name."""

    def remember(self, request: MemoryWriteRequest) -> MemoryGatewayWriteResult:
        """Govern and write one memory request."""

    def retrieve(self, query: MemoryQuery) -> MemoryGatewayRetrievalResult:
        """Govern and retrieve explainable memory results."""

    def get(self, memory_id: str) -> MemoryRecord | None:
        """Get one record by id through gateway."""

    def delete(self, memory_id: str) -> bool:
        """Delete one record by id through gateway."""

    def clear(self) -> None:
        """Clear memory through gateway when allowed."""

    def snapshot(self) -> MemoryGatewaySnapshot:
        """Return gateway diagnostics."""


class GovernedMemoryGateway:
    """
    Policy-controlled gateway in front of MemoryStore.

    Responsibilities:
    - be the only cognition-facing memory entry point
    - govern writes through MemoryWritePolicy
    - govern retrieval before reaching store
    - preserve explainable retrieval results
    - keep diagnostics and audit-friendly counters
    - prevent direct sensitive retrieval unless enabled

    Non-responsibilities:
    - no embeddings
    - no persistence implementation
    - no vector search
    - no LLM calls
    - no cognition logic
    """

    def __init__(
        self,
        *,
        store: MemoryStore,
        config: MemoryGatewayConfig | None = None,
    ) -> None:
        self._config = config or MemoryGatewayConfig()
        self._config.validate()

        self._store = store
        self._write_policy = MemoryWritePolicy(
            config=MemoryWritePolicyConfig(
                allow_sensitive_writes=self._config.allow_sensitive_writes,
            )
        )
        self._lock = RLock()
        self._logger = get_logger("memory.gateway")

        self._write_count = 0
        self._write_allowed_count = 0
        self._write_blocked_count = 0
        self._retrieve_count = 0
        self._retrieve_allowed_count = 0
        self._retrieve_blocked_count = 0
        self._delete_count = 0
        self._delete_blocked_count = 0
        self._clear_count = 0
        self._clear_blocked_count = 0
        self._last_memory_id: str | None = None
        self._last_query_id: str | None = None
        self._last_error: str | None = None

    @property
    def name(self) -> str:
        return self._config.name

    @property
    def store(self) -> MemoryStore:
        """
        Expose store for diagnostics and tests only.

        Cognition code should still depend on MemoryGateway, not MemoryStore.
        """

        return self._store

    @property
    def write_policy(self) -> MemoryWritePolicy:
        """
        Expose write policy for diagnostics and tests only.
        """

        return self._write_policy

    def remember(self, request: MemoryWriteRequest) -> MemoryGatewayWriteResult:
        """
        Govern and write one memory request.
        """

        with self._lock:
            self._write_count += 1
            self._last_error = None

        decision = self._write_policy.evaluate(request)

        if decision.blocked or not decision.allowed:
            result = MemoryGatewayWriteResult(
                request=request,
                record=None,
                allowed=False,
                blocked=True,
                reason=decision.reason,
                policy_classification=decision.policy_classification,
            )
            self._record_write_blocked(request=request, reason=result.reason)

            return result

        effective_request = decision.effective_request or request
        record = self._store.write(effective_request)

        with self._lock:
            self._write_allowed_count += 1
            self._last_memory_id = record.memory_id
            self._last_error = None

        self._logger.info(
            "memory_gateway_write_allowed",
            gateway=self.name,
            memory_id=record.memory_id,
            kind=record.kind.value,
            sensitivity=record.sensitivity.value,
            reason=decision.reason,
        )

        return MemoryGatewayWriteResult(
            request=request,
            record=record,
            allowed=True,
            blocked=False,
            reason=decision.reason,
            policy_classification=decision.policy_classification,
        )

    def retrieve(self, query: MemoryQuery) -> MemoryGatewayRetrievalResult:
        """
        Govern and retrieve explainable memory results.
        """

        with self._lock:
            self._retrieve_count += 1
            self._last_query_id = query.query_id
            self._last_error = None

        governed_query = self._govern_query(query)
        retrieval = self._store.retrieve(governed_query)

        with self._lock:
            self._retrieve_allowed_count += 1

        self._logger.info(
            "memory_gateway_retrieve_allowed",
            gateway=self.name,
            query_id=query.query_id,
            result_count=retrieval.result_count,
            include_sensitive=governed_query.include_sensitive,
        )

        return MemoryGatewayRetrievalResult(
            query=governed_query,
            retrieval=retrieval,
            allowed=True,
            blocked=False,
            reason=self._retrieval_reason(original=query, governed=governed_query),
            policy_classification=MemoryPolicyClassification.ALLOWED,
        )

    def get(self, memory_id: str) -> MemoryRecord | None:
        """
        Get one memory through the gateway.
        """

        return self._store.get(memory_id)

    def delete(self, memory_id: str) -> bool:
        """
        Delete one memory through the gateway.
        """

        cleaned = memory_id.strip()

        if not cleaned:
            return False

        if not self._config.allow_delete:
            with self._lock:
                self._delete_blocked_count += 1
                self._last_error = "delete blocked by gateway policy"

            self._logger.info(
                "memory_gateway_delete_blocked",
                gateway=self.name,
                memory_id=cleaned,
            )

            return False

        deleted = self._store.delete(cleaned)

        if deleted:
            with self._lock:
                self._delete_count += 1
                self._last_memory_id = cleaned
                self._last_error = None

            self._logger.info(
                "memory_gateway_delete_allowed",
                gateway=self.name,
                memory_id=cleaned,
            )

        return deleted

    def clear(self) -> None:
        """
        Clear memory only when gateway policy allows it.
        """

        if not self._config.allow_clear:
            with self._lock:
                self._clear_blocked_count += 1
                self._last_error = "clear blocked by gateway policy"

            self._logger.info("memory_gateway_clear_blocked", gateway=self.name)

            return

        self._store.clear()

        with self._lock:
            self._clear_count += 1
            self._last_memory_id = None
            self._last_error = None

        self._logger.info("memory_gateway_clear_allowed", gateway=self.name)

    def snapshot(self) -> MemoryGatewaySnapshot:
        """
        Return gateway diagnostics.
        """

        with self._lock:
            return MemoryGatewaySnapshot(
                name=self.name,
                write_count=self._write_count,
                write_allowed_count=self._write_allowed_count,
                write_blocked_count=self._write_blocked_count,
                retrieve_count=self._retrieve_count,
                retrieve_allowed_count=self._retrieve_allowed_count,
                retrieve_blocked_count=self._retrieve_blocked_count,
                delete_count=self._delete_count,
                delete_blocked_count=self._delete_blocked_count,
                clear_count=self._clear_count,
                clear_blocked_count=self._clear_blocked_count,
                last_memory_id=self._last_memory_id,
                last_query_id=self._last_query_id,
                last_error=self._last_error,
                store_snapshot=self._store.snapshot(),
                write_policy_snapshot=self._write_policy.snapshot(),
            )

    def _record_write_blocked(
        self,
        *,
        request: MemoryWriteRequest,
        reason: str,
    ) -> None:
        with self._lock:
            self._write_blocked_count += 1
            self._last_error = reason

        self._logger.info(
            "memory_gateway_write_blocked",
            gateway=self.name,
            request_id=request.request_id,
            sensitivity=request.sensitivity.value,
            reason=reason,
        )

    def _govern_query(self, query: MemoryQuery) -> MemoryQuery:
        if query.include_sensitive and not self._config.allow_sensitive_retrieval:
            return query.model_copy(update={"include_sensitive": False})

        return query

    def _retrieval_reason(
        self,
        *,
        original: MemoryQuery,
        governed: MemoryQuery,
    ) -> str:
        if original.include_sensitive and not governed.include_sensitive:
            return "memory retrieval allowed with sensitive results filtered"

        return "memory retrieval allowed by gateway policy"