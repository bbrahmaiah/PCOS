from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Protocol, Self, runtime_checkable

from pydantic import Field, model_validator

from jarvis.cognition.models import (
    CognitionFailure,
    CognitionModel,
    CognitionRequest,
    CognitionResponse,
    CognitionToken,
)


class CognitionAdapterCapability(StrEnum):
    """
    Capabilities exposed by a cognition adapter.

    These are runtime capabilities, not model marketing labels. Workers and
    engines can use them to route safely without knowing adapter internals.
    """

    NON_STREAMING = "non_streaming"
    STREAMING = "streaming"
    CANCELLATION = "cancellation"
    MEMORY_CONTEXT = "memory_context"
    TOOL_PLANNING = "tool_planning"


class CognitionAdapterStatus(StrEnum):
    """
    Runtime health status for a cognition adapter.
    """

    READY = "ready"
    BUSY = "busy"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"


def utc_now() -> datetime:
    return datetime.now(UTC)


def duration_ms_between(started_at: datetime, finished_at: datetime) -> float:
    """
    Return bounded duration in milliseconds.

    Clock skew should not happen with monotonic measurement, but adapter result
    timestamps are wall-clock values. Clamp to zero to keep diagnostics safe.
    """

    return max(0.0, (finished_at - started_at).total_seconds() * 1000.0)


class CognitionAdapterResult(CognitionModel):
    """
    Result returned by a non-streaming cognition adapter.

    Exactly one of response or failure must be set. This keeps worker logic
    deterministic and avoids ambiguous "partial success" states.
    """

    request_id: str
    response: CognitionResponse | None = None
    failure: CognitionFailure | None = None
    started_at: datetime
    finished_at: datetime
    duration_ms: float = Field(ge=0.0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_result_shape(self) -> Self:
        has_response = self.response is not None
        has_failure = self.failure is not None

        if has_response == has_failure:
            raise ValueError("exactly one of response or failure must be set.")

        if self.response is not None and self.response.request_id != self.request_id:
            raise ValueError("response request_id must match result request_id.")

        if self.failure is not None and self.failure.request_id != self.request_id:
            raise ValueError("failure request_id must match result request_id.")

        if self.finished_at < self.started_at:
            raise ValueError("finished_at cannot be earlier than started_at.")

        return self

    @property
    def succeeded(self) -> bool:
        return self.response is not None

    @property
    def failed(self) -> bool:
        return self.failure is not None


class CognitionAdapterSnapshot(CognitionModel):
    """
    Observable adapter snapshot.

    Workers can expose this without leaking model/backend implementation details.
    """

    name: str
    status: CognitionAdapterStatus = CognitionAdapterStatus.READY
    capabilities: tuple[CognitionAdapterCapability, ...] = ()
    request_count: int = Field(default=0, ge=0)
    success_count: int = Field(default=0, ge=0)
    failure_count: int = Field(default=0, ge=0)
    cancelled_count: int = Field(default=0, ge=0)
    streaming_count: int = Field(default=0, ge=0)
    last_request_id: str | None = None
    last_error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


@runtime_checkable
class CognitionAdapter(Protocol):
    """
    Non-streaming cognition adapter contract.

    CognitionWorker will depend on this protocol, not on any concrete LLM
    implementation.
    """

    @property
    def name(self) -> str:
        """Stable adapter name."""

    @property
    def capabilities(self) -> tuple[CognitionAdapterCapability, ...]:
        """Supported adapter capabilities."""

    def generate(self, request: CognitionRequest) -> CognitionAdapterResult:
        """
        Generate a final cognition response for one request.
        """

    def snapshot(self) -> CognitionAdapterSnapshot:
        """
        Return an observable adapter snapshot.
        """


@runtime_checkable
class StreamingCognitionAdapter(CognitionAdapter, Protocol):
    """
    Streaming cognition adapter contract.

    Streaming adapters produce tokens first and later a final response through
    the normal result path.
    """

    def stream(self, request: CognitionRequest) -> Iterator[CognitionToken]:
        """
        Stream cognition tokens for one request.
        """


@runtime_checkable
class CancellableCognitionAdapter(Protocol):
    """
    Optional cancellation contract.

    Not every backend can truly cancel generation. Workers should check this
    protocol before attempting backend-level cancellation.
    """

    def cancel(self, *, request_id: str, reason: str | None = None) -> bool:
        """
        Attempt to cancel an active request.
        """


def adapter_success_result(
    *,
    request: CognitionRequest,
    response: CognitionResponse,
    started_at: datetime,
    finished_at: datetime,
    metadata: dict[str, Any] | None = None,
) -> CognitionAdapterResult:
    """
    Build a validated successful adapter result.
    """

    return CognitionAdapterResult(
        request_id=request.request_id,
        response=response,
        started_at=started_at,
        finished_at=finished_at,
        duration_ms=duration_ms_between(started_at, finished_at),
        metadata=metadata or {},
    )


def adapter_failure_result(
    *,
    request: CognitionRequest,
    failure: CognitionFailure,
    started_at: datetime,
    finished_at: datetime,
    metadata: dict[str, Any] | None = None,
) -> CognitionAdapterResult:
    """
    Build a validated failed adapter result.
    """

    return CognitionAdapterResult(
        request_id=request.request_id,
        failure=failure,
        started_at=started_at,
        finished_at=finished_at,
        duration_ms=duration_ms_between(started_at, finished_at),
        metadata=metadata or {},
    )


def adapter_supports(
    adapter: CognitionAdapter,
    capability: CognitionAdapterCapability,
) -> bool:
    """
    Return whether an adapter declares a capability.
    """

    return capability in adapter.capabilities