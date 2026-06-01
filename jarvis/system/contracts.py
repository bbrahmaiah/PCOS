from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from jarvis.cognition.models import CognitionResponse
from jarvis.runtime.workers.worker import WorkerSnapshot


def utc_now() -> datetime:
    return datetime.now(UTC)


def new_system_id(prefix: str) -> str:
    clean_prefix = prefix.strip()
    if not clean_prefix:
        raise ValueError("prefix cannot be empty.")
    return f"{clean_prefix}_{uuid4().hex}"


class JarvisSystemStatus(StrEnum):
    CREATED = "created"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"


class JarvisSubsystemKind(StrEnum):
    MEMORY = "memory"
    COGNITION = "cognition"


class JarvisAskStatus(StrEnum):
    ANSWERED = "answered"
    REJECTED = "rejected"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class JarvisSystemRequest:
    text: str
    session_id: str = "default"
    request_id: str = ""
    max_memory_results: int = 5
    metadata: dict[str, object] | None = None

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("text cannot be empty.")
        if not self.session_id.strip():
            raise ValueError("session_id cannot be empty.")
        if self.max_memory_results < 0:
            raise ValueError("max_memory_results cannot be negative.")


@dataclass(frozen=True, slots=True)
class JarvisSystemResponse:
    request_id: str
    session_id: str
    status: JarvisAskStatus
    text: str
    cognition_response: CognitionResponse | None
    memory_result_count: int
    used_memory: bool
    used_cognition: bool
    reason: str
    created_at: datetime
    metadata: dict[str, object]

    @property
    def succeeded(self) -> bool:
        return self.status == JarvisAskStatus.ANSWERED


@dataclass(frozen=True, slots=True)
class JarvisSubsystemHealth:
    kind: JarvisSubsystemKind
    worker: WorkerSnapshot
    subsystem_snapshot: Any | None = None


@dataclass(frozen=True, slots=True)
class JarvisSystemSnapshot:
    name: str
    status: JarvisSystemStatus
    started_at: datetime | None
    stopped_at: datetime | None
    memory_worker: WorkerSnapshot | None
    cognition_worker: WorkerSnapshot | None
    subsystem_health: tuple[JarvisSubsystemHealth, ...]
    kernel_snapshot: Any | None
    ask_count: int
    failure_count: int
    last_error: str | None