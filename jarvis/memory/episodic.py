from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from threading import RLock
from typing import Any

from pydantic import Field, field_validator

from jarvis.memory.gateway import (
    MemoryGateway,
    MemoryGatewayRetrievalResult,
    MemoryGatewayWriteResult,
)
from jarvis.memory.models import (
    MemoryImportance,
    MemoryKind,
    MemoryModel,
    MemoryQuery,
    MemoryRetention,
    MemoryScope,
    MemorySensitivity,
    MemorySource,
    MemoryWriteRequest,
    utc_now,
)
from jarvis.runtime.observability.structured_logger import get_logger


class EpisodicMemoryEventKind(StrEnum):
    """
    Type of event captured by episodic memory.
    """

    USER_MESSAGE = "user_message"
    ASSISTANT_RESPONSE = "assistant_response"
    DECISION = "decision"
    MILESTONE = "milestone"
    ERROR = "error"
    DEBUGGING = "debugging"
    TASK_PROGRESS = "task_progress"
    SYSTEM_EVENT = "system_event"


class EpisodicMemoryActor(StrEnum):
    """
    Actor responsible for an episodic event.
    """

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class EpisodicMemoryEvent(MemoryModel):
    """
    One event intended for episodic memory.

    This model is not the stored MemoryRecord. It is the runtime event contract
    that gets converted into a governed MemoryWriteRequest through the gateway.
    """

    event_id: str
    kind: EpisodicMemoryEventKind
    actor: EpisodicMemoryActor
    text: str
    occurred_at: datetime = Field(default_factory=utc_now)
    importance: MemoryImportance = MemoryImportance.NORMAL
    sensitivity: MemorySensitivity = MemorySensitivity.PRIVATE
    retention: MemoryRetention = MemoryRetention.PERSISTENT
    tags: tuple[str, ...] = ()
    source: MemorySource = MemorySource.CONVERSATION
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("event_id", "text")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned

    @field_validator("tags")
    @classmethod
    def _clean_tags(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        cleaned = tuple(tag.strip().casefold() for tag in value if tag.strip())

        return tuple(dict.fromkeys(cleaned))

    def to_write_request(self) -> MemoryWriteRequest:
        """
        Convert event into a governed memory write request.
        """

        return MemoryWriteRequest(
            kind=MemoryKind.EPISODIC,
            scope=MemoryScope.USER,
            text=self.text,
            source=self.source,
            sensitivity=self.sensitivity,
            importance=self.importance,
            retention=self.retention,
            tags=(
                "episodic",
                self.kind.value,
                self.actor.value,
                *self.tags,
            ),
            metadata={
                **self.metadata,
                "event_id": self.event_id,
                "event_kind": self.kind.value,
                "actor": self.actor.value,
                "occurred_at": self.occurred_at.isoformat(),
            },
        )


class EpisodicMemoryQuery(MemoryModel):
    """
    Query contract for episodic memory.
    """

    text: str | None = None
    event_kinds: tuple[EpisodicMemoryEventKind, ...] = ()
    actors: tuple[EpisodicMemoryActor, ...] = ()
    tags: tuple[str, ...] = ()
    max_results: int = Field(default=8, ge=1, le=100)
    include_sensitive: bool = False
    include_expired: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("text")
    @classmethod
    def _clean_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None

        cleaned = value.strip()

        return cleaned or None

    @field_validator("tags")
    @classmethod
    def _clean_tags(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        cleaned = tuple(tag.strip().casefold() for tag in value if tag.strip())

        return tuple(dict.fromkeys(cleaned))

    def to_memory_query(self) -> MemoryQuery:
        """
        Convert episodic query into a normal MemoryQuery.
        """

        tags = [
            "episodic",
            *(kind.value for kind in self.event_kinds),
            *(actor.value for actor in self.actors),
            *self.tags,
        ]

        return MemoryQuery(
            text=self.text,
            kinds=(MemoryKind.EPISODIC,),
            tags=tuple(dict.fromkeys(tags)),
            max_results=self.max_results,
            include_sensitive=self.include_sensitive,
            include_expired=self.include_expired,
            metadata={
                **self.metadata,
                "episodic_query": True,
            },
        )


@dataclass(frozen=True, slots=True)
class EpisodicMemoryRuntimeConfig:
    """
    Configuration for EpisodicMemoryRuntime.
    """

    name: str = "episodic_memory_runtime"
    default_importance: MemoryImportance = MemoryImportance.NORMAL
    default_retention: MemoryRetention = MemoryRetention.PERSISTENT
    default_sensitivity: MemorySensitivity = MemorySensitivity.PRIVATE

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("name cannot be empty.")


@dataclass(frozen=True, slots=True)
class EpisodicMemoryRuntimeSnapshot:
    """
    Observable diagnostics for episodic memory runtime.
    """

    name: str
    captured_count: int
    captured_allowed_count: int
    captured_blocked_count: int
    retrieved_count: int
    last_event_id: str | None
    last_error: str | None


class EpisodicMemoryRuntime:
    """
    Runtime for episodic memory.

    Responsibilities:
    - capture timeline-style events
    - convert events into governed memory writes
    - retrieve episodic memories through MemoryGateway only
    - preserve explainability and policy boundaries
    - keep diagnostics

    Non-responsibilities:
    - no direct store access
    - no vector search
    - no summarization
    - no semantic memory extraction
    - no LLM calls
    """

    def __init__(
        self,
        *,
        gateway: MemoryGateway,
        config: EpisodicMemoryRuntimeConfig | None = None,
    ) -> None:
        self._config = config or EpisodicMemoryRuntimeConfig()
        self._config.validate()

        self._gateway = gateway
        self._lock = RLock()
        self._logger = get_logger("memory.episodic_runtime")

        self._captured_count = 0
        self._captured_allowed_count = 0
        self._captured_blocked_count = 0
        self._retrieved_count = 0
        self._last_event_id: str | None = None
        self._last_error: str | None = None

    @property
    def name(self) -> str:
        return self._config.name

    def capture(self, event: EpisodicMemoryEvent) -> MemoryGatewayWriteResult:
        """
        Capture one episodic event through the MemoryGateway.
        """

        with self._lock:
            self._captured_count += 1
            self._last_event_id = event.event_id
            self._last_error = None

        result = self._gateway.remember(event.to_write_request())

        with self._lock:
            if result.allowed:
                self._captured_allowed_count += 1

            else:
                self._captured_blocked_count += 1
                self._last_error = result.reason

        self._logger.info(
            "episodic_memory_event_captured",
            runtime=self.name,
            event_id=event.event_id,
            event_kind=event.kind.value,
            actor=event.actor.value,
            allowed=result.allowed,
            blocked=result.blocked,
        )

        return result

    def capture_text(
        self,
        text: str,
        *,
        event_id: str,
        kind: EpisodicMemoryEventKind = EpisodicMemoryEventKind.SYSTEM_EVENT,
        actor: EpisodicMemoryActor = EpisodicMemoryActor.SYSTEM,
        importance: MemoryImportance | None = None,
        sensitivity: MemorySensitivity | None = None,
        retention: MemoryRetention | None = None,
        tags: tuple[str, ...] = (),
        metadata: dict[str, Any] | None = None,
    ) -> MemoryGatewayWriteResult:
        """
        Convenience method for capturing text as an episodic event.
        """

        event = EpisodicMemoryEvent(
            event_id=event_id,
            kind=kind,
            actor=actor,
            text=text,
            importance=importance or self._config.default_importance,
            sensitivity=sensitivity or self._config.default_sensitivity,
            retention=retention or self._config.default_retention,
            tags=tags,
            metadata=metadata or {},
        )

        return self.capture(event)

    def retrieve(
        self,
        query: EpisodicMemoryQuery,
    ) -> MemoryGatewayRetrievalResult:
        """
        Retrieve episodic memories through the MemoryGateway.
        """

        with self._lock:
            self._retrieved_count += 1
            self._last_error = None

        result = self._gateway.retrieve(query.to_memory_query())

        self._logger.info(
            "episodic_memory_retrieved",
            runtime=self.name,
            result_count=result.result_count,
            allowed=result.allowed,
            reason=result.reason,
        )

        return result

    def snapshot(self) -> EpisodicMemoryRuntimeSnapshot:
        """
        Return episodic runtime diagnostics.
        """

        with self._lock:
            return EpisodicMemoryRuntimeSnapshot(
                name=self.name,
                captured_count=self._captured_count,
                captured_allowed_count=self._captured_allowed_count,
                captured_blocked_count=self._captured_blocked_count,
                retrieved_count=self._retrieved_count,
                last_event_id=self._last_event_id,
                last_error=self._last_error,
            )

    def reset(self) -> None:
        """
        Reset runtime diagnostics.
        """

        with self._lock:
            self._captured_count = 0
            self._captured_allowed_count = 0
            self._captured_blocked_count = 0
            self._retrieved_count = 0
            self._last_event_id = None
            self._last_error = None

        self._logger.info("episodic_memory_runtime_reset", runtime=self.name)