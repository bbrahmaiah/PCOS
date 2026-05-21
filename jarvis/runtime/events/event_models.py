from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Self
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from jarvis.runtime.events.priorities import priority_for_event
from jarvis.runtime.shared.enums import EventCategory, EventPriority, EventType


def utc_now() -> datetime:
    return datetime.now(UTC)


def new_id() -> str:
    return uuid4().hex


class EventMetadata(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    tags: tuple[str, ...] = ()
    trace_name: str | None = None
    source_file: str | None = None
    source_line: int | None = Field(default=None, ge=1)


class RuntimeEvent(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        arbitrary_types_allowed=True,
    )

    event_id: str = Field(default_factory=new_id)
    correlation_id: str = Field(default_factory=new_id)
    causation_id: str | None = None

    event_type: EventType
    category: EventCategory
    source: str

    priority: EventPriority | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    metadata: EventMetadata = Field(default_factory=EventMetadata)

    created_at: datetime = Field(default_factory=utc_now)
    deadline_at: datetime | None = None
    cancellable: bool = True
    schema_version: str = "1.0"

    @model_validator(mode="before")
    @classmethod
    def set_default_priority(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        if data.get("priority") is not None:
            return data

        if "event_type" not in data:
            return data

        updated = dict(data)

        try:
            event_type = EventType(updated["event_type"])
        except (TypeError, ValueError):
            return updated

        updated["priority"] = priority_for_event(event_type)
        return updated

    @field_validator("event_id", "correlation_id")
    @classmethod
    def validate_required_ids(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("event IDs cannot be empty.")

        return cleaned

    @field_validator("causation_id")
    @classmethod
    def validate_optional_causation_id(cls, value: str | None) -> str | None:
        if value is None:
            return None

        cleaned = value.strip()

        if not cleaned:
            raise ValueError("event.causation_id cannot be empty when provided.")

        return cleaned

    @field_validator("source")
    @classmethod
    def validate_source(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("event.source cannot be empty.")

        return cleaned

    @field_validator("schema_version")
    @classmethod
    def validate_schema_version(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("event.schema_version cannot be empty.")

        return cleaned

    @model_validator(mode="after")
    def validate_event_integrity(self) -> Self:
        if self.priority is None:
            raise ValueError("event.priority must be resolved before model creation.")

        if self.deadline_at is not None and self.deadline_at <= self.created_at:
            raise ValueError("event.deadline_at must be later than event.created_at.")

        return self

    def child(
        self,
        event_type: EventType,
        category: EventCategory,
        source: str,
        payload: dict[str, Any] | None = None,
        priority: EventPriority | None = None,
        cancellable: bool | None = None,
    ) -> RuntimeEvent:
        return RuntimeEvent(
            correlation_id=self.correlation_id,
            causation_id=self.event_id,
            event_type=event_type,
            category=category,
            source=source,
            priority=priority,
            payload=payload or {},
            cancellable=self.cancellable if cancellable is None else cancellable,
        )

    @property
    def age_ms(self) -> float:
        return (utc_now() - self.created_at).total_seconds() * 1000

    @property
    def is_expired(self) -> bool:
        return self.deadline_at is not None and utc_now() >= self.deadline_at