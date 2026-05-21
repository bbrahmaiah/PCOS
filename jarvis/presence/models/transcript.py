from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator


def utc_now() -> datetime:
    return datetime.now(UTC)


def new_transcript_id() -> str:
    return uuid4().hex


class TranscriptKind(StrEnum):
    PARTIAL = "partial"
    FINAL = "final"
    REJECTED = "rejected"


class Transcript(BaseModel):
    """
    Speech-to-text result.

    STT workers will emit Transcript objects. Router/Cognition will later
    consume final transcripts.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    transcript_id: str = Field(default_factory=new_transcript_id)
    segment_id: str

    text: str
    kind: TranscriptKind
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    language: str = "en"

    alternatives: tuple[str, ...] = ()
    created_at: datetime = Field(default_factory=utc_now)

    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("transcript_id", "segment_id", "language")
    @classmethod
    def validate_non_empty_string(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("transcript string fields cannot be empty.")

        return cleaned

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("transcript text cannot be empty.")

        return cleaned

    @property
    def is_final(self) -> bool:
        return self.kind == TranscriptKind.FINAL

    @property
    def is_partial(self) -> bool:
        return self.kind == TranscriptKind.PARTIAL

    @property
    def is_rejected(self) -> bool:
        return self.kind == TranscriptKind.REJECTED