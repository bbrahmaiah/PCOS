from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator


class CognitionRequestKind(StrEnum):
    """
    Type of cognition work requested.

    USER_UTTERANCE:
        Normal spoken/user text that needs an assistant response.

    SYSTEM_TASK:
        Runtime/internal task. Later useful for summaries, reminders, and
        maintenance work.

    FOLLOW_UP:
        A continuation of an active conversation turn.
    """

    USER_UTTERANCE = "user_utterance"
    SYSTEM_TASK = "system_task"
    FOLLOW_UP = "follow_up"


class CognitionResponseKind(StrEnum):
    """
    Shape of cognition output.
    """

    SPOKEN_REPLY = "spoken_reply"
    CLARIFICATION = "clarification"
    REFUSAL = "refusal"
    ERROR_FALLBACK = "error_fallback"


class CognitionPlanKind(StrEnum):
    """
    High-level plan classification before final response generation.
    """

    DIRECT_ANSWER = "direct_answer"
    ASK_CLARIFICATION = "ask_clarification"
    SAFE_REFUSAL = "safe_refusal"
    TOOL_PLANNING_REQUIRED = "tool_planning_required"


class CognitionFailureKind(StrEnum):
    """
    Typed failure category for cognition runtime failures.
    """

    VALIDATION_ERROR = "validation_error"
    ADAPTER_ERROR = "adapter_error"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


class CognitionTokenKind(StrEnum):
    """
    Token stream shape.

    TEXT:
        Normal generated text token/chunk.

    SENTENCE_BOUNDARY:
        Useful later for streaming TTS.

    FINAL:
        Marks the stream as complete.
    """

    TEXT = "text"
    SENTENCE_BOUNDARY = "sentence_boundary"
    FINAL = "final"


class SpokenResponseStyle(StrEnum):
    """
    Voice-native response style.

    CONCISE:
        Short spoken response. Default for JARVIS-like conversation.

    NORMAL:
        Balanced response.

    DETAILED:
        Longer explanation when explicitly requested.
    """

    CONCISE = "concise"
    NORMAL = "normal"
    DETAILED = "detailed"


def new_id() -> str:
    return uuid4().hex


def utc_now() -> datetime:
    return datetime.now(UTC)


class CognitionModel(BaseModel):
    """
    Base model for cognition contracts.

    Frozen models prevent accidental mutation across concurrent workers. Extra
    fields are rejected so runtime contracts stay explicit and debuggable.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        str_strip_whitespace=True,
    )


class CognitionRuntimePolicy(CognitionModel):
    """
    Runtime policy for one cognition request.

    This is not a prompt. It is execution control: cancellation, streaming,
    response length, tools, and spoken style.
    """

    cancellable: bool = True
    streaming_enabled: bool = False
    allow_tools: bool = False
    allow_memory_lookup: bool = False
    max_response_chars: int = Field(default=1_200, ge=1, le=20_000)
    timeout_ms: int = Field(default=30_000, ge=100, le=300_000)
    spoken_style: SpokenResponseStyle = SpokenResponseStyle.CONCISE
    metadata: dict[str, Any] = Field(default_factory=dict)


class CognitionContextItem(CognitionModel):
    """
    One context item supplied to cognition.

    Later this can represent session memory, retrieved memory, screen context,
    or tool results. Step 1 keeps it generic but typed.
    """

    item_id: str = Field(default_factory=new_id)
    kind: str
    text: str
    score: float | None = Field(default=None, ge=0.0, le=1.0)
    source: str = "runtime"
    created_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("kind", "text", "source")
    @classmethod
    def _non_empty_text_fields(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("field cannot be empty.")

        return value


class CognitionContext(CognitionModel):
    """
    Context bundle for cognition.

    It is intentionally bounded. We never blindly dump unlimited history into
    the brain runtime.
    """

    session_id: str | None = None
    turn_id: str | None = None
    items: tuple[CognitionContextItem, ...] = ()
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def item_count(self) -> int:
        return len(self.items)


class CognitionRequest(CognitionModel):
    """
    Request sent into the cognition runtime.

    This is the typed input contract for CognitionWorker and CognitionEngine.
    """

    request_id: str = Field(default_factory=new_id)
    kind: CognitionRequestKind = CognitionRequestKind.USER_UTTERANCE
    text: str
    source: str = "dialogue"
    turn_id: str | None = None
    transcript_id: str | None = None
    correlation_id: str | None = None
    context: CognitionContext = Field(default_factory=CognitionContext)
    policy: CognitionRuntimePolicy = Field(default_factory=CognitionRuntimePolicy)
    created_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("text", "source")
    @classmethod
    def _required_text_fields(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("field cannot be empty.")

        return value


class CognitionPlan(CognitionModel):
    """
    Response plan produced before or during reasoning.

    This lets cognition decide how it should answer before generating long text.
    """

    plan_id: str = Field(default_factory=new_id)
    request_id: str
    kind: CognitionPlanKind = CognitionPlanKind.DIRECT_ANSWER
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    needs_clarification: bool = False
    allowed_tool_names: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("request_id")
    @classmethod
    def _request_id_required(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("request_id cannot be empty.")

        return value


class CognitionToken(CognitionModel):
    """
    One streamed cognition token/chunk.

    Streaming token text must preserve whitespace.

    Real streaming model backends often emit token pieces like:
        " am"
        " the"
        "\\nNext"

    If we strip token text, final streamed output can become broken:
        "I am" -> "Iam"

    So we preserve token.text exactly, while still rejecting fully blank tokens.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        str_strip_whitespace=False,
    )

    token_id: str = Field(default_factory=new_id)
    request_id: str
    index: int = Field(ge=0)
    text: str
    kind: CognitionTokenKind = CognitionTokenKind.TEXT
    final: bool = False
    created_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("request_id")
    @classmethod
    def _request_id_required(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("request_id cannot be empty.")

        return cleaned

    @field_validator("text")
    @classmethod
    def _token_text_required(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text cannot be empty.")

        return value


class CognitionResponse(CognitionModel):
    """
    Final cognition response.

    This becomes dialogue.response_ready later, which Presence speaks.
    """

    response_id: str = Field(default_factory=new_id)
    request_id: str
    text: str
    kind: CognitionResponseKind = CognitionResponseKind.SPOKEN_REPLY
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    plan: CognitionPlan | None = None
    token_count: int = Field(default=0, ge=0)
    created_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("request_id", "text")
    @classmethod
    def _required_response_fields(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("field cannot be empty.")

        return value


class CognitionFailure(CognitionModel):
    """
    Typed failure response for cognition.

    Failures must be explicit, observable, and safe to convert into fallback
    spoken responses.
    """

    failure_id: str = Field(default_factory=new_id)
    request_id: str
    kind: CognitionFailureKind = CognitionFailureKind.UNKNOWN
    message: str
    recoverable: bool = True
    created_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("request_id", "message")
    @classmethod
    def _required_failure_fields(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("field cannot be empty.")

        return value


class CognitionSnapshot(CognitionModel):
    """
    Lightweight observable cognition runtime snapshot.

    The state store in Step 2 will produce this shape.
    """

    active_request_id: str | None = None
    active_turn_id: str | None = None
    running: bool = False
    streaming: bool = False
    cancelling: bool = False
    completed_count: int = 0
    failed_count: int = 0
    cancelled_count: int = 0
    last_response_id: str | None = None
    last_error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)