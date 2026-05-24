from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator


def new_conversation_id() -> str:
    return uuid4().hex


def utc_now() -> datetime:
    return datetime.now(UTC)


class ConversationModel(BaseModel):
    """
    Base model for Adaptive Conversational Runtime contracts.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        validate_assignment=True,
    )


class ConversationMode(StrEnum):
    """
    Current conversational mode.

    The mode changes endpointing patience. Commands should finalize faster.
    Discussion should wait longer because the user may pause to think.
    """

    UNKNOWN = "unknown"
    COMMAND = "command"
    QUESTION = "question"
    DISCUSSION = "discussion"
    DICTATION = "dictation"


class TurnInputSource(StrEnum):
    """
    Source that produced the turn signal.
    """

    MICROPHONE = "microphone"
    STT_PARTIAL = "stt_partial"
    VAD = "vad"
    INTERRUPTION_WORKER = "interruption_worker"
    SYSTEM = "system"


class TranscriptCompleteness(StrEnum):
    """
    Semantic completion estimate for the current transcript.
    """

    EMPTY = "empty"
    INCOMPLETE = "incomplete"
    LIKELY_COMPLETE = "likely_complete"
    COMPLETE = "complete"


class TurnUrgency(StrEnum):
    """
    Urgency class for endpointing.
    """

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class InterruptionIntent(StrEnum):
    """
    Interruption intent detected from speech.
    """

    NONE = "none"
    STOP = "stop"
    PAUSE = "pause"
    CANCEL = "cancel"
    WAIT = "wait"
    CORRECTION = "correction"
    BARGE_IN = "barge_in"


class TurnDecisionKind(StrEnum):
    """
    Decision produced by TurnDetector.
    """

    WAIT = "wait"
    MAYBE_COMPLETE = "maybe_complete"
    FINALIZE = "finalize"
    INTERRUPT = "interrupt"
    CANCEL = "cancel"


class TurnEndpointReason(StrEnum):
    """
    Reason behind the turn decision.
    """

    SPEECH_ACTIVE = "speech_active"
    EMPTY_TRANSCRIPT = "empty_transcript"
    INCOMPLETE_TRANSCRIPT = "incomplete_transcript"
    LOW_SILENCE = "low_silence"
    MAYBE_COMPLETE = "maybe_complete"
    COMPLETE_COMMAND = "complete_command"
    COMPLETE_QUESTION = "complete_question"
    COMPLETE_DISCUSSION = "complete_discussion"
    MAX_WAIT_REACHED = "max_wait_reached"
    INTERRUPTION_INTENT = "interruption_intent"
    BARGE_IN = "barge_in"
    CANCEL_INTENT = "cancel_intent"


class TurnDetectionInput(ConversationModel):
    """
    Input snapshot for one turn-detection evaluation.

    This object is intentionally multimodal-ready. For now it can be driven by
    fake tests, VAD, and STT partials. Later it can include prosody and richer
    speech timing.
    """

    turn_id: str = Field(default_factory=new_conversation_id)
    source: TurnInputSource = TurnInputSource.STT_PARTIAL
    transcript: str = ""
    is_speech_active: bool = False
    is_assistant_speaking: bool = False
    silence_ms: int = Field(default=0, ge=0)
    speech_ms: int = Field(default=0, ge=0)
    vad_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    transcript_stability: float = Field(default=0.0, ge=0.0, le=1.0)
    conversation_mode: ConversationMode = ConversationMode.UNKNOWN
    received_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("turn_id")
    @classmethod
    def _turn_id_required(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("turn_id cannot be empty.")

        return cleaned

    @field_validator("transcript")
    @classmethod
    def _clean_transcript(cls, value: str) -> str:
        return " ".join(value.strip().split())


class TurnDetectionDecision(ConversationModel):
    """
    Decision produced by the adaptive turn detector.
    """

    turn_id: str
    decision: TurnDecisionKind
    reason: TurnEndpointReason
    transcript: str
    completeness: TranscriptCompleteness
    interruption_intent: InterruptionIntent = InterruptionIntent.NONE
    urgency: TurnUrgency = TurnUrgency.NORMAL
    confidence: float = Field(ge=0.0, le=1.0)
    should_start_cognition: bool = False
    should_cancel_response: bool = False
    should_keep_listening: bool = True
    endpoint_delay_ms: int = Field(default=0, ge=0)
    evaluated_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("turn_id")
    @classmethod
    def _turn_id_required(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("turn_id cannot be empty.")

        return cleaned

    @property
    def finalized(self) -> bool:
        return self.decision == TurnDecisionKind.FINALIZE

    @property
    def interrupting(self) -> bool:
        return self.decision in {
            TurnDecisionKind.INTERRUPT,
            TurnDecisionKind.CANCEL,
        }

    @property
    def waiting(self) -> bool:
        return self.decision == TurnDecisionKind.WAIT