from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from threading import RLock

from pydantic import Field, field_validator

from jarvis.conversation.models import (
    ConversationMode,
    ConversationModel,
    InterruptionIntent,
    TranscriptCompleteness,
    TurnDecisionKind,
    TurnDetectionDecision,
    TurnDetectionInput,
    TurnEndpointReason,
    TurnUrgency,
    utc_now,
)
from jarvis.conversation.state_machine import ConversationState
from jarvis.conversation.turn_detection import (
    AdaptiveTurnDetector,
    AdaptiveTurnDetectorConfig,
)
from jarvis.runtime.observability.structured_logger import get_logger


class EndpointAction(StrEnum):
    """
    Runtime action recommended by the endpointing engine.
    """

    KEEP_LISTENING = "keep_listening"
    WAIT_FOR_USER = "wait_for_user"
    PREPARE_RESPONSE = "prepare_response"
    START_COGNITION = "start_cognition"
    INTERRUPT_RESPONSE = "interrupt_response"
    CANCEL_RESPONSE = "cancel_response"


class EndpointConfidenceBand(StrEnum):
    """
    Confidence band for endpointing decisions.
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class EndpointPacing(StrEnum):
    """
    Human pacing mode selected for this endpointing decision.
    """

    FAST_COMMAND = "fast_command"
    NORMAL_QUESTION = "normal_question"
    PATIENT_DISCUSSION = "patient_discussion"
    THINKING_PAUSE = "thinking_pause"
    INTERRUPT_IMMEDIATE = "interrupt_immediate"


class EndpointingInput(ConversationModel):
    """
    Input for adaptive endpointing.

    This combines the latest speech/turn signal with the current conversation
    state. Endpointing should never make decisions from silence alone.
    """

    signal: TurnDetectionInput
    conversation_state: ConversationState
    previous_transcript: str | None = None
    consecutive_maybe_complete_count: int = Field(default=0, ge=0)
    user_pause_count: int = Field(default=0, ge=0)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("previous_transcript")
    @classmethod
    def _clean_optional_transcript(cls, value: str | None) -> str | None:
        if value is None:
            return None

        cleaned = " ".join(value.strip().split())

        return cleaned or None


class EndpointingDecision(ConversationModel):
    """
    Final adaptive endpointing decision.

    This is the object a future conversation orchestrator should consume.
    """

    turn_decision: TurnDetectionDecision
    action: EndpointAction
    pacing: EndpointPacing
    confidence_band: EndpointConfidenceBand
    conversation_state: ConversationState
    should_start_cognition: bool
    should_cancel_response: bool
    should_wait_for_more_audio: bool
    endpoint_delay_ms: int = Field(default=0, ge=0)
    reason: str
    decided_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("reason")
    @classmethod
    def _reason_required(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("reason cannot be empty.")

        return cleaned

    @property
    def finalized(self) -> bool:
        return self.action in {
            EndpointAction.PREPARE_RESPONSE,
            EndpointAction.START_COGNITION,
        }

    @property
    def interrupting(self) -> bool:
        return self.action in {
            EndpointAction.INTERRUPT_RESPONSE,
            EndpointAction.CANCEL_RESPONSE,
        }

    @property
    def waiting(self) -> bool:
        return self.action in {
            EndpointAction.KEEP_LISTENING,
            EndpointAction.WAIT_FOR_USER,
        }


@dataclass(frozen=True, slots=True)
class AdaptiveEndpointingEngineConfig:
    """
    Configuration for adaptive endpointing.

    This layer adds state-aware patience on top of the turn detector.
    """

    name: str = "adaptive_endpointing_engine"
    turn_detector_config: AdaptiveTurnDetectorConfig | None = None

    maybe_complete_to_prepare_count: int = 2
    thinking_pause_extra_ms: int = 700
    follow_up_extra_ms: int = 300
    speaking_barge_in_min_confidence: float = 0.65
    low_confidence_wait_ms: int = 400

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("name cannot be empty.")

        if self.maybe_complete_to_prepare_count < 1:
            raise ValueError("maybe_complete_to_prepare_count must be >= 1.")

        if self.thinking_pause_extra_ms < 0:
            raise ValueError("thinking_pause_extra_ms cannot be negative.")

        if self.follow_up_extra_ms < 0:
            raise ValueError("follow_up_extra_ms cannot be negative.")

        if (
            self.speaking_barge_in_min_confidence < 0.0
            or self.speaking_barge_in_min_confidence > 1.0
        ):
            raise ValueError(
                "speaking_barge_in_min_confidence must be between 0 and 1."
            )

        if self.low_confidence_wait_ms < 0:
            raise ValueError("low_confidence_wait_ms cannot be negative.")


@dataclass(frozen=True, slots=True)
class AdaptiveEndpointingEngineSnapshot:
    """
    Observable diagnostics for endpointing engine.
    """

    name: str
    evaluation_count: int
    finalized_count: int
    wait_count: int
    interrupt_count: int
    cancel_count: int
    last_action: EndpointAction | None
    last_pacing: EndpointPacing | None
    last_state: ConversationState | None
    last_error: str | None


class AdaptiveEndpointingEngine:
    """
    State-aware adaptive endpointing engine.

    Responsibilities:
    - combine turn detection with conversation state
    - prevent premature responses during thinking pauses
    - fast-path urgent commands and interruptions
    - delay politely in discussion/follow-up states
    - produce orchestration-ready endpoint actions
    - expose diagnostics

    Non-responsibilities:
    - no STT
    - no VAD implementation
    - no LLM calls
    - no TTS control
    - no tool execution
    """

    def __init__(
        self,
        *,
        config: AdaptiveEndpointingEngineConfig | None = None,
        turn_detector: AdaptiveTurnDetector | None = None,
    ) -> None:
        self._config = config or AdaptiveEndpointingEngineConfig()
        self._config.validate()

        self._turn_detector = turn_detector or AdaptiveTurnDetector(
            config=self._config.turn_detector_config
        )
        self._lock = RLock()
        self._logger = get_logger("conversation.endpointing")

        self._evaluation_count = 0
        self._finalized_count = 0
        self._wait_count = 0
        self._interrupt_count = 0
        self._cancel_count = 0
        self._last_action: EndpointAction | None = None
        self._last_pacing: EndpointPacing | None = None
        self._last_state: ConversationState | None = None
        self._last_error: str | None = None

    @property
    def name(self) -> str:
        return self._config.name

    def evaluate(self, request: EndpointingInput) -> EndpointingDecision:
        """
        Evaluate endpointing from current signal and conversation state.
        """

        with self._lock:
            self._evaluation_count += 1
            self._last_state = request.conversation_state
            self._last_error = None

        try:
            turn_decision = self._turn_detector.evaluate(request.signal)
            decision = self._decide(
                request=request,
                turn_decision=turn_decision,
            )
            self._record_decision(decision)

            self._logger.info(
                "adaptive_endpointing_evaluated",
                engine=self.name,
                turn_id=request.signal.turn_id,
                state=request.conversation_state.value,
                turn_decision=turn_decision.decision.value,
                action=decision.action.value,
                pacing=decision.pacing.value,
                confidence_band=decision.confidence_band.value,
                endpoint_delay_ms=decision.endpoint_delay_ms,
            )

            return decision

        except Exception as exc:
            with self._lock:
                self._last_error = f"{type(exc).__name__}: {exc}"

            raise

    def snapshot(self) -> AdaptiveEndpointingEngineSnapshot:
        """
        Return engine diagnostics.
        """

        with self._lock:
            return AdaptiveEndpointingEngineSnapshot(
                name=self.name,
                evaluation_count=self._evaluation_count,
                finalized_count=self._finalized_count,
                wait_count=self._wait_count,
                interrupt_count=self._interrupt_count,
                cancel_count=self._cancel_count,
                last_action=self._last_action,
                last_pacing=self._last_pacing,
                last_state=self._last_state,
                last_error=self._last_error,
            )

    def reset(self) -> None:
        """
        Reset endpointing engine diagnostics.
        """

        with self._lock:
            self._evaluation_count = 0
            self._finalized_count = 0
            self._wait_count = 0
            self._interrupt_count = 0
            self._cancel_count = 0
            self._last_action = None
            self._last_pacing = None
            self._last_state = None
            self._last_error = None

        self._turn_detector.reset()
        self._logger.info("adaptive_endpointing_engine_reset", engine=self.name)

    def _decide(
        self,
        *,
        request: EndpointingInput,
        turn_decision: TurnDetectionDecision,
    ) -> EndpointingDecision:
        state = request.conversation_state
        pacing = self._pacing(request=request, turn_decision=turn_decision)
        band = self._confidence_band(turn_decision.confidence)

        if turn_decision.decision == TurnDecisionKind.CANCEL:
            return self._decision(
                request=request,
                turn_decision=turn_decision,
                action=EndpointAction.CANCEL_RESPONSE,
                pacing=EndpointPacing.INTERRUPT_IMMEDIATE,
                confidence_band=EndpointConfidenceBand.CRITICAL,
                reason="cancel intent requires immediate response cancellation",
                endpoint_delay_ms=0,
            )

        if turn_decision.decision == TurnDecisionKind.INTERRUPT:
            return self._handle_interrupt(
                request=request,
                turn_decision=turn_decision,
                pacing=pacing,
                confidence_band=band,
            )

        if turn_decision.decision == TurnDecisionKind.FINALIZE:
            return self._handle_finalize(
                request=request,
                turn_decision=turn_decision,
                pacing=pacing,
                confidence_band=band,
            )

        if turn_decision.decision == TurnDecisionKind.MAYBE_COMPLETE:
            return self._handle_maybe_complete(
                request=request,
                turn_decision=turn_decision,
                pacing=pacing,
                confidence_band=band,
            )

        if state == ConversationState.USER_THINKING:
            return self._decision(
                request=request,
                turn_decision=turn_decision,
                action=EndpointAction.WAIT_FOR_USER,
                pacing=EndpointPacing.THINKING_PAUSE,
                confidence_band=band,
                reason="user appears to be thinking; wait patiently",
                endpoint_delay_ms=(
                    turn_decision.endpoint_delay_ms
                    + self._config.thinking_pause_extra_ms
                ),
            )

        return self._decision(
            request=request,
            turn_decision=turn_decision,
            action=EndpointAction.KEEP_LISTENING,
            pacing=pacing,
            confidence_band=band,
            reason="turn is not complete; keep listening",
            endpoint_delay_ms=turn_decision.endpoint_delay_ms,
        )

    def _handle_interrupt(
        self,
        *,
        request: EndpointingInput,
        turn_decision: TurnDetectionDecision,
        pacing: EndpointPacing,
        confidence_band: EndpointConfidenceBand,
    ) -> EndpointingDecision:
        if request.conversation_state == ConversationState.SPEAKING:
            return self._decision(
                request=request,
                turn_decision=turn_decision,
                action=EndpointAction.INTERRUPT_RESPONSE,
                pacing=EndpointPacing.INTERRUPT_IMMEDIATE,
                confidence_band=EndpointConfidenceBand.CRITICAL,
                reason="user barge-in while assistant is speaking",
                endpoint_delay_ms=0,
            )

        return self._decision(
            request=request,
            turn_decision=turn_decision,
            action=EndpointAction.INTERRUPT_RESPONSE,
            pacing=pacing,
            confidence_band=confidence_band,
            reason="interruption intent detected",
            endpoint_delay_ms=0,
        )

    def _handle_finalize(
        self,
        *,
        request: EndpointingInput,
        turn_decision: TurnDetectionDecision,
        pacing: EndpointPacing,
        confidence_band: EndpointConfidenceBand,
    ) -> EndpointingDecision:
        if request.conversation_state in {
            ConversationState.IDLE,
            ConversationState.WAITING,
            ConversationState.FOLLOW_UP,
            ConversationState.LISTENING,
            ConversationState.USER_THINKING,
        }:
            action = (
                EndpointAction.START_COGNITION
                if turn_decision.confidence >= 0.75
                else EndpointAction.PREPARE_RESPONSE
            )
            reason = "turn finalized; cognition can start"

            return self._decision(
                request=request,
                turn_decision=turn_decision,
                action=action,
                pacing=pacing,
                confidence_band=confidence_band,
                reason=reason,
                endpoint_delay_ms=0,
            )

        if request.conversation_state == ConversationState.SPEAKING:
            return self._decision(
                request=request,
                turn_decision=turn_decision,
                action=EndpointAction.INTERRUPT_RESPONSE,
                pacing=EndpointPacing.INTERRUPT_IMMEDIATE,
                confidence_band=confidence_band,
                reason="new finalized user turn arrived during assistant speech",
                endpoint_delay_ms=0,
            )

        return self._decision(
            request=request,
            turn_decision=turn_decision,
            action=EndpointAction.WAIT_FOR_USER,
            pacing=pacing,
            confidence_band=confidence_band,
            reason="turn finalized but runtime state is not ready for cognition",
            endpoint_delay_ms=self._config.low_confidence_wait_ms,
        )

    def _handle_maybe_complete(
        self,
        *,
        request: EndpointingInput,
        turn_decision: TurnDetectionDecision,
        pacing: EndpointPacing,
        confidence_band: EndpointConfidenceBand,
    ) -> EndpointingDecision:
        if (
            request.consecutive_maybe_complete_count
            >= self._config.maybe_complete_to_prepare_count
            and turn_decision.completeness
            in {
                TranscriptCompleteness.LIKELY_COMPLETE,
                TranscriptCompleteness.COMPLETE,
            }
        ):
            return self._decision(
                request=request,
                turn_decision=turn_decision,
                action=EndpointAction.PREPARE_RESPONSE,
                pacing=pacing,
                confidence_band=confidence_band,
                reason="stable maybe-complete turn can prepare response",
                endpoint_delay_ms=turn_decision.endpoint_delay_ms,
            )

        delay = turn_decision.endpoint_delay_ms

        if request.conversation_state == ConversationState.FOLLOW_UP:
            delay += self._config.follow_up_extra_ms

        return self._decision(
            request=request,
            turn_decision=turn_decision,
            action=EndpointAction.WAIT_FOR_USER,
            pacing=pacing,
            confidence_band=confidence_band,
            reason="maybe complete; wait for more evidence before responding",
            endpoint_delay_ms=delay,
        )

    def _decision(
        self,
        *,
        request: EndpointingInput,
        turn_decision: TurnDetectionDecision,
        action: EndpointAction,
        pacing: EndpointPacing,
        confidence_band: EndpointConfidenceBand,
        reason: str,
        endpoint_delay_ms: int,
    ) -> EndpointingDecision:
        return EndpointingDecision(
            turn_decision=turn_decision,
            action=action,
            pacing=pacing,
            confidence_band=confidence_band,
            conversation_state=request.conversation_state,
            should_start_cognition=action == EndpointAction.START_COGNITION,
            should_cancel_response=action
            in {
                EndpointAction.INTERRUPT_RESPONSE,
                EndpointAction.CANCEL_RESPONSE,
            },
            should_wait_for_more_audio=action
            in {
                EndpointAction.KEEP_LISTENING,
                EndpointAction.WAIT_FOR_USER,
                EndpointAction.PREPARE_RESPONSE,
            },
            endpoint_delay_ms=max(0, endpoint_delay_ms),
            reason=reason,
            metadata={
                "engine": self.name,
                "turn_reason": turn_decision.reason.value,
                "turn_confidence": turn_decision.confidence,
                "conversation_mode": request.signal.conversation_mode.value,
                "consecutive_maybe_complete_count": (
                    request.consecutive_maybe_complete_count
                ),
                "user_pause_count": request.user_pause_count,
            },
        )

    def _pacing(
        self,
        *,
        request: EndpointingInput,
        turn_decision: TurnDetectionDecision,
    ) -> EndpointPacing:
        if turn_decision.interruption_intent != InterruptionIntent.NONE:
            return EndpointPacing.INTERRUPT_IMMEDIATE

        if turn_decision.urgency in {
            TurnUrgency.HIGH,
            TurnUrgency.CRITICAL,
        }:
            return EndpointPacing.FAST_COMMAND

        if request.conversation_state == ConversationState.USER_THINKING:
            return EndpointPacing.THINKING_PAUSE

        if turn_decision.completeness == TranscriptCompleteness.INCOMPLETE:
            return EndpointPacing.THINKING_PAUSE

        if request.signal.conversation_mode == ConversationMode.QUESTION:
            return EndpointPacing.NORMAL_QUESTION

        if request.signal.conversation_mode == ConversationMode.DISCUSSION:
            return EndpointPacing.PATIENT_DISCUSSION

        if turn_decision.reason == TurnEndpointReason.COMPLETE_QUESTION:
            return EndpointPacing.NORMAL_QUESTION

        return EndpointPacing.PATIENT_DISCUSSION

    @staticmethod
    def _confidence_band(confidence: float) -> EndpointConfidenceBand:
        if confidence >= 0.92:
            return EndpointConfidenceBand.CRITICAL

        if confidence >= 0.8:
            return EndpointConfidenceBand.HIGH

        if confidence >= 0.55:
            return EndpointConfidenceBand.MEDIUM

        return EndpointConfidenceBand.LOW

    def _record_decision(self, decision: EndpointingDecision) -> None:
        with self._lock:
            self._last_action = decision.action
            self._last_pacing = decision.pacing

            if decision.action == EndpointAction.START_COGNITION:
                self._finalized_count += 1

            elif decision.action == EndpointAction.CANCEL_RESPONSE:
                self._cancel_count += 1

            elif decision.action == EndpointAction.INTERRUPT_RESPONSE:
                self._interrupt_count += 1

            elif decision.waiting:
                self._wait_count += 1