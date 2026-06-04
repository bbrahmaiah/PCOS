from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum

from jarvis.live.contracts import (
    LiveEventKind,
    LiveEventPriority,
    LiveInteractionState,
    LiveSessionConfig,
    LiveSubsystem,
    LiveWakeState,
    default_live_session_config,
    make_live_event,
    utc_now,
)
from jarvis.live.event_bridge import (
    LiveEventBridgeRequest,
    LiveEventBridgeResult,
    LiveEventBridgeRuntime,
    LiveEventBridgeStatus,
)
from jarvis.live.session_state import LiveSessionStateRuntime


class LiveWakeEngagementStatus(StrEnum):
    READY = "ready"
    BLOCKED = "blocked"


class LiveWakeEngagementOperation(StrEnum):
    EVALUATE = "evaluate"
    ENGAGE = "engage"
    DISENGAGE = "disengage"
    SNAPSHOT = "snapshot"


class LiveEngagementDecision(StrEnum):
    IGNORE = "ignore"
    LISTEN_FOR_WAKE = "listen_for_wake"
    ENGAGE = "engage"
    CONTINUE_ACTIVE_SESSION = "continue_active_session"
    DISENGAGE = "disengage"
    INTERRUPT = "interrupt"


class LiveEngagementReason(StrEnum):
    NO_SPEECH = "no_speech"
    WAKE_WORD_DETECTED = "wake_word_detected"
    ALREADY_ENGAGED = "already_engaged"
    USER_CONTINUED = "user_continued"
    USER_DISMISSED = "user_dismissed"
    USER_INTERRUPTED = "user_interrupted"
    BACKGROUND_SPEECH = "background_speech"
    LOW_CONFIDENCE = "low_confidence"


@dataclass(frozen=True, slots=True)
class LiveWakeEngagementPolicy:
    wake_word: str = "jarvis"
    min_speech_probability: float = 0.35
    require_wake_word_when_sleeping: bool = True
    allow_continuation_when_engaged: bool = True
    interrupt_words: tuple[str, ...] = (
        "stop",
        "wait",
        "hold on",
        "pause",
        "cancel",
    )
    dismiss_words: tuple[str, ...] = (
        "sleep",
        "standby",
        "go quiet",
        "stop listening",
    )

    def __post_init__(self) -> None:
        if not self.wake_word.strip():
            raise ValueError("wake engagement wake_word cannot be empty.")
        if not 0.0 <= self.min_speech_probability <= 1.0:
            raise ValueError(
                "min_speech_probability must be between 0 and 1."
            )


@dataclass(frozen=True, slots=True)
class LiveWakeEngagementRequest:
    text: str
    speech_probability: float
    assistant_is_speaking: bool = False
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.speech_probability <= 1.0:
            raise ValueError("speech_probability must be between 0 and 1.")


@dataclass(frozen=True, slots=True)
class LiveWakeEngagementResult:
    status: LiveWakeEngagementStatus
    operation: LiveWakeEngagementOperation
    decision: LiveEngagementDecision
    reason: LiveEngagementReason
    bridge_result: LiveEventBridgeResult | None
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def succeeded(self) -> bool:
        return self.status == LiveWakeEngagementStatus.READY

    @property
    def engaged(self) -> bool:
        return self.decision in {
            LiveEngagementDecision.ENGAGE,
            LiveEngagementDecision.CONTINUE_ACTIVE_SESSION,
        }


@dataclass(frozen=True, slots=True)
class LiveWakeEngagementSnapshot:
    status: LiveWakeEngagementStatus
    wake_state: LiveWakeState
    evaluated_count: int
    engaged_count: int
    ignored_count: int
    interrupted_count: int
    disengaged_count: int
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)


class LiveWakeEngagementRuntime:
    """
    Step 50E Wake / Engagement Runtime.

    Decides whether the live session should engage, ignore, continue,
    interrupt, or disengage.

    It does not:
    - generate conversational responses
    - call TTS
    - execute tools
    - access memory directly
    - bypass LiveEventBridgeRuntime

    This runtime only emits typed live events into the live bridge.
    """

    def __init__(
        self,
        *,
        live_state: LiveSessionStateRuntime | None = None,
        bridge: LiveEventBridgeRuntime | None = None,
        config: LiveSessionConfig | None = None,
        policy: LiveWakeEngagementPolicy | None = None,
    ) -> None:
        self._config = config or default_live_session_config()
        self._state = live_state or LiveSessionStateRuntime(config=self._config)
        self._bridge = bridge or LiveEventBridgeRuntime(live_state=self._state)
        self._policy = policy or LiveWakeEngagementPolicy(
            wake_word=self._config.wake_word
        )
        self._evaluated_count = 0
        self._engaged_count = 0
        self._ignored_count = 0
        self._interrupted_count = 0
        self._disengaged_count = 0

    @property
    def live_state(self) -> LiveSessionStateRuntime:
        return self._state

    @property
    def bridge(self) -> LiveEventBridgeRuntime:
        return self._bridge

    def evaluate(
        self,
        request: LiveWakeEngagementRequest,
    ) -> LiveWakeEngagementResult:
        self._evaluated_count += 1

        text = _normalize(request.text)
        if not text:
            return self._ignored(
                reason=LiveEngagementReason.NO_SPEECH,
                metadata=request.metadata,
            )

        if request.speech_probability < self._policy.min_speech_probability:
            return self._ignored(
                reason=LiveEngagementReason.LOW_CONFIDENCE,
                metadata=request.metadata,
            )

        if _contains_any(text, self._policy.interrupt_words):
            return self._interrupt(request=request)

        if _contains_any(text, self._policy.dismiss_words):
            return self._disengage(request=request)

        if _contains_wake_word(text, self._policy.wake_word):
            return self._engage(
                request=request,
                reason=LiveEngagementReason.WAKE_WORD_DETECTED,
            )

        if self._is_active_session():
            if self._policy.allow_continuation_when_engaged:
                return self._continue_active(request=request)

        if self._policy.require_wake_word_when_sleeping:
            return self._ignored(
                reason=LiveEngagementReason.BACKGROUND_SPEECH,
                metadata=request.metadata,
            )

        return self._engage(
            request=request,
            reason=LiveEngagementReason.USER_CONTINUED,
        )

    def engage(
        self,
        *,
        reason: str,
        metadata: dict[str, object] | None = None,
    ) -> LiveWakeEngagementResult:
        if not reason.strip():
            raise ValueError("wake engagement reason cannot be empty.")

        request = LiveWakeEngagementRequest(
            text=reason,
            speech_probability=1.0,
            metadata=metadata or {},
        )
        return self._engage(
            request=request,
            reason=LiveEngagementReason.WAKE_WORD_DETECTED,
        )

    def disengage(
        self,
        *,
        reason: str,
        metadata: dict[str, object] | None = None,
    ) -> LiveWakeEngagementResult:
        if not reason.strip():
            raise ValueError("wake disengagement reason cannot be empty.")

        request = LiveWakeEngagementRequest(
            text=reason,
            speech_probability=1.0,
            metadata=metadata or {},
        )
        return self._disengage(request=request)

    def snapshot(self) -> LiveWakeEngagementSnapshot:
        return LiveWakeEngagementSnapshot(
            status=LiveWakeEngagementStatus.READY,
            wake_state=self._state.state.wake_state,
            evaluated_count=self._evaluated_count,
            engaged_count=self._engaged_count,
            ignored_count=self._ignored_count,
            interrupted_count=self._interrupted_count,
            disengaged_count=self._disengaged_count,
            created_at=utc_now(),
        )

    def _engage(
        self,
        *,
        request: LiveWakeEngagementRequest,
        reason: LiveEngagementReason,
    ) -> LiveWakeEngagementResult:
        event = make_live_event(
            kind=LiveEventKind.WAKE_DETECTED,
            priority=LiveEventPriority.NORMAL,
            source=LiveSubsystem.WAKE,
            title="Wake engagement event",
            summary="Wake engagement requested through live event bridge.",
            metadata={
                **request.metadata,
                "engagement_reason": reason.value,
                "speech_probability": request.speech_probability,
            },
        )
        bridge_result = self._bridge.bridge_event(
            LiveEventBridgeRequest(
                event=event,
                update_live_state=True,
                update_cognitive_state=True,
                allow_interruptions=True,
                metadata=request.metadata,
            )
        )

        if bridge_result.status == LiveEventBridgeStatus.BLOCKED:
            return self._blocked(
                decision=LiveEngagementDecision.ENGAGE,
                reason=reason,
                bridge_result=bridge_result,
                metadata=request.metadata,
            )

        self._engaged_count += 1
        return LiveWakeEngagementResult(
            status=LiveWakeEngagementStatus.READY,
            operation=LiveWakeEngagementOperation.ENGAGE,
            decision=LiveEngagementDecision.ENGAGE,
            reason=reason,
            bridge_result=bridge_result,
            created_at=utc_now(),
            metadata=request.metadata,
        )

    def _continue_active(
        self,
        *,
        request: LiveWakeEngagementRequest,
    ) -> LiveWakeEngagementResult:
        event = make_live_event(
            kind=LiveEventKind.USER_SPEECH_STARTED,
            priority=LiveEventPriority.NORMAL,
            source=LiveSubsystem.PRESENCE,
            title="Active session continued",
            summary="User continued within the active live session.",
            metadata={
                **request.metadata,
                "speech_probability": request.speech_probability,
            },
        )
        bridge_result = self._bridge.bridge_event(
            LiveEventBridgeRequest(
                event=event,
                update_live_state=True,
                update_cognitive_state=True,
                allow_interruptions=True,
                metadata=request.metadata,
            )
        )

        if bridge_result.status == LiveEventBridgeStatus.BLOCKED:
            return self._blocked(
                decision=LiveEngagementDecision.CONTINUE_ACTIVE_SESSION,
                reason=LiveEngagementReason.USER_CONTINUED,
                bridge_result=bridge_result,
                metadata=request.metadata,
            )

        self._engaged_count += 1
        return LiveWakeEngagementResult(
            status=LiveWakeEngagementStatus.READY,
            operation=LiveWakeEngagementOperation.EVALUATE,
            decision=LiveEngagementDecision.CONTINUE_ACTIVE_SESSION,
            reason=LiveEngagementReason.USER_CONTINUED,
            bridge_result=bridge_result,
            created_at=utc_now(),
            metadata=request.metadata,
        )

    def _interrupt(
        self,
        *,
        request: LiveWakeEngagementRequest,
    ) -> LiveWakeEngagementResult:
        event = make_live_event(
            kind=LiveEventKind.INTERRUPTION_REQUESTED,
            priority=LiveEventPriority.CRITICAL,
            source=LiveSubsystem.INTERRUPTION,
            title="Wake engagement interruption",
            summary="User requested interruption during live engagement.",
            metadata={
                **request.metadata,
                "speech_probability": request.speech_probability,
                "assistant_is_speaking": request.assistant_is_speaking,
            },
        )
        bridge_result = self._bridge.bridge_event(
            LiveEventBridgeRequest(
                event=event,
                update_live_state=True,
                update_cognitive_state=True,
                allow_interruptions=True,
                metadata=request.metadata,
            )
        )

        if bridge_result.status == LiveEventBridgeStatus.BLOCKED:
            return self._blocked(
                decision=LiveEngagementDecision.INTERRUPT,
                reason=LiveEngagementReason.USER_INTERRUPTED,
                bridge_result=bridge_result,
                metadata=request.metadata,
            )

        self._interrupted_count += 1
        return LiveWakeEngagementResult(
            status=LiveWakeEngagementStatus.READY,
            operation=LiveWakeEngagementOperation.EVALUATE,
            decision=LiveEngagementDecision.INTERRUPT,
            reason=LiveEngagementReason.USER_INTERRUPTED,
            bridge_result=bridge_result,
            created_at=utc_now(),
            metadata=request.metadata,
        )

    def _disengage(
        self,
        *,
        request: LiveWakeEngagementRequest,
    ) -> LiveWakeEngagementResult:
        event = make_live_event(
            kind=LiveEventKind.SESSION_STOP_REQUESTED,
            priority=LiveEventPriority.NORMAL,
            source=LiveSubsystem.WAKE,
            title="Wake disengagement event",
            summary="User requested live engagement disengagement.",
            metadata={
                **request.metadata,
                "speech_probability": request.speech_probability,
            },
        )
        bridge_result = self._bridge.bridge_event(
            LiveEventBridgeRequest(
                event=event,
                update_live_state=True,
                update_cognitive_state=True,
                allow_interruptions=False,
                metadata=request.metadata,
            )
        )

        if bridge_result.status == LiveEventBridgeStatus.BLOCKED:
            return self._blocked(
                decision=LiveEngagementDecision.DISENGAGE,
                reason=LiveEngagementReason.USER_DISMISSED,
                bridge_result=bridge_result,
                metadata=request.metadata,
            )

        self._disengaged_count += 1
        return LiveWakeEngagementResult(
            status=LiveWakeEngagementStatus.READY,
            operation=LiveWakeEngagementOperation.DISENGAGE,
            decision=LiveEngagementDecision.DISENGAGE,
            reason=LiveEngagementReason.USER_DISMISSED,
            bridge_result=bridge_result,
            created_at=utc_now(),
            metadata=request.metadata,
        )

    def _ignored(
        self,
        *,
        reason: LiveEngagementReason,
        metadata: dict[str, object],
    ) -> LiveWakeEngagementResult:
        self._ignored_count += 1
        return LiveWakeEngagementResult(
            status=LiveWakeEngagementStatus.READY,
            operation=LiveWakeEngagementOperation.EVALUATE,
            decision=LiveEngagementDecision.IGNORE,
            reason=reason,
            bridge_result=None,
            created_at=utc_now(),
            metadata=metadata,
        )

    def _blocked(
        self,
        *,
        decision: LiveEngagementDecision,
        reason: LiveEngagementReason,
        bridge_result: LiveEventBridgeResult,
        metadata: dict[str, object],
    ) -> LiveWakeEngagementResult:
        return LiveWakeEngagementResult(
            status=LiveWakeEngagementStatus.BLOCKED,
            operation=LiveWakeEngagementOperation.EVALUATE,
            decision=decision,
            reason=reason,
            bridge_result=bridge_result,
            created_at=utc_now(),
            metadata=metadata,
        )

    def _is_active_session(self) -> bool:
        state = self._state.state

        has_real_dialogue_context = (
            state.current_turn_id is not None
            or state.last_transcript is not None
            or state.last_response is not None
            or state.assistant_speaking
        )

        return (
            state.conversation_active
            and has_real_dialogue_context
            and state.interaction_state
            in {
                LiveInteractionState.LISTENING,
                LiveInteractionState.WAITING_FOR_USER,
                LiveInteractionState.USER_SPEAKING,
                LiveInteractionState.THINKING,
                LiveInteractionState.SPEAKING,
            }
        )


def _normalize(text: str) -> str:
    return " ".join(text.casefold().strip().split())


def _contains_wake_word(text: str, wake_word: str) -> bool:
    normalized_wake = _normalize(wake_word)
    words = text.split()
    wake_words = normalized_wake.split()

    if not wake_words:
        return False

    if len(wake_words) == 1:
        return normalized_wake in words

    return normalized_wake in text


def _contains_any(text: str, candidates: tuple[str, ...]) -> bool:
    return any(_normalize(candidate) in text for candidate in candidates)