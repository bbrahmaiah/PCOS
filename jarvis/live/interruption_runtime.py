from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum

from jarvis.live.audio_runtime import LiveAudioRuntime, LiveAudioRuntimeResult
from jarvis.live.contracts import (
    LiveEventKind,
    LiveEventPriority,
    LiveResponse,
    LiveSessionConfig,
    LiveSubsystem,
    LiveTranscript,
    LiveTranscriptKind,
    LiveTurnId,
    default_live_session_config,
    make_live_event,
    make_live_transcript,
    utc_now,
)
from jarvis.live.dialogue_runtime import (
    LiveDialogueRequest,
    LiveDialogueResult,
    LiveDialogueRuntime,
    LiveDialogueRuntimeStatus,
)
from jarvis.live.event_bridge import (
    LiveEventBridgeRequest,
    LiveEventBridgeResult,
    LiveEventBridgeRuntime,
    LiveEventBridgeStatus,
)
from jarvis.live.response_boundary import LiveResponseSurface
from jarvis.live.session_state import LiveSessionStateRuntime


class LiveInterruptionRuntimeStatus(StrEnum):
    READY = "ready"
    BLOCKED = "blocked"


class LiveInterruptionOperation(StrEnum):
    CAPTURE_CONTEXT = "capture_context"
    REQUEST_INTERRUPT = "request_interrupt"
    HANDLE_INTERRUPT_TRANSCRIPT = "handle_interrupt_transcript"
    RESUME = "resume"
    CANCEL_RESUME = "cancel_resume"
    SNAPSHOT = "snapshot"


class LiveInterruptionKind(StrEnum):
    STOP = "stop"
    PAUSE = "pause"
    CANCEL = "cancel"
    CORRECTION = "correction"
    QUESTION = "question"
    REPEAT = "repeat"
    CONTINUE = "continue"
    UNKNOWN = "unknown"


class LiveInterruptionDisposition(StrEnum):
    STOP_OUTPUT = "stop_output"
    PAUSE_AND_LISTEN = "pause_and_listen"
    CANCEL_CURRENT_TURN = "cancel_current_turn"
    ANSWER_INTERRUPTION = "answer_interruption"
    REPEAT_LAST = "repeat_last"
    RESUME_PREVIOUS = "resume_previous"
    IGNORE = "ignore"


@dataclass(frozen=True, slots=True)
class LiveInterruptedContext:
    turn_id: LiveTurnId | None
    active_topic: str | None
    transcript: LiveTranscript | None
    response: LiveResponse | None
    resume_summary: str
    captured_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def has_resume_point(self) -> bool:
        return bool(
            self.turn_id is not None
            or self.active_topic
            or self.transcript is not None
            or self.response is not None
        )


@dataclass(frozen=True, slots=True)
class LiveInterruptionPolicy:
    stop_words: tuple[str, ...] = ("stop", "shut up")
    pause_words: tuple[str, ...] = ("wait", "hold on", "pause")
    cancel_words: tuple[str, ...] = ("cancel", "forget that")
    repeat_words: tuple[str, ...] = ("repeat", "repeat that", "say that again")
    continue_words: tuple[str, ...] = (
        "continue",
        "go on",
        "continue from there",
        "continue where you stopped",
    )
    correction_markers: tuple[str, ...] = (
        "no",
        "actually",
        "wait actually",
        "not that",
    )
    min_confidence: float = 0.35
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("interruption min_confidence must be between 0 and 1.")


@dataclass(frozen=True, slots=True)
class LiveInterruptionRequest:
    text: str
    confidence: float
    assistant_is_speaking: bool
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("interruption confidence must be between 0 and 1.")


@dataclass(frozen=True, slots=True)
class LiveInterruptionResult:
    status: LiveInterruptionRuntimeStatus
    operation: LiveInterruptionOperation
    kind: LiveInterruptionKind
    disposition: LiveInterruptionDisposition
    interrupted_context: LiveInterruptedContext | None
    bridge_result: LiveEventBridgeResult | None
    audio_result: LiveAudioRuntimeResult | None
    dialogue_result: LiveDialogueResult | None
    reason: str
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def succeeded(self) -> bool:
        return self.status == LiveInterruptionRuntimeStatus.READY

    @property
    def should_resume(self) -> bool:
        return self.disposition == LiveInterruptionDisposition.RESUME_PREVIOUS


@dataclass(frozen=True, slots=True)
class LiveInterruptionSnapshot:
    status: LiveInterruptionRuntimeStatus
    active_context: LiveInterruptedContext | None
    captured_context_count: int
    interruption_count: int
    resume_count: int
    cancelled_count: int
    blocked_count: int
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)


class LiveInterruptionRuntime:
    """
    Step 50G Live Interruption Runtime.

    Handles natural interruption behavior:
    - stop current output
    - preserve unfinished context
    - route interruption transcript through dialogue
    - support repeat / continue / resume
    - keep all final words generated through response boundary

    It does not:
    - hardcode user-facing speech
    - call TTS directly
    - execute tools
    - access memory directly
    - bypass live event bridge
    """

    def __init__(
        self,
        *,
        live_state: LiveSessionStateRuntime | None = None,
        bridge: LiveEventBridgeRuntime | None = None,
        audio: LiveAudioRuntime | None = None,
        dialogue: LiveDialogueRuntime | None = None,
        config: LiveSessionConfig | None = None,
        policy: LiveInterruptionPolicy | None = None,
    ) -> None:
        self._config = config or default_live_session_config()
        self._state = live_state or LiveSessionStateRuntime(config=self._config)
        self._bridge = bridge or LiveEventBridgeRuntime(live_state=self._state)
        self._audio = audio
        self._dialogue = dialogue or LiveDialogueRuntime(
            live_state=self._state,
            bridge=self._bridge,
        )
        self._policy = policy or LiveInterruptionPolicy()
        self._active_context: LiveInterruptedContext | None = None
        self._captured_context_count = 0
        self._interruption_count = 0
        self._resume_count = 0
        self._cancelled_count = 0
        self._blocked_count = 0

    @property
    def live_state(self) -> LiveSessionStateRuntime:
        return self._state

    @property
    def active_context(self) -> LiveInterruptedContext | None:
        return self._active_context

    def capture_context(
        self,
        *,
        reason: str,
        metadata: dict[str, object] | None = None,
    ) -> LiveInterruptionResult:
        if not reason.strip():
            raise ValueError("interruption capture reason cannot be empty.")

        context = self._capture_context(
            reason=reason.strip(),
            metadata=metadata or {},
        )
        self._active_context = context
        self._captured_context_count += 1

        return LiveInterruptionResult(
            status=LiveInterruptionRuntimeStatus.READY,
            operation=LiveInterruptionOperation.CAPTURE_CONTEXT,
            kind=LiveInterruptionKind.UNKNOWN,
            disposition=LiveInterruptionDisposition.PAUSE_AND_LISTEN,
            interrupted_context=context,
            bridge_result=None,
            audio_result=None,
            dialogue_result=None,
            reason="interrupted context captured",
            created_at=utc_now(),
            metadata=metadata or {},
        )

    def request_interrupt(
        self,
        request: LiveInterruptionRequest,
    ) -> LiveInterruptionResult:
        if request.confidence < self._policy.min_confidence:
            return self._blocked(
                operation=LiveInterruptionOperation.REQUEST_INTERRUPT,
                kind=LiveInterruptionKind.UNKNOWN,
                disposition=LiveInterruptionDisposition.IGNORE,
                reason="interruption confidence below policy threshold",
                metadata=request.metadata,
            )

        kind = _classify_interruption(request.text, self._policy)
        disposition = _disposition_for_kind(kind)

        context = self._capture_context(
            reason=request.text,
            metadata=request.metadata,
        )
        self._active_context = context
        self._captured_context_count += 1

        audio_result: LiveAudioRuntimeResult | None = None
        if self._audio is not None and request.assistant_is_speaking:
            audio_result = self._audio.stop_output(reason=request.text)

        event = make_live_event(
            kind=LiveEventKind.INTERRUPTION_REQUESTED,
            priority=LiveEventPriority.CRITICAL,
            source=LiveSubsystem.INTERRUPTION,
            title="Live interruption requested",
            summary="Live interruption was requested by user speech.",
            metadata={
                **request.metadata,
                "interruption_kind": kind.value,
                "disposition": disposition.value,
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
            self._blocked_count += 1
            return LiveInterruptionResult(
                status=LiveInterruptionRuntimeStatus.BLOCKED,
                operation=LiveInterruptionOperation.REQUEST_INTERRUPT,
                kind=kind,
                disposition=disposition,
                interrupted_context=context,
                bridge_result=bridge_result,
                audio_result=audio_result,
                dialogue_result=None,
                reason=bridge_result.reason,
                created_at=utc_now(),
                metadata=request.metadata,
            )

        self._interruption_count += 1

        return LiveInterruptionResult(
            status=LiveInterruptionRuntimeStatus.READY,
            operation=LiveInterruptionOperation.REQUEST_INTERRUPT,
            kind=kind,
            disposition=disposition,
            interrupted_context=context,
            bridge_result=bridge_result,
            audio_result=audio_result,
            dialogue_result=None,
            reason="live interruption routed through bridge",
            created_at=utc_now(),
            metadata=request.metadata,
        )

    def handle_interrupt_transcript(
        self,
        transcript: LiveTranscript,
        *,
        metadata: dict[str, object] | None = None,
    ) -> LiveInterruptionResult:
        kind = _classify_interruption(transcript.text, self._policy)
        disposition = _disposition_for_kind(kind)

        if kind == LiveInterruptionKind.CONTINUE:
            return self.resume(metadata=metadata or {})

        if kind == LiveInterruptionKind.REPEAT:
            return self._handle_repeat(
                transcript=transcript,
                metadata=metadata or {},
            )

        dialogue_result = self._dialogue.process_transcript(
            LiveDialogueRequest(
                transcript=transcript,
                user_is_speaking=True,
                assistant_is_speaking=False,
                response_surface=LiveResponseSurface.VOICE,
                metadata={
                    **(metadata or {}),
                    "interruption_kind": kind.value,
                    "interrupted_context": (
                        self._active_context.resume_summary
                        if self._active_context is not None
                        else ""
                    ),
                },
            )
        )

        if dialogue_result.status == LiveDialogueRuntimeStatus.BLOCKED:
            self._blocked_count += 1
            return LiveInterruptionResult(
                status=LiveInterruptionRuntimeStatus.BLOCKED,
                operation=LiveInterruptionOperation.HANDLE_INTERRUPT_TRANSCRIPT,
                kind=kind,
                disposition=disposition,
                interrupted_context=self._active_context,
                bridge_result=None,
                audio_result=None,
                dialogue_result=dialogue_result,
                reason=dialogue_result.reason,
                created_at=utc_now(),
                metadata=metadata or {},
            )

        return LiveInterruptionResult(
            status=LiveInterruptionRuntimeStatus.READY,
            operation=LiveInterruptionOperation.HANDLE_INTERRUPT_TRANSCRIPT,
            kind=kind,
            disposition=disposition,
            interrupted_context=self._active_context,
            bridge_result=None,
            audio_result=None,
            dialogue_result=dialogue_result,
            reason="interruption transcript handled through dialogue runtime",
            created_at=utc_now(),
            metadata=metadata or {},
        )

    def resume(
        self,
        *,
        metadata: dict[str, object] | None = None,
    ) -> LiveInterruptionResult:
        if self._active_context is None or not self._active_context.has_resume_point:
            return self._blocked(
                operation=LiveInterruptionOperation.RESUME,
                kind=LiveInterruptionKind.CONTINUE,
                disposition=LiveInterruptionDisposition.RESUME_PREVIOUS,
                reason="no interrupted context is available to resume",
                metadata=metadata or {},
            )

        turn_id = self._active_context.turn_id or self._state.state.current_turn_id
        if turn_id is None:
            return self._blocked(
                operation=LiveInterruptionOperation.RESUME,
                kind=LiveInterruptionKind.CONTINUE,
                disposition=LiveInterruptionDisposition.RESUME_PREVIOUS,
                reason="no turn id is available for resume",
                metadata=metadata or {},
            )

        transcript = make_live_transcript(
            turn_id=turn_id,
            kind=LiveTranscriptKind.FINAL,
            text=(
                "continue from the preserved interrupted context: "
                f"{self._active_context.resume_summary}"
            ),
            confidence=1.0,
            metadata={
                **(metadata or {}),
                "resume": True,
                "interrupted_context": self._active_context.resume_summary,
            },
        )
        dialogue_result = self._dialogue.process_transcript(
            LiveDialogueRequest(
                transcript=transcript,
                response_surface=LiveResponseSurface.VOICE,
                metadata={
                    **(metadata or {}),
                    "resume": True,
                    "interrupted_context": self._active_context.resume_summary,
                },
            )
        )

        if dialogue_result.status == LiveDialogueRuntimeStatus.BLOCKED:
            self._blocked_count += 1
            return LiveInterruptionResult(
                status=LiveInterruptionRuntimeStatus.BLOCKED,
                operation=LiveInterruptionOperation.RESUME,
                kind=LiveInterruptionKind.CONTINUE,
                disposition=LiveInterruptionDisposition.RESUME_PREVIOUS,
                interrupted_context=self._active_context,
                bridge_result=None,
                audio_result=None,
                dialogue_result=dialogue_result,
                reason=dialogue_result.reason,
                created_at=utc_now(),
                metadata=metadata or {},
            )

        self._resume_count += 1
        return LiveInterruptionResult(
            status=LiveInterruptionRuntimeStatus.READY,
            operation=LiveInterruptionOperation.RESUME,
            kind=LiveInterruptionKind.CONTINUE,
            disposition=LiveInterruptionDisposition.RESUME_PREVIOUS,
            interrupted_context=self._active_context,
            bridge_result=None,
            audio_result=None,
            dialogue_result=dialogue_result,
            reason="interrupted context resumed through dialogue runtime",
            created_at=utc_now(),
            metadata=metadata or {},
        )

    def cancel_resume(
        self,
        *,
        reason: str,
        metadata: dict[str, object] | None = None,
    ) -> LiveInterruptionResult:
        if not reason.strip():
            raise ValueError("cancel resume reason cannot be empty.")

        context = self._active_context
        self._active_context = None
        self._cancelled_count += 1

        return LiveInterruptionResult(
            status=LiveInterruptionRuntimeStatus.READY,
            operation=LiveInterruptionOperation.CANCEL_RESUME,
            kind=LiveInterruptionKind.CANCEL,
            disposition=LiveInterruptionDisposition.CANCEL_CURRENT_TURN,
            interrupted_context=context,
            bridge_result=None,
            audio_result=None,
            dialogue_result=None,
            reason="interrupted context cancelled",
            created_at=utc_now(),
            metadata={**(metadata or {}), "cancel_reason": reason.strip()},
        )

    def snapshot(self) -> LiveInterruptionSnapshot:
        return LiveInterruptionSnapshot(
            status=LiveInterruptionRuntimeStatus.READY,
            active_context=self._active_context,
            captured_context_count=self._captured_context_count,
            interruption_count=self._interruption_count,
            resume_count=self._resume_count,
            cancelled_count=self._cancelled_count,
            blocked_count=self._blocked_count,
            created_at=utc_now(),
        )

    def _handle_repeat(
        self,
        *,
        transcript: LiveTranscript,
        metadata: dict[str, object],
    ) -> LiveInterruptionResult:
        if self._active_context is None or self._active_context.response is None:
            return self._blocked(
                operation=LiveInterruptionOperation.HANDLE_INTERRUPT_TRANSCRIPT,
                kind=LiveInterruptionKind.REPEAT,
                disposition=LiveInterruptionDisposition.REPEAT_LAST,
                reason="no previous response is available to repeat",
                metadata=metadata,
            )

        dialogue_result = self._dialogue.process_transcript(
            LiveDialogueRequest(
                transcript=transcript,
                response_surface=LiveResponseSurface.VOICE,
                metadata={
                    **metadata,
                    "repeat": True,
                    "previous_response_id": str(
                        self._active_context.response.response_id
                    ),
                    "previous_response_context": self._active_context.resume_summary,
                },
            )
        )

        if dialogue_result.status == LiveDialogueRuntimeStatus.BLOCKED:
            self._blocked_count += 1
            return LiveInterruptionResult(
                status=LiveInterruptionRuntimeStatus.BLOCKED,
                operation=LiveInterruptionOperation.HANDLE_INTERRUPT_TRANSCRIPT,
                kind=LiveInterruptionKind.REPEAT,
                disposition=LiveInterruptionDisposition.REPEAT_LAST,
                interrupted_context=self._active_context,
                bridge_result=None,
                audio_result=None,
                dialogue_result=dialogue_result,
                reason=dialogue_result.reason,
                created_at=utc_now(),
                metadata=metadata,
            )

        return LiveInterruptionResult(
            status=LiveInterruptionRuntimeStatus.READY,
            operation=LiveInterruptionOperation.HANDLE_INTERRUPT_TRANSCRIPT,
            kind=LiveInterruptionKind.REPEAT,
            disposition=LiveInterruptionDisposition.REPEAT_LAST,
            interrupted_context=self._active_context,
            bridge_result=None,
            audio_result=None,
            dialogue_result=dialogue_result,
            reason="repeat request routed through dialogue runtime",
            created_at=utc_now(),
            metadata=metadata,
        )

    def _capture_context(
        self,
        *,
        reason: str,
        metadata: dict[str, object],
    ) -> LiveInterruptedContext:
        state = self._state.state
        resume_summary = _build_resume_summary(
            active_topic=state.active_topic,
            transcript=state.last_transcript,
            response=state.last_response,
            reason=reason,
        )
        return LiveInterruptedContext(
            turn_id=state.current_turn_id,
            active_topic=state.active_topic,
            transcript=state.last_transcript,
            response=state.last_response,
            resume_summary=resume_summary,
            captured_at=utc_now(),
            metadata=metadata,
        )

    def _blocked(
        self,
        *,
        operation: LiveInterruptionOperation,
        kind: LiveInterruptionKind,
        disposition: LiveInterruptionDisposition,
        reason: str,
        metadata: dict[str, object],
    ) -> LiveInterruptionResult:
        self._blocked_count += 1
        return LiveInterruptionResult(
            status=LiveInterruptionRuntimeStatus.BLOCKED,
            operation=operation,
            kind=kind,
            disposition=disposition,
            interrupted_context=self._active_context,
            bridge_result=None,
            audio_result=None,
            dialogue_result=None,
            reason=reason,
            created_at=utc_now(),
            metadata=metadata,
        )


def _classify_interruption(
    text: str,
    policy: LiveInterruptionPolicy,
) -> LiveInterruptionKind:
    normalized = _normalize(text)

    if _contains_any(normalized, policy.repeat_words):
        return LiveInterruptionKind.REPEAT

    if _contains_any(normalized, policy.continue_words):
        return LiveInterruptionKind.CONTINUE

    if _contains_any(normalized, policy.cancel_words):
        return LiveInterruptionKind.CANCEL

    if _contains_any(normalized, policy.pause_words):
        return LiveInterruptionKind.PAUSE

    if _contains_any(normalized, policy.stop_words):
        return LiveInterruptionKind.STOP

    if _contains_any(normalized, policy.correction_markers):
        return LiveInterruptionKind.CORRECTION

    if "?" in text:
        return LiveInterruptionKind.QUESTION

    return LiveInterruptionKind.UNKNOWN


def _disposition_for_kind(
    kind: LiveInterruptionKind,
) -> LiveInterruptionDisposition:
    if kind == LiveInterruptionKind.STOP:
        return LiveInterruptionDisposition.STOP_OUTPUT
    if kind == LiveInterruptionKind.PAUSE:
        return LiveInterruptionDisposition.PAUSE_AND_LISTEN
    if kind == LiveInterruptionKind.CANCEL:
        return LiveInterruptionDisposition.CANCEL_CURRENT_TURN
    if kind == LiveInterruptionKind.REPEAT:
        return LiveInterruptionDisposition.REPEAT_LAST
    if kind == LiveInterruptionKind.CONTINUE:
        return LiveInterruptionDisposition.RESUME_PREVIOUS
    if kind in {LiveInterruptionKind.CORRECTION, LiveInterruptionKind.QUESTION}:
        return LiveInterruptionDisposition.ANSWER_INTERRUPTION
    return LiveInterruptionDisposition.PAUSE_AND_LISTEN


def _build_resume_summary(
    *,
    active_topic: str | None,
    transcript: LiveTranscript | None,
    response: LiveResponse | None,
    reason: str,
) -> str:
    parts: list[str] = []

    if active_topic:
        parts.append(f"topic={active_topic}")

    if transcript is not None:
        parts.append(f"transcript={transcript.text}")

    if response is not None:
        parts.append(f"response_id={response.response_id}")

    parts.append(f"interruption={reason}")

    return " | ".join(parts)


def _contains_any(text: str, candidates: tuple[str, ...]) -> bool:
    return any(_normalize(candidate) in text for candidate in candidates)


def _normalize(text: str) -> str:
    return " ".join(text.casefold().strip().split())