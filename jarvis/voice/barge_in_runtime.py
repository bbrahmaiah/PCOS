from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Protocol

from jarvis.voice.contracts import (
    VoiceInterruptKind,
    VoiceInterruptSignal,
    VoicePlaybackState,
    VoiceTranscript,
    make_voice_interrupt_id,
    utc_now,
)
from jarvis.voice.playback_runtime import (
    VoicePlaybackResult,
    VoicePlaybackRuntimeStatus,
    VoicePlaybackSnapshot,
)


class VoiceBargeInRuntimeStatus(StrEnum):
    CREATED = "created"
    READY = "ready"
    MONITORING = "monitoring"
    INTERRUPTING = "interrupting"
    INTERRUPTED = "interrupted"
    IGNORED = "ignored"
    DEGRADED = "degraded"
    FAILED = "failed"


class VoiceBargeInOperation(StrEnum):
    PREPARE = "prepare"
    EVALUATE_TRANSCRIPT = "evaluate_transcript"
    EVALUATE_SIGNAL = "evaluate_signal"
    INTERRUPT_PLAYBACK = "interrupt_playback"
    RESET = "reset"
    SNAPSHOT = "snapshot"


class VoiceBargeInDisposition(StrEnum):
    IGNORE = "ignore"
    STOP_PLAYBACK = "stop_playback"
    PAUSE_PLAYBACK = "pause_playback"
    CANCEL_RESPONSE = "cancel_response"
    USER_CORRECTION = "user_correction"
    NEW_QUESTION = "new_question"


@dataclass(frozen=True, slots=True)
class VoiceInterruptedSpeechContext:
    response_text: str
    transcript_text: str
    playback_state: VoicePlaybackState | None
    interrupted_at: datetime
    disposition: VoiceBargeInDisposition
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.response_text.strip():
            raise ValueError("interrupted response_text cannot be empty.")
        if not self.transcript_text.strip():
            raise ValueError("interrupted transcript_text cannot be empty.")


@dataclass(frozen=True, slots=True)
class VoiceBargeInPolicy:
    min_confidence: float = 0.55
    emergency_stop_min_confidence: float = 0.20
    min_semantic_barge_in_stability: float = 0.62
    min_question_words: int = 3
    min_correction_words: int = 2
    require_assistant_speaking: bool = True
    max_stop_latency_ms: int = 200
    stop_phrases: tuple[str, ...] = (
        "stop",
        "wait",
        "hold",
        "hold on",
        "pause",
        "shut up",
        "quiet",
        "enough",
    )
    cancel_phrases: tuple[str, ...] = (
        "cancel",
        "cancel that",
        "forget it",
        "never mind",
        "nevermind",
    )
    correction_markers: tuple[str, ...] = (
        "actually",
        "no",
        "not that",
        "i mean",
        "instead",
        "wait",
    )
    question_markers: tuple[str, ...] = (
        "what",
        "why",
        "how",
        "when",
        "where",
        "explain",
        "compare",
        "tell me",
    )
    wake_words: tuple[str, ...] = ("jarvis", "jervis", "jarves", "bob")
    known_silence_hallucinations: tuple[str, ...] = (
        "thank you",
        "thank you bye bye",
        "bye bye",
        "buh bye",
        "goodbye",
        "see you",
        "see you next time",
        "we ll see you next time",
        "tick",
        "ticks",
        "happy holidays",
        "good luck",
        "good luck yall",
        "good job",
        "i ll be right back",
        "i ll be right back bye",
        "ill be right back",
        "ill be right back bye",
        "wait thank you",
        "wait thank you bye bye",
    )
    assistant_echo_min_overlap: float = 0.72
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be 0..1.")
        if not 0.0 <= self.emergency_stop_min_confidence <= 1.0:
            raise ValueError("emergency_stop_min_confidence must be 0..1.")
        if not 0.0 <= self.min_semantic_barge_in_stability <= 1.0:
            raise ValueError("min_semantic_barge_in_stability must be 0..1.")
        if self.min_question_words < 1:
            raise ValueError("min_question_words must be positive.")
        if self.min_correction_words < 1:
            raise ValueError("min_correction_words must be positive.")
        if self.max_stop_latency_ms <= 0:
            raise ValueError("max_stop_latency_ms must be positive.")
        if not self.stop_phrases:
            raise ValueError("stop_phrases cannot be empty.")
        if not 0.0 <= self.assistant_echo_min_overlap <= 1.0:
            raise ValueError("assistant_echo_min_overlap must be 0..1.")


@dataclass(frozen=True, slots=True)
class VoiceBargeInRequest:
    transcript: VoiceTranscript
    assistant_speaking: bool
    active_response_text: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class VoiceBargeInResult:
    status: VoiceBargeInRuntimeStatus
    operation: VoiceBargeInOperation
    disposition: VoiceBargeInDisposition
    signal: VoiceInterruptSignal | None
    interrupted_context: VoiceInterruptedSpeechContext | None
    playback_result: VoicePlaybackResult | None
    message: str
    latency_ms: float
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def interrupted(self) -> bool:
        return self.status == VoiceBargeInRuntimeStatus.INTERRUPTED


@dataclass(frozen=True, slots=True)
class VoiceBargeInSnapshot:
    status: VoiceBargeInRuntimeStatus
    evaluated_transcripts: int
    ignored: int
    interruptions: int
    failed_interruptions: int
    last_disposition: VoiceBargeInDisposition | None
    last_signal_text: str | None
    last_latency_ms: float | None
    last_stop_latency_ms: float | None
    last_error: str | None
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)


class VoiceBargeInPlaybackController(Protocol):
    def stop(self) -> VoicePlaybackResult:
        raise NotImplementedError

    def snapshot(self) -> VoicePlaybackSnapshot:
        raise NotImplementedError


class VoiceBargeInRuntime:
    """
    Step 51H barge-in interruption runtime.

    This runtime detects user interruption while JARVIS is speaking and stops
    playback quickly. It preserves interrupted speech context so the dialogue
    runtime can continue naturally.

    It does not:
    - generate responses
    - synthesize TTS
    - execute tools
    - bypass Step 50 dialogue/interruption policy
    """

    def __init__(
        self,
        *,
        playback: VoiceBargeInPlaybackController | None = None,
        policy: VoiceBargeInPolicy | None = None,
    ) -> None:
        self._playback = playback
        self._policy = policy or VoiceBargeInPolicy()
        self._status = VoiceBargeInRuntimeStatus.CREATED
        self._evaluated_transcripts = 0
        self._ignored = 0
        self._interruptions = 0
        self._failed_interruptions = 0
        self._last_disposition: VoiceBargeInDisposition | None = None
        self._last_signal_text: str | None = None
        self._last_latency_ms: float | None = None
        self._last_stop_latency_ms: float | None = None
        self._last_error: str | None = None

    def prepare(
        self,
        playback: VoiceBargeInPlaybackController | None = None,
    ) -> VoiceBargeInResult:
        started = time.perf_counter()
        if playback is not None:
            self._playback = playback

        if self._playback is None:
            self._status = VoiceBargeInRuntimeStatus.DEGRADED
            return self._result(
                operation=VoiceBargeInOperation.PREPARE,
                status=self._status,
                disposition=VoiceBargeInDisposition.IGNORE,
                signal=None,
                interrupted_context=None,
                playback_result=None,
                message="barge-in runtime has no playback controller",
                started=started,
            )

        self._status = VoiceBargeInRuntimeStatus.READY
        self._last_error = None
        return self._result(
            operation=VoiceBargeInOperation.PREPARE,
            status=self._status,
            disposition=VoiceBargeInDisposition.IGNORE,
            signal=None,
            interrupted_context=None,
            playback_result=None,
            message="barge-in runtime prepared",
            started=started,
        )

    def evaluate_transcript(
        self,
        request: VoiceBargeInRequest,
    ) -> VoiceBargeInResult:
        started = time.perf_counter()
        self._evaluated_transcripts += 1
        self._status = VoiceBargeInRuntimeStatus.MONITORING

        decision = _classify_barge_in(
            text=request.transcript.text,
            confidence=request.transcript.confidence,
            policy=self._policy,
        )

        self._last_disposition = decision

        if self._policy.require_assistant_speaking and not request.assistant_speaking:
            self._ignored += 1
            self._status = VoiceBargeInRuntimeStatus.IGNORED
            return self._result(
                operation=VoiceBargeInOperation.EVALUATE_TRANSCRIPT,
                status=self._status,
                disposition=VoiceBargeInDisposition.IGNORE,
                signal=None,
                interrupted_context=None,
                playback_result=None,
                message="ignored because assistant is not speaking",
                started=started,
            )

        if decision == VoiceBargeInDisposition.IGNORE:
            self._ignored += 1
            self._status = VoiceBargeInRuntimeStatus.IGNORED
            return self._result(
                operation=VoiceBargeInOperation.EVALUATE_TRANSCRIPT,
                status=self._status,
                disposition=decision,
                signal=None,
                interrupted_context=None,
                playback_result=None,
                message="transcript is not a barge-in interruption",
                started=started,
                metadata={
                    "confidence": request.transcript.confidence,
                    "kind": request.transcript.kind.value,
                },
            )

        if _looks_like_assistant_echo(
            transcript_text=request.transcript.text,
            active_response_text=request.active_response_text,
            policy=self._policy,
        ):
            self._ignored += 1
            self._status = VoiceBargeInRuntimeStatus.IGNORED
            return self._result(
                operation=VoiceBargeInOperation.EVALUATE_TRANSCRIPT,
                status=self._status,
                disposition=VoiceBargeInDisposition.IGNORE,
                signal=None,
                interrupted_context=None,
                playback_result=None,
                message="barge-in ignored as assistant playback echo",
                started=started,
                metadata={
                    "candidate_disposition": decision.value,
                    "confidence": request.transcript.confidence,
                    "kind": request.transcript.kind.value,
                    "echo_guard": True,
                },
            )

        if not _semantic_barge_in_is_stable(
            transcript=request.transcript,
            disposition=decision,
            policy=self._policy,
        ):
            self._ignored += 1
            self._status = VoiceBargeInRuntimeStatus.IGNORED
            return self._result(
                operation=VoiceBargeInOperation.EVALUATE_TRANSCRIPT,
                status=self._status,
                disposition=VoiceBargeInDisposition.IGNORE,
                signal=None,
                interrupted_context=None,
                playback_result=None,
                message="semantic barge-in ignored until transcript is stable",
                started=started,
                metadata={
                    "candidate_disposition": decision.value,
                    "confidence": request.transcript.confidence,
                    "kind": request.transcript.kind.value,
                    "perception": request.transcript.metadata.get("perception"),
                },
            )

        signal = _make_signal_from_transcript(
            transcript=request.transcript,
            disposition=decision,
        )
        self._last_signal_text = signal.text

        return self.interrupt_playback(
            signal=signal,
            disposition=decision,
            active_response_text=request.active_response_text,
            started=started,
            metadata=request.metadata,
        )

    def evaluate_signal(
        self,
        signal: VoiceInterruptSignal,
        *,
        active_response_text: str | None = None,
    ) -> VoiceBargeInResult:
        started = time.perf_counter()
        disposition = _disposition_from_signal(signal)
        self._last_disposition = disposition
        self._last_signal_text = signal.text

        if disposition == VoiceBargeInDisposition.IGNORE:
            self._ignored += 1
            self._status = VoiceBargeInRuntimeStatus.IGNORED
            return self._result(
                operation=VoiceBargeInOperation.EVALUATE_SIGNAL,
                status=self._status,
                disposition=disposition,
                signal=signal,
                interrupted_context=None,
                playback_result=None,
                message="interrupt signal ignored",
                started=started,
            )

        return self.interrupt_playback(
            signal=signal,
            disposition=disposition,
            active_response_text=active_response_text,
            started=started,
        )

    def interrupt_playback(
        self,
        *,
        signal: VoiceInterruptSignal,
        disposition: VoiceBargeInDisposition,
        active_response_text: str | None,
        started: float | None = None,
        metadata: dict[str, object] | None = None,
    ) -> VoiceBargeInResult:
        operation_started = started if started is not None else time.perf_counter()

        if self._playback is None:
            self._failed_interruptions += 1
            self._status = VoiceBargeInRuntimeStatus.FAILED
            self._last_error = "playback controller unavailable"
            return self._result(
                operation=VoiceBargeInOperation.INTERRUPT_PLAYBACK,
                status=self._status,
                disposition=disposition,
                signal=signal,
                interrupted_context=None,
                playback_result=None,
                message="cannot interrupt playback without playback controller",
                started=operation_started,
            )

        self._status = VoiceBargeInRuntimeStatus.INTERRUPTING
        stop_started = time.perf_counter()

        try:
            playback_result = self._playback.stop()
        except Exception as exc:
            self._failed_interruptions += 1
            self._status = VoiceBargeInRuntimeStatus.FAILED
            self._last_error = str(exc)
            return self._result(
                operation=VoiceBargeInOperation.INTERRUPT_PLAYBACK,
                status=self._status,
                disposition=disposition,
                signal=signal,
                interrupted_context=None,
                playback_result=None,
                message="playback stop raised during barge-in",
                started=operation_started,
                metadata={"error": str(exc)},
            )

        stop_latency_ms = (time.perf_counter() - stop_started) * 1000.0
        self._last_stop_latency_ms = stop_latency_ms

        if playback_result.status not in {
            VoicePlaybackRuntimeStatus.STOPPED,
            VoicePlaybackRuntimeStatus.READY,
        }:
            self._failed_interruptions += 1
            self._status = VoiceBargeInRuntimeStatus.FAILED
            self._last_error = playback_result.message

            failure_metadata: dict[str, object] = {
                "playback_status": playback_result.status.value,
                "playback_message": playback_result.message,
                **playback_result.metadata,
            }

            return self._result(
                operation=VoiceBargeInOperation.INTERRUPT_PLAYBACK,
                status=self._status,
                disposition=disposition,
                signal=signal,
                interrupted_context=None,
                playback_result=playback_result,
                message="playback did not stop cleanly",
                started=operation_started,
                metadata=failure_metadata,
            )

        playback_state = playback_result.playback_state
        context = _make_interrupted_context(
            signal=signal,
            disposition=disposition,
            active_response_text=active_response_text,
            playback_state=playback_state,
            metadata=metadata or {},
        )

        self._interruptions += 1
        self._status = VoiceBargeInRuntimeStatus.INTERRUPTED
        self._last_error = None

        result_metadata: dict[str, object] = {
            "stop_latency_ms": stop_latency_ms,
            "within_stop_budget": (
                stop_latency_ms <= self._policy.max_stop_latency_ms
            ),
        }

        return self._result(
            operation=VoiceBargeInOperation.INTERRUPT_PLAYBACK,
            status=self._status,
            disposition=disposition,
            signal=signal,
            interrupted_context=context,
            playback_result=playback_result,
            message="playback interrupted by user barge-in",
            started=operation_started,
            metadata=result_metadata,
        )

    def reset(self) -> VoiceBargeInResult:
        started = time.perf_counter()
        self._status = VoiceBargeInRuntimeStatus.CREATED
        self._last_disposition = None
        self._last_signal_text = None
        self._last_latency_ms = None
        self._last_stop_latency_ms = None
        self._last_error = None
        return self._result(
            operation=VoiceBargeInOperation.RESET,
            status=self._status,
            disposition=VoiceBargeInDisposition.IGNORE,
            signal=None,
            interrupted_context=None,
            playback_result=None,
            message="barge-in runtime reset",
            started=started,
        )

    def snapshot(self) -> VoiceBargeInSnapshot:
        return VoiceBargeInSnapshot(
            status=self._status,
            evaluated_transcripts=self._evaluated_transcripts,
            ignored=self._ignored,
            interruptions=self._interruptions,
            failed_interruptions=self._failed_interruptions,
            last_disposition=self._last_disposition,
            last_signal_text=self._last_signal_text,
            last_latency_ms=self._last_latency_ms,
            last_stop_latency_ms=self._last_stop_latency_ms,
            last_error=self._last_error,
            created_at=utc_now(),
        )

    def _result(
        self,
        *,
        operation: VoiceBargeInOperation,
        status: VoiceBargeInRuntimeStatus,
        disposition: VoiceBargeInDisposition,
        signal: VoiceInterruptSignal | None,
        interrupted_context: VoiceInterruptedSpeechContext | None,
        playback_result: VoicePlaybackResult | None,
        message: str,
        started: float,
        metadata: dict[str, object] | None = None,
    ) -> VoiceBargeInResult:
        latency_ms = (time.perf_counter() - started) * 1000.0
        self._last_latency_ms = latency_ms
        return VoiceBargeInResult(
            status=status,
            operation=operation,
            disposition=disposition,
            signal=signal,
            interrupted_context=interrupted_context,
            playback_result=playback_result,
            message=message,
            latency_ms=latency_ms,
            created_at=utc_now(),
            metadata=metadata or {},
        )


def _classify_barge_in(
    *,
    text: str,
    confidence: float,
    policy: VoiceBargeInPolicy,
) -> VoiceBargeInDisposition:
    normalized = _normalize(text)

    if not normalized:
        return VoiceBargeInDisposition.IGNORE

    if _is_known_silence_hallucination(normalized, policy):
        return VoiceBargeInDisposition.IGNORE

    if _contains_phrase(normalized, policy.cancel_phrases):
        if confidence >= policy.emergency_stop_min_confidence:
            return VoiceBargeInDisposition.CANCEL_RESPONSE
        return VoiceBargeInDisposition.IGNORE

    if _contains_phrase(normalized, policy.stop_phrases):
        if confidence >= policy.emergency_stop_min_confidence:
            if _contains_question_marker(normalized, policy.question_markers):
                return VoiceBargeInDisposition.NEW_QUESTION
            return VoiceBargeInDisposition.STOP_PLAYBACK
        return VoiceBargeInDisposition.IGNORE

    if confidence < policy.min_confidence:
        return VoiceBargeInDisposition.IGNORE

    if _contains_phrase(normalized, policy.correction_markers):
        if len(normalized.split()) < policy.min_correction_words:
            return VoiceBargeInDisposition.IGNORE
        return VoiceBargeInDisposition.USER_CORRECTION

    if _looks_like_direct_question(normalized, policy.question_markers):
        if len(normalized.split()) < policy.min_question_words:
            return VoiceBargeInDisposition.IGNORE
        return VoiceBargeInDisposition.NEW_QUESTION

    return VoiceBargeInDisposition.IGNORE


def _semantic_barge_in_is_stable(
    *,
    transcript: VoiceTranscript,
    disposition: VoiceBargeInDisposition,
    policy: VoiceBargeInPolicy,
) -> bool:
    if disposition in {
        VoiceBargeInDisposition.STOP_PLAYBACK,
        VoiceBargeInDisposition.CANCEL_RESPONSE,
        VoiceBargeInDisposition.PAUSE_PLAYBACK,
    }:
        return True

    perception = transcript.metadata.get("perception")
    if not isinstance(perception, dict):
        return transcript.confidence >= policy.min_confidence

    state = str(perception.get("intent_state") or "")
    if state in {"noise", "capturing", "stabilizing"}:
        return False

    stability = perception.get("stability")
    if isinstance(stability, int | float):
        return float(stability) >= policy.min_semantic_barge_in_stability

    return transcript.confidence >= policy.min_confidence


def _looks_like_assistant_echo(
    *,
    transcript_text: str,
    active_response_text: str | None,
    policy: VoiceBargeInPolicy,
) -> bool:
    if active_response_text is None:
        return False

    transcript = _normalize(transcript_text)
    response = _normalize(active_response_text)
    if not transcript or not response:
        return False

    if _contains_phrase(transcript, policy.wake_words):
        return False

    if _phrase_match(response, transcript):
        return True

    transcript_words = transcript.split()
    response_words = set(response.split())
    if len(transcript_words) < 3 or not response_words:
        return False

    overlap = sum(1 for word in transcript_words if word in response_words)
    return (overlap / len(transcript_words)) >= policy.assistant_echo_min_overlap


def _is_known_silence_hallucination(
    normalized: str,
    policy: VoiceBargeInPolicy,
) -> bool:
    if normalized in {_normalize(item) for item in policy.known_silence_hallucinations}:
        return True
    if normalized.startswith("thank you") and len(normalized.split()) <= 5:
        return True
    if normalized.startswith("good luck") and len(normalized.split()) <= 5:
        return True
    if normalized.startswith("see you") and len(normalized.split()) <= 5:
        return True
    return False


def _make_signal_from_transcript(
    *,
    transcript: VoiceTranscript,
    disposition: VoiceBargeInDisposition,
) -> VoiceInterruptSignal:
    kind = _signal_kind_for_disposition(disposition)
    return VoiceInterruptSignal(
        interrupt_id=make_voice_interrupt_id(),
        session_id=transcript.session_id,
        kind=kind,
        text=transcript.text,
        confidence=transcript.confidence,
        created_at=utc_now(),
        metadata={
            "transcript_id": str(transcript.transcript_id),
            "transcript_kind": transcript.kind.value,
            "disposition": disposition.value,
        },
    )


def _signal_kind_for_disposition(
    disposition: VoiceBargeInDisposition,
) -> VoiceInterruptKind:
    if disposition == VoiceBargeInDisposition.CANCEL_RESPONSE:
        return VoiceInterruptKind.CANCEL
    if disposition == VoiceBargeInDisposition.PAUSE_PLAYBACK:
        return VoiceInterruptKind.PAUSE
    if disposition == VoiceBargeInDisposition.USER_CORRECTION:
        return VoiceInterruptKind.USER_CORRECTION
    if disposition == VoiceBargeInDisposition.NEW_QUESTION:
        return VoiceInterruptKind.BARGE_IN
    if disposition == VoiceBargeInDisposition.STOP_PLAYBACK:
        return VoiceInterruptKind.STOP
    return VoiceInterruptKind.UNKNOWN


def _disposition_from_signal(
    signal: VoiceInterruptSignal,
) -> VoiceBargeInDisposition:
    if signal.kind == VoiceInterruptKind.STOP:
        return VoiceBargeInDisposition.STOP_PLAYBACK
    if signal.kind == VoiceInterruptKind.PAUSE:
        return VoiceBargeInDisposition.PAUSE_PLAYBACK
    if signal.kind == VoiceInterruptKind.CANCEL:
        return VoiceBargeInDisposition.CANCEL_RESPONSE
    if signal.kind == VoiceInterruptKind.USER_CORRECTION:
        return VoiceBargeInDisposition.USER_CORRECTION
    if signal.kind == VoiceInterruptKind.BARGE_IN:
        return VoiceBargeInDisposition.NEW_QUESTION
    return VoiceBargeInDisposition.IGNORE


def _make_interrupted_context(
    *,
    signal: VoiceInterruptSignal,
    disposition: VoiceBargeInDisposition,
    active_response_text: str | None,
    playback_state: VoicePlaybackState | None,
    metadata: dict[str, object],
) -> VoiceInterruptedSpeechContext:
    response_text = active_response_text or "active voice response"
    return VoiceInterruptedSpeechContext(
        response_text=response_text,
        transcript_text=signal.text,
        playback_state=playback_state,
        interrupted_at=utc_now(),
        disposition=disposition,
        metadata={
            "signal_kind": signal.kind.value,
            "confidence": signal.confidence,
            **metadata,
        },
    )


def _normalize(text: str) -> str:
    lowered = text.casefold().strip()
    lowered = re.sub(r"[^a-z0-9\s']", " ", lowered)
    return " ".join(lowered.split())


def _contains_phrase(
    text: str,
    phrases: tuple[str, ...],
) -> bool:
    return any(_phrase_match(text, phrase) for phrase in phrases)


def _phrase_match(
    text: str,
    phrase: str,
) -> bool:
    normalized_phrase = _normalize(phrase)
    if not normalized_phrase:
        return False
    pattern = rf"(^|\s){re.escape(normalized_phrase)}($|\s)"
    return re.search(pattern, text) is not None


def _contains_question_marker(
    text: str,
    markers: tuple[str, ...],
) -> bool:
    return any(_phrase_match(text, marker) for marker in markers)


def _looks_like_direct_question(
    text: str,
    markers: tuple[str, ...],
) -> bool:
    words = text.split()
    if not words:
        return False
    if words[0] in {"what", "why", "how", "when", "where"}:
        return True
    return _contains_question_marker(text, markers) and len(words) >= 3
