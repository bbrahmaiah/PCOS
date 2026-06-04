from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum

from jarvis.cognitive.contracts import CognitiveSessionState
from jarvis.live.contracts import (
    LiveResponse,
    LiveResponseKind,
    LiveResponseSafety,
    LiveSessionConfig,
    LiveTranscript,
    LiveTranscriptKind,
    LiveTurnId,
    default_live_session_config,
    make_live_turn_id,
    utc_now,
)
from jarvis.live.event_bridge import (
    LiveEventBridgeRuntime,
    LiveEventBridgeStatus,
    LiveResponseBridgeRequest,
    LiveTranscriptBridgeRequest,
)
from jarvis.live.response_boundary import (
    LiveResponseBoundaryResult,
    LiveResponseBoundaryRuntime,
    LiveResponseContext,
    LiveResponseGenerationRequest,
    LiveResponseGenerator,
    LiveResponseIntent,
    LiveResponseSurface,
)
from jarvis.live.session_state import (
    LiveSessionStateRuntime,
    LiveSessionStateRuntimeResult,
    LiveSessionStateRuntimeStatus,
)


class LiveDialogueRuntimeStatus(StrEnum):
    READY = "ready"
    BLOCKED = "blocked"
    DEGRADED = "degraded"


class LiveDialogueOperation(StrEnum):
    START_TURN = "start_turn"
    PROCESS_TRANSCRIPT = "process_transcript"
    FINISH_RESPONSE = "finish_response"
    SNAPSHOT = "snapshot"


class LiveDialogueTurnStatus(StrEnum):
    STARTED = "started"
    TRANSCRIPT_ACCEPTED = "transcript_accepted"
    RESPONSE_GENERATED = "response_generated"
    RESPONSE_STARTED = "response_started"
    RESPONSE_FINISHED = "response_finished"
    BLOCKED = "blocked"


@dataclass(frozen=True, slots=True)
class LiveDialoguePolicy:
    max_response_sentences: int = 3
    default_surface: LiveResponseSurface = LiveResponseSurface.VOICE
    allow_interruptions: bool = True
    require_response_generation: bool = True
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.max_response_sentences < 1:
            raise ValueError("max_response_sentences must be at least 1.")


@dataclass(frozen=True, slots=True)
class LiveDialogueTurn:
    turn_id: LiveTurnId
    transcript: LiveTranscript | None
    response: LiveResponse | None
    status: LiveDialogueTurnStatus
    started_at: datetime
    updated_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class LiveDialogueRequest:
    transcript: LiveTranscript
    user_is_speaking: bool = False
    assistant_is_speaking: bool = False
    response_surface: LiveResponseSurface | None = None
    safety: LiveResponseSafety = LiveResponseSafety.SAFE_TO_SPEAK
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class LiveDialogueResult:
    status: LiveDialogueRuntimeStatus
    operation: LiveDialogueOperation
    turn: LiveDialogueTurn | None
    live_state_result: LiveSessionStateRuntimeResult | None
    bridge_result: object | None
    response_boundary_result: LiveResponseBoundaryResult | None
    reason: str
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def succeeded(self) -> bool:
        return self.status == LiveDialogueRuntimeStatus.READY


@dataclass(frozen=True, slots=True)
class LiveDialogueSnapshot:
    status: LiveDialogueRuntimeStatus
    active_turn: LiveDialogueTurn | None
    completed_turns: int
    blocked_turns: int
    generated_responses: int
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)


class LiveDialogueRuntime:
    """
    Step 50F Live Dialogue Runtime.

    Turns live transcripts into generated LiveResponse objects.

    It does not:
    - hardcode conversation
    - call TTS directly
    - access memory directly
    - execute tools
    - bypass the LiveResponseBoundaryRuntime

    All normal dialogue output must be generated through 50A.5.
    """

    def __init__(
        self,
        *,
        live_state: LiveSessionStateRuntime | None = None,
        bridge: LiveEventBridgeRuntime | None = None,
        response_boundary: LiveResponseBoundaryRuntime | None = None,
        response_generator: LiveResponseGenerator | None = None,
        config: LiveSessionConfig | None = None,
        policy: LiveDialoguePolicy | None = None,
    ) -> None:
        self._config = config or default_live_session_config()
        self._state = live_state or LiveSessionStateRuntime(config=self._config)
        self._response_boundary = response_boundary or LiveResponseBoundaryRuntime(
            generator=response_generator
        )
        self._bridge = bridge or LiveEventBridgeRuntime(
            live_state=self._state,
            response_boundary=self._response_boundary,
        )
        self._policy = policy or LiveDialoguePolicy()
        self._active_turn: LiveDialogueTurn | None = None
        self._completed_turns = 0
        self._blocked_turns = 0
        self._generated_responses = 0

    @property
    def live_state(self) -> LiveSessionStateRuntime:
        return self._state

    @property
    def bridge(self) -> LiveEventBridgeRuntime:
        return self._bridge

    def start_turn(self) -> LiveDialogueResult:
        state_result = self._state.start_user_turn()
        if state_result.status == LiveSessionStateRuntimeStatus.BLOCKED:
            self._blocked_turns += 1
            return self._blocked(
                operation=LiveDialogueOperation.START_TURN,
                reason=state_result.reason,
                live_state_result=state_result,
            )

        if state_result.state.current_turn_id is None:
            self._blocked_turns += 1
            return self._blocked(
                operation=LiveDialogueOperation.START_TURN,
                reason="live session did not produce a turn id",
                live_state_result=state_result,
            )

        now = utc_now()
        self._active_turn = LiveDialogueTurn(
            turn_id=state_result.state.current_turn_id,
            transcript=None,
            response=None,
            status=LiveDialogueTurnStatus.STARTED,
            started_at=now,
            updated_at=now,
        )

        return LiveDialogueResult(
            status=LiveDialogueRuntimeStatus.READY,
            operation=LiveDialogueOperation.START_TURN,
            turn=self._active_turn,
            live_state_result=state_result,
            bridge_result=None,
            response_boundary_result=None,
            reason="dialogue turn started",
            created_at=utc_now(),
        )

    def process_transcript(
        self,
        request: LiveDialogueRequest,
    ) -> LiveDialogueResult:
        bridge_result = self._bridge.bridge_transcript(
            LiveTranscriptBridgeRequest(
                transcript=request.transcript,
                user_is_speaking=request.user_is_speaking,
                assistant_is_speaking=request.assistant_is_speaking,
                allow_interruptions=self._policy.allow_interruptions,
                metadata=request.metadata,
            )
        )

        if bridge_result.status == LiveEventBridgeStatus.BLOCKED:
            self._blocked_turns += 1
            return self._blocked(
                operation=LiveDialogueOperation.PROCESS_TRANSCRIPT,
                reason="transcript rejected by live event bridge",
                bridge_result=bridge_result,
            )

        thinking_result = self._state.start_thinking()
        if thinking_result.status == LiveSessionStateRuntimeStatus.BLOCKED:
            self._blocked_turns += 1
            return self._blocked(
                operation=LiveDialogueOperation.PROCESS_TRANSCRIPT,
                reason=thinking_result.reason,
                live_state_result=thinking_result,
                bridge_result=bridge_result,
            )

        response_request = self._build_response_request(
            request=request,
            cognitive_state=(
                bridge_result.cognitive_result.session_result.session
                if bridge_result.cognitive_result is not None
                else None
            ),
        )
        boundary_result = self._response_boundary.generate(response_request)

        if not boundary_result.succeeded or boundary_result.response is None:
            self._blocked_turns += 1
            return self._blocked(
                operation=LiveDialogueOperation.PROCESS_TRANSCRIPT,
                reason=boundary_result.reason,
                live_state_result=thinking_result,
                bridge_result=bridge_result,
                response_boundary_result=boundary_result,
            )

        response_bridge = self._bridge.bridge_response(
            LiveResponseBridgeRequest(
                response=boundary_result.response,
                validate_for_tts=True,
                metadata=request.metadata,
            )
        )

        if response_bridge.status == LiveEventBridgeStatus.BLOCKED:
            self._blocked_turns += 1
            return self._blocked(
                operation=LiveDialogueOperation.PROCESS_TRANSCRIPT,
                reason=response_bridge.reason,
                live_state_result=thinking_result,
                bridge_result=response_bridge,
                response_boundary_result=boundary_result,
            )

        turn = LiveDialogueTurn(
            turn_id=request.transcript.turn_id,
            transcript=request.transcript,
            response=boundary_result.response,
            status=LiveDialogueTurnStatus.RESPONSE_STARTED,
            started_at=(
                self._active_turn.started_at
                if self._active_turn is not None
                else utc_now()
            ),
            updated_at=utc_now(),
            metadata={
                **request.metadata,
                "intent": response_request.intent.value,
                "surface": response_request.surface.value,
            },
        )
        self._active_turn = turn
        self._generated_responses += 1

        return LiveDialogueResult(
            status=LiveDialogueRuntimeStatus.READY,
            operation=LiveDialogueOperation.PROCESS_TRANSCRIPT,
            turn=turn,
            live_state_result=thinking_result,
            bridge_result=response_bridge,
            response_boundary_result=boundary_result,
            reason="dialogue response generated through boundary",
            created_at=utc_now(),
            metadata=turn.metadata,
        )

    def finish_response(self) -> LiveDialogueResult:
        state_result = self._state.finish_speaking()
        if state_result.status == LiveSessionStateRuntimeStatus.BLOCKED:
            return self._blocked(
                operation=LiveDialogueOperation.FINISH_RESPONSE,
                reason=state_result.reason,
                live_state_result=state_result,
            )

        if self._active_turn is None:
            return LiveDialogueResult(
                status=LiveDialogueRuntimeStatus.READY,
                operation=LiveDialogueOperation.FINISH_RESPONSE,
                turn=None,
                live_state_result=state_result,
                bridge_result=None,
                response_boundary_result=None,
                reason="no active dialogue turn to finish",
                created_at=utc_now(),
            )

        finished = LiveDialogueTurn(
            turn_id=self._active_turn.turn_id,
            transcript=self._active_turn.transcript,
            response=self._active_turn.response,
            status=LiveDialogueTurnStatus.RESPONSE_FINISHED,
            started_at=self._active_turn.started_at,
            updated_at=utc_now(),
            metadata=self._active_turn.metadata,
        )
        self._active_turn = finished
        self._completed_turns += 1

        return LiveDialogueResult(
            status=LiveDialogueRuntimeStatus.READY,
            operation=LiveDialogueOperation.FINISH_RESPONSE,
            turn=finished,
            live_state_result=state_result,
            bridge_result=None,
            response_boundary_result=None,
            reason="dialogue response finished",
            created_at=utc_now(),
        )

    def snapshot(self) -> LiveDialogueSnapshot:
        return LiveDialogueSnapshot(
            status=LiveDialogueRuntimeStatus.READY,
            active_turn=self._active_turn,
            completed_turns=self._completed_turns,
            blocked_turns=self._blocked_turns,
            generated_responses=self._generated_responses,
            created_at=utc_now(),
        )

    def _build_response_request(
        self,
        *,
        request: LiveDialogueRequest,
        cognitive_state: CognitiveSessionState | None,
    ) -> LiveResponseGenerationRequest:
        return LiveResponseGenerationRequest(
            turn_id=request.transcript.turn_id,
            intent=_intent_from_transcript(request.transcript),
            surface=request.response_surface or self._policy.default_surface,
            context=LiveResponseContext(
                live_state=self._state.state,
                cognitive_state=cognitive_state,
                user_text=request.transcript.text,
                situation_summary=_situation_summary(request.transcript),
                memory_context=_metadata_tuple(request.metadata, "memory"),
                working_memory_context=_metadata_tuple(
                    request.metadata,
                    "working_memory",
                ),
                attention_context=_metadata_tuple(request.metadata, "attention"),
                goal_context=_metadata_tuple(request.metadata, "goal"),
                planning_context=_metadata_tuple(request.metadata, "planning"),
                environment_context=_metadata_tuple(
                    request.metadata,
                    "environment",
                ),
                developer_context=_metadata_tuple(request.metadata, "developer"),
                metadata=request.metadata,
            ),
            response_kind=LiveResponseKind.CONVERSATIONAL,
            safety=request.safety,
            max_sentences=self._policy.max_response_sentences,
            metadata={
                **request.metadata,
                "dialogue_runtime": "50F",
                "transcript_id": str(request.transcript.transcript_id),
                "transcript_kind": request.transcript.kind.value,
            },
        )

    def _blocked(
        self,
        *,
        operation: LiveDialogueOperation,
        reason: str,
        live_state_result: LiveSessionStateRuntimeResult | None = None,
        bridge_result: object | None = None,
        response_boundary_result: LiveResponseBoundaryResult | None = None,
    ) -> LiveDialogueResult:
        return LiveDialogueResult(
            status=LiveDialogueRuntimeStatus.BLOCKED,
            operation=operation,
            turn=self._active_turn,
            live_state_result=live_state_result,
            bridge_result=bridge_result,
            response_boundary_result=response_boundary_result,
            reason=reason,
            created_at=utc_now(),
        )


def make_dialogue_turn_id() -> LiveTurnId:
    return make_live_turn_id()


def _intent_from_transcript(
    transcript: LiveTranscript,
) -> LiveResponseIntent:
    text = _normalize(transcript.text)

    if transcript.kind == LiveTranscriptKind.INTERRUPTION:
        return LiveResponseIntent.INTERRUPTION

    if text in {"repeat", "repeat that", "say that again"}:
        return LiveResponseIntent.REPEAT

    if text in {"continue", "go on", "continue from there"}:
        return LiveResponseIntent.CONTINUE

    if "?" in transcript.text:
        return LiveResponseIntent.ANSWER

    if any(word in text for word in ("explain", "teach", "learn")):
        return LiveResponseIntent.LEARNING

    if any(word in text for word in ("shutdown", "shut down", "sleep")):
        return LiveResponseIntent.SHUTDOWN_REQUEST

    return LiveResponseIntent.ANSWER


def _situation_summary(transcript: LiveTranscript) -> str:
    return (
        "Live dialogue turn received a transcript and requires generated "
        f"response. Transcript kind: {transcript.kind.value}."
    )


def _metadata_tuple(
    metadata: dict[str, object],
    key: str,
) -> tuple[str, ...]:
    value = metadata.get(key)
    if value is None:
        return ()

    if isinstance(value, str):
        return (value,)

    if isinstance(value, tuple):
        return tuple(str(item) for item in value)

    if isinstance(value, list):
        return tuple(str(item) for item in value)

    return (str(value),)


def _normalize(text: str) -> str:
    return " ".join(text.casefold().strip().split())