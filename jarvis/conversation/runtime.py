from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from threading import RLock

from pydantic import Field, field_validator

from jarvis.conversation.attention import (
    AttentionDecision,
    AttentionPriority,
    AttentionRuntime,
    AttentionSignal,
    AttentionSignalKind,
)
from jarvis.conversation.endpointing import (
    AdaptiveEndpointingEngine,
    EndpointAction,
    EndpointingDecision,
    EndpointingInput,
)
from jarvis.conversation.interrupt_controller import (
    InterruptController,
    InterruptDecision,
)
from jarvis.conversation.models import (
    ConversationMode,
    ConversationModel,
    TurnDetectionInput,
    TurnInputSource,
    TurnUrgency,
    new_conversation_id,
    utc_now,
)
from jarvis.conversation.session_runtime import (
    ConversationSessionRuntime,
    ConversationSessionSnapshotModel,
)
from jarvis.conversation.state_machine import (
    ConversationState,
    ConversationStateEvent,
    ConversationStateEventKind,
    ConversationStateMachine,
    ConversationStateTransition,
)
from jarvis.conversation.streaming import (
    StreamingConversationCoordinator,
    StreamingConversationCoordinatorConfig,
    StreamingConversationEvent,
    StreamingCoordinatorOutput,
    StreamingEventKind,
    StreamingLifecycle,
)
from jarvis.runtime.observability.structured_logger import get_logger


class RealConversationRuntimeAction(StrEnum):
    """
    High-level action emitted by RealConversationRuntime.
    """

    KEEP_LISTENING = "keep_listening"
    WAIT_FOR_USER = "wait_for_user"
    START_COGNITION = "start_cognition"
    STREAM_TOKEN = "stream_token"
    START_TTS = "start_tts"
    CANCEL_ACTIVE_WORK = "cancel_active_work"
    UPDATE_SESSION = "update_session"
    UPDATE_ATTENTION = "update_attention"
    COMPLETE_RESPONSE = "complete_response"
    RESET_RUNTIME = "reset_runtime"


class RealConversationRuntimeStatus(StrEnum):
    """
    Lifecycle status for the assembled real conversation runtime.
    """

    READY = "ready"
    LISTENING = "listening"
    USER_THINKING = "user_thinking"
    COGNITION_READY = "cognition_ready"
    STREAMING_RESPONSE = "streaming_response"
    SPEAKING = "speaking"
    INTERRUPTED = "interrupted"
    COMPLETED = "completed"
    RESET = "reset"


class RealConversationInput(ConversationModel):
    """
    One live input snapshot entering the assembled runtime.

    This is designed to be driven by Presence/STT/VAD now and upgraded later
    by the full continuous voice loop.
    """

    turn_id: str = Field(default_factory=new_conversation_id)
    transcript: str = ""
    source: TurnInputSource = TurnInputSource.STT_PARTIAL
    is_speech_active: bool = False
    is_assistant_speaking: bool = False
    silence_ms: int = Field(default=0, ge=0)
    speech_ms: int = Field(default=0, ge=0)
    vad_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    transcript_stability: float = Field(default=0.0, ge=0.0, le=1.0)
    conversation_mode: ConversationMode = ConversationMode.UNKNOWN
    previous_transcript: str | None = None
    consecutive_maybe_complete_count: int = Field(default=0, ge=0)
    user_pause_count: int = Field(default=0, ge=0)
    created_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("turn_id")
    @classmethod
    def _turn_id_required(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("turn_id cannot be empty.")

        return cleaned

    @field_validator("transcript", "previous_transcript")
    @classmethod
    def _clean_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None

        return " ".join(value.strip().split())


class RealConversationRuntimeOutput(ConversationModel):
    """
    One assembled runtime output.

    This is the object future orchestration workers should consume.
    """

    turn_id: str
    status: RealConversationRuntimeStatus
    actions: tuple[RealConversationRuntimeAction, ...]
    endpointing_decision: EndpointingDecision | None = None
    state_transitions: tuple[ConversationStateTransition, ...] = ()
    streaming_output: StreamingCoordinatorOutput | None = None
    interrupt_decision: InterruptDecision | None = None
    session_snapshot: ConversationSessionSnapshotModel | None = None
    attention_decision: AttentionDecision | None = None
    should_start_cognition: bool = False
    should_start_tts: bool = False
    should_cancel_active_work: bool = False
    should_keep_listening: bool = True
    cognition_context_block: str | None = None
    reason: str
    created_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("turn_id", "reason")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned


@dataclass(frozen=True, slots=True)
class RealConversationRuntimeConfig:
    """
    Configuration for the assembled runtime.
    """

    name: str = "real_conversation_runtime"
    auto_start_listening: bool = True
    capture_finalized_user_turns: bool = True
    update_attention_from_session: bool = True

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("name cannot be empty.")


@dataclass(frozen=True, slots=True)
class RealConversationRuntimeSnapshot:
    """
    Observable diagnostics for RealConversationRuntime.
    """

    name: str
    status: RealConversationRuntimeStatus
    input_count: int
    cognition_start_count: int
    tts_start_count: int
    interrupt_count: int
    session_update_count: int
    attention_update_count: int
    current_state: ConversationState
    streaming_lifecycle: StreamingLifecycle
    current_focus_id: str | None
    last_turn_id: str | None
    last_actions: tuple[RealConversationRuntimeAction, ...]
    last_error: str | None


class RealConversationRuntime:
    """
    Final assembly runtime for Phase 4.5.

    Responsibilities:
    - coordinate turn detection through endpointing
    - update conversation state machine
    - coordinate streaming cognition/TTS chunks
    - route interruptions to interrupt controller
    - update short-term session continuity
    - update attention runtime
    - produce one orchestration-ready output object

    Non-responsibilities:
    - no direct microphone access
    - no direct STT implementation
    - no direct LLM call
    - no direct TTS playback
    - no direct laptop/tool execution
    - no direct long-term memory writes

    Future workers should consume output actions and call:
    - Presence Runtime for audio/STT/TTS
    - Cognition Runtime for generation
    - Memory Runtime through Memory Gateway only
    - Tool Runtime through policy-gated action execution only
    """

    def __init__(
        self,
        *,
        config: RealConversationRuntimeConfig | None = None,
        endpointing_engine: AdaptiveEndpointingEngine | None = None,
        state_machine: ConversationStateMachine | None = None,
        streaming_coordinator: StreamingConversationCoordinator | None = None,
        interrupt_controller: InterruptController | None = None,
        session_runtime: ConversationSessionRuntime | None = None,
        attention_runtime: AttentionRuntime | None = None,
    ) -> None:
        self._config = config or RealConversationRuntimeConfig()
        self._config.validate()

        self._state_machine = state_machine or ConversationStateMachine()
        self._streaming_coordinator = (
            streaming_coordinator
            or StreamingConversationCoordinator(
                config=StreamingConversationCoordinatorConfig(
                    name="real_conversation_streaming_coordinator",
                    min_speech_chunk_chars=20,
                    max_speech_chunk_chars=180,
                    emit_on_sentence_boundary=True,
                    comma_soft_boundary=True,
                )
            )
        )
        self._interrupt_controller = interrupt_controller or InterruptController(
            state_machine=self._state_machine,
            streaming_coordinator=self._streaming_coordinator,
        )
        self._endpointing_engine = endpointing_engine or AdaptiveEndpointingEngine()
        self._session_runtime = session_runtime or ConversationSessionRuntime()
        self._attention_runtime = attention_runtime or AttentionRuntime()

        self._lock = RLock()
        self._logger = get_logger("conversation.real_runtime")

        self._status = RealConversationRuntimeStatus.READY
        self._input_count = 0
        self._cognition_start_count = 0
        self._tts_start_count = 0
        self._interrupt_count = 0
        self._session_update_count = 0
        self._attention_update_count = 0
        self._last_turn_id: str | None = None
        self._last_actions: tuple[RealConversationRuntimeAction, ...] = ()
        self._last_error: str | None = None

    @property
    def name(self) -> str:
        return self._config.name

    def accept_input(
        self,
        signal: RealConversationInput,
    ) -> RealConversationRuntimeOutput:
        """
        Accept one live user/STT/VAD signal and run the assembled path.

        This is the main entrypoint for future real-time voice orchestration.
        """

        with self._lock:
            self._input_count += 1
            self._last_turn_id = signal.turn_id
            self._last_error = None

        try:
            state_transitions: list[ConversationStateTransition] = []
            start_transition = self._maybe_start_listening(signal)

            if start_transition is not None:
                state_transitions.append(start_transition)

            endpointing = self._endpoint(signal)
            interrupt_decision: InterruptDecision | None = None
            streaming_output: StreamingCoordinatorOutput | None = None
            session_snapshot: ConversationSessionSnapshotModel | None = None
            attention_decision: AttentionDecision | None = None

            if endpointing.action in {
                EndpointAction.INTERRUPT_RESPONSE,
                EndpointAction.CANCEL_RESPONSE,
            }:
                interrupt_decision = (
                    self._interrupt_controller.handle_endpointing_decision(
                        endpointing
                    )
                )
                attention_decision = self._submit_interrupt_attention(
                    signal=signal,
                    endpointing=endpointing,
                )
                output = self._interrupt_output(
                    signal=signal,
                    endpointing=endpointing,
                    interrupt_decision=interrupt_decision,
                    attention_decision=attention_decision,
                )
                self._record_output(output)

                return output

            transition = self._apply_endpoint_state(endpointing)

            if transition is not None:
                state_transitions.append(transition)

            streaming_output = (
                self._streaming_coordinator.accept_endpointing_decision(endpointing)
            )

            if self._should_capture_user_turn(endpointing):
                session_snapshot = self._session_runtime.add_user_turn(
                    endpointing.turn_decision.transcript,
                    topic=self._safe_topic(signal),
                    conversation_mode=signal.conversation_mode,
                    state=self._state_machine.current_state,
                    metadata={
                        "turn_id": signal.turn_id,
                        "endpoint_action": endpointing.action.value,
                        "endpoint_reason": endpointing.reason,
                    },
                )
                self._session_update_count += 1

            if (
                session_snapshot is not None
                and self._config.update_attention_from_session
            ):
                attention_decision = self._attention_runtime.update_from_session(
                    session_snapshot
                )
                self._attention_update_count += 1

            output = self._normal_output(
                signal=signal,
                endpointing=endpointing,
                state_transitions=tuple(state_transitions),
                streaming_output=streaming_output,
                session_snapshot=session_snapshot,
                attention_decision=attention_decision,
            )
            self._record_output(output)

            return output

        except Exception as exc:
            with self._lock:
                self._last_error = f"{type(exc).__name__}: {exc}"

            raise

    def add_cognition_token(
        self,
        *,
        turn_id: str,
        token: str,
        sequence: int = 0,
    ) -> RealConversationRuntimeOutput:
        """
        Add one cognition token and possibly emit an early speech chunk.
        """

        streaming_output = self._streaming_coordinator.add_cognition_token(
            turn_id=turn_id,
            token=token,
            sequence=sequence,
        )
        transitions = self._maybe_mark_response_started(streaming_output)
        actions = [RealConversationRuntimeAction.STREAM_TOKEN]

        if streaming_output.should_start_tts:
            actions.append(RealConversationRuntimeAction.START_TTS)

        output = RealConversationRuntimeOutput(
            turn_id=turn_id,
            status=self._status_from_streaming(streaming_output),
            actions=tuple(actions),
            state_transitions=tuple(transitions),
            streaming_output=streaming_output,
            should_start_tts=streaming_output.should_start_tts,
            should_keep_listening=streaming_output.should_keep_listening,
            reason=streaming_output.reason,
            metadata={
                "runtime": self.name,
            },
        )
        self._record_output(output)

        return output

    def complete_cognition(
        self,
        *,
        turn_id: str,
    ) -> RealConversationRuntimeOutput:
        """
        Mark cognition stream complete and emit remaining final speech chunk.
        """

        streaming_output = self._streaming_coordinator.complete_cognition(
            turn_id=turn_id
        )
        transitions = self._maybe_mark_response_started(streaming_output)
        actions = [RealConversationRuntimeAction.COMPLETE_RESPONSE]

        if streaming_output.should_start_tts:
            actions.append(RealConversationRuntimeAction.START_TTS)

        output = RealConversationRuntimeOutput(
            turn_id=turn_id,
            status=self._status_from_streaming(streaming_output),
            actions=tuple(actions),
            state_transitions=tuple(transitions),
            streaming_output=streaming_output,
            should_start_tts=streaming_output.should_start_tts,
            should_keep_listening=True,
            reason=streaming_output.reason,
            metadata={
                "runtime": self.name,
            },
        )
        self._record_output(output)

        return output

    def add_assistant_response(
        self,
        text: str,
        *,
        turn_id: str | None = None,
        expects_follow_up: bool = False,
    ) -> RealConversationRuntimeOutput:
        """
        Add final assistant response into session continuity.
        """

        resolved_turn_id = turn_id or self._last_turn_id or new_conversation_id()
        session_snapshot = self._session_runtime.add_assistant_turn(
            text,
            state=self._state_machine.current_state,
            expects_follow_up=expects_follow_up,
            metadata={
                "turn_id": resolved_turn_id,
            },
        )
        attention_decision = self._attention_runtime.update_from_session(
            session_snapshot
        )

        with self._lock:
            self._session_update_count += 1
            self._attention_update_count += 1

        output = RealConversationRuntimeOutput(
            turn_id=resolved_turn_id,
            status=RealConversationRuntimeStatus.COMPLETED,
            actions=(
                RealConversationRuntimeAction.UPDATE_SESSION,
                RealConversationRuntimeAction.UPDATE_ATTENTION,
            ),
            session_snapshot=session_snapshot,
            attention_decision=attention_decision,
            should_keep_listening=True,
            cognition_context_block=self.context_block(),
            reason="assistant response added to session continuity",
            metadata={
                "runtime": self.name,
            },
        )
        self._record_output(output)

        return output

    def speech_chunk_started(self, *, turn_id: str) -> RealConversationRuntimeOutput:
        streaming_output = self._streaming_coordinator.accept_event(
            StreamingConversationEvent(
                turn_id=turn_id,
                kind=StreamingEventKind.SPEECH_CHUNK_STARTED,
            )
        )
        transition = self._transition_if_allowed(
            ConversationStateEventKind.RESPONSE_STARTED,
            turn_id=turn_id,
            reason="speech chunk started",
        )
        transitions = (transition,) if transition is not None else ()

        output = RealConversationRuntimeOutput(
            turn_id=turn_id,
            status=RealConversationRuntimeStatus.SPEAKING,
            actions=(RealConversationRuntimeAction.START_TTS,),
            state_transitions=transitions,
            streaming_output=streaming_output,
            should_keep_listening=True,
            reason="speech chunk playback started",
            metadata={
                "runtime": self.name,
            },
        )
        self._record_output(output)

        return output

    def speech_chunk_completed(
        self,
        *,
        turn_id: str,
    ) -> RealConversationRuntimeOutput:
        streaming_output = self._streaming_coordinator.accept_event(
            StreamingConversationEvent(
                turn_id=turn_id,
                kind=StreamingEventKind.SPEECH_CHUNK_COMPLETED,
            )
        )
        transition = self._transition_if_allowed(
            ConversationStateEventKind.RESPONSE_COMPLETED,
            turn_id=turn_id,
            reason="speech chunk completed",
        )
        transitions = (transition,) if transition is not None else ()

        output = RealConversationRuntimeOutput(
            turn_id=turn_id,
            status=self._status_from_streaming(streaming_output),
            actions=(RealConversationRuntimeAction.KEEP_LISTENING,),
            state_transitions=transitions,
            streaming_output=streaming_output,
            should_keep_listening=True,
            reason="speech chunk playback completed",
            metadata={
                "runtime": self.name,
            },
        )
        self._record_output(output)

        return output

    def context_block(self) -> str:
        """
        Build compact cognition-ready runtime context.
        """

        return "\n\n".join(
            (
                self._session_runtime.snapshot_model().as_context_block(),
                self._attention_runtime.as_context_block(),
            )
        )

    def snapshot(self) -> RealConversationRuntimeSnapshot:
        """
        Return observable diagnostics.
        """

        with self._lock:
            focus = self._attention_runtime.current_focus()

            return RealConversationRuntimeSnapshot(
                name=self.name,
                status=self._status,
                input_count=self._input_count,
                cognition_start_count=self._cognition_start_count,
                tts_start_count=self._tts_start_count,
                interrupt_count=self._interrupt_count,
                session_update_count=self._session_update_count,
                attention_update_count=self._attention_update_count,
                current_state=self._state_machine.current_state,
                streaming_lifecycle=self._streaming_coordinator.lifecycle,
                current_focus_id=focus.target_id if focus else None,
                last_turn_id=self._last_turn_id,
                last_actions=self._last_actions,
                last_error=self._last_error,
            )

    def reset(self) -> RealConversationRuntimeOutput:
        """
        Reset the assembled conversation runtime.
        """

        self._state_machine.reset()
        self._streaming_coordinator.reset()
        self._interrupt_controller.reset()
        self._session_runtime.reset()
        self._attention_runtime.reset()

        with self._lock:
            self._status = RealConversationRuntimeStatus.RESET
            self._input_count = 0
            self._cognition_start_count = 0
            self._tts_start_count = 0
            self._interrupt_count = 0
            self._session_update_count = 0
            self._attention_update_count = 0
            self._last_turn_id = None
            self._last_actions = (RealConversationRuntimeAction.RESET_RUNTIME,)
            self._last_error = None

        return RealConversationRuntimeOutput(
            turn_id=new_conversation_id(),
            status=RealConversationRuntimeStatus.RESET,
            actions=(RealConversationRuntimeAction.RESET_RUNTIME,),
            should_keep_listening=True,
            reason="real conversation runtime reset",
            metadata={
                "runtime": self.name,
            },
        )

    def _endpoint(
        self,
        signal: RealConversationInput,
    ) -> EndpointingDecision:
        turn_signal = TurnDetectionInput(
            turn_id=signal.turn_id,
            source=signal.source,
            transcript=signal.transcript,
            is_speech_active=signal.is_speech_active,
            is_assistant_speaking=signal.is_assistant_speaking,
            silence_ms=signal.silence_ms,
            speech_ms=signal.speech_ms,
            vad_confidence=signal.vad_confidence,
            transcript_stability=signal.transcript_stability,
            conversation_mode=signal.conversation_mode,
            metadata=signal.metadata,
        )
        return self._endpointing_engine.evaluate(
            EndpointingInput(
                signal=turn_signal,
                conversation_state=self._state_machine.current_state,
                previous_transcript=signal.previous_transcript,
                consecutive_maybe_complete_count=(
                    signal.consecutive_maybe_complete_count
                ),
                user_pause_count=signal.user_pause_count,
                metadata={
                    "runtime": self.name,
                },
            )
        )

    def _maybe_start_listening(
        self,
        signal: RealConversationInput,
    ) -> ConversationStateTransition | None:
        if not self._config.auto_start_listening:
            return None

        if self._state_machine.current_state not in {
            ConversationState.IDLE,
            ConversationState.WAITING,
            ConversationState.FOLLOW_UP,
            ConversationState.INTERRUPTED,
        }:
            return None

        if not signal.transcript and not signal.is_speech_active:
            return None

        return self._transition_if_allowed(
            ConversationStateEventKind.START_LISTENING,
            turn_id=signal.turn_id,
            reason="runtime auto-started listening",
        )

    def _apply_endpoint_state(
        self,
        endpointing: EndpointingDecision,
    ) -> ConversationStateTransition | None:
        if endpointing.turn_decision.decision in {
            endpointing.turn_decision.decision.WAIT,
        } and endpointing.turn_decision.transcript:
            if self._state_machine.can_transition(
                state=self._state_machine.current_state,
                event_kind=ConversationStateEventKind.USER_PAUSED,
            ):
                return self._state_machine.transition(
                    ConversationStateEvent(
                        turn_id=endpointing.turn_decision.turn_id,
                        kind=ConversationStateEventKind.USER_PAUSED,
                        reason="runtime observed user pause",
                    )
                )

            return None

        if self._state_machine.can_transition(
            state=self._state_machine.current_state,
            event_kind=self._event_kind_from_endpoint(endpointing),
        ):
            return self._state_machine.apply_turn_decision(
                endpointing.turn_decision
            )

        return None

    def _event_kind_from_endpoint(
        self,
        endpointing: EndpointingDecision,
    ) -> ConversationStateEventKind:
        if endpointing.action == EndpointAction.START_COGNITION:
            return ConversationStateEventKind.TURN_FINALIZED

        if endpointing.action == EndpointAction.PREPARE_RESPONSE:
            return ConversationStateEventKind.TURN_MAYBE_COMPLETE

        if endpointing.action == EndpointAction.INTERRUPT_RESPONSE:
            return ConversationStateEventKind.INTERRUPTED

        if endpointing.action == EndpointAction.CANCEL_RESPONSE:
            return ConversationStateEventKind.CANCELLED

        return ConversationStateEventKind.USER_PAUSED

    def _transition_if_allowed(
        self,
        event_kind: ConversationStateEventKind,
        *,
        turn_id: str,
        reason: str,
    ) -> ConversationStateTransition | None:
        if not self._state_machine.can_transition(
            state=self._state_machine.current_state,
            event_kind=event_kind,
        ):
            return None

        return self._state_machine.transition(
            ConversationStateEvent(
                turn_id=turn_id,
                kind=event_kind,
                reason=reason,
                metadata={
                    "runtime": self.name,
                },
            )
        )

    def _should_capture_user_turn(
        self,
        endpointing: EndpointingDecision,
    ) -> bool:
        return (
            self._config.capture_finalized_user_turns
            and endpointing.action
            in {
                EndpointAction.START_COGNITION,
                EndpointAction.PREPARE_RESPONSE,
            }
            and bool(endpointing.turn_decision.transcript)
        )

    def _normal_output(
        self,
        *,
        signal: RealConversationInput,
        endpointing: EndpointingDecision,
        state_transitions: tuple[ConversationStateTransition, ...],
        streaming_output: StreamingCoordinatorOutput,
        session_snapshot: ConversationSessionSnapshotModel | None,
        attention_decision: AttentionDecision | None,
    ) -> RealConversationRuntimeOutput:
        actions: list[RealConversationRuntimeAction] = []

        if streaming_output.should_start_cognition:
            actions.append(RealConversationRuntimeAction.START_COGNITION)

        if streaming_output.should_start_tts:
            actions.append(RealConversationRuntimeAction.START_TTS)

        if session_snapshot is not None:
            actions.append(RealConversationRuntimeAction.UPDATE_SESSION)

        if attention_decision is not None:
            actions.append(RealConversationRuntimeAction.UPDATE_ATTENTION)

        if not actions:
            if endpointing.action == EndpointAction.WAIT_FOR_USER:
                actions.append(RealConversationRuntimeAction.WAIT_FOR_USER)
            else:
                actions.append(RealConversationRuntimeAction.KEEP_LISTENING)

        status = self._status_from_endpointing(
            endpointing=endpointing,
            streaming_output=streaming_output,
        )

        return RealConversationRuntimeOutput(
            turn_id=signal.turn_id,
            status=status,
            actions=tuple(actions),
            endpointing_decision=endpointing,
            state_transitions=state_transitions,
            streaming_output=streaming_output,
            session_snapshot=session_snapshot,
            attention_decision=attention_decision,
            should_start_cognition=streaming_output.should_start_cognition,
            should_start_tts=streaming_output.should_start_tts,
            should_cancel_active_work=False,
            should_keep_listening=streaming_output.should_keep_listening,
            cognition_context_block=(
                self.context_block()
                if session_snapshot is not None or attention_decision is not None
                else None
            ),
            reason=streaming_output.reason,
            metadata={
                "runtime": self.name,
                "endpoint_action": endpointing.action.value,
            },
        )

    def _interrupt_output(
        self,
        *,
        signal: RealConversationInput,
        endpointing: EndpointingDecision,
        interrupt_decision: InterruptDecision,
        attention_decision: AttentionDecision | None,
    ) -> RealConversationRuntimeOutput:
        return RealConversationRuntimeOutput(
            turn_id=signal.turn_id,
            status=RealConversationRuntimeStatus.INTERRUPTED,
            actions=(RealConversationRuntimeAction.CANCEL_ACTIVE_WORK,),
            endpointing_decision=endpointing,
            interrupt_decision=interrupt_decision,
            attention_decision=attention_decision,
            should_start_cognition=False,
            should_start_tts=False,
            should_cancel_active_work=True,
            should_keep_listening=interrupt_decision.should_return_to_listening,
            cognition_context_block=self.context_block(),
            reason="runtime handled interruption and cancelled active work",
            metadata={
                "runtime": self.name,
                "interrupt_reason": interrupt_decision.reason.value,
            },
        )

    def _submit_interrupt_attention(
        self,
        *,
        signal: RealConversationInput,
        endpointing: EndpointingDecision,
    ) -> AttentionDecision:
        decision = self._attention_runtime.submit_signal(
            AttentionSignal(
                kind=AttentionSignalKind.INTERRUPTION,
                text=signal.transcript or "interruption",
                priority=AttentionPriority.CRITICAL,
                urgency=TurnUrgency.CRITICAL,
                source="real_conversation_runtime",
                state=self._state_machine.current_state,
                target_id=signal.turn_id,
                metadata={
                    "endpoint_action": endpointing.action.value,
                },
            )
        )
        with self._lock:
            self._attention_update_count += 1

        return decision

    def _maybe_mark_response_started(
        self,
        streaming_output: StreamingCoordinatorOutput,
    ) -> list[ConversationStateTransition]:
        transitions: list[ConversationStateTransition] = []

        if not streaming_output.should_start_tts:
            return transitions

        event = streaming_output.event

        if event is None:
            return transitions

        transition = self._transition_if_allowed(
            ConversationStateEventKind.RESPONSE_STARTED,
            turn_id=event.turn_id,
            reason="runtime response streaming started TTS",
        )

        if transition is not None:
            transitions.append(transition)

        return transitions

    def _status_from_endpointing(
        self,
        *,
        endpointing: EndpointingDecision,
        streaming_output: StreamingCoordinatorOutput,
    ) -> RealConversationRuntimeStatus:
        if endpointing.action == EndpointAction.START_COGNITION:
            return RealConversationRuntimeStatus.COGNITION_READY

        if endpointing.action == EndpointAction.PREPARE_RESPONSE:
            return RealConversationRuntimeStatus.USER_THINKING

        if streaming_output.lifecycle == StreamingLifecycle.LISTENING:
            return RealConversationRuntimeStatus.LISTENING

        return self._status_from_streaming(streaming_output)

    @staticmethod
    def _status_from_streaming(
        streaming_output: StreamingCoordinatorOutput,
    ) -> RealConversationRuntimeStatus:
        if streaming_output.lifecycle == StreamingLifecycle.THINKING:
            return RealConversationRuntimeStatus.COGNITION_READY

        if streaming_output.lifecycle == StreamingLifecycle.RESPONSE_STREAMING:
            return RealConversationRuntimeStatus.STREAMING_RESPONSE

        if streaming_output.lifecycle == StreamingLifecycle.SPEAKING:
            return RealConversationRuntimeStatus.SPEAKING

        if streaming_output.lifecycle in {
            StreamingLifecycle.INTERRUPTED,
            StreamingLifecycle.CANCELLED,
        }:
            return RealConversationRuntimeStatus.INTERRUPTED

        if streaming_output.lifecycle == StreamingLifecycle.COMPLETED:
            return RealConversationRuntimeStatus.COMPLETED

        if streaming_output.lifecycle == StreamingLifecycle.LISTENING:
            return RealConversationRuntimeStatus.LISTENING

        return RealConversationRuntimeStatus.READY

    def _record_output(self, output: RealConversationRuntimeOutput) -> None:
        with self._lock:
            self._status = output.status
            self._last_actions = output.actions
            self._last_turn_id = output.turn_id

            if output.should_start_cognition:
                self._cognition_start_count += 1

            if output.should_start_tts:
                self._tts_start_count += 1

            if output.should_cancel_active_work:
                self._interrupt_count += 1

        self._logger.info(
            "real_conversation_runtime_output",
            runtime=self.name,
            turn_id=output.turn_id,
            status=output.status.value,
            actions=tuple(action.value for action in output.actions),
            should_start_cognition=output.should_start_cognition,
            should_start_tts=output.should_start_tts,
            should_cancel_active_work=output.should_cancel_active_work,
        )

    @staticmethod
    def _safe_topic(signal: RealConversationInput) -> str | None:
        text = signal.transcript.strip()

        if not text:
            return None

        words = [
            word.strip(".,?!:;()[]{}\"'").casefold()
            for word in text.split()
            if len(word.strip(".,?!:;()[]{}\"'")) >= 4
        ]

        if not words:
            return None

        return " ".join(words[:4])