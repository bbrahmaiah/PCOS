from __future__ import annotations

import pytest
from pydantic import ValidationError

from jarvis.conversation import (
    AdaptiveEndpointingEngine,
    ConversationMode,
    ConversationState,
    ConversationStateMachine,
    ConversationStateMachineConfig,
    EndpointingDecision,
    EndpointingInput,
    InterruptAction,
    InterruptController,
    InterruptControllerConfig,
    InterruptionIntent,
    InterruptPriority,
    InterruptReason,
    InterruptRequest,
    InterruptScope,
    StreamingConversationCoordinator,
    StreamingLifecycle,
    TurnDetectionInput,
)


def request(
    *,
    intent: InterruptionIntent = InterruptionIntent.BARGE_IN,
    priority: InterruptPriority = InterruptPriority.HIGH,
    scope: InterruptScope = InterruptScope.SPEECH_AND_COGNITION,
    assistant_was_speaking: bool = True,
    cognition_was_active: bool = True,
    tools_were_active: bool = False,
) -> InterruptRequest:
    return InterruptRequest(
        turn_id="turn-1",
        transcript="stop",
        intent=intent,
        priority=priority,
        scope=scope,
        assistant_was_speaking=assistant_was_speaking,
        cognition_was_active=cognition_was_active,
        tools_were_active=tools_were_active,
    )


def endpointing_decision_for_stop() -> EndpointingDecision:
    engine = AdaptiveEndpointingEngine()

    return engine.evaluate(
        EndpointingInput(
            signal=TurnDetectionInput(
                turn_id="turn-1",
                transcript="stop",
                conversation_mode=ConversationMode.COMMAND,
            ),
            conversation_state=ConversationState.SPEAKING,
        )
    )


def test_interrupt_controller_config_rejects_invalid_name() -> None:
    with pytest.raises(ValueError):
        InterruptControllerConfig(name=" ").validate()


def test_interrupt_request_requires_turn_id() -> None:
    with pytest.raises(ValidationError):
        InterruptRequest(turn_id=" ", intent=InterruptionIntent.STOP)


def test_interrupt_request_cleans_transcript() -> None:
    item = InterruptRequest(
        turn_id="turn-1",
        transcript="  stop    now ",
        intent=InterruptionIntent.STOP,
    )

    assert item.transcript == "stop now"


def test_no_interrupt_is_ignored() -> None:
    controller = InterruptController()

    decision = controller.handle_request(
        request(intent=InterruptionIntent.NONE)
    )
    snapshot = controller.snapshot()

    assert decision.interrupted is False
    assert decision.reason == InterruptReason.NO_INTERRUPT
    assert decision.actions == (InterruptAction.NONE,)
    assert snapshot.ignored_count == 1


def test_barge_in_cancels_speech_cognition_and_streaming() -> None:
    controller = InterruptController()

    decision = controller.handle_request(
        request(intent=InterruptionIntent.BARGE_IN)
    )
    snapshot = controller.snapshot()

    assert decision.interrupted is True
    assert decision.reason == InterruptReason.USER_BARGE_IN
    assert decision.should_cancel_speech is True
    assert decision.should_cancel_cognition is True
    assert decision.should_cancel_streaming is True
    assert decision.should_return_to_listening is True
    assert snapshot.interrupt_count == 1
    assert snapshot.speech_cancel_count == 1
    assert snapshot.cognition_cancel_count == 1
    assert snapshot.streaming_cancel_count == 1


def test_cancel_intent_cancels_all_active_work() -> None:
    controller = InterruptController()

    decision = controller.handle_request(
        request(
            intent=InterruptionIntent.CANCEL,
            priority=InterruptPriority.CRITICAL,
            scope=InterruptScope.ALL_ACTIVE_WORK,
            tools_were_active=True,
        )
    )

    assert decision.reason == InterruptReason.USER_CANCEL_INTENT
    assert decision.should_cancel_speech is True
    assert decision.should_cancel_cognition is True
    assert decision.should_cancel_streaming is True
    assert decision.should_cancel_tools is True
    assert InterruptAction.CANCEL_TOOLS in decision.actions


def test_pause_intent_can_pause_and_cancel_speech() -> None:
    controller = InterruptController()

    decision = controller.handle_request(
        request(
            intent=InterruptionIntent.PAUSE,
            scope=InterruptScope.SPEECH_ONLY,
            cognition_was_active=False,
        )
    )

    assert decision.reason == InterruptReason.USER_PAUSE_INTENT
    assert InterruptAction.PAUSE_SPEECH in decision.actions
    assert decision.should_cancel_speech is True
    assert decision.should_cancel_cognition is False


def test_correction_can_request_clarification_when_enabled() -> None:
    controller = InterruptController(
        config=InterruptControllerConfig(
            correction_requests_clarification=True
        )
    )

    decision = controller.handle_request(
        request(intent=InterruptionIntent.CORRECTION)
    )

    assert decision.reason == InterruptReason.USER_CORRECTION
    assert decision.should_request_clarification is True
    assert InterruptAction.REQUEST_CLARIFICATION in decision.actions


def test_endpointing_decision_routes_to_interrupt() -> None:
    controller = InterruptController()
    endpointing = endpointing_decision_for_stop()

    decision = controller.handle_endpointing_decision(endpointing)

    assert decision.interrupted is True
    assert decision.reason == InterruptReason.USER_STOP_INTENT
    assert decision.should_cancel_speech is True
    assert decision.should_cancel_cognition is True


def test_interrupt_from_streaming_event() -> None:
    from jarvis.conversation import StreamingConversationEvent, StreamingEventKind

    controller = InterruptController()
    decision = controller.interrupt_from_streaming_event(
        StreamingConversationEvent(
            turn_id="turn-1",
            kind=StreamingEventKind.INTERRUPT_REQUESTED,
            text="stop",
        )
    )

    assert decision.interrupted is True
    assert decision.reason == InterruptReason.USER_BARGE_IN
    assert decision.priority == InterruptPriority.CRITICAL


def test_controller_coordinates_streaming_coordinator() -> None:
    streaming = StreamingConversationCoordinator()
    controller = InterruptController(streaming_coordinator=streaming)

    controller.handle_request(request(intent=InterruptionIntent.BARGE_IN))
    snapshot = streaming.snapshot()

    assert snapshot.lifecycle == StreamingLifecycle.INTERRUPTED
    assert snapshot.interrupt_count == 1


def test_controller_coordinates_state_machine() -> None:
    machine = ConversationStateMachine(
        config=ConversationStateMachineConfig(
            initial_state=ConversationState.SPEAKING
        )
    )
    controller = InterruptController(state_machine=machine)

    decision = controller.handle_request(request(intent=InterruptionIntent.STOP))

    assert decision.should_return_to_listening is True
    assert machine.current_state == ConversationState.LISTENING


def test_controller_coordinates_state_machine_and_streaming() -> None:
    machine = ConversationStateMachine(
        config=ConversationStateMachineConfig(
            initial_state=ConversationState.SPEAKING
        )
    )
    streaming = StreamingConversationCoordinator()
    controller = InterruptController(
        state_machine=machine,
        streaming_coordinator=streaming,
    )

    decision = controller.handle_request(request(intent=InterruptionIntent.CANCEL))

    assert decision.reason == InterruptReason.USER_CANCEL_INTENT
    assert machine.current_state == ConversationState.LISTENING
    assert streaming.snapshot().lifecycle == StreamingLifecycle.CANCELLED


def test_snapshot_and_reset() -> None:
    controller = InterruptController()

    controller.handle_request(request(intent=InterruptionIntent.STOP))
    snapshot = controller.snapshot()

    assert snapshot.interrupt_count == 1
    assert snapshot.last_reason == InterruptReason.USER_STOP_INTENT

    controller.reset()
    reset_snapshot = controller.snapshot()

    assert reset_snapshot.interrupt_count == 0
    assert reset_snapshot.last_reason is None


def test_interrupt_decision_cancelled_anything_property() -> None:
    controller = InterruptController()

    decision = controller.handle_request(request(intent=InterruptionIntent.STOP))

    assert decision.cancelled_anything is True


def test_interrupt_enum_values_are_stable() -> None:
    assert InterruptPriority.CRITICAL.value == "critical"
    assert InterruptScope.ALL_ACTIVE_WORK.value == "all_active_work"
    assert InterruptAction.CANCEL_SPEECH.value == "cancel_speech"
    assert InterruptAction.CANCEL_COGNITION.value == "cancel_cognition"
    assert InterruptAction.CANCEL_STREAMING.value == "cancel_streaming"
    assert InterruptReason.USER_BARGE_IN.value == "user_barge_in"