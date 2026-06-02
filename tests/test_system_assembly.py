from __future__ import annotations

import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import cast

import pytest

from jarvis.cognition.adapters import (
    CognitionAdapter,
    CognitionAdapterCapability,
    CognitionAdapterResult,
    CognitionAdapterSnapshot,
)
from jarvis.cognition.models import (
    CognitionRequest,
    CognitionResponse,
    CognitionResponseKind,
)
from jarvis.cognition.worker import CognitionWorker
from jarvis.conversation.models import ConversationMode, TurnInputSource
from jarvis.conversation.runtime import (
    RealConversationInput,
    RealConversationRuntime,
    RealConversationRuntimeAction,
)
from jarvis.memory.gateway import (
    MemoryGateway,
    MemoryGatewayRetrievalResult,
    MemoryGatewayWriteResult,
)
from jarvis.memory.models import (
    MemoryKind,
    MemoryPolicyClassification,
    MemoryQuery,
    MemoryRecord,
    MemoryRetrievalExplanation,
    MemoryRetrievalResult,
    MemoryScope,
    MemorySearchResult,
    MemorySensitivity,
    MemorySource,
    MemoryWriteRequest,
)
from jarvis.presence import PresenceEngine, PresenceEngineAdapters
from jarvis.presence.adapters import (
    FakeAudioPlaybackAdapter,
    FakeMicrophoneAdapter,
    FakeSpeechToTextAdapter,
    FakeTextToSpeechAdapter,
    FakeVoiceActivityAdapter,
    FakeWakeWordAdapter,
    make_fake_audio_frame,
)
from jarvis.presence.models import VoiceActivityState
from jarvis.runtime.events import EventBus
from jarvis.runtime.kernel.runtime_kernel import RuntimeKernel
from jarvis.runtime.shared.enums import WorkerStatus
from jarvis.system import (
    CognitionRuntimeWorker,
    ConversationRuntimeWorker,
    JarvisAskStatus,
    JarvisMemoryWriteStatus,
    JarvisSystem,
    JarvisSystemStatus,
    MemoryRuntimeWorker,
    PresenceRuntimeWorker,
)


def _now() -> datetime:
    return datetime.now(UTC)


class FakeMemoryGateway:
    def __init__(
        self,
        records: tuple[MemoryRecord, ...] = (),
        *,
        block_writes: bool = False,
    ) -> None:
        self.records = records
        self.block_writes = block_writes
        self.queries: list[MemoryQuery] = []
        self.writes: list[MemoryWriteRequest] = []

    @property
    def name(self) -> str:
        return "fake_memory_gateway"

    def retrieve(
        self,
        query: MemoryQuery,
    ) -> MemoryGatewayRetrievalResult:
        self.queries.append(query)
        results = tuple(
            MemorySearchResult(
                record=record,
                score=record.confidence,
                explanation=MemoryRetrievalExplanation(
                    source=MemorySource.CONVERSATION,
                    reason="test retrieval",
                    confidence=record.confidence,
                    policy_classification=MemoryPolicyClassification.ALLOWED,
                ),
            )
            for record in self.records
        )
        return MemoryGatewayRetrievalResult(
            query=query,
            retrieval=MemoryRetrievalResult(
                query=query,
                results=results,
            ),
            allowed=True,
            blocked=False,
            reason="test retrieval allowed",
            policy_classification=MemoryPolicyClassification.ALLOWED,
        )

    def remember(
        self,
        request: MemoryWriteRequest,
    ) -> MemoryGatewayWriteResult:
        self.writes.append(request)

        if self.block_writes:
            return MemoryGatewayWriteResult(
                request=request,
                record=None,
                allowed=False,
                blocked=True,
                reason="test write blocked",
                policy_classification=MemoryPolicyClassification.BLOCKED,
            )

        return MemoryGatewayWriteResult(
            request=request,
            record=request.to_record(),
            allowed=True,
            blocked=False,
            reason="test write allowed",
            policy_classification=MemoryPolicyClassification.ALLOWED,
        )

    def snapshot(self) -> dict[str, object]:
        return {
            "query_count": len(self.queries),
            "write_count": len(self.writes),
            "record_count": len(self.records),
        }


class FakeCognitionAdapter:
    def __init__(self) -> None:
        self.requests: list[CognitionRequest] = []

    @property
    def name(self) -> str:
        return "fake_cognition_adapter"

    @property
    def capabilities(self) -> tuple[CognitionAdapterCapability, ...]:
        return ()

    def generate(
        self,
        request: CognitionRequest,
    ) -> CognitionAdapterResult:
        self.requests.append(request)
        memory_text = "no memory"

        if request.context.items:
            memory_text = request.context.items[0].text

        started_at = _now()
        finished_at = _now()

        return CognitionAdapterResult(
            request_id=request.request_id,
            response=CognitionResponse(
                request_id=request.request_id,
                text=f"Short answer using {memory_text}",
                kind=CognitionResponseKind.SPOKEN_REPLY,
                confidence=0.95,
            ),
            failure=None,
            started_at=started_at,
            finished_at=finished_at,
            duration_ms=1,
            metadata={},
        )

    def snapshot(self) -> CognitionAdapterSnapshot:
        return CognitionAdapterSnapshot(
            name=self.name,
            capabilities=self.capabilities,
            metadata={"request_count": len(self.requests)},
        )


def _memory_gateway(
    gateway: FakeMemoryGateway,
) -> MemoryGateway:
    return cast(MemoryGateway, gateway)


def _cognition_adapter(
    adapter: FakeCognitionAdapter,
) -> CognitionAdapter:
    return cast(CognitionAdapter, adapter)

def test_conversation_runtime_worker_is_base_worker_compatible() -> None:
    conversation = RealConversationRuntime()
    event_bus = EventBus(name="test_event_bus")
    worker = ConversationRuntimeWorker(
        conversation_runtime=conversation,
        event_bus=event_bus,
        tick_interval_seconds=0.01,
    )

    worker.start()

    try:
        assert _eventually(lambda: worker.wait_until_ready())
    finally:
        worker.stop()

    snapshot = worker.snapshot()

    assert snapshot.name == "conversation_runtime"
    assert snapshot.status == WorkerStatus.STOPPED


def test_memory_runtime_worker_is_base_worker_compatible() -> None:
    gateway = FakeMemoryGateway()
    event_bus = EventBus(name="test_event_bus")
    worker = MemoryRuntimeWorker(
        memory_gateway=_memory_gateway(gateway),
        event_bus=event_bus,
        tick_interval_seconds=0.01,
    )

    worker.start()
    worker.stop()

    snapshot = worker.snapshot()

    assert snapshot.name == "memory_runtime"
    assert snapshot.status == WorkerStatus.STOPPED


def test_cognition_runtime_worker_starts_inner_worker() -> None:
    adapter = FakeCognitionAdapter()
    cognition = CognitionWorker(adapter=_cognition_adapter(adapter))
    event_bus = EventBus(name="test_event_bus")
    worker = CognitionRuntimeWorker(
        cognition_worker=cognition,
        event_bus=event_bus,
        tick_interval_seconds=0.01,
    )

    worker.start()

    try:
        assert _eventually(lambda: cognition.started)
    finally:
        worker.stop()

    assert cognition.started is False


def test_jarvis_system_rejects_empty_name() -> None:
    with pytest.raises(ValueError):
        JarvisSystem(
            name=" ",
            memory_gateway=_memory_gateway(FakeMemoryGateway()),
            cognition_worker=CognitionWorker(
                adapter=_cognition_adapter(FakeCognitionAdapter())
            ),
            kernel=RuntimeKernel(),
        )


def test_jarvis_system_start_registers_memory_and_cognition_workers() -> None:
    system = JarvisSystem(
        memory_gateway=_memory_gateway(FakeMemoryGateway()),
        cognition_worker=CognitionWorker(
            adapter=_cognition_adapter(FakeCognitionAdapter())
        ),
        kernel=RuntimeKernel(),
    )

    system.start()
    snapshot = system.snapshot()

    assert system.status == JarvisSystemStatus.RUNNING
    assert snapshot.memory_worker is not None
    assert snapshot.cognition_worker is not None
    assert snapshot.memory_worker.running is True
    assert snapshot.cognition_worker.running is True

    system.stop()

    stopped = system.snapshot()

    assert stopped.status == JarvisSystemStatus.STOPPED
    assert stopped.memory_worker is not None
    assert stopped.cognition_worker is not None
    assert stopped.memory_worker.status == WorkerStatus.STOPPED
    assert stopped.cognition_worker.status == WorkerStatus.STOPPED


def test_jarvis_system_start_is_idempotent() -> None:
    system = JarvisSystem(
        memory_gateway=_memory_gateway(FakeMemoryGateway()),
        cognition_worker=CognitionWorker(
            adapter=_cognition_adapter(FakeCognitionAdapter())
        ),
        kernel=RuntimeKernel(),
    )

    system.start()
    system.start()
    snapshot = system.snapshot()

    assert snapshot.status == JarvisSystemStatus.RUNNING
    assert snapshot.kernel_snapshot is not None

    system.stop()


def test_jarvis_system_ask_builds_memory_context_and_cognition_request() -> None:
    record = _memory_record(
        text="Bala is building a living JARVIS OS.",
    )
    gateway = FakeMemoryGateway(records=(record,))
    adapter = FakeCognitionAdapter()
    system = JarvisSystem(
        memory_gateway=_memory_gateway(gateway),
        cognition_worker=CognitionWorker(adapter=_cognition_adapter(adapter)),
        kernel=RuntimeKernel(),
    )

    system.start()
    response = system.ask("What am I building?", session_id="session")
    system.stop()

    assert response.status == JarvisAskStatus.ANSWERED
    assert response.used_memory is True
    assert response.used_cognition is True
    assert response.memory_result_count == 1
    assert "living JARVIS OS" in response.text
    assert len(gateway.queries) == 1
    assert gateway.queries[0].text == "What am I building?"
    assert len(adapter.requests) == 1

    cognition_request = adapter.requests[0]

    assert cognition_request.source == "jarvis_system"
    assert cognition_request.context.session_id == "session"
    assert cognition_request.context.item_count == 1
    assert cognition_request.context.items[0].text == (
        "Bala is building a living JARVIS OS."
    )


def test_jarvis_system_ask_uses_concise_spoken_policy() -> None:
    adapter = FakeCognitionAdapter()
    system = JarvisSystem(
        memory_gateway=_memory_gateway(FakeMemoryGateway()),
        cognition_worker=CognitionWorker(adapter=_cognition_adapter(adapter)),
        kernel=RuntimeKernel(),
    )

    system.start()
    system.ask("What is Python?")
    system.stop()

    request = adapter.requests[0]

    assert request.policy.cancellable is True
    assert request.policy.allow_tools is False
    assert request.policy.max_response_chars == 600
    assert request.policy.metadata["voice_native_default"] is True


def test_jarvis_system_snapshot_tracks_ask_count() -> None:
    system = JarvisSystem(
        memory_gateway=_memory_gateway(FakeMemoryGateway()),
        cognition_worker=CognitionWorker(
            adapter=_cognition_adapter(FakeCognitionAdapter())
        ),
        kernel=RuntimeKernel(),
    )

    system.start()
    system.ask("hello")
    snapshot = system.snapshot()
    system.stop()

    assert snapshot.ask_count == 1
    assert snapshot.failure_count == 0
    assert len(snapshot.subsystem_health) == 2


def test_jarvis_system_request_rejects_empty_text() -> None:
    system = JarvisSystem(
        memory_gateway=_memory_gateway(FakeMemoryGateway()),
        cognition_worker=CognitionWorker(
            adapter=_cognition_adapter(FakeCognitionAdapter())
        ),
        kernel=RuntimeKernel(),
    )

    with pytest.raises(ValueError):
        system.ask(" ")

def test_step_44b_writes_explicit_user_memory_after_cognition() -> None:
    gateway = FakeMemoryGateway()
    system = JarvisSystem(
        memory_gateway=_memory_gateway(gateway),
        cognition_worker=CognitionWorker(
            adapter=_cognition_adapter(FakeCognitionAdapter())
        ),
        kernel=RuntimeKernel(),
    )

    system.start()
    response = system.ask("Remember that my favorite editor is VS Code.")
    system.stop()

    assert response.status == JarvisAskStatus.ANSWERED
    assert response.wrote_memory is True
    assert response.memory_write.status == JarvisMemoryWriteStatus.WRITTEN
    assert len(gateway.queries) == 1
    assert len(gateway.writes) == 1

    write = gateway.writes[0]

    assert write.text == "my favorite editor is VS Code"
    assert write.kind == MemoryKind.PREFERENCE
    assert write.scope == MemoryScope.USER
    assert write.source == MemorySource.USER_EXPLICIT
    assert write.sensitivity == MemorySensitivity.PRIVATE

def test_step_44b_surfaces_blocked_memory_write() -> None:
    gateway = FakeMemoryGateway(block_writes=True)
    system = JarvisSystem(
        memory_gateway=_memory_gateway(gateway),
        cognition_worker=CognitionWorker(
            adapter=_cognition_adapter(FakeCognitionAdapter())
        ),
        kernel=RuntimeKernel(),
    )

    system.start()
    response = system.ask("Remember that this is sensitive.")
    system.stop()

    assert response.status == JarvisAskStatus.ANSWERED
    assert response.wrote_memory is False
    assert response.memory_write.status == JarvisMemoryWriteStatus.BLOCKED
    assert len(gateway.writes) == 1
    assert response.memory_write.result is not None
    assert response.memory_write.result.blocked is True

def test_step_44b_does_not_write_without_explicit_memory_intent() -> None:
    gateway = FakeMemoryGateway()
    system = JarvisSystem(
        memory_gateway=_memory_gateway(gateway),
        cognition_worker=CognitionWorker(
            adapter=_cognition_adapter(FakeCognitionAdapter())
        ),
        kernel=RuntimeKernel(),
    )

    system.start()
    response = system.ask("What is Python?")
    system.stop()

    assert response.status == JarvisAskStatus.ANSWERED
    assert response.wrote_memory is False
    assert response.memory_write.status == JarvisMemoryWriteStatus.NOT_REQUESTED
    assert len(gateway.queries) == 1
    assert len(gateway.writes) == 0


def test_enum_values_are_stable() -> None:
    assert JarvisSystemStatus.RUNNING.value == "running"
    assert JarvisAskStatus.ANSWERED.value == "answered"
    assert JarvisMemoryWriteStatus.WRITTEN.value == "written"
    assert RealConversationRuntimeAction.CANCEL_ACTIVE_WORK.value == (
        "cancel_active_work"
    )
    assert VoiceActivityState.SPEECH_STARTED.value == "speech_started"

def _eventually(
    predicate: Callable[[], bool],
    *,
    timeout_seconds: float = 1.0,
    interval_seconds: float = 0.01,
) -> bool:
    deadline = time.monotonic() + timeout_seconds

    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval_seconds)

    return False

def _presence_engine() -> PresenceEngine:
    frames = (
        make_fake_audio_frame(frame_index=0),
        make_fake_audio_frame(frame_index=1),
        make_fake_audio_frame(frame_index=2),
    )
    adapters = PresenceEngineAdapters(
        microphone=FakeMicrophoneAdapter(frames=frames),
        wake_word=FakeWakeWordAdapter(detection_pattern=(True, False, False)),
        vad=FakeVoiceActivityAdapter(
            states=(
                VoiceActivityState.SPEECH_STARTED,
                VoiceActivityState.SPEECH_CONTINUING,
                VoiceActivityState.SPEECH_ENDED,
            ),
        ),
        stt=FakeSpeechToTextAdapter(
            text="hello jarvis",
            confidence=0.98,
        ),
        tts=FakeTextToSpeechAdapter(),
        playback=FakeAudioPlaybackAdapter(),
    )

    return PresenceEngine(
        name="test_presence_engine",
        adapters=adapters,
    )


def _memory_record(*, text: str) -> MemoryRecord:
    return MemoryRecord(
        kind=MemoryKind.SEMANTIC,
        scope=MemoryScope.USER,
        source=MemorySource.CONVERSATION,
        text=text,
        sensitivity=MemorySensitivity.PRIVATE,
        confidence=0.9,
        metadata={},
    )

def test_jarvis_system_start_registers_conversation_worker_when_configured() -> None:
    system = JarvisSystem(
        memory_gateway=_memory_gateway(FakeMemoryGateway()),
        cognition_worker=CognitionWorker(
            adapter=_cognition_adapter(FakeCognitionAdapter())
        ),
        conversation_runtime=RealConversationRuntime(),
        kernel=RuntimeKernel(),
    )

    system.start()
    snapshot = system.snapshot()

    assert system.status == JarvisSystemStatus.RUNNING
    assert snapshot.conversation_worker is not None
    assert snapshot.conversation_worker.running is True
    assert len(snapshot.subsystem_health) == 3

    system.stop()

    stopped = system.snapshot()

    assert stopped.conversation_worker is not None
    assert stopped.conversation_worker.status == WorkerStatus.STOPPED

def test_jarvis_system_accepts_conversation_input() -> None:
    system = JarvisSystem(
        memory_gateway=_memory_gateway(FakeMemoryGateway()),
        cognition_worker=CognitionWorker(
            adapter=_cognition_adapter(FakeCognitionAdapter())
        ),
        conversation_runtime=RealConversationRuntime(),
        kernel=RuntimeKernel(),
    )

    system.start()
    output = system.accept_conversation_input(
        RealConversationInput(
            transcript="What is Python?",
            source=TurnInputSource.STT_PARTIAL,
            is_speech_active=False,
            silence_ms=900,
            speech_ms=1200,
            vad_confidence=0.95,
            transcript_stability=0.95,
            conversation_mode=ConversationMode.QUESTION,
        )
    )
    system.stop()

    assert output.should_keep_listening is True
    assert output.actions

def test_jarvis_system_conversation_interruption_emits_cancel() -> None:
    system = JarvisSystem(
        memory_gateway=_memory_gateway(FakeMemoryGateway()),
        cognition_worker=CognitionWorker(
            adapter=_cognition_adapter(FakeCognitionAdapter())
        ),
        conversation_runtime=RealConversationRuntime(),
        kernel=RuntimeKernel(),
    )

    system.start()
    output = system.accept_conversation_input(
        RealConversationInput(
            transcript="stop",
            source=TurnInputSource.INTERRUPTION_WORKER,
            is_speech_active=True,
            is_assistant_speaking=True,
            silence_ms=0,
            speech_ms=250,
            vad_confidence=0.99,
            transcript_stability=1.0,
            conversation_mode=ConversationMode.COMMAND,
        )
    )
    system.stop()

    assert output.should_cancel_active_work is True
    assert RealConversationRuntimeAction.CANCEL_ACTIVE_WORK in output.actions

def test_jarvis_system_ask_updates_conversation_when_configured() -> None:
    adapter = FakeCognitionAdapter()
    system = JarvisSystem(
        memory_gateway=_memory_gateway(FakeMemoryGateway()),
        cognition_worker=CognitionWorker(adapter=_cognition_adapter(adapter)),
        conversation_runtime=RealConversationRuntime(),
        kernel=RuntimeKernel(),
    )

    system.start()
    response = system.ask("What is Python?")
    snapshot = system.snapshot()
    system.stop()

    assert response.status == JarvisAskStatus.ANSWERED
    assert response.metadata["conversation_updated"] is True

    conversation_health = [
        health
        for health in snapshot.subsystem_health
        if health.kind.value == "conversation"
    ]

    assert len(conversation_health) == 1


def test_presence_runtime_worker_is_base_worker_compatible() -> None:
    event_bus = EventBus(name="test_event_bus")
    worker = PresenceRuntimeWorker(
        presence_engine=_presence_engine(),
        event_bus=event_bus,
        tick_interval_seconds=0.01,
    )

    worker.start()

    try:
        assert _eventually(lambda: worker.wait_until_ready())
    finally:
        worker.stop()

    snapshot = worker.snapshot()

    assert snapshot.name == "presence_runtime"
    assert snapshot.status == WorkerStatus.STOPPED


def test_jarvis_system_start_registers_presence_worker_when_configured() -> None:
    system = JarvisSystem(
        memory_gateway=_memory_gateway(FakeMemoryGateway()),
        cognition_worker=CognitionWorker(
            adapter=_cognition_adapter(FakeCognitionAdapter())
        ),
        conversation_runtime=RealConversationRuntime(),
        presence_engine=_presence_engine(),
        kernel=RuntimeKernel(),
    )

    system.start()
    snapshot = system.snapshot()

    assert system.status == JarvisSystemStatus.RUNNING
    assert snapshot.presence_worker is not None
    assert snapshot.presence_worker.running is True
    assert len(snapshot.subsystem_health) == 4

    system.stop()

    stopped = system.snapshot()

    assert stopped.presence_worker is not None
    assert stopped.presence_worker.status == WorkerStatus.STOPPED


def test_jarvis_system_ask_updates_presence_when_configured() -> None:
    adapter = FakeCognitionAdapter()
    system = JarvisSystem(
        memory_gateway=_memory_gateway(FakeMemoryGateway()),
        cognition_worker=CognitionWorker(adapter=_cognition_adapter(adapter)),
        conversation_runtime=RealConversationRuntime(),
        presence_engine=_presence_engine(),
        kernel=RuntimeKernel(),
    )

    system.start()
    response = system.ask("What is Python?")
    snapshot = system.snapshot()
    system.stop()

    assert response.status == JarvisAskStatus.ANSWERED
    assert response.metadata["conversation_updated"] is True
    assert response.metadata["presence_updated"] is True

    presence_health = [
        health
        for health in snapshot.subsystem_health
        if health.kind.value == "presence"
    ]

    assert len(presence_health) == 1

def test_jarvis_system_can_publish_presence_response_ready() -> None:
    system = JarvisSystem(
        memory_gateway=_memory_gateway(FakeMemoryGateway()),
        cognition_worker=CognitionWorker(
            adapter=_cognition_adapter(FakeCognitionAdapter())
        ),
        presence_engine=_presence_engine(),
        kernel=RuntimeKernel(),
    )

    system.start()
    system.publish_presence_response_ready(text="Ready.")
    snapshot = system.snapshot()
    system.stop()

    assert snapshot.presence_worker is not None