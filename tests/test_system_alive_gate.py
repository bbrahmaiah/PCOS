from __future__ import annotations

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
from jarvis.conversation.runtime import RealConversationRuntime
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
from jarvis.runtime.kernel.runtime_kernel import RuntimeKernel
from jarvis.system import (
    JarvisAliveGate,
    JarvisAliveGateCheckKind,
    JarvisAliveGateConfig,
    JarvisAliveGateStatus,
    JarvisSystemFactoryBundle,
)


def _now() -> datetime:
    return datetime.now(UTC)


class FakeMemoryGateway:
    def __init__(self, records: tuple[MemoryRecord, ...] = ()) -> None:
        self.records = records
        self.queries: list[MemoryQuery] = []
        self.writes: list[MemoryWriteRequest] = []

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
                    reason="alive gate retrieval",
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
            reason="alive gate retrieval allowed",
            policy_classification=MemoryPolicyClassification.ALLOWED,
        )

    def remember(
        self,
        request: MemoryWriteRequest,
    ) -> MemoryGatewayWriteResult:
        self.writes.append(request)
        return MemoryGatewayWriteResult(
            request=request,
            record=request.to_record(),
            allowed=True,
            blocked=False,
            reason="alive gate write allowed",
            policy_classification=MemoryPolicyClassification.ALLOWED,
        )


class FakeCognitionAdapter:
    def __init__(self) -> None:
        self.requests: list[CognitionRequest] = []

    @property
    def name(self) -> str:
        return "alive_gate_fake_cognition_adapter"

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
                text=f"Alive response using {memory_text}",
                kind=CognitionResponseKind.SPOKEN_REPLY,
                confidence=0.96,
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


class FakeOrchestrationRuntime:
    def __init__(self) -> None:
        self.started = False
        self.stopped = False
        self.heartbeat_count = 0

    def start(self) -> None:
        self.started = True
        self.stopped = False

    def stop(self) -> None:
        self.started = False
        self.stopped = True

    def heartbeat(self) -> None:
        self.heartbeat_count += 1

    def snapshot(self) -> dict[str, object]:
        return {
            "started": self.started,
            "stopped": self.stopped,
            "heartbeat_count": self.heartbeat_count,
        }


def test_alive_gate_rejects_empty_name() -> None:
    with pytest.raises(ValueError):
        JarvisAliveGateConfig(name=" ")


def test_alive_gate_passes_full_living_runtime_path() -> None:
    memory = FakeMemoryGateway(
        records=(
            _memory_record(
                text="Bala is building a living real-time cognition JARVIS OS."
            ),
        )
    )
    adapter = FakeCognitionAdapter()
    orchestration = FakeOrchestrationRuntime()

    gate = JarvisAliveGate(
        config=JarvisAliveGateConfig(
            name="test_alive_gate",
            session_id="session",
            attach_conversation=True,
            attach_presence=True,
            attach_orchestration=True,
        ),
        factories=JarvisSystemFactoryBundle(
            memory_gateway=lambda: _memory_gateway(memory),
            cognition_worker=lambda: CognitionWorker(
                adapter=_cognition_adapter(adapter)
            ),
            conversation_runtime=RealConversationRuntime,
            presence_engine=_presence_engine,
            orchestration_runtime=lambda: orchestration,
            kernel=RuntimeKernel,
        ),
    )

    report = gate.run()

    assert report.status == JarvisAliveGateStatus.PASSED
    assert report.passed is True
    assert report.failed_count == 0
    assert len(memory.queries) >= 2
    assert len(memory.writes) == 1
    assert len(adapter.requests) >= 2
    assert orchestration.stopped is True

    check_kinds = {check.kind for check in report.checks}

    assert JarvisAliveGateCheckKind.BOOT in check_kinds
    assert JarvisAliveGateCheckKind.SNAPSHOT_RUNNING in check_kinds
    assert JarvisAliveGateCheckKind.NORMAL_PIPELINE in check_kinds
    assert JarvisAliveGateCheckKind.GOVERNED_MEMORY_WRITE in check_kinds
    assert JarvisAliveGateCheckKind.INTERRUPTION in check_kinds
    assert JarvisAliveGateCheckKind.SHUTDOWN in check_kinds


def test_alive_gate_reports_failed_boot() -> None:
    def fail_memory() -> MemoryGateway:
        raise RuntimeError("memory boot failed")

    gate = JarvisAliveGate(
        config=JarvisAliveGateConfig(name="failed_alive_gate"),
        factories=JarvisSystemFactoryBundle(
            memory_gateway=fail_memory,
            cognition_worker=lambda: CognitionWorker(
                adapter=_cognition_adapter(FakeCognitionAdapter())
            ),
            conversation_runtime=RealConversationRuntime,
            presence_engine=None,
            orchestration_runtime=None,
            kernel=RuntimeKernel,
        ),
    )

    report = gate.run()

    assert report.status == JarvisAliveGateStatus.FAILED
    assert report.passed is False
    assert report.error is not None
    assert "memory boot failed" in report.error


def test_alive_gate_enum_values_are_stable() -> None:
    assert JarvisAliveGateStatus.PASSED.value == "passed"
    assert JarvisAliveGateCheckKind.INTERRUPTION.value == "interruption"


def _memory_gateway(gateway: FakeMemoryGateway) -> MemoryGateway:
    return cast(MemoryGateway, gateway)


def _cognition_adapter(adapter: FakeCognitionAdapter) -> CognitionAdapter:
    return cast(CognitionAdapter, adapter)


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
        name="alive_gate_presence_engine",
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