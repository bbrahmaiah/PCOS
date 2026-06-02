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
    JarvisSystemFactoryBundle,
    LiveDependencyProfile,
    LiveDependencyWiring,
    LiveDependencyWiringConfig,
    RuntimeReadinessCheckKind,
    RuntimeReadinessConfig,
    RuntimeReadinessReview,
    RuntimeReadinessStatus,
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
                    reason="runtime readiness retrieval",
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
            reason="runtime readiness retrieval allowed",
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
            reason="runtime readiness write allowed",
            policy_classification=MemoryPolicyClassification.ALLOWED,
        )


class FakeCognitionAdapter:
    def __init__(self) -> None:
        self.requests: list[CognitionRequest] = []

    @property
    def name(self) -> str:
        return "runtime_readiness_fake_cognition_adapter"

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
                text=f"Runtime readiness response using {memory_text}",
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


def test_runtime_readiness_config_rejects_empty_session() -> None:
    with pytest.raises(ValueError):
        RuntimeReadinessConfig(session_id=" ")


def test_runtime_readiness_review_passes_full_runtime_path() -> None:
    memory = FakeMemoryGateway(
        records=(
            _memory_record(
                text="Bala is building a living personal cognition OS."
            ),
        )
    )
    adapter = FakeCognitionAdapter()
    orchestration = FakeOrchestrationRuntime()

    review = RuntimeReadinessReview(
        config=RuntimeReadinessConfig(
            session_id="session",
            require_presence=True,
            require_orchestration=True,
            minimum_subsystems=5,
        ),
        wiring=LiveDependencyWiring(
            config=LiveDependencyWiringConfig(
                name="runtime_readiness_test",
                profile=LiveDependencyProfile.TEST,
                dry_run=False,
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
        ),
    )

    report = review.run()

    assert report.status == RuntimeReadinessStatus.PASSED
    assert report.passed is True
    assert report.failed_count == 0
    assert len(memory.queries) >= 3
    assert len(memory.writes) == 1
    assert len(adapter.requests) >= 3
    assert orchestration.stopped is True

    check_kinds = {check.kind for check in report.checks}

    assert RuntimeReadinessCheckKind.DEPENDENCY_GRAPH in check_kinds
    assert RuntimeReadinessCheckKind.BOOT in check_kinds
    assert RuntimeReadinessCheckKind.SYSTEM_RUNNING in check_kinds
    assert RuntimeReadinessCheckKind.WORKERS_REGISTERED in check_kinds
    assert RuntimeReadinessCheckKind.EVENTBUS_STARTED in check_kinds
    assert RuntimeReadinessCheckKind.COGNITION_PIPELINE in check_kinds
    assert RuntimeReadinessCheckKind.MEMORY_RETRIEVE in check_kinds
    assert RuntimeReadinessCheckKind.MEMORY_WRITE in check_kinds
    assert RuntimeReadinessCheckKind.PRESENCE_OUTPUT in check_kinds
    assert RuntimeReadinessCheckKind.INTERRUPTION in check_kinds
    assert RuntimeReadinessCheckKind.RECOVERY_BASELINE in check_kinds
    assert RuntimeReadinessCheckKind.SHUTDOWN in check_kinds


def test_runtime_readiness_review_fails_when_dependencies_block() -> None:
    def failing_memory() -> MemoryGateway:
        raise RuntimeError("should not be called")

    review = RuntimeReadinessReview(
        config=RuntimeReadinessConfig(
            session_id="session",
            minimum_subsystems=3,
        ),
        wiring=LiveDependencyWiring(
            config=LiveDependencyWiringConfig(
                name="runtime_readiness_blocked",
                profile=LiveDependencyProfile.TEST,
                dry_run=False,
                attach_conversation=False,
                attach_presence=False,
                attach_orchestration=False,
            ),
            factories=JarvisSystemFactoryBundle(
                memory_gateway=failing_memory,
                cognition_worker=lambda: CognitionWorker(
                    adapter=_cognition_adapter(FakeCognitionAdapter())
                ),
                conversation_runtime=None,
                presence_engine=None,
                orchestration_runtime=None,
                kernel=RuntimeKernel,
            ),
        ),
    )

    report = review.run()

    assert report.status == RuntimeReadinessStatus.FAILED
    assert report.passed is False
    assert report.error is not None
    assert any(
        check.kind == RuntimeReadinessCheckKind.BOOT
        for check in report.checks
    )


def test_runtime_readiness_review_fails_when_presence_required_missing() -> None:
    review = RuntimeReadinessReview(
        config=RuntimeReadinessConfig(
            session_id="session",
            require_presence=True,
            minimum_subsystems=3,
        ),
        wiring=LiveDependencyWiring(
            config=LiveDependencyWiringConfig(
                name="runtime_readiness_no_presence",
                profile=LiveDependencyProfile.TEST,
                dry_run=False,
                attach_conversation=True,
                attach_presence=False,
                attach_orchestration=False,
            ),
            factories=JarvisSystemFactoryBundle(
                memory_gateway=lambda: _memory_gateway(FakeMemoryGateway()),
                cognition_worker=lambda: CognitionWorker(
                    adapter=_cognition_adapter(FakeCognitionAdapter())
                ),
                conversation_runtime=RealConversationRuntime,
                presence_engine=None,
                orchestration_runtime=None,
                kernel=RuntimeKernel,
            ),
        ),
    )

    report = review.run()

    assert report.status == RuntimeReadinessStatus.FAILED
    presence_checks = [
        check
        for check in report.checks
        if check.kind == RuntimeReadinessCheckKind.PRESENCE_OUTPUT
    ]

    assert len(presence_checks) == 1
    assert presence_checks[0].passed is False


def test_runtime_readiness_enum_values_are_stable() -> None:
    assert RuntimeReadinessStatus.PASSED.value == "passed"
    assert RuntimeReadinessCheckKind.SHUTDOWN.value == "shutdown"


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
        name="runtime_readiness_presence_engine",
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