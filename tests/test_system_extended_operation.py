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
    CognitionFailure,
    CognitionFailureKind,
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
    ExtendedOperationConfig,
    ExtendedOperationEventKind,
    ExtendedOperationRunMode,
    ExtendedOperationStatus,
    ExtendedOperationValidator,
    JarvisSystemFactoryBundle,
    LiveDependencyProfile,
    LiveDependencyWiring,
    LiveDependencyWiringConfig,
    profile_config,
)


def _now() -> datetime:
    return datetime.now(UTC)


class HealthyMemoryGateway:
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
                    reason="extended operation retrieval",
                    confidence=record.confidence,
                    policy_classification=MemoryPolicyClassification.ALLOWED,
                ),
            )
            for record in self.records
        )
        return MemoryGatewayRetrievalResult(
            query=query,
            retrieval=MemoryRetrievalResult(query=query, results=results),
            allowed=True,
            blocked=False,
            reason="extended operation retrieval allowed",
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
            reason="extended operation write allowed",
            policy_classification=MemoryPolicyClassification.ALLOWED,
        )


class HealthyCognitionAdapter:
    def __init__(self) -> None:
        self.requests: list[CognitionRequest] = []

    @property
    def name(self) -> str:
        return "extended_operation_healthy_adapter"

    @property
    def capabilities(self) -> tuple[CognitionAdapterCapability, ...]:
        return ()

    def generate(
        self,
        request: CognitionRequest,
    ) -> CognitionAdapterResult:
        self.requests.append(request)
        started_at = _now()
        finished_at = _now()
        memory_text = (
            request.context.items[0].text
            if request.context.items
            else "no memory"
        )

        return CognitionAdapterResult(
            request_id=request.request_id,
            response=CognitionResponse(
                request_id=request.request_id,
                text=f"Extended operation response using {memory_text}",
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


class FailingCognitionAdapter:
    @property
    def name(self) -> str:
        return "extended_operation_failing_adapter"

    @property
    def capabilities(self) -> tuple[CognitionAdapterCapability, ...]:
        return ()

    def generate(
        self,
        request: CognitionRequest,
    ) -> CognitionAdapterResult:
        started_at = _now()
        finished_at = _now()

        return CognitionAdapterResult(
            request_id=request.request_id,
            response=None,
            failure=CognitionFailure(
                request_id=request.request_id,
                kind=_failure_kind(),
                message="extended operation injected failure",
                recoverable=True,
            ),
            started_at=started_at,
            finished_at=finished_at,
            duration_ms=1,
            metadata={},
        )

    def snapshot(self) -> CognitionAdapterSnapshot:
        return CognitionAdapterSnapshot(
            name=self.name,
            capabilities=self.capabilities,
            metadata={},
        )


def test_extended_operation_config_rejects_bad_values() -> None:
    with pytest.raises(ValueError):
        ExtendedOperationConfig(cycle_count=0)

    with pytest.raises(ValueError):
        ExtendedOperationConfig(cycle_delay_seconds=-1.0)


def test_profile_config_builds_long_run_profiles() -> None:
    smoke = profile_config(ExtendedOperationRunMode.SMOKE)
    two_hours = profile_config(ExtendedOperationRunMode.TWO_HOURS)

    assert smoke.cycle_count == 5
    assert two_hours.cycle_count > smoke.cycle_count
    assert two_hours.cycle_delay_seconds == 10.0


def test_extended_operation_validator_passes_smoke_runtime() -> None:
    memory = HealthyMemoryGateway(
        records=(
            _memory_record(
                text="Bala is building a reliable cognition runtime."
            ),
        )
    )
    adapter = HealthyCognitionAdapter()

    validator = ExtendedOperationValidator(
        config=ExtendedOperationConfig(
            run_mode=ExtendedOperationRunMode.SMOKE,
            session_id="session",
            cycle_count=3,
            max_average_latency_ms=2_000.0,
        ),
        wiring=_wiring(
            memory_gateway=memory,
            cognition_adapter=adapter,
            include_presence=True,
        ),
    )

    report = validator.run()

    assert report.status == ExtendedOperationStatus.PASSED
    assert report.passed is True
    assert report.metrics is not None
    assert report.metrics.completed_cycles == 3
    assert report.metrics.failed_cycles == 0
    assert report.metrics.interruption_count == 1
    assert report.metrics.memory_write_count == 1
    assert report.metrics.worker_count >= 4
    assert len(memory.queries) >= 4
    assert len(memory.writes) == 1
    assert len(adapter.requests) >= 4

    event_kinds = {event.kind for event in report.events}

    assert ExtendedOperationEventKind.BOOT in event_kinds
    assert ExtendedOperationEventKind.CYCLE_COMPLETED in event_kinds
    assert ExtendedOperationEventKind.INTERRUPTION_COMPLETED in event_kinds
    assert ExtendedOperationEventKind.MEMORY_WRITE_COMPLETED in event_kinds
    assert ExtendedOperationEventKind.SHUTDOWN in event_kinds


def test_extended_operation_validator_fails_when_cycles_fail() -> None:
    validator = ExtendedOperationValidator(
        config=ExtendedOperationConfig(
            session_id="session",
            cycle_count=2,
            max_failure_count=0,
        ),
        wiring=_wiring(
            memory_gateway=HealthyMemoryGateway(),
            cognition_adapter=FailingCognitionAdapter(),
            include_presence=False,
        ),
    )

    report = validator.run()

    assert report.status == ExtendedOperationStatus.FAILED
    assert report.metrics is not None
    assert report.metrics.failed_cycles >= 1
    assert report.error is not None


def test_extended_operation_enum_values_are_stable() -> None:
    assert ExtendedOperationStatus.PASSED.value == "passed"
    assert ExtendedOperationRunMode.SMOKE.value == "smoke"
    assert ExtendedOperationEventKind.SHUTDOWN.value == "shutdown"


def _wiring(
    *,
    memory_gateway: object,
    cognition_adapter: object,
    include_presence: bool,
) -> LiveDependencyWiring:
    return LiveDependencyWiring(
        config=LiveDependencyWiringConfig(
            name="extended_operation_test",
            profile=LiveDependencyProfile.TEST,
            dry_run=False,
            attach_conversation=True,
            attach_presence=include_presence,
            attach_orchestration=False,
        ),
        factories=JarvisSystemFactoryBundle(
            memory_gateway=lambda: cast(MemoryGateway, memory_gateway),
            cognition_worker=lambda: CognitionWorker(
                adapter=cast(CognitionAdapter, cognition_adapter)
            ),
            conversation_runtime=RealConversationRuntime,
            presence_engine=_presence_engine if include_presence else None,
            orchestration_runtime=None,
            kernel=RuntimeKernel,
        ),
    )


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
    return PresenceEngine(name="extended_operation_presence", adapters=adapters)


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


def _failure_kind() -> CognitionFailureKind:
    for candidate in CognitionFailureKind:
        return candidate

    raise AssertionError("CognitionFailureKind has no values.")