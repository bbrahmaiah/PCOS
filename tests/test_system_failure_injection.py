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
    MemoryPolicyClassification,
    MemoryQuery,
    MemoryRetrievalResult,
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
    FailureInjectionConfig,
    FailureInjectionOutcome,
    FailureInjectionReview,
    FailureInjectionScenarioKind,
    FailureInjectionStatus,
    JarvisCompositionOverride,
    JarvisSystemFactoryBundle,
    LiveDependencyProfile,
    LiveDependencyWiring,
    LiveDependencyWiringConfig,
    default_failure_scenarios,
)


def _now() -> datetime:
    return datetime.now(UTC)


class FailingCognitionAdapter:
    @property
    def name(self) -> str:
        return "failing_cognition_adapter"

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
                kind=CognitionFailureKind.ADAPTER_ERROR,
                message="injected cognition failure",
                recoverable=True,
            ),
            started_at=started_at,
            finished_at=finished_at,
            duration_ms=1,
            metadata={"injected": True},
        )

    def snapshot(self) -> CognitionAdapterSnapshot:
        return CognitionAdapterSnapshot(
            name=self.name,
            capabilities=self.capabilities,
            metadata={},
        )


class HealthyCognitionAdapter:
    @property
    def name(self) -> str:
        return "healthy_cognition_adapter"

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
            response=CognitionResponse(
                request_id=request.request_id,
                text="healthy failure injection response",
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
            metadata={},
        )


class MemoryWriteFailingGateway:
    def retrieve(
        self,
        query: MemoryQuery,
    ) -> MemoryGatewayRetrievalResult:
        return MemoryGatewayRetrievalResult(
            query=query,
            retrieval=MemoryRetrievalResult(query=query, results=()),
            allowed=True,
            blocked=False,
            reason="failure injection retrieval allowed",
            policy_classification=MemoryPolicyClassification.ALLOWED,
        )

    def remember(
        self,
        request: MemoryWriteRequest,
    ) -> MemoryGatewayWriteResult:
        raise RuntimeError("injected memory write failure")


class HealthyMemoryGateway:
    def retrieve(
        self,
        query: MemoryQuery,
    ) -> MemoryGatewayRetrievalResult:
        return MemoryGatewayRetrievalResult(
            query=query,
            retrieval=MemoryRetrievalResult(query=query, results=()),
            allowed=True,
            blocked=False,
            reason="healthy retrieval allowed",
            policy_classification=MemoryPolicyClassification.ALLOWED,
        )

    def remember(
        self,
        request: MemoryWriteRequest,
    ) -> MemoryGatewayWriteResult:
        return MemoryGatewayWriteResult(
            request=request,
            record=request.to_record(),
            allowed=True,
            blocked=False,
            reason="healthy write allowed",
            policy_classification=MemoryPolicyClassification.ALLOWED,
        )


class FailingPresenceEngine:
    def start(self) -> None:
        return None

    def stop(self) -> None:
        return None

    def publish_response_ready(self, *, text: str) -> None:
        raise RuntimeError("injected presence output failure")

    def snapshot(self) -> dict[str, object]:
        return {"presence": "failing"}


class FailingOrchestrationRuntime:
    def start(self) -> None:
        raise RuntimeError("injected orchestration start failure")

    def stop(self) -> None:
        return None

    def snapshot(self) -> dict[str, object]:
        return {"orchestration": "failing"}


def test_failure_injection_config_rejects_empty_session() -> None:
    with pytest.raises(ValueError):
        FailureInjectionConfig(session_id=" ")


def test_default_failure_scenarios_cover_required_kinds() -> None:
    scenarios = default_failure_scenarios(
        FailureInjectionConfig(session_id="session")
    )

    kinds = {scenario.kind for scenario in scenarios}

    assert FailureInjectionScenarioKind.MEMORY_FACTORY_FAILURE in kinds
    assert FailureInjectionScenarioKind.COGNITION_RUNTIME_FAILURE in kinds
    assert FailureInjectionScenarioKind.MEMORY_WRITE_FAILURE in kinds
    assert FailureInjectionScenarioKind.PRESENCE_OUTPUT_FAILURE in kinds
    assert FailureInjectionScenarioKind.ORCHESTRATION_START_FAILURE in kinds
    assert FailureInjectionScenarioKind.INTERRUPTION_DURING_FAILURE in kinds
    assert FailureInjectionScenarioKind.SHUTDOWN_AFTER_FAILURE in kinds


def test_failure_injection_memory_factory_failure_is_detected() -> None:
    review = FailureInjectionReview(
        config=FailureInjectionConfig(session_id="session"),
        wiring=_failing_memory_wiring(),
    )

    report = review.run()

    memory_factory = [
        result
        for result in report.results
        if result.kind == FailureInjectionScenarioKind.MEMORY_FACTORY_FAILURE
    ][0]

    assert memory_factory.passed is True
    assert memory_factory.outcome == FailureInjectionOutcome.DETECTED




def test_failure_injection_cognition_failure_is_contained() -> None:
    review = FailureInjectionReview(
        config=FailureInjectionConfig(session_id="session"),
        wiring=_failing_cognition_wiring(),
    )

    report = review.run()

    cognition = [
        result
        for result in report.results
        if result.kind
        == FailureInjectionScenarioKind.COGNITION_RUNTIME_FAILURE
    ][0]

    assert cognition.passed is True
    assert cognition.outcome == FailureInjectionOutcome.CONTAINED


def test_failure_injection_memory_write_failure_is_contained() -> None:
    review = FailureInjectionReview(
        config=FailureInjectionConfig(session_id="session"),
        wiring=_memory_write_failure_wiring(),
    )

    report = review.run()

    memory_write = [
        result
        for result in report.results
        if result.kind == FailureInjectionScenarioKind.MEMORY_WRITE_FAILURE
    ][0]

    assert memory_write.passed is True
    assert memory_write.outcome == FailureInjectionOutcome.CONTAINED


def test_failure_injection_presence_failure_is_contained() -> None:
    review = FailureInjectionReview(
        config=FailureInjectionConfig(session_id="session"),
        wiring=_presence_failure_wiring(),
    )

    report = review.run()

    presence = [
        result
        for result in report.results
        if result.kind == FailureInjectionScenarioKind.PRESENCE_OUTPUT_FAILURE
    ][0]

    assert presence.passed is True
    assert presence.outcome == FailureInjectionOutcome.CONTAINED


def test_failure_injection_enum_values_are_stable() -> None:
    assert FailureInjectionStatus.PASSED.value == "passed"
    assert FailureInjectionOutcome.CONTAINED.value == "contained"


def _memory_gateway(obj: object) -> MemoryGateway:
    return cast(MemoryGateway, obj)


def _cognition_adapter(obj: object) -> CognitionAdapter:
    return cast(CognitionAdapter, obj)


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
    return PresenceEngine(name="failure_injection_presence", adapters=adapters)


def _base_wiring(
    *,
    memory_gateway: object,
    cognition_adapter: object,
    presence_engine: object | None = None,
    orchestration_runtime: object | None = None,
    overrides: tuple[JarvisCompositionOverride, ...] = (),
) -> LiveDependencyWiring:
    return LiveDependencyWiring(
        config=LiveDependencyWiringConfig(
            name="failure_injection",
            profile=LiveDependencyProfile.TEST,
            dry_run=False,
            attach_conversation=True,
            attach_presence=presence_engine is not None,
            attach_orchestration=orchestration_runtime is not None,
        ),
        factories=JarvisSystemFactoryBundle(
            memory_gateway=lambda: _memory_gateway(memory_gateway),
            cognition_worker=lambda: CognitionWorker(
                adapter=_cognition_adapter(cognition_adapter)
            ),
            conversation_runtime=RealConversationRuntime,
            presence_engine=(
                (lambda: cast(PresenceEngine, presence_engine))
                if presence_engine is not None
                else None
            ),
            orchestration_runtime=(
                (lambda: orchestration_runtime)
                if orchestration_runtime is not None
                else None
            ),
            kernel=RuntimeKernel,
        ),
        overrides=overrides,
    )


def _failing_memory_wiring() -> LiveDependencyWiring:
    return _base_wiring(
        memory_gateway=HealthyMemoryGateway(),
        cognition_adapter=HealthyCognitionAdapter(),
        presence_engine=FailingPresenceEngine(),
        orchestration_runtime=FailingOrchestrationRuntime(),
        overrides=(
            JarvisCompositionOverride(
                component_id="memory",
                factory_present=False,
            ),
        ),
    )


def _failing_cognition_wiring() -> LiveDependencyWiring:
    return _base_wiring(
        memory_gateway=HealthyMemoryGateway(),
        cognition_adapter=FailingCognitionAdapter(),
        presence_engine=_presence_engine(),
        orchestration_runtime=None,
    )


def _memory_write_failure_wiring() -> LiveDependencyWiring:
    return _base_wiring(
        memory_gateway=MemoryWriteFailingGateway(),
        cognition_adapter=HealthyCognitionAdapter(),
        presence_engine=_presence_engine(),
        orchestration_runtime=None,
    )


def _presence_failure_wiring() -> LiveDependencyWiring:
    return _base_wiring(
        memory_gateway=HealthyMemoryGateway(),
        cognition_adapter=HealthyCognitionAdapter(),
        presence_engine=FailingPresenceEngine(),
        orchestration_runtime=None,
    )

def _orchestration_failure_wiring() -> LiveDependencyWiring:
    return _base_wiring(
        memory_gateway=HealthyMemoryGateway(),
        cognition_adapter=HealthyCognitionAdapter(),
        presence_engine=_presence_engine(),
        orchestration_runtime=FailingOrchestrationRuntime(),
    )

def test_failure_injection_orchestration_start_failure_is_detected() -> None:
    review = FailureInjectionReview(
        config=FailureInjectionConfig(session_id="session"),
        wiring=_orchestration_failure_wiring(),
    )

    report = review.run()

    orchestration = [
        result
        for result in report.results
        if result.kind
        == FailureInjectionScenarioKind.ORCHESTRATION_START_FAILURE
    ][0]

    assert orchestration.passed is True
    assert orchestration.outcome == FailureInjectionOutcome.DETECTED