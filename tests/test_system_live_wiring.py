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
    MemoryPolicyClassification,
    MemoryQuery,
    MemoryRetrievalResult,
    MemoryWriteRequest,
)
from jarvis.runtime.kernel.runtime_kernel import RuntimeKernel
from jarvis.system import (
    JarvisCompositionOverride,
    JarvisSystemFactoryBundle,
    LiveDependencyProfile,
    LiveDependencyWiring,
    LiveDependencyWiringConfig,
    LiveDependencyWiringStatus,
)


def _now() -> datetime:
    return datetime.now(UTC)


class FakeMemoryGateway:
    def __init__(self) -> None:
        self.queries: list[MemoryQuery] = []
        self.writes: list[MemoryWriteRequest] = []

    def retrieve(
        self,
        query: MemoryQuery,
    ) -> MemoryGatewayRetrievalResult:
        self.queries.append(query)
        return MemoryGatewayRetrievalResult(
            query=query,
            retrieval=MemoryRetrievalResult(
                query=query,
                results=(),
            ),
            allowed=True,
            blocked=False,
            reason="live wiring retrieval allowed",
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
            reason="live wiring write allowed",
            policy_classification=MemoryPolicyClassification.ALLOWED,
        )


class FakeCognitionAdapter:
    @property
    def name(self) -> str:
        return "live_wiring_fake_cognition_adapter"

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
                text="live wiring cognition ready",
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


class FakeOrchestrationRuntime:
    def __init__(self) -> None:
        self.started = False
        self.stopped = False

    def start(self) -> None:
        self.started = True
        self.stopped = False

    def stop(self) -> None:
        self.started = False
        self.stopped = True

    def snapshot(self) -> dict[str, object]:
        return {
            "started": self.started,
            "stopped": self.stopped,
        }


def _memory_gateway() -> MemoryGateway:
    return cast(MemoryGateway, FakeMemoryGateway())


def _cognition_worker() -> CognitionWorker:
    return CognitionWorker(
        adapter=cast(CognitionAdapter, FakeCognitionAdapter())
    )


def _factories(
    *,
    include_conversation: bool = True,
    include_orchestration: bool = True,
) -> JarvisSystemFactoryBundle:
    return JarvisSystemFactoryBundle(
        memory_gateway=_memory_gateway,
        cognition_worker=_cognition_worker,
        conversation_runtime=(
            RealConversationRuntime if include_conversation else None
        ),
        presence_engine=None,
        orchestration_runtime=(
            FakeOrchestrationRuntime if include_orchestration else None
        ),
        kernel=RuntimeKernel,
    )


def test_live_dependency_wiring_config_rejects_empty_name() -> None:
    with pytest.raises(ValueError):
        LiveDependencyWiringConfig(name=" ")


def test_live_dependency_wiring_validates_graph() -> None:
    wiring = LiveDependencyWiring(
        config=LiveDependencyWiringConfig(
            name="live_wiring_test",
            profile=LiveDependencyProfile.TEST,
            dry_run=True,
            attach_conversation=True,
            attach_presence=False,
            attach_orchestration=True,
        ),
        factories=_factories(),
    )

    report = wiring.validate()

    assert report.status == LiveDependencyWiringStatus.VALIDATED
    assert report.succeeded is True
    assert report.dependency_graph is not None
    assert report.dependency_graph.boot_allowed is True
    assert report.composition_snapshot is not None


def test_live_dependency_wiring_builds_bootstrap_after_validation() -> None:
    wiring = LiveDependencyWiring(
        config=LiveDependencyWiringConfig(
            name="live_wiring_test",
            profile=LiveDependencyProfile.TEST,
            dry_run=True,
            attach_conversation=True,
            attach_presence=False,
            attach_orchestration=False,
        ),
        factories=_factories(include_orchestration=False),
    )

    report = wiring.validate_bootstrap_ready()

    assert report.status == LiveDependencyWiringStatus.BOOTSTRAP_READY
    assert report.succeeded is True
    assert wiring.build_bootstrap() is not None


def test_live_dependency_wiring_run_dry_boot() -> None:
    wiring = LiveDependencyWiring(
        config=LiveDependencyWiringConfig(
            name="live_wiring_test",
            profile=LiveDependencyProfile.TEST,
            dry_run=True,
            attach_conversation=True,
            attach_presence=False,
            attach_orchestration=True,
        ),
        factories=_factories(),
    )

    report = wiring.run_dry_boot()

    assert report.status == LiveDependencyWiringStatus.DRY_RUN_PASSED
    assert report.succeeded is True
    assert report.bootstrap_result is not None
    assert report.bootstrap_result.succeeded is True


def test_live_dependency_wiring_blocks_required_missing_memory() -> None:
    def missing_memory() -> MemoryGateway:
        raise RuntimeError("missing memory should not be called here")

    wiring = LiveDependencyWiring(
        config=LiveDependencyWiringConfig(
            name="live_wiring_test",
            profile=LiveDependencyProfile.TEST,
            dry_run=True,
            attach_conversation=False,
            attach_presence=False,
            attach_orchestration=False,
        ),
        factories=JarvisSystemFactoryBundle(
            memory_gateway=missing_memory,
            cognition_worker=_cognition_worker,
            conversation_runtime=None,
            presence_engine=None,
            orchestration_runtime=None,
            kernel=RuntimeKernel,
        ),
        overrides=(
            JarvisCompositionOverride(
                component_id="memory",
                factory_present=False,
            ),
        ),
    )

    report = wiring.validate()

    assert report.status == LiveDependencyWiringStatus.BLOCKED
    assert report.succeeded is False
    assert report.dependency_graph is not None
    assert report.dependency_graph.boot_allowed is False


def test_live_dependency_wiring_degrades_missing_presence() -> None:
    wiring = LiveDependencyWiring(
        config=LiveDependencyWiringConfig(
            name="live_wiring_test",
            profile=LiveDependencyProfile.TEST,
            dry_run=True,
            attach_conversation=True,
            attach_presence=True,
            attach_orchestration=False,
            allow_degraded_presence=True,
        ),
        factories=_factories(include_orchestration=False),
    )

    report = wiring.validate()

    assert report.succeeded is True
    assert report.dependency_graph is not None
    assert report.dependency_graph.boot_allowed is True
    assert report.dependency_graph.degraded_count >= 0


def test_live_dependency_wiring_reports_failed_dry_boot() -> None:
    def failing_memory() -> MemoryGateway:
        raise RuntimeError("memory factory failed")

    wiring = LiveDependencyWiring(
        config=LiveDependencyWiringConfig(
            name="live_wiring_test",
            profile=LiveDependencyProfile.TEST,
            dry_run=True,
            attach_conversation=False,
            attach_presence=False,
            attach_orchestration=False,
        ),
        factories=JarvisSystemFactoryBundle(
            memory_gateway=failing_memory,
            cognition_worker=_cognition_worker,
            conversation_runtime=None,
            presence_engine=None,
            orchestration_runtime=None,
            kernel=RuntimeKernel,
        ),
    )

    report = wiring.run_dry_boot()

    assert report.status == LiveDependencyWiringStatus.FAILED
    assert report.succeeded is False
    assert report.error is not None
    assert "memory factory failed" in report.error


def test_live_dependency_wiring_enum_values_are_stable() -> None:
    assert LiveDependencyProfile.LOCAL.value == "local"
    assert LiveDependencyWiringStatus.DRY_RUN_PASSED.value == "dry_run_passed"