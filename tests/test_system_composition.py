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
from jarvis.memory.gateway import (
    MemoryGateway,
    MemoryGatewayRetrievalResult,
)
from jarvis.memory.models import (
    MemoryPolicyClassification,
    MemoryQuery,
    MemoryRetrievalResult,
)
from jarvis.runtime.kernel.runtime_kernel import RuntimeKernel
from jarvis.system import (
    JarvisBootstrapConfig,
    JarvisComponentKind,
    JarvisComponentMode,
    JarvisComponentRequirement,
    JarvisComponentSpec,
    JarvisComponentValidationStatus,
    JarvisCompositionOverride,
    JarvisCompositionRoot,
    JarvisDependencyGraphStatus,
    JarvisSystemFactoryBundle,
    validate_dependency_graph,
)


def _now() -> datetime:
    return datetime.now(UTC)


class FakeMemoryGateway:
    def retrieve(
        self,
        query: MemoryQuery,
    ) -> MemoryGatewayRetrievalResult:
        return MemoryGatewayRetrievalResult(
            query=query,
            retrieval=MemoryRetrievalResult(
                query=query,
                results=(),
            ),
            allowed=True,
            blocked=False,
            reason="composition test retrieval allowed",
            policy_classification=MemoryPolicyClassification.ALLOWED,
        )


class FakeCognitionAdapter:
    @property
    def name(self) -> str:
        return "composition_fake_cognition_adapter"

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
                text="composition cognition ready",
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


def _memory_gateway() -> MemoryGateway:
    return cast(MemoryGateway, FakeMemoryGateway())


def _cognition_worker() -> CognitionWorker:
    return CognitionWorker(
        adapter=cast(CognitionAdapter, FakeCognitionAdapter())
    )


def _factories() -> JarvisSystemFactoryBundle:
    return JarvisSystemFactoryBundle(
        memory_gateway=_memory_gateway,
        cognition_worker=_cognition_worker,
        conversation_runtime=None,
        presence_engine=None,
        orchestration_runtime=None,
        kernel=RuntimeKernel,
    )


def test_component_spec_rejects_empty_id() -> None:
    with pytest.raises(ValueError):
        JarvisComponentSpec(
            component_id=" ",
            kind=JarvisComponentKind.MEMORY,
            requirement=JarvisComponentRequirement.REQUIRED,
        )


def test_component_spec_rejects_self_dependency() -> None:
    with pytest.raises(ValueError):
        JarvisComponentSpec(
            component_id="memory",
            kind=JarvisComponentKind.MEMORY,
            requirement=JarvisComponentRequirement.REQUIRED,
            dependencies=("memory",),
        )


def test_dependency_graph_ready_for_required_components() -> None:
    report = validate_dependency_graph(
        (
            JarvisComponentSpec(
                component_id="config",
                kind=JarvisComponentKind.CONFIG,
                requirement=JarvisComponentRequirement.REQUIRED,
            ),
            JarvisComponentSpec(
                component_id="memory",
                kind=JarvisComponentKind.MEMORY,
                requirement=JarvisComponentRequirement.REQUIRED,
                dependencies=("config",),
            ),
        )
    )

    assert report.status == JarvisDependencyGraphStatus.READY
    assert report.boot_allowed is True
    assert report.ready_count == 2
    assert report.blocked_count == 0


def test_dependency_graph_blocks_required_missing_dependency() -> None:
    report = validate_dependency_graph(
        (
            JarvisComponentSpec(
                component_id="cognition",
                kind=JarvisComponentKind.COGNITION,
                requirement=JarvisComponentRequirement.REQUIRED,
                dependencies=("memory",),
            ),
        )
    )

    assert report.status == JarvisDependencyGraphStatus.BLOCKED
    assert report.boot_allowed is False
    assert report.blocked_count == 1
    assert report.components[0].missing_dependencies == ("memory",)


def test_dependency_graph_degrades_optional_component() -> None:
    report = validate_dependency_graph(
        (
            JarvisComponentSpec(
                component_id="presence",
                kind=JarvisComponentKind.PRESENCE,
                requirement=JarvisComponentRequirement.OPTIONAL,
                dependencies=("conversation",),
                can_degrade=True,
            ),
        )
    )

    assert report.status == JarvisDependencyGraphStatus.DEGRADED
    assert report.boot_allowed is True
    assert report.degraded_count == 1
    assert report.components[0].status == (
        JarvisComponentValidationStatus.DEGRADED
    )


def test_composition_root_validates_default_graph() -> None:
    root = JarvisCompositionRoot(
        config=JarvisBootstrapConfig(
            name="composition_test",
            dry_run=True,
            attach_conversation=False,
            attach_presence=False,
            attach_orchestration=False,
        ),
        factories=_factories(),
    )

    report = root.validate_dependency_graph()

    assert report.boot_allowed is True
    assert report.status == JarvisDependencyGraphStatus.READY
    assert report.ready_count >= 5
    assert report.disabled_count == 3


def test_composition_root_blocks_boot_when_required_memory_missing() -> None:
    root = JarvisCompositionRoot(
        config=JarvisBootstrapConfig(
            name="composition_test",
            dry_run=True,
        ),
        factories=_factories(),
        overrides=(
            JarvisCompositionOverride(
                component_id="memory",
                factory_present=False,
            ),
        ),
    )

    report = root.validate_dependency_graph()

    assert report.status == JarvisDependencyGraphStatus.BLOCKED
    assert report.boot_allowed is False

    with pytest.raises(RuntimeError):
        root.build_bootstrap()


def test_composition_root_allows_degraded_presence() -> None:
    root = JarvisCompositionRoot(
        config=JarvisBootstrapConfig(
            name="composition_test",
            dry_run=True,
            attach_conversation=True,
            attach_presence=True,
            attach_orchestration=False,
        ),
        factories=_factories(),
        overrides=(
            JarvisCompositionOverride(
                component_id="presence",
                mode=JarvisComponentMode.ENABLED,
                factory_present=False,
                can_degrade=True,
            ),
        ),
    )

    report = root.validate_dependency_graph()

    assert report.status == JarvisDependencyGraphStatus.DEGRADED
    assert report.boot_allowed is True

    presence = [
        component
        for component in report.components
        if component.component_id == "presence"
    ][0]

    assert presence.status == JarvisComponentValidationStatus.DEGRADED


def test_composition_root_builds_bootstrap_when_graph_allows_boot() -> None:
    root = JarvisCompositionRoot(
        config=JarvisBootstrapConfig(
            name="composition_test",
            dry_run=True,
            attach_conversation=False,
            attach_presence=False,
            attach_orchestration=False,
        ),
        factories=_factories(),
    )

    bootstrap = root.build_bootstrap()
    snapshot = root.snapshot()

    assert bootstrap is not None
    assert snapshot.boot_allowed is True
    assert snapshot.dependency_graph.boot_allowed is True


def test_unknown_override_fails_fast() -> None:
    root = JarvisCompositionRoot(
        config=JarvisBootstrapConfig(name="composition_test"),
        factories=_factories(),
        overrides=(
            JarvisCompositionOverride(
                component_id="unknown",
                factory_present=False,
            ),
        ),
    )

    with pytest.raises(ValueError):
        root.validate_dependency_graph()


def test_composition_enum_values_are_stable() -> None:
    assert JarvisComponentKind.MEMORY.value == "memory"
    assert JarvisComponentMode.DEGRADED.value == "degraded"
    assert JarvisDependencyGraphStatus.BLOCKED.value == "blocked"