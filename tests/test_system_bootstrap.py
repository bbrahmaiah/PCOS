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
    JarvisBootstrapConfig,
    JarvisBootstrapStatus,
    JarvisSystemBootstrap,
    JarvisSystemFactoryBundle,
    JarvisSystemStatus,
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
            reason="bootstrap test retrieval allowed",
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
            reason="bootstrap test write allowed",
            policy_classification=MemoryPolicyClassification.ALLOWED,
        )


class FakeCognitionAdapter:
    @property
    def name(self) -> str:
        return "bootstrap_fake_cognition_adapter"

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
                text="Bootstrap cognition ready.",
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


def _memory_gateway(gateway: FakeMemoryGateway) -> MemoryGateway:
    return cast(MemoryGateway, gateway)


def _cognition_adapter(adapter: FakeCognitionAdapter) -> CognitionAdapter:
    return cast(CognitionAdapter, adapter)


def test_bootstrap_config_rejects_empty_name() -> None:
    with pytest.raises(ValueError):
        JarvisBootstrapConfig(name=" ")


def test_system_bootstrap_dry_run_starts_and_stops_jarvis_system() -> None:
    memory = FakeMemoryGateway()
    orchestration = FakeOrchestrationRuntime()

    bootstrap = JarvisSystemBootstrap(
        config=JarvisBootstrapConfig(
            name="test_jarvis_system",
            dry_run=True,
            attach_conversation=True,
            attach_presence=False,
            attach_orchestration=True,
        ),
        factories=JarvisSystemFactoryBundle(
            memory_gateway=lambda: _memory_gateway(memory),
            cognition_worker=lambda: CognitionWorker(
                adapter=_cognition_adapter(FakeCognitionAdapter())
            ),
            conversation_runtime=RealConversationRuntime,
            presence_engine=None,
            orchestration_runtime=lambda: orchestration,
            kernel=RuntimeKernel,
        ),
    )

    result = bootstrap.start()

    assert result.status == JarvisBootstrapStatus.STOPPED
    assert result.succeeded is True
    assert result.system_snapshot is not None
    assert result.system_snapshot.status == JarvisSystemStatus.STOPPED
    assert result.system_snapshot.conversation_worker is not None
    assert result.system_snapshot.orchestration_worker is not None
    assert orchestration.stopped is True


def test_system_bootstrap_non_dry_run_leaves_system_running_until_stop() -> None:
    bootstrap = JarvisSystemBootstrap(
        config=JarvisBootstrapConfig(
            name="test_jarvis_system",
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
    )

    result = bootstrap.start()

    assert result.status == JarvisBootstrapStatus.STARTED
    assert result.succeeded is True
    assert bootstrap.system is not None
    assert bootstrap.system.status == JarvisSystemStatus.RUNNING

    stopped = bootstrap.stop()

    assert stopped.status == JarvisBootstrapStatus.STOPPED
    assert stopped.succeeded is True
    assert stopped.system_snapshot is not None
    assert stopped.system_snapshot.status == JarvisSystemStatus.STOPPED


def test_system_bootstrap_reports_factory_failure() -> None:
    def fail_memory() -> MemoryGateway:
        raise RuntimeError("memory factory failed")

    bootstrap = JarvisSystemBootstrap(
        config=JarvisBootstrapConfig(
            name="test_jarvis_system",
            dry_run=True,
        ),
        factories=JarvisSystemFactoryBundle(
            memory_gateway=fail_memory,
            cognition_worker=lambda: CognitionWorker(
                adapter=_cognition_adapter(FakeCognitionAdapter())
            ),
            kernel=RuntimeKernel,
        ),
    )

    result = bootstrap.start()

    assert result.status == JarvisBootstrapStatus.FAILED
    assert result.succeeded is False
    assert result.error is not None
    assert "memory factory failed" in result.error


def test_bootstrap_stop_before_start_is_safe() -> None:
    bootstrap = JarvisSystemBootstrap(
        config=JarvisBootstrapConfig(
            name="test_jarvis_system",
            dry_run=False,
        ),
        factories=JarvisSystemFactoryBundle(
            memory_gateway=lambda: _memory_gateway(FakeMemoryGateway()),
            cognition_worker=lambda: CognitionWorker(
                adapter=_cognition_adapter(FakeCognitionAdapter())
            ),
            kernel=RuntimeKernel,
        ),
    )

    result = bootstrap.stop()

    assert result.status == JarvisBootstrapStatus.STOPPED
    assert result.succeeded is True
    assert result.metadata["already_stopped"] is True


def test_bootstrap_enum_values_are_stable() -> None:
    assert JarvisBootstrapStatus.STARTED.value == "started"
    assert JarvisBootstrapStatus.STOPPED.value == "stopped"
    assert JarvisBootstrapStatus.FAILED.value == "failed"