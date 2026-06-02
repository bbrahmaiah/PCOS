from __future__ import annotations

from threading import Event
from typing import Any, cast

from jarvis.cognition.models import CognitionRequest
from jarvis.cognition.worker import (
    CognitionWorker,
    CognitionWorkerResult,
    CognitionWorkerSnapshot,
)
from jarvis.conversation.runtime import (
    RealConversationInput,
    RealConversationRuntime,
    RealConversationRuntimeOutput,
    RealConversationRuntimeSnapshot,
)
from jarvis.memory.gateway import (
    MemoryGateway,
    MemoryGatewayRetrievalResult,
    MemoryGatewayWriteResult,
)
from jarvis.memory.models import MemoryQuery, MemoryWriteRequest
from jarvis.presence import PresenceEngine
from jarvis.runtime.events import EventBus
from jarvis.runtime.workers.worker import BaseWorker


class MemoryRuntimeWorker(BaseWorker):
    """
    Runtime citizen wrapper for MemoryGateway.

    Step 44A keeps memory read-only. It exposes governed retrieval through the
    MemoryGateway and does not write memories.
    """

    def __init__(
        self,
        *,
        memory_gateway: MemoryGateway,
        event_bus: EventBus,
        name: str = "memory_runtime",
        tick_interval_seconds: float = 0.05,
    ) -> None:
        super().__init__(
            name=name,
            event_bus=event_bus,
            tick_interval_seconds=tick_interval_seconds,
        )
        self._memory_gateway = memory_gateway
        self._retrieve_count = 0
        self._write_count = 0
       

    @property
    def memory_gateway(self) -> MemoryGateway:
        return self._memory_gateway

    def run_once(self) -> None:
        return None

    def retrieve(
        self,
        query: MemoryQuery,
    ) -> MemoryGatewayRetrievalResult:
        self._retrieve_count += 1
        return self._memory_gateway.retrieve(query)

    def remember(
        self,
        request: MemoryWriteRequest,
    ) -> MemoryGatewayWriteResult:
        self._write_count += 1
        return self._memory_gateway.remember(request)

    def memory_snapshot(self) -> Any | None:
        snapshot = getattr(self._memory_gateway, "snapshot", None)
        if callable(snapshot):
            return snapshot()

        return {
            "retrieve_count": self._retrieve_count,
            "write_count": self._write_count,
            "gateway": type(self._memory_gateway).__name__,
        }


class CognitionRuntimeWorker(BaseWorker):
    """
    Runtime citizen wrapper for CognitionWorker.

    This preserves the existing CognitionWorker contract:
    CognitionWorker.process_request(CognitionRequest).
    """

    def __init__(
        self,
        *,
        cognition_worker: CognitionWorker,
        event_bus: EventBus,
        name: str = "cognition_runtime",
        tick_interval_seconds: float = 0.05,
    ) -> None:
        super().__init__(
            name=name,
            event_bus=event_bus,
            tick_interval_seconds=tick_interval_seconds,
        )
        self._cognition_worker = cognition_worker
        self._ready = Event()

    @property
    def cognition_worker(self) -> CognitionWorker:
        return self._cognition_worker

    def on_start(self) -> None:
        self._cognition_worker.on_start()
        self._ready.set()

    def on_stop(self) -> None:
        try:
            self._cognition_worker.on_stop()
        finally:
            self._ready.clear()

    def run_once(self) -> None:
        return None

    def process_request(
        self,
        request: CognitionRequest,
    ) -> CognitionWorkerResult:
        if not self.wait_until_ready(timeout_seconds=2.0):
            raise RuntimeError("cognition runtime worker is not ready.")

        return self._cognition_worker.process_request(request)

    def request_cancel(
        self,
        *,
        request_id: str | None = None,
        reason: str | None = None,
    ) -> bool:
        return self._cognition_worker.request_cancel(
            request_id=request_id,
            reason=reason,
        )

    def wait_until_ready(
        self,
        *,
        timeout_seconds: float = 2.0,
    ) -> bool:
        return self._ready.wait(timeout=timeout_seconds)

    def cognition_snapshot(self) -> CognitionWorkerSnapshot:
        return self._cognition_worker.snapshot()

class ConversationRuntimeWorker(BaseWorker):
    """
    Runtime citizen wrapper for RealConversationRuntime.

    Step 44C attaches the conversation organ to RuntimeKernel without adding
    voice, presence, orchestration, or bootstrap changes yet.

    Conversation remains responsible for:
    - turn state
    - endpointing output
    - interruption actions
    - session continuity
    - attention update signals

    It does not directly call cognition, memory, tools, or TTS.
    """

    def __init__(
        self,
        *,
        conversation_runtime: RealConversationRuntime,
        event_bus: EventBus,
        name: str = "conversation_runtime",
        tick_interval_seconds: float = 0.05,
    ) -> None:
        super().__init__(
            name=name,
            event_bus=event_bus,
            tick_interval_seconds=tick_interval_seconds,
        )
        self._conversation_runtime = conversation_runtime
        self._ready = Event()

    @property
    def conversation_runtime(self) -> RealConversationRuntime:
        return self._conversation_runtime

    def on_start(self) -> None:
        self._ready.set()

    def on_stop(self) -> None:
        self._ready.clear()

    def run_once(self) -> None:
        return None

    def wait_until_ready(
        self,
        *,
        timeout_seconds: float = 2.0,
    ) -> bool:
        return self._ready.wait(timeout=timeout_seconds)

    def accept_input(
        self,
        signal: RealConversationInput,
    ) -> RealConversationRuntimeOutput:
        if not self.wait_until_ready(timeout_seconds=2.0):
            raise RuntimeError("conversation runtime worker is not ready.")

        return self._conversation_runtime.accept_input(signal)

    def add_assistant_response(
        self,
        text: str,
        *,
        turn_id: str | None = None,
        expects_follow_up: bool = False,
    ) -> RealConversationRuntimeOutput:
        if not self.wait_until_ready(timeout_seconds=2.0):
            raise RuntimeError("conversation runtime worker is not ready.")

        return self._conversation_runtime.add_assistant_response(
            text,
            turn_id=turn_id,
            expects_follow_up=expects_follow_up,
        )

    def conversation_snapshot(self) -> RealConversationRuntimeSnapshot:
        return self._conversation_runtime.snapshot()


class PresenceRuntimeWorker(BaseWorker):
    """
    Runtime citizen wrapper for PresenceEngine.

    Step 44D attaches the voice/presence organ to RuntimeKernel without
    letting Presence call cognition, memory, tools, or orchestration directly.

    Presence remains responsible for:
    - microphone/wake/VAD/STT pipeline
    - TTS/playback pipeline
    - interruption detection
    - presence state

    EventBus full routing comes later in Step 44F.
    """

    def __init__(
        self,
        *,
        presence_engine: PresenceEngine,
        event_bus: EventBus,
        name: str = "presence_runtime",
        tick_interval_seconds: float = 0.05,
    ) -> None:
        super().__init__(
            name=name,
            event_bus=event_bus,
            tick_interval_seconds=tick_interval_seconds,
        )
        self._presence_engine = presence_engine
        self._ready = Event()

    @property
    def presence_engine(self) -> PresenceEngine:
        return self._presence_engine

    def on_start(self) -> None:
        self._presence_engine.start()
        self._ready.set()

    def on_stop(self) -> None:
        try:
            self._presence_engine.stop()
        finally:
            self._ready.clear()

    def run_once(self) -> None:
        return None

    def wait_until_ready(
        self,
        *,
        timeout_seconds: float = 2.0,
    ) -> bool:
        return self._ready.wait(timeout=timeout_seconds)

    def publish_response_ready(self, *, text: str) -> None:
        if not self.wait_until_ready(timeout_seconds=2.0):
            raise RuntimeError("presence runtime worker is not ready.")

        self._presence_engine.publish_response_ready(text=text)

    def presence_snapshot(self) -> object:
        return self._presence_engine.snapshot()

class OrchestrationRuntimeWorker(BaseWorker):
    """
    Runtime citizen wrapper for the orchestration subsystem.

    Step 44E attaches orchestration as the conductor organ without giving it
    direct hidden control over cognition, memory, tools, presence, or desktop
    actions.

    Full event-driven coordination comes in Step 44F.
    """

    def __init__(
        self,
        *,
        orchestration_runtime: object,
        event_bus: EventBus,
        name: str = "orchestration_runtime",
        tick_interval_seconds: float = 0.05,
    ) -> None:
        super().__init__(
            name=name,
            event_bus=event_bus,
            tick_interval_seconds=tick_interval_seconds,
        )
        self._orchestration_runtime = orchestration_runtime
        self._ready = Event()

    @property
    def orchestration_runtime(self) -> object:
        return self._orchestration_runtime

    def on_start(self) -> None:
        _call_optional_lifecycle(self._orchestration_runtime, "start")
        self._ready.set()

    def on_stop(self) -> None:
        try:
            _call_optional_lifecycle(self._orchestration_runtime, "stop")
        finally:
            self._ready.clear()

    def run_once(self) -> None:
        heartbeat = getattr(self._orchestration_runtime, "heartbeat", None)
        if callable(heartbeat):
            heartbeat()

    def wait_until_ready(
        self,
        *,
        timeout_seconds: float = 2.0,
    ) -> bool:
        return self._ready.wait(timeout=timeout_seconds)

    def orchestration_snapshot(self) -> object | None:
        snapshot = getattr(self._orchestration_runtime, "snapshot", None)
        if callable(snapshot):
            return cast(object, snapshot())

        return {
            "runtime": type(self._orchestration_runtime).__name__,
            "ready": self._ready.is_set(),
        }


def _call_optional_lifecycle(subsystem: object, method_name: str) -> None:
    method = getattr(subsystem, method_name, None)
    if callable(method):
        method()