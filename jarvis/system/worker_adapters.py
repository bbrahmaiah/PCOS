from __future__ import annotations

from threading import Event
from typing import Any

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