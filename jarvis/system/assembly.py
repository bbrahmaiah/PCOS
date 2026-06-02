from __future__ import annotations

from datetime import datetime
from typing import Any

from jarvis.cognition.models import (
    CognitionContext,
    CognitionContextItem,
    CognitionRequest,
    CognitionRuntimePolicy,
    SpokenResponseStyle,
)
from jarvis.cognition.worker import CognitionWorker
from jarvis.memory.gateway import MemoryGateway, MemoryGatewayRetrievalResult
from jarvis.memory.models import (
    MemoryImportance,
    MemoryKind,
    MemoryQuery,
    MemoryRetention,
    MemoryScope,
    MemorySensitivity,
    MemorySource,
    MemoryWriteRequest,
)
from jarvis.runtime.kernel.runtime_kernel import RuntimeKernel
from jarvis.system.contracts import (
    JarvisAskStatus,
    JarvisMemoryWriteDecision,
    JarvisMemoryWriteStatus,
    JarvisSubsystemHealth,
    JarvisSubsystemKind,
    JarvisSystemRequest,
    JarvisSystemResponse,
    JarvisSystemSnapshot,
    JarvisSystemStatus,
    new_system_id,
    utc_now,
)
from jarvis.system.worker_adapters import (
    CognitionRuntimeWorker,
    MemoryRuntimeWorker,
)


class JarvisSystem:
    """
    Step 44A system assembly.

    This is the first runtime spine:
    RuntimeKernel + MemoryRuntimeWorker + CognitionRuntimeWorker.

    It intentionally does not include:
    - bootstrap.py replacement
    - voice
    - presence
    - conversation
    - orchestration
    - memory writes
    """

    def __init__(
        self,
        *,
        memory_gateway: MemoryGateway,
        cognition_worker: CognitionWorker,
        kernel: RuntimeKernel | None = None,
        name: str = "jarvis_system",
    ) -> None:
        clean_name = name.strip()
        if not clean_name:
            raise ValueError("name cannot be empty.")

        self._name = clean_name
        self._kernel = kernel or RuntimeKernel()
        self._memory_worker = MemoryRuntimeWorker(
            memory_gateway=memory_gateway,
            event_bus=self._kernel.event_bus,
        )
        self._cognition_worker = CognitionRuntimeWorker(
            cognition_worker=cognition_worker,
            event_bus=self._kernel.event_bus,
        )

        self._status = JarvisSystemStatus.CREATED
        self._started_at: datetime | None = None
        self._stopped_at: datetime | None = None
        self._ask_count = 0
        self._failure_count = 0
        self._last_error: str | None = None
        self._registered = False

    @property
    def name(self) -> str:
        return self._name

    @property
    def status(self) -> JarvisSystemStatus:
        return self._status

    @property
    def kernel(self) -> RuntimeKernel:
        return self._kernel

    @property
    def memory_worker(self) -> MemoryRuntimeWorker:
        return self._memory_worker

    @property
    def cognition_worker(self) -> CognitionRuntimeWorker:
        return self._cognition_worker

    def start(self) -> None:
        if self._status == JarvisSystemStatus.RUNNING:
            return

        self._status = JarvisSystemStatus.STARTING

        try:
            self._register_workers_once()
            self._kernel.start()
            self._wait_until_ready()
        except Exception as exc:
            self._failure_count += 1
            self._last_error = f"{type(exc).__name__}: {exc}"
            self._status = JarvisSystemStatus.FAILED
            raise

        self._status = JarvisSystemStatus.RUNNING
        self._started_at = utc_now()
        self._stopped_at = None
        self._last_error = None

    def stop(self) -> None:
        if self._status in {
            JarvisSystemStatus.CREATED,
            JarvisSystemStatus.STOPPED,
        }:
            self._status = JarvisSystemStatus.STOPPED
            self._stopped_at = utc_now()
            return

        self._status = JarvisSystemStatus.STOPPING

        try:
            self._kernel.stop()
        finally:
            self._status = JarvisSystemStatus.STOPPED
            self._stopped_at = utc_now()


    def ask(
        self,
        text: str,
        *,
        session_id: str = "default",
        max_memory_results: int = 5,
        metadata: dict[str, object] | None = None,
    ) -> JarvisSystemResponse:
        request = JarvisSystemRequest(
            request_id=new_system_id("jarvis_request"),
            text=text,
            session_id=session_id,
            max_memory_results=max_memory_results,
            metadata=metadata or {},
        )

        self._ask_count += 1

        try:
            memory_result = self._retrieve_memory(request)
            context = _memory_result_to_cognition_context(
                memory_result=memory_result,
                session_id=request.session_id,
            )
            cognition_request = _build_cognition_request(
                system_request=request,
                context=context,
            )
            cognition_result = self._cognition_worker.process_request(
                cognition_request
            )

            if cognition_result.response is not None:
                memory_write = self._maybe_write_memory_after_cognition(
                    system_request=request,
                    cognition_response_text=cognition_result.response.text,
                )

                return JarvisSystemResponse(
                    request_id=request.request_id,
                    session_id=request.session_id,
                    status=JarvisAskStatus.ANSWERED,
                    text=cognition_result.response.text,
                    cognition_response=cognition_result.response,
                    memory_result_count=context.item_count,
                    used_memory=True,
                    used_cognition=True,
                    memory_write=memory_write,
                    wrote_memory=memory_write.wrote_memory,
                    reason="cognition response produced",
                    created_at=utc_now(),
                    metadata={
                        "cognition_request_id": cognition_request.request_id,
                        "cognition_worker_request_id": (
                            cognition_result.request_id
                        ),
                        "memory_write_status": memory_write.status.value,
                    },
                )

            if cognition_result.failure is not None:
                self._failure_count += 1
                self._last_error = cognition_result.failure.message
                return JarvisSystemResponse(
                    request_id=request.request_id,
                    session_id=request.session_id,
                    status=JarvisAskStatus.FAILED,
                    text="I could not complete that cognition request.",
                    cognition_response=None,
                    memory_result_count=context.item_count,
                    used_memory=True,
                    used_cognition=True,
                    memory_write=_no_memory_write(
                        reason="cognition did not produce a successful response"
                    ),
                    wrote_memory=False,
                    reason=cognition_result.failure.message,
                    created_at=utc_now(),
                    metadata={
                        "failure_kind": cognition_result.failure.kind.value,
                    },
                )

            self._failure_count += 1
            self._last_error = cognition_result.reason
            return JarvisSystemResponse(
                request_id=request.request_id,
                session_id=request.session_id,
                status=JarvisAskStatus.REJECTED,
                text="I am not ready to process that yet.",
                cognition_response=None,
                memory_result_count=context.item_count,
                used_memory=True,
                used_cognition=True,
                memory_write=_no_memory_write(
                    reason="cognition did not produce a successful response"
                ),
                wrote_memory=False,
                reason=cognition_result.reason or "cognition request rejected",
                created_at=utc_now(),
                metadata={},
            )

        except Exception as exc:
            self._failure_count += 1
            self._last_error = f"{type(exc).__name__}: {exc}"
            raise

    def snapshot(self) -> JarvisSystemSnapshot:
        memory_snapshot = self._memory_worker.snapshot()
        cognition_snapshot = self._cognition_worker.snapshot()

        return JarvisSystemSnapshot(
            name=self._name,
            status=self._status,
            started_at=self._started_at,
            stopped_at=self._stopped_at,
            memory_worker=memory_snapshot,
            cognition_worker=cognition_snapshot,
            subsystem_health=(
                JarvisSubsystemHealth(
                    kind=JarvisSubsystemKind.MEMORY,
                    worker=memory_snapshot,
                    subsystem_snapshot=self._memory_worker.memory_snapshot(),
                ),
                JarvisSubsystemHealth(
                    kind=JarvisSubsystemKind.COGNITION,
                    worker=cognition_snapshot,
                    subsystem_snapshot=(
                        self._cognition_worker.cognition_snapshot()
                    ),
                ),
            ),
            kernel_snapshot=_safe_kernel_snapshot(self._kernel),
            ask_count=self._ask_count,
            failure_count=self._failure_count,
            last_error=self._last_error,
        )

    def _register_workers_once(self) -> None:
        if self._registered:
            return

        self._kernel.register_worker(self._memory_worker)
        self._kernel.register_worker(self._cognition_worker)
        self._registered = True

    def _wait_until_ready(self) -> None:
        if not self._cognition_worker.wait_until_ready(timeout_seconds=2.0):
            raise RuntimeError("cognition runtime worker did not become ready.")

    def _retrieve_memory(
        self,
        request: JarvisSystemRequest,
    ) -> MemoryGatewayRetrievalResult:
        return self._memory_worker.retrieve(
            MemoryQuery(
                text=request.text,
                max_results=request.max_memory_results,
            )
        )


    def _maybe_write_memory_after_cognition(
        self,
        *,
        system_request: JarvisSystemRequest,
        cognition_response_text: str,
    ) -> JarvisMemoryWriteDecision:
        write_request = _build_memory_write_request_if_explicit(
            system_request=system_request,
            cognition_response_text=cognition_response_text,
        )

        if write_request is None:
            return _no_memory_write(
                reason="no explicit user memory intent detected"
            )

        try:
            result = self._memory_worker.remember(write_request)
        except Exception as exc:
            return JarvisMemoryWriteDecision(
                status=JarvisMemoryWriteStatus.FAILED,
                should_write=True,
                reason=f"{type(exc).__name__}: {exc}",
                request=write_request,
                result=None,
            )

        if result.allowed and not result.blocked:
            return JarvisMemoryWriteDecision(
                status=JarvisMemoryWriteStatus.WRITTEN,
                should_write=True,
                reason=result.reason,
                request=write_request,
                result=result,
            )

        return JarvisMemoryWriteDecision(
            status=JarvisMemoryWriteStatus.BLOCKED,
            should_write=True,
            reason=result.reason,
            request=write_request,
            result=result,
        )


def _build_cognition_request(
    *,
    system_request: JarvisSystemRequest,
    context: CognitionContext,
) -> CognitionRequest:
    return CognitionRequest(
        text=system_request.text,
        source="jarvis_system",
        correlation_id=system_request.request_id,
        context=context,
        policy=CognitionRuntimePolicy(
            cancellable=True,
            streaming_enabled=False,
            allow_tools=False,
            allow_memory_lookup=False,
            max_response_chars=600,
            spoken_style=SpokenResponseStyle.CONCISE,
            metadata={
                "system_request_id": system_request.request_id,
                "voice_native_default": True,
            },
        ),
        metadata={
            "session_id": system_request.session_id,
            **(system_request.metadata or {}),
        },
    )


def _memory_result_to_cognition_context(
    *,
    memory_result: MemoryGatewayRetrievalResult,
    session_id: str,
) -> CognitionContext:
    records = _records_from_memory_result(memory_result)

    items = tuple(
        CognitionContextItem(
            kind=_memory_context_kind(record),
            text=str(record.text),
            score=_memory_confidence(record),
            source=_memory_source(record),
            metadata=_memory_metadata(record),
        )
        for record in records
        if str(getattr(record, "text", "")).strip()
    )

    return CognitionContext(
        session_id=session_id,
        items=items,
        metadata={
            "memory_allowed": memory_result.allowed,
            "memory_blocked": memory_result.blocked,
            "memory_reason": memory_result.reason,
            "memory_result_count": len(items),
            "policy_classification": (
                memory_result.policy_classification.value
            ),
        },
    )


def _records_from_memory_result(
    memory_result: MemoryGatewayRetrievalResult,
) -> tuple[Any, ...]:
    records = getattr(memory_result, "records", None)
    if records is not None:
        return tuple(records)

    retrieval = getattr(memory_result, "retrieval", None)
    if retrieval is None:
        return ()

    retrieval_records = getattr(retrieval, "records", None)
    if retrieval_records is not None:
        return tuple(retrieval_records)

    results = getattr(retrieval, "results", None)
    if results is None:
        return ()

    extracted: list[Any] = []
    for result in results:
        record = getattr(result, "record", None)
        if record is not None:
            extracted.append(record)

    return tuple(extracted)


def _memory_context_kind(record: Any) -> str:
    kind = getattr(record, "kind", "memory")
    value = getattr(kind, "value", kind)
    return f"memory:{value}"


def _memory_confidence(record: Any) -> float | None:
    confidence = getattr(record, "confidence", None)
    if isinstance(confidence, int | float):
        return max(0.0, min(1.0, float(confidence)))

    return None


def _memory_source(record: Any) -> str:
    source = getattr(record, "source", "memory_gateway")
    value = getattr(source, "value", source)
    text = str(value).strip()
    return text or "memory_gateway"


def _memory_metadata(record: Any) -> dict[str, object]:
    metadata: dict[str, object] = {}

    for field_name in (
        "memory_id",
        "scope",
        "importance",
        "policy_classification",
        "created_at",
        "updated_at",
    ):
        value = getattr(record, field_name, None)
        if value is None:
            continue

        metadata[field_name] = getattr(value, "value", value)

    return metadata

def _build_memory_write_request_if_explicit(
    *,
    system_request: JarvisSystemRequest,
    cognition_response_text: str,
) -> MemoryWriteRequest | None:
    memory_text = _extract_explicit_memory_text(system_request.text)

    if memory_text is None:
        return None

    return MemoryWriteRequest(
        kind=_classify_memory_kind(memory_text),
        scope=MemoryScope.USER,
        source=MemorySource.USER_EXPLICIT,
        text=memory_text,
        sensitivity=MemorySensitivity.PRIVATE,
        importance=_classify_memory_importance(memory_text),
        retention=MemoryRetention.PERSISTENT,
        confidence=0.95,
        tags=("jarvis-system", "explicit-user-memory"),
        metadata={
            "system_request_id": system_request.request_id,
            "session_id": system_request.session_id,
            "write_reason": "explicit user memory request",
            "cognition_response_preview": cognition_response_text[:240],
        },
    )


def _extract_explicit_memory_text(text: str) -> str | None:
    cleaned = " ".join(text.strip().split())
    lowered = cleaned.casefold()

    prefixes = (
        "remember that ",
        "remember ",
        "note that ",
        "store that ",
        "store this ",
        "save that ",
        "save this ",
    )

    for prefix in prefixes:
        if lowered.startswith(prefix):
            extracted = cleaned[len(prefix):].strip(" .")
            return extracted or None

    return None


def _classify_memory_kind(text: str) -> MemoryKind:
    lowered = text.casefold()

    if any(marker in lowered for marker in ("favorite", "prefer", "preference")):
        return MemoryKind.PREFERENCE

    if any(marker in lowered for marker in ("project", "jarvis", "workspace")):
        return MemoryKind.PROJECT

    return MemoryKind.SEMANTIC


def _classify_memory_importance(text: str) -> MemoryImportance:
    lowered = text.casefold()

    if any(marker in lowered for marker in ("always", "never", "important")):
        return MemoryImportance.HIGH

    return MemoryImportance.NORMAL


def _no_memory_write(*, reason: str) -> JarvisMemoryWriteDecision:
    return JarvisMemoryWriteDecision(
        status=JarvisMemoryWriteStatus.NOT_REQUESTED,
        should_write=False,
        reason=reason,
        request=None,
        result=None,
    )


def _safe_kernel_snapshot(kernel: RuntimeKernel) -> object | None:
    try:
        return kernel.snapshot()
    except Exception as exc:
        return {"snapshot_error": f"{type(exc).__name__}: {exc}"}