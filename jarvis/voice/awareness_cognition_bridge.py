from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import StrEnum
from typing import Any, Protocol, cast

from jarvis.voice.awareness_runtime import (
    VoiceAwarenessPacket,
    VoiceAwarenessRequest,
    VoiceAwarenessRuntime,
    VoiceAwarenessStatus,
)
from jarvis.voice.cognition_response import (
    VoiceCognitionRequest,
    VoiceCognitionResponseRuntime,
    VoiceCognitionResult,
)
from jarvis.voice.contracts import utc_now


class VoiceAwarenessCognitionBridgeStatus(StrEnum):
    READY = "ready"
    DEGRADED = "degraded"
    FAILED = "failed"


class VoiceAwarenessCognitionBridgeOperation(StrEnum):
    PREPARE = "prepare"
    PREFETCH_FROM_PARTIAL = "prefetch_from_partial"
    THINK_FROM_TRANSCRIPT = "think_from_transcript"
    SNAPSHOT = "snapshot"


@dataclass(frozen=True, slots=True)
class VoiceAwarenessCognitionBridgePolicy:
    allow_degraded_awareness: bool = True
    block_on_failed_awareness: bool = True
    attach_prompt_context: bool = True
    attach_awareness_signature: bool = True
    attach_awareness_metadata: bool = True
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class VoiceAwarenessCognitionBridgeResult:
    status: VoiceAwarenessCognitionBridgeStatus
    operation: VoiceAwarenessCognitionBridgeOperation
    cognition_result: VoiceCognitionResult | None
    awareness_packet: VoiceAwarenessPacket | None
    reason: str
    latency_ms: float
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def succeeded(self) -> bool:
        return self.status in {
            VoiceAwarenessCognitionBridgeStatus.READY,
            VoiceAwarenessCognitionBridgeStatus.DEGRADED,
        }


@dataclass(frozen=True, slots=True)
class VoiceAwarenessCognitionBridgeSnapshot:
    status: VoiceAwarenessCognitionBridgeStatus
    prepared: bool
    awareness_builds: int
    cognition_calls: int
    degraded_awareness_count: int
    failed_awareness_count: int
    last_awareness_signature: str | None
    last_awareness_status: VoiceAwarenessStatus | None
    last_latency_ms: float | None
    last_error: str | None
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)


class VoiceAwarenessBuilder(Protocol):
    def build(self, request: VoiceAwarenessRequest) -> VoiceAwarenessPacket:
        raise NotImplementedError


class VoiceCognitionEngine(Protocol):
    def prepare(self, *, user_label: str, assistant_name: str) -> object:
        raise NotImplementedError

    def prefetch_from_partial(
        self,
        request: VoiceCognitionRequest,
    ) -> object:
        raise NotImplementedError

    def think_from_transcript(
        self,
        request: VoiceCognitionRequest,
    ) -> VoiceCognitionResult:
        raise NotImplementedError

    def snapshot(self) -> object:
        raise NotImplementedError


class VoiceAwarenessCognitionBridge:
    """
    Wires JARVIS awareness into voice cognition.

    This bridge never creates final user-facing speech.
    It enriches the cognition request with awareness context, then delegates
    final response generation to the cognition/Ollama path and response boundary.
    """

    def __init__(
        self,
        *,
        awareness: VoiceAwarenessBuilder | None = None,
        cognition: VoiceCognitionEngine | None = None,
        policy: VoiceAwarenessCognitionBridgePolicy | None = None,
    ) -> None:
        self._awareness = awareness or VoiceAwarenessRuntime()
        self._cognition = cognition or VoiceCognitionResponseRuntime()
        self._policy = policy or VoiceAwarenessCognitionBridgePolicy()
        self._status = VoiceAwarenessCognitionBridgeStatus.READY
        self._prepared = False
        self._awareness_builds = 0
        self._cognition_calls = 0
        self._degraded_awareness_count = 0
        self._failed_awareness_count = 0
        self._last_awareness_signature: str | None = None
        self._last_awareness_status: VoiceAwarenessStatus | None = None
        self._last_latency_ms: float | None = None
        self._last_error: str | None = None

    def prepare(self, *, user_label: str, assistant_name: str) -> object:
        started = time.perf_counter()
        try:
            prepared = self._cognition.prepare(
                user_label=user_label,
                assistant_name=assistant_name,
            )
        except Exception as exc:
            self._status = VoiceAwarenessCognitionBridgeStatus.FAILED
            self._last_error = str(exc)
            raise

        self._prepared = True
        self._status = VoiceAwarenessCognitionBridgeStatus.READY
        self._last_latency_ms = (time.perf_counter() - started) * 1000.0
        return prepared

    def prefetch_from_partial(
        self,
        request: VoiceCognitionRequest,
    ) -> object:
        started = time.perf_counter()
        packet = self._build_awareness(request)

        enriched = self._enrich_request(
            request=request,
            awareness_packet=packet,
            partial=True,
        )

        result = self._cognition.prefetch_from_partial(enriched)
        self._last_latency_ms = (time.perf_counter() - started) * 1000.0
        return result

    def think_from_transcript(
        self,
        request: VoiceCognitionRequest,
    ) -> VoiceCognitionResult:
        bridge_result = self.think_with_awareness(request)

        if bridge_result.cognition_result is None:
            raise RuntimeError(bridge_result.reason)

        return bridge_result.cognition_result

    def think_with_awareness(
        self,
        request: VoiceCognitionRequest,
    ) -> VoiceAwarenessCognitionBridgeResult:
        started = time.perf_counter()

        try:
            packet = self._build_awareness(request)
        except Exception as exc:
            self._status = VoiceAwarenessCognitionBridgeStatus.FAILED
            self._last_error = str(exc)
            return self._result(
                operation=VoiceAwarenessCognitionBridgeOperation.THINK_FROM_TRANSCRIPT,
                status=VoiceAwarenessCognitionBridgeStatus.FAILED,
                cognition_result=None,
                awareness_packet=None,
                reason="awareness build raised",
                started=started,
                metadata={"error": str(exc)},
            )

        if packet.status == VoiceAwarenessStatus.FAILED:
            self._failed_awareness_count += 1
            if self._policy.block_on_failed_awareness:
                self._status = VoiceAwarenessCognitionBridgeStatus.FAILED
                return self._result(
                    operation=VoiceAwarenessCognitionBridgeOperation.THINK_FROM_TRANSCRIPT,
                    status=VoiceAwarenessCognitionBridgeStatus.FAILED,
                    cognition_result=None,
                    awareness_packet=packet,
                    reason="awareness failed; cognition blocked",
                    started=started,
                )

        if packet.status == VoiceAwarenessStatus.DEGRADED:
            self._degraded_awareness_count += 1
            if not self._policy.allow_degraded_awareness:
                self._status = VoiceAwarenessCognitionBridgeStatus.FAILED
                return self._result(
                    operation=VoiceAwarenessCognitionBridgeOperation.THINK_FROM_TRANSCRIPT,
                    status=VoiceAwarenessCognitionBridgeStatus.FAILED,
                    cognition_result=None,
                    awareness_packet=packet,
                    reason="awareness degraded; policy blocked cognition",
                    started=started,
                )

        enriched = self._enrich_request(
            request=request,
            awareness_packet=packet,
            partial=False,
        )

        try:
            cognition_result = self._cognition.think_from_transcript(enriched)
        except Exception as exc:
            self._status = VoiceAwarenessCognitionBridgeStatus.FAILED
            self._last_error = str(exc)
            return self._result(
                operation=VoiceAwarenessCognitionBridgeOperation.THINK_FROM_TRANSCRIPT,
                status=VoiceAwarenessCognitionBridgeStatus.FAILED,
                cognition_result=None,
                awareness_packet=packet,
                reason="cognition raised after awareness enrichment",
                started=started,
                metadata={"error": str(exc)},
            )

        self._cognition_calls += 1
        self._status = (
            VoiceAwarenessCognitionBridgeStatus.DEGRADED
            if packet.status == VoiceAwarenessStatus.DEGRADED
            else VoiceAwarenessCognitionBridgeStatus.READY
        )

        return self._result(
            operation=VoiceAwarenessCognitionBridgeOperation.THINK_FROM_TRANSCRIPT,
            status=self._status,
            cognition_result=cognition_result,
            awareness_packet=packet,
            reason="cognition completed with awareness",
            started=started,
        )

    def snapshot(self) -> VoiceAwarenessCognitionBridgeSnapshot:
        return VoiceAwarenessCognitionBridgeSnapshot(
            status=self._status,
            prepared=self._prepared,
            awareness_builds=self._awareness_builds,
            cognition_calls=self._cognition_calls,
            degraded_awareness_count=self._degraded_awareness_count,
            failed_awareness_count=self._failed_awareness_count,
            last_awareness_signature=self._last_awareness_signature,
            last_awareness_status=self._last_awareness_status,
            last_latency_ms=self._last_latency_ms,
            last_error=self._last_error,
            created_at=utc_now(),
        )

    def _build_awareness(
        self,
        request: VoiceCognitionRequest,
    ) -> VoiceAwarenessPacket:
        awareness_request = VoiceAwarenessRequest(
            transcript=request.transcript,
            session_id=str(request.transcript.session_id),
            user_label=request.user_label,
            assistant_name=request.assistant_name,
        )
        packet = self._awareness.build(awareness_request)
        self._awareness_builds += 1
        self._last_awareness_signature = packet.signature
        self._last_awareness_status = packet.status
        return packet

    def _enrich_request(
        self,
        *,
        request: VoiceCognitionRequest,
        awareness_packet: VoiceAwarenessPacket,
        partial: bool,
    ) -> VoiceCognitionRequest:
        existing_metadata = _extract_metadata(request)
        awareness_metadata: dict[str, object] = {
            **existing_metadata,
            "awareness_enabled": True,
            "awareness_partial": partial,
            "awareness_status": awareness_packet.status.value,
        }

        if self._policy.attach_prompt_context:
            awareness_metadata["awareness_context"] = (
                awareness_packet.cognition_context
            )
        if self._policy.attach_awareness_signature:
            awareness_metadata["awareness_signature"] = (
                awareness_packet.signature
            )
            awareness_metadata["awareness_highest_priority"] = (
                awareness_packet.highest_priority.value
            )
        if self._policy.attach_awareness_metadata:
            awareness_metadata["awareness_missing_sources"] = tuple(
                source.value for source in awareness_packet.missing_sources
            )
            awareness_metadata["awareness_provider_errors"] = (
                awareness_packet.provider_errors
            )

        return _replace_request_metadata(
            request=request,
            metadata=awareness_metadata,
        )

    def _result(
        self,
        *,
        operation: VoiceAwarenessCognitionBridgeOperation,
        status: VoiceAwarenessCognitionBridgeStatus,
        cognition_result: VoiceCognitionResult | None,
        awareness_packet: VoiceAwarenessPacket | None,
        reason: str,
        started: float,
        metadata: dict[str, object] | None = None,
    ) -> VoiceAwarenessCognitionBridgeResult:
        latency_ms = (time.perf_counter() - started) * 1000.0
        self._last_latency_ms = latency_ms

        return VoiceAwarenessCognitionBridgeResult(
            status=status,
            operation=operation,
            cognition_result=cognition_result,
            awareness_packet=awareness_packet,
            reason=reason,
            latency_ms=latency_ms,
            created_at=utc_now(),
            metadata=metadata or {},
        )


def _extract_metadata(request: VoiceCognitionRequest) -> dict[str, object]:
    metadata = getattr(request, "metadata", None)
    if isinstance(metadata, dict):
        return dict(metadata)
    return {}


def _replace_request_metadata(
    *,
    request: VoiceCognitionRequest,
    metadata: dict[str, object],
) -> VoiceCognitionRequest:
    try:
        return replace(request, metadata=metadata)
    except TypeError:
        mutable_request = cast(Any, request)
        try:
            mutable_request.metadata = metadata
            return request
        except Exception as exc:
            raise RuntimeError(
                "VoiceCognitionRequest does not support awareness metadata. "
                "Add metadata: dict[str, object] to the request contract."
            ) from exc