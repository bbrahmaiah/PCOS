from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import StrEnum
from typing import Any, Protocol, cast

from jarvis.voice.awareness_runtime import (
    VoiceAwarenessFact,
    VoiceAwarenessPacket,
    VoiceAwarenessPriority,
    VoiceAwarenessProvider,
    VoiceAwarenessRequest,
    VoiceAwarenessRuntime,
    VoiceAwarenessSource,
    VoiceAwarenessStatus,
)
from jarvis.voice.cognition_response import (
    VoiceCognitionRequest,
    VoiceCognitionResult,
)
from jarvis.voice.contracts import utc_now


class VoiceRealAwarenessStatus(StrEnum):
    READY = "ready"
    DEGRADED = "degraded"
    FAILED = "failed"


class VoiceRealAwarenessOperation(StrEnum):
    COLLECT = "collect"
    INJECT = "inject"
    VALIDATE = "validate"
    COMPLETE = "complete"


class VoiceRealAwarenessProviderKind(StrEnum):
    MEMORY = "memory"
    ENVIRONMENT = "environment"
    GOALS = "goals"
    PERSONALITY = "personality"
    TOOLS = "tools"
    DEVELOPER = "developer"
    HEALTH = "health"
    SAFETY = "safety"


@dataclass(frozen=True, slots=True)
class VoiceAwarenessRecord:
    key: str
    value: str
    confidence: float = 0.9
    priority: VoiceAwarenessPriority = VoiceAwarenessPriority.NORMAL
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.key.strip():
            raise ValueError("awareness record key cannot be empty.")
        if not self.value.strip():
            raise ValueError("awareness record value cannot be empty.")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("awareness record confidence must be between 0 and 1.")


class VoiceRealAwarenessSourceClient(Protocol):
    def collect_records(
        self,
        request: VoiceAwarenessRequest,
    ) -> tuple[VoiceAwarenessRecord, ...]:
        raise NotImplementedError


@dataclass(slots=True)
class VoiceRealAwarenessProvider:
    source: VoiceAwarenessSource
    client: VoiceRealAwarenessSourceClient
    provider_kind: VoiceRealAwarenessProviderKind
    required: bool = True

    def collect(
        self,
        request: VoiceAwarenessRequest,
    ) -> tuple[VoiceAwarenessFact, ...]:
        records = self.client.collect_records(request)

        return tuple(
            VoiceAwarenessFact(
                source=self.source,
                key=record.key,
                value=record.value,
                confidence=record.confidence,
                priority=record.priority,
                created_at=utc_now(),
                metadata={
                    "provider_kind": self.provider_kind.value,
                    "required": self.required,
                    **record.metadata,
                },
            )
            for record in records
        )


@dataclass(frozen=True, slots=True)
class StaticAwarenessSourceClient:
    records: tuple[VoiceAwarenessRecord, ...]

    def collect_records(
        self,
        request: VoiceAwarenessRequest,
    ) -> tuple[VoiceAwarenessRecord, ...]:
        return self.records


class VoiceRealAwarenessPromptInjectionMode(StrEnum):
    METADATA_ONLY = "metadata_only"
    CONTEXT_AND_METADATA = "context_and_metadata"


@dataclass(frozen=True, slots=True)
class VoiceRealAwarenessPromptInjectorConfig:
    mode: VoiceRealAwarenessPromptInjectionMode = (
        VoiceRealAwarenessPromptInjectionMode.CONTEXT_AND_METADATA
    )
    max_context_chars: int = 6000
    require_ready_awareness: bool = True
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.max_context_chars < 500:
            raise ValueError("max_context_chars must be at least 500.")


@dataclass(frozen=True, slots=True)
class VoiceRealAwarenessInjectionResult:
    status: VoiceRealAwarenessStatus
    request: VoiceCognitionRequest
    awareness_packet: VoiceAwarenessPacket
    reason: str
    latency_ms: float
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)


class VoiceRealAwarenessPromptInjector:
    def __init__(
        self,
        *,
        config: VoiceRealAwarenessPromptInjectorConfig | None = None,
    ) -> None:
        self._config = config or VoiceRealAwarenessPromptInjectorConfig()

    def inject(
        self,
        *,
        request: VoiceCognitionRequest,
        awareness_packet: VoiceAwarenessPacket,
    ) -> VoiceRealAwarenessInjectionResult:
        started = time.perf_counter()

        if (
            self._config.require_ready_awareness
            and awareness_packet.status == VoiceAwarenessStatus.FAILED
        ):
            return VoiceRealAwarenessInjectionResult(
                status=VoiceRealAwarenessStatus.FAILED,
                request=request,
                awareness_packet=awareness_packet,
                reason="awareness packet failed; injection blocked",
                latency_ms=(time.perf_counter() - started) * 1000.0,
                created_at=utc_now(),
            )

        metadata = _request_metadata(request)
        metadata.update(
            {
                "awareness_enabled": True,
                "awareness_status": awareness_packet.status.value,
                "awareness_signature": awareness_packet.signature,
                "awareness_highest_priority": (
                    awareness_packet.highest_priority.value
                ),
                "awareness_missing_sources": tuple(
                    source.value for source in awareness_packet.missing_sources
                ),
                "awareness_provider_errors": awareness_packet.provider_errors,
                "response_origin_required": "cognition_response_boundary",
                "scripted_conversation_allowed": False,
            }
        )

        if (
            self._config.mode
            == VoiceRealAwarenessPromptInjectionMode.CONTEXT_AND_METADATA
        ):
            metadata["awareness_context"] = _trim_context(
                awareness_packet.cognition_context,
                max_chars=self._config.max_context_chars,
            )

        metadata.update(self._config.metadata)

        enriched = _replace_request_metadata(request, metadata)

        return VoiceRealAwarenessInjectionResult(
            status=(
                VoiceRealAwarenessStatus.DEGRADED
                if awareness_packet.status == VoiceAwarenessStatus.DEGRADED
                else VoiceRealAwarenessStatus.READY
            ),
            request=enriched,
            awareness_packet=awareness_packet,
            reason="awareness injected into cognition request",
            latency_ms=(time.perf_counter() - started) * 1000.0,
            created_at=utc_now(),
            metadata={"injection_mode": self._config.mode.value},
        )


@dataclass(frozen=True, slots=True)
class VoiceRealAwarenessBoundaryConfig:
    require_awareness_enabled: bool = True
    require_awareness_context: bool = True
    require_response_origin: bool = True
    require_generated_response: bool = True
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class VoiceRealAwarenessBoundaryResult:
    status: VoiceRealAwarenessStatus
    passed: bool
    reason: str
    violations: tuple[str, ...]
    latency_ms: float
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)


class VoiceRealAwarenessResponseBoundary:
    """
    Validates that voice cognition used awareness wiring.

    This boundary does not inspect or create final wording. It verifies origin,
    awareness metadata, response existence, and that no runtime path explicitly
    marked the response as scripted.
    """

    def __init__(
        self,
        *,
        config: VoiceRealAwarenessBoundaryConfig | None = None,
    ) -> None:
        self._config = config or VoiceRealAwarenessBoundaryConfig()

    def validate(
        self,
        *,
        request: VoiceCognitionRequest,
        result: VoiceCognitionResult,
    ) -> VoiceRealAwarenessBoundaryResult:
        started = time.perf_counter()
        metadata = _request_metadata(request)
        violations: list[str] = []

        if self._config.require_awareness_enabled and not metadata.get(
            "awareness_enabled"
        ):
            violations.append("awareness metadata missing")

        if self._config.require_awareness_context and not str(
            metadata.get("awareness_context", "")
        ).strip():
            violations.append("awareness context missing")

        if self._config.require_response_origin and (
            metadata.get("response_origin_required")
            != "cognition_response_boundary"
        ):
            violations.append("response origin boundary missing")

        if metadata.get("scripted_conversation_allowed") is True:
            violations.append("scripted conversation was allowed")

        response = getattr(result, "response", None)
        response_text = str(getattr(response, "text", "")).strip()

        if self._config.require_generated_response and not response_text:
            violations.append("generated response missing")

        passed = not violations

        return VoiceRealAwarenessBoundaryResult(
            status=(
                VoiceRealAwarenessStatus.READY
                if passed
                else VoiceRealAwarenessStatus.FAILED
            ),
            passed=passed,
            reason=(
                "awareness response boundary passed"
                if passed
                else "awareness response boundary failed"
            ),
            violations=tuple(violations),
            latency_ms=(time.perf_counter() - started) * 1000.0,
            created_at=utc_now(),
            metadata=self._config.metadata,
        )


@dataclass(frozen=True, slots=True)
class VoiceRealAwarenessIntegrationConfig:
    allow_degraded_awareness: bool = True
    require_boundary_pass: bool = True
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class VoiceRealAwarenessIntegrationResult:
    status: VoiceRealAwarenessStatus
    cognition_request: VoiceCognitionRequest
    awareness_packet: VoiceAwarenessPacket
    injection_result: VoiceRealAwarenessInjectionResult
    boundary_result: VoiceRealAwarenessBoundaryResult | None
    reason: str
    latency_ms: float
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)


class VoiceRealAwarenessIntegration:
    """
    Balanced 51K.7 + 51K.8 + 51K.9 integration.

    51K.7: collect real awareness providers.
    51K.8: inject awareness into cognition request.
    51K.9: validate awareness response boundary.

    This class never generates final user-facing speech.
    """

    def __init__(
        self,
        *,
        awareness_runtime: VoiceAwarenessRuntime,
        injector: VoiceRealAwarenessPromptInjector | None = None,
        boundary: VoiceRealAwarenessResponseBoundary | None = None,
        config: VoiceRealAwarenessIntegrationConfig | None = None,
    ) -> None:
        self._awareness_runtime = awareness_runtime
        self._injector = injector or VoiceRealAwarenessPromptInjector()
        self._boundary = boundary or VoiceRealAwarenessResponseBoundary()
        self._config = config or VoiceRealAwarenessIntegrationConfig()

    def prepare_request(
        self,
        request: VoiceCognitionRequest,
    ) -> VoiceRealAwarenessIntegrationResult:
        started = time.perf_counter()
        awareness_request = VoiceAwarenessRequest(
            transcript=request.transcript,
            session_id=str(request.transcript.session_id),
            user_label=request.user_label,
            assistant_name=request.assistant_name,
        )
        packet = self._awareness_runtime.build(awareness_request)

        if packet.status == VoiceAwarenessStatus.FAILED:
            return self._integration_result(
                status=VoiceRealAwarenessStatus.FAILED,
                cognition_request=request,
                awareness_packet=packet,
                injection_result=VoiceRealAwarenessInjectionResult(
                    status=VoiceRealAwarenessStatus.FAILED,
                    request=request,
                    awareness_packet=packet,
                    reason="awareness failed before prompt injection",
                    latency_ms=0.0,
                    created_at=utc_now(),
                ),
                boundary_result=None,
                reason="awareness failed",
                started=started,
            )

        if (
            packet.status == VoiceAwarenessStatus.DEGRADED
            and not self._config.allow_degraded_awareness
        ):
            return self._integration_result(
                status=VoiceRealAwarenessStatus.FAILED,
                cognition_request=request,
                awareness_packet=packet,
                injection_result=VoiceRealAwarenessInjectionResult(
                    status=VoiceRealAwarenessStatus.FAILED,
                    request=request,
                    awareness_packet=packet,
                    reason="degraded awareness blocked by policy",
                    latency_ms=0.0,
                    created_at=utc_now(),
                ),
                boundary_result=None,
                reason="degraded awareness blocked",
                started=started,
            )

        injection = self._injector.inject(
            request=request,
            awareness_packet=packet,
        )

        return self._integration_result(
            status=injection.status,
            cognition_request=injection.request,
            awareness_packet=packet,
            injection_result=injection,
            boundary_result=None,
            reason="awareness request prepared for cognition",
            started=started,
        )

    def validate_response(
        self,
        *,
        request: VoiceCognitionRequest,
        result: VoiceCognitionResult,
        awareness_packet: VoiceAwarenessPacket,
        injection_result: VoiceRealAwarenessInjectionResult,
    ) -> VoiceRealAwarenessIntegrationResult:
        started = time.perf_counter()
        boundary = self._boundary.validate(request=request, result=result)

        status = (
            VoiceRealAwarenessStatus.READY
            if boundary.passed
            else VoiceRealAwarenessStatus.FAILED
        )

        if (
            self._config.require_boundary_pass
            and status == VoiceRealAwarenessStatus.FAILED
        ):
            reason = "awareness boundary blocked response"
        else:
            reason = "awareness response validated"

        return self._integration_result(
            status=status,
            cognition_request=request,
            awareness_packet=awareness_packet,
            injection_result=injection_result,
            boundary_result=boundary,
            reason=reason,
            started=started,
        )

    def _integration_result(
        self,
        *,
        status: VoiceRealAwarenessStatus,
        cognition_request: VoiceCognitionRequest,
        awareness_packet: VoiceAwarenessPacket,
        injection_result: VoiceRealAwarenessInjectionResult,
        boundary_result: VoiceRealAwarenessBoundaryResult | None,
        reason: str,
        started: float,
    ) -> VoiceRealAwarenessIntegrationResult:
        return VoiceRealAwarenessIntegrationResult(
            status=status,
            cognition_request=cognition_request,
            awareness_packet=awareness_packet,
            injection_result=injection_result,
            boundary_result=boundary_result,
            reason=reason,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            created_at=utc_now(),
            metadata=self._config.metadata,
        )


def make_real_awareness_runtime(
    *,
    providers: Sequence[VoiceAwarenessProvider],
) -> VoiceAwarenessRuntime:
    return VoiceAwarenessRuntime(providers=tuple(providers))


def make_real_awareness_provider(
    *,
    source: VoiceAwarenessSource,
    provider_kind: VoiceRealAwarenessProviderKind,
    records: tuple[VoiceAwarenessRecord, ...],
    required: bool = True,
) -> VoiceRealAwarenessProvider:
    return VoiceRealAwarenessProvider(
        source=source,
        provider_kind=provider_kind,
        client=StaticAwarenessSourceClient(records=records),
        required=required,
    )


def _request_metadata(request: VoiceCognitionRequest) -> dict[str, object]:
    metadata = getattr(request, "metadata", None)
    if isinstance(metadata, Mapping):
        return dict(metadata)
    return {}


def _replace_request_metadata(
    request: VoiceCognitionRequest,
    metadata: dict[str, object],
) -> VoiceCognitionRequest:
    try:
        return replace(request, metadata=metadata)
    except TypeError:
        mutable = cast(Any, request)
        mutable.metadata = metadata
        return request


def _trim_context(context: str, *, max_chars: int) -> str:
    if len(context) <= max_chars:
        return context
    return context[: max_chars - 24].rstrip() + "\n...context_trimmed"