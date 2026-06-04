from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Protocol

from jarvis.cognitive.contracts import CognitiveSessionState
from jarvis.live.contracts import (
    LiveResponse,
    LiveResponseGenerationSource,
    LiveResponseKind,
    LiveResponseSafety,
    LiveSessionState,
    LiveTurnId,
    make_live_response,
    utc_now,
)


class LiveResponseBoundaryStatus(StrEnum):
    READY = "ready"
    BLOCKED = "blocked"


class LiveResponseIntent(StrEnum):
    GREETING = "greeting"
    ANSWER = "answer"
    CLARIFICATION = "clarification"
    WARNING = "warning"
    INTERRUPTION = "interruption"
    CONTINUE = "continue"
    REPEAT = "repeat"
    STATUS = "status"
    SHUTDOWN_REQUEST = "shutdown_request"
    LEARNING = "learning"


class LiveResponseSurface(StrEnum):
    VOICE = "voice"
    DISPLAY = "display"
    VOICE_AND_DISPLAY = "voice_and_display"


class LiveResponseBoundaryViolation(StrEnum):
    NONE = "none"
    MISSING_GENERATOR = "missing_generator"
    EMPTY_GENERATED_TEXT = "empty_generated_text"
    CONVERSATIONAL_STATIC_TEXT = "conversational_static_text"
    INVALID_GENERATION_SOURCE = "invalid_generation_source"
    BLOCKED_BY_SAFETY = "blocked_by_safety"
    DIAGNOSTIC_KIND_REQUIRED = "diagnostic_kind_required"


@dataclass(frozen=True, slots=True)
class LiveResponseContext:
    live_state: LiveSessionState
    cognitive_state: CognitiveSessionState | None = None
    user_text: str = ""
    situation_summary: str = ""
    memory_context: tuple[str, ...] = ()
    working_memory_context: tuple[str, ...] = ()
    attention_context: tuple[str, ...] = ()
    goal_context: tuple[str, ...] = ()
    planning_context: tuple[str, ...] = ()
    environment_context: tuple[str, ...] = ()
    developer_context: tuple[str, ...] = ()
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class LiveResponseGenerationRequest:
    turn_id: LiveTurnId
    intent: LiveResponseIntent
    surface: LiveResponseSurface
    context: LiveResponseContext
    response_kind: LiveResponseKind = LiveResponseKind.CONVERSATIONAL
    safety: LiveResponseSafety = LiveResponseSafety.SAFE_TO_SPEAK
    max_sentences: int = 3
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.turn_id).strip():
            raise ValueError("live response generation turn_id cannot be empty.")
        if self.max_sentences < 1:
            raise ValueError("live response max_sentences must be at least 1.")


@dataclass(frozen=True, slots=True)
class LiveResponseDraft:
    text: str
    generation_source: LiveResponseGenerationSource
    token_count: int | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("live response draft text cannot be empty.")
        if self.token_count is not None and self.token_count < 0:
            raise ValueError("live response draft token_count cannot be negative.")


@dataclass(frozen=True, slots=True)
class LiveDeterministicSystemMessage:
    turn_id: LiveTurnId
    kind: LiveResponseKind
    text: str
    source: LiveResponseGenerationSource
    safety: LiveResponseSafety = LiveResponseSafety.SAFE_TO_SPEAK
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.turn_id).strip():
            raise ValueError("deterministic message turn_id cannot be empty.")
        if not self.text.strip():
            raise ValueError("deterministic message text cannot be empty.")
        if self.kind == LiveResponseKind.CONVERSATIONAL:
            raise ValueError(
                "deterministic system message cannot be conversational."
            )
        if self.source not in {
            LiveResponseGenerationSource.DIAGNOSTIC_SYSTEM,
            LiveResponseGenerationSource.EMERGENCY_FALLBACK,
        }:
            raise ValueError(
                "deterministic system message requires diagnostic or fallback source."
            )


@dataclass(frozen=True, slots=True)
class LiveResponseBoundaryPolicy:
    require_generated_conversation: bool = True
    allow_deterministic_diagnostics: bool = True
    allowed_deterministic_kinds: tuple[LiveResponseKind, ...] = (
        LiveResponseKind.DIAGNOSTIC,
        LiveResponseKind.SAFETY,
        LiveResponseKind.RECOVERY,
        LiveResponseKind.SHUTDOWN,
    )
    allowed_conversational_sources: tuple[LiveResponseGenerationSource, ...] = (
        LiveResponseGenerationSource.RESPONSE_GENERATOR,
        LiveResponseGenerationSource.COGNITION_RUNTIME,
    )


@dataclass(frozen=True, slots=True)
class LiveResponseBoundaryResult:
    status: LiveResponseBoundaryStatus
    response: LiveResponse | None
    violation: LiveResponseBoundaryViolation
    reason: str
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def succeeded(self) -> bool:
        return self.status == LiveResponseBoundaryStatus.READY


@dataclass(frozen=True, slots=True)
class LiveResponseBoundarySnapshot:
    status: LiveResponseBoundaryStatus
    generated_count: int
    deterministic_system_count: int
    blocked_count: int
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)


class LiveResponseGenerator(Protocol):
    def generate(
        self,
        request: LiveResponseGenerationRequest,
    ) -> LiveResponseDraft:
        """Generate final user-facing conversational text."""


class LiveResponseBoundaryRuntime:
    """
    Step 50A.5 Live Response Generation Boundary.

    This runtime protects live JARVIS from becoming a scripted voice demo.

    Conversational speech must be generated through an injected response
    generator or cognition runtime boundary.

    Deterministic text is allowed only for non-conversational diagnostics,
    recovery, safety, shutdown, or emergency fallback messages.
    """

    def __init__(
        self,
        *,
        generator: LiveResponseGenerator | None = None,
        policy: LiveResponseBoundaryPolicy | None = None,
    ) -> None:
        self._generator = generator
        self._policy = policy or LiveResponseBoundaryPolicy()
        self._generated_count = 0
        self._deterministic_system_count = 0
        self._blocked_count = 0

    def generate(
        self,
        request: LiveResponseGenerationRequest,
    ) -> LiveResponseBoundaryResult:
        if request.response_kind != LiveResponseKind.CONVERSATIONAL:
            self._blocked_count += 1
            return _blocked(
                violation=LiveResponseBoundaryViolation.DIAGNOSTIC_KIND_REQUIRED,
                reason=(
                    "generate() is only for conversational responses; "
                    "use system_message() for deterministic diagnostics"
                ),
                metadata=request.metadata,
            )

        if request.safety == LiveResponseSafety.BLOCKED:
            self._blocked_count += 1
            return _blocked(
                violation=LiveResponseBoundaryViolation.BLOCKED_BY_SAFETY,
                reason="response generation blocked by safety policy",
                metadata=request.metadata,
            )

        if self._policy.require_generated_conversation:
            if self._generator is None:
                self._blocked_count += 1
                return _blocked(
                    violation=LiveResponseBoundaryViolation.MISSING_GENERATOR,
                    reason=(
                        "conversational live speech requires a response "
                        "generator"
                    ),
                    metadata=request.metadata,
                )

        if self._generator is None:
            self._blocked_count += 1
            return _blocked(
                violation=LiveResponseBoundaryViolation.MISSING_GENERATOR,
                reason="no response generator configured",
                metadata=request.metadata,
            )

        draft = self._generator.generate(request)

        if not draft.text.strip():
            self._blocked_count += 1
            return _blocked(
                violation=LiveResponseBoundaryViolation.EMPTY_GENERATED_TEXT,
                reason="response generator returned empty text",
                metadata=request.metadata,
            )

        if draft.generation_source not in self._policy.allowed_conversational_sources:
            self._blocked_count += 1
            return _blocked(
                violation=(
                    LiveResponseBoundaryViolation.INVALID_GENERATION_SOURCE
                ),
                reason="conversational response came from invalid source",
                metadata={
                    **request.metadata,
                    "generation_source": draft.generation_source.value,
                },
            )

        response = make_live_response(
            turn_id=request.turn_id,
            kind=LiveResponseKind.CONVERSATIONAL,
            text=draft.text,
            generation_source=draft.generation_source,
            safety=request.safety,
            token_count=draft.token_count,
            metadata={
                **request.metadata,
                **draft.metadata,
                "intent": request.intent.value,
                "surface": request.surface.value,
                "boundary": "live_response_generation",
            },
        )
        self._generated_count += 1

        return LiveResponseBoundaryResult(
            status=LiveResponseBoundaryStatus.READY,
            response=response,
            violation=LiveResponseBoundaryViolation.NONE,
            reason="conversational response generated through boundary",
            created_at=utc_now(),
            metadata=response.metadata,
        )

    def system_message(
        self,
        message: LiveDeterministicSystemMessage,
    ) -> LiveResponseBoundaryResult:
        if not self._policy.allow_deterministic_diagnostics:
            self._blocked_count += 1
            return _blocked(
                violation=(
                    LiveResponseBoundaryViolation.CONVERSATIONAL_STATIC_TEXT
                ),
                reason="deterministic system messages are disabled by policy",
                metadata=message.metadata,
            )

        if message.kind not in self._policy.allowed_deterministic_kinds:
            self._blocked_count += 1
            return _blocked(
                violation=LiveResponseBoundaryViolation.DIAGNOSTIC_KIND_REQUIRED,
                reason="deterministic message kind is not allowed",
                metadata=message.metadata,
            )

        response = make_live_response(
            turn_id=message.turn_id,
            kind=message.kind,
            text=message.text,
            generation_source=message.source,
            safety=message.safety,
            metadata={
                **message.metadata,
                "boundary": "deterministic_system_message",
            },
        )
        self._deterministic_system_count += 1

        return LiveResponseBoundaryResult(
            status=LiveResponseBoundaryStatus.READY,
            response=response,
            violation=LiveResponseBoundaryViolation.NONE,
            reason="deterministic non-conversational system message accepted",
            created_at=utc_now(),
            metadata=response.metadata,
        )

    def validate_for_tts(
        self,
        response: LiveResponse,
    ) -> LiveResponseBoundaryResult:
        if response.safety == LiveResponseSafety.BLOCKED:
            self._blocked_count += 1
            return _blocked(
                violation=LiveResponseBoundaryViolation.BLOCKED_BY_SAFETY,
                reason="blocked response cannot be sent to TTS",
                metadata=response.metadata,
            )

        if response.is_conversational:
            if response.generation_source not in (
                self._policy.allowed_conversational_sources
            ):
                self._blocked_count += 1
                return _blocked(
                    violation=(
                        LiveResponseBoundaryViolation.INVALID_GENERATION_SOURCE
                    ),
                    reason=(
                        "conversational TTS response must come from "
                        "response generator or cognition runtime"
                    ),
                    metadata=response.metadata,
                )

        if response.deterministic_system_response:
            if response.kind not in self._policy.allowed_deterministic_kinds:
                self._blocked_count += 1
                return _blocked(
                    violation=(
                        LiveResponseBoundaryViolation.DIAGNOSTIC_KIND_REQUIRED
                    ),
                    reason=(
                        "deterministic response kind is not allowed for TTS"
                    ),
                    metadata=response.metadata,
                )

        return LiveResponseBoundaryResult(
            status=LiveResponseBoundaryStatus.READY,
            response=response,
            violation=LiveResponseBoundaryViolation.NONE,
            reason="response accepted for TTS",
            created_at=utc_now(),
            metadata=response.metadata,
        )

    def snapshot(self) -> LiveResponseBoundarySnapshot:
        return LiveResponseBoundarySnapshot(
            status=LiveResponseBoundaryStatus.READY,
            generated_count=self._generated_count,
            deterministic_system_count=self._deterministic_system_count,
            blocked_count=self._blocked_count,
            created_at=utc_now(),
        )


def _blocked(
    *,
    violation: LiveResponseBoundaryViolation,
    reason: str,
    metadata: dict[str, object],
) -> LiveResponseBoundaryResult:
    return LiveResponseBoundaryResult(
        status=LiveResponseBoundaryStatus.BLOCKED,
        response=None,
        violation=violation,
        reason=reason,
        created_at=utc_now(),
        metadata=metadata,
    )