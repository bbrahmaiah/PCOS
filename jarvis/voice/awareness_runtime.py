from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Protocol

from jarvis.voice.contracts import VoiceTranscript, utc_now


class VoiceAwarenessStatus(StrEnum):
    READY = "ready"
    DEGRADED = "degraded"
    FAILED = "failed"


class VoiceAwarenessSource(StrEnum):
    TRANSCRIPT = "transcript"
    SESSION = "session"
    MEMORY = "memory"
    GOALS = "goals"
    PERSONALITY = "personality"
    ENVIRONMENT = "environment"
    TOOLS = "tools"
    DEVELOPER = "developer"
    HEALTH = "health"
    SAFETY = "safety"
    RESPONSE_BOUNDARY = "response_boundary"


class VoiceAwarenessPriority(StrEnum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass(frozen=True, slots=True)
class VoiceAwarenessFact:
    source: VoiceAwarenessSource
    key: str
    value: str
    confidence: float
    priority: VoiceAwarenessPriority = VoiceAwarenessPriority.NORMAL
    created_at: datetime = field(default_factory=utc_now)
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.key.strip():
            raise ValueError("awareness fact key cannot be empty.")
        if not self.value.strip():
            raise ValueError("awareness fact value cannot be empty.")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("awareness fact confidence must be between 0 and 1.")


@dataclass(frozen=True, slots=True)
class VoiceAwarenessRequest:
    transcript: VoiceTranscript
    session_id: str
    user_label: str
    assistant_name: str
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.session_id.strip():
            raise ValueError("session_id cannot be empty.")
        if not self.user_label.strip():
            raise ValueError("user_label cannot be empty.")
        if not self.assistant_name.strip():
            raise ValueError("assistant_name cannot be empty.")


@dataclass(frozen=True, slots=True)
class VoiceAwarenessPacket:
    status: VoiceAwarenessStatus
    request: VoiceAwarenessRequest
    facts: tuple[VoiceAwarenessFact, ...]
    cognition_context: str
    signature: str
    highest_priority: VoiceAwarenessPriority
    missing_sources: tuple[VoiceAwarenessSource, ...]
    provider_errors: tuple[str, ...]
    latency_ms: float
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def ready(self) -> bool:
        return self.status == VoiceAwarenessStatus.READY

    @property
    def fact_count(self) -> int:
        return len(self.facts)


@dataclass(frozen=True, slots=True)
class VoiceAwarenessRuntimeConfig:
    max_facts_per_source: int = 8
    max_context_chars: int = 6000
    min_confidence: float = 0.35
    require_memory: bool = True
    require_environment: bool = True
    require_goals: bool = True
    require_personality: bool = True
    require_response_boundary: bool = True
    include_low_confidence: bool = False
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.max_facts_per_source < 1:
            raise ValueError("max_facts_per_source must be positive.")
        if self.max_context_chars < 500:
            raise ValueError("max_context_chars must be at least 500.")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be between 0 and 1.")


class VoiceAwarenessProvider(Protocol):
    source: VoiceAwarenessSource

    def collect(
        self,
        request: VoiceAwarenessRequest,
    ) -> tuple[VoiceAwarenessFact, ...]:
        raise NotImplementedError


class VoiceAwarenessRuntime:
    """
    Builds JARVIS awareness for the voice path.

    This runtime never creates final user-facing speech.
    It only prepares awareness for cognition/Ollama.
    """

    def __init__(
        self,
        *,
        providers: tuple[VoiceAwarenessProvider, ...] = (),
        config: VoiceAwarenessRuntimeConfig | None = None,
    ) -> None:
        self._providers = providers
        self._config = config or VoiceAwarenessRuntimeConfig()

    def build(self, request: VoiceAwarenessRequest) -> VoiceAwarenessPacket:
        started = time.perf_counter()

        facts: list[VoiceAwarenessFact] = [
            VoiceAwarenessFact(
                source=VoiceAwarenessSource.TRANSCRIPT,
                key="current_user_transcript",
                value=request.transcript.text,
                confidence=request.transcript.confidence,
                priority=VoiceAwarenessPriority.NORMAL,
                metadata={"kind": request.transcript.kind.value},
            ),
            VoiceAwarenessFact(
                source=VoiceAwarenessSource.SESSION,
                key="active_voice_session",
                value=request.session_id,
                confidence=1.0,
                priority=VoiceAwarenessPriority.NORMAL,
            ),
        ]

        provider_errors: list[str] = []

        for provider in self._providers:
            try:
                facts.extend(self._filter_facts(provider.collect(request)))
            except Exception as exc:
                provider_errors.append(
                    f"{provider.source.value}:{type(exc).__name__}:{exc}"
                )

        compact_facts = _dedupe_limit_and_sort(
            tuple(facts),
            max_facts_per_source=self._config.max_facts_per_source,
        )
        missing_sources = _missing_required_sources(
            compact_facts,
            config=self._config,
        )

        status = _packet_status(
            missing_sources=missing_sources,
            provider_errors=tuple(provider_errors),
        )
        cognition_context = _render_cognition_context(
            compact_facts,
            max_chars=self._config.max_context_chars,
        )

        return VoiceAwarenessPacket(
            status=status,
            request=request,
            facts=compact_facts,
            cognition_context=cognition_context,
            signature=_awareness_signature(compact_facts),
            highest_priority=_highest_priority(compact_facts),
            missing_sources=missing_sources,
            provider_errors=tuple(provider_errors),
            latency_ms=(time.perf_counter() - started) * 1000.0,
            created_at=utc_now(),
            metadata={
                "provider_count": len(self._providers),
                **self._config.metadata,
            },
        )

    def _filter_facts(
        self,
        facts: tuple[VoiceAwarenessFact, ...],
    ) -> tuple[VoiceAwarenessFact, ...]:
        if self._config.include_low_confidence:
            return facts
        return tuple(
            fact for fact in facts if fact.confidence >= self._config.min_confidence
        )


@dataclass(slots=True)
class StaticVoiceAwarenessProvider:
    source: VoiceAwarenessSource
    facts: tuple[VoiceAwarenessFact, ...]

    def collect(
        self,
        request: VoiceAwarenessRequest,
    ) -> tuple[VoiceAwarenessFact, ...]:
        return self.facts


class VoiceMemoryAwarenessProvider:
    source = VoiceAwarenessSource.MEMORY

    def __init__(self, facts: tuple[VoiceAwarenessFact, ...]) -> None:
        self._facts = facts

    def collect(
        self,
        request: VoiceAwarenessRequest,
    ) -> tuple[VoiceAwarenessFact, ...]:
        return self._facts


class VoiceEnvironmentAwarenessProvider:
    source = VoiceAwarenessSource.ENVIRONMENT

    def __init__(self, facts: tuple[VoiceAwarenessFact, ...]) -> None:
        self._facts = facts

    def collect(
        self,
        request: VoiceAwarenessRequest,
    ) -> tuple[VoiceAwarenessFact, ...]:
        return self._facts


class VoiceGoalAwarenessProvider:
    source = VoiceAwarenessSource.GOALS

    def __init__(self, facts: tuple[VoiceAwarenessFact, ...]) -> None:
        self._facts = facts

    def collect(
        self,
        request: VoiceAwarenessRequest,
    ) -> tuple[VoiceAwarenessFact, ...]:
        return self._facts


class VoicePersonalityAwarenessProvider:
    source = VoiceAwarenessSource.PERSONALITY

    def __init__(self, facts: tuple[VoiceAwarenessFact, ...]) -> None:
        self._facts = facts

    def collect(
        self,
        request: VoiceAwarenessRequest,
    ) -> tuple[VoiceAwarenessFact, ...]:
        return self._facts


class VoiceToolAwarenessProvider:
    source = VoiceAwarenessSource.TOOLS

    def __init__(self, facts: tuple[VoiceAwarenessFact, ...]) -> None:
        self._facts = facts

    def collect(
        self,
        request: VoiceAwarenessRequest,
    ) -> tuple[VoiceAwarenessFact, ...]:
        return self._facts


class VoiceHealthAwarenessProvider:
    source = VoiceAwarenessSource.HEALTH

    def __init__(self, facts: tuple[VoiceAwarenessFact, ...]) -> None:
        self._facts = facts

    def collect(
        self,
        request: VoiceAwarenessRequest,
    ) -> tuple[VoiceAwarenessFact, ...]:
        return self._facts


class VoiceResponseBoundaryAwarenessProvider:
    source = VoiceAwarenessSource.RESPONSE_BOUNDARY

    def collect(
        self,
        request: VoiceAwarenessRequest,
    ) -> tuple[VoiceAwarenessFact, ...]:
        return (
            VoiceAwarenessFact(
                source=VoiceAwarenessSource.RESPONSE_BOUNDARY,
                key="final_speech_origin",
                value="cognition_response_boundary",
                confidence=1.0,
                priority=VoiceAwarenessPriority.HIGH,
                metadata={"fixed_conversational_response_allowed": False},
            ),
        )


def _missing_required_sources(
    facts: tuple[VoiceAwarenessFact, ...],
    *,
    config: VoiceAwarenessRuntimeConfig,
) -> tuple[VoiceAwarenessSource, ...]:
    present = {fact.source for fact in facts}
    required = {
        VoiceAwarenessSource.TRANSCRIPT,
        VoiceAwarenessSource.SESSION,
    }

    if config.require_memory:
        required.add(VoiceAwarenessSource.MEMORY)
    if config.require_environment:
        required.add(VoiceAwarenessSource.ENVIRONMENT)
    if config.require_goals:
        required.add(VoiceAwarenessSource.GOALS)
    if config.require_personality:
        required.add(VoiceAwarenessSource.PERSONALITY)
    if config.require_response_boundary:
        required.add(VoiceAwarenessSource.RESPONSE_BOUNDARY)

    return tuple(sorted(required - present, key=lambda source: source.value))


def _packet_status(
    *,
    missing_sources: tuple[VoiceAwarenessSource, ...],
    provider_errors: tuple[str, ...],
) -> VoiceAwarenessStatus:
    if missing_sources:
        return VoiceAwarenessStatus.FAILED
    if provider_errors:
        return VoiceAwarenessStatus.DEGRADED
    return VoiceAwarenessStatus.READY


def _dedupe_limit_and_sort(
    facts: tuple[VoiceAwarenessFact, ...],
    *,
    max_facts_per_source: int,
) -> tuple[VoiceAwarenessFact, ...]:
    seen: set[tuple[VoiceAwarenessSource, str, str]] = set()
    counts: dict[VoiceAwarenessSource, int] = {}
    output: list[VoiceAwarenessFact] = []

    sorted_facts = sorted(
        facts,
        key=lambda fact: (
            _priority_rank(fact.priority),
            fact.confidence,
            fact.created_at.timestamp(),
        ),
        reverse=True,
    )

    for fact in sorted_facts:
        identity = (fact.source, fact.key, fact.value)
        if identity in seen:
            continue

        count = counts.get(fact.source, 0)
        if count >= max_facts_per_source:
            continue

        seen.add(identity)
        counts[fact.source] = count + 1
        output.append(fact)

    return tuple(output)


def _render_cognition_context(
    facts: tuple[VoiceAwarenessFact, ...],
    *,
    max_chars: int,
) -> str:
    lines = [
        "JARVIS_AWARENESS_CONTEXT",
        "Use current awareness to reason before answering.",
        "Final spoken words must be generated by cognition.",
        "Do not answer from transcript alone.",
    ]

    for fact in facts:
        lines.append(
            "- "
            f"source={fact.source.value}; "
            f"priority={fact.priority.value}; "
            f"confidence={fact.confidence:.2f}; "
            f"{fact.key}={fact.value}"
        )

    rendered = "\n".join(lines).strip()
    if len(rendered) <= max_chars:
        return rendered

    return rendered[: max_chars - 24].rstrip() + "\n...awareness_truncated"


def _awareness_signature(facts: tuple[VoiceAwarenessFact, ...]) -> str:
    raw = "\n".join(
        f"{fact.source.value}:{fact.key}:{fact.value}:{fact.priority.value}"
        for fact in facts
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _highest_priority(
    facts: tuple[VoiceAwarenessFact, ...],
) -> VoiceAwarenessPriority:
    if not facts:
        return VoiceAwarenessPriority.LOW
    return max(facts, key=lambda fact: _priority_rank(fact.priority)).priority


def _priority_rank(priority: VoiceAwarenessPriority) -> int:
    ranks = {
        VoiceAwarenessPriority.LOW: 0,
        VoiceAwarenessPriority.NORMAL: 1,
        VoiceAwarenessPriority.HIGH: 2,
        VoiceAwarenessPriority.CRITICAL: 3,
    }
    return ranks[priority]