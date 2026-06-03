from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum

from jarvis.cognitive.contracts import (
    BehaviorPolicy,
    BehaviorTone,
    PersonalityProfile,
    utc_now,
)


class BehaviorRuntimeStatus(StrEnum):
    READY = "ready"
    BLOCKED = "blocked"


class BehaviorIntent(StrEnum):
    CONFIRMATION = "confirmation"
    CLARIFICATION = "clarification"
    WARNING = "warning"
    CHALLENGE = "challenge"
    STATUS = "status"
    INTERRUPTION = "interruption"
    SILENCE = "silence"
    HUMOR = "humor"


class BehaviorRisk(StrEnum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class BehaviorStance(StrEnum):
    SUPPORTIVE = "supportive"
    PROTECTIVE = "protective"
    DIRECT = "direct"
    CAUTIOUS = "cautious"
    SILENT = "silent"


@dataclass(frozen=True, slots=True)
class BehaviorRequest:
    intent: BehaviorIntent
    message: str = ""
    risk: BehaviorRisk = BehaviorRisk.NONE
    instruction_complete: bool = True
    user_is_busy: bool = False
    allow_humor: bool = False
    requires_truth_challenge: bool = False
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class BehaviorDirective:
    tone: BehaviorTone
    stance: BehaviorStance
    max_sentences: int
    should_speak: bool
    should_warn: bool
    should_clarify: bool
    should_challenge: bool
    allow_humor: bool
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class BehaviorRuntimeResult:
    status: BehaviorRuntimeStatus
    intent: BehaviorIntent
    text: str
    directive: BehaviorDirective
    reason: str
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def should_speak(self) -> bool:
        return self.directive.should_speak


@dataclass(frozen=True, slots=True)
class BehaviorRuntimeSnapshot:
    status: BehaviorRuntimeStatus
    profile: PersonalityProfile
    policy: BehaviorPolicy
    decision_count: int
    warning_count: int
    clarification_count: int
    challenge_count: int
    silence_count: int
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)


class PersonalityRuntime:
    """
    Phase 9 / Step 49E Personality & Behavior Runtime.

    This runtime shapes JARVIS behavior:
    - calm
    - respectful
    - concise
    - protective
    - socially aware
    - lightly witty when appropriate
    - truthful
    - able to challenge Balu carefully

    It does not change facts.
    It does not execute tools.
    It does not fake human emotions.
    It does not override safety.
    """

    def __init__(
        self,
        *,
        profile: PersonalityProfile | None = None,
        policy: BehaviorPolicy | None = None,
    ) -> None:
        self._profile = profile or default_jarvis_personality()
        self._policy = policy or default_behavior_policy()
        self._decision_count = 0
        self._warning_count = 0
        self._clarification_count = 0
        self._challenge_count = 0
        self._silence_count = 0

    @property
    def profile(self) -> PersonalityProfile:
        return self._profile

    @property
    def policy(self) -> BehaviorPolicy:
        return self._policy

    def respond(self, request: BehaviorRequest) -> BehaviorRuntimeResult:
        self._decision_count += 1

        directive = _directive_for_request(
            request=request,
            profile=self._profile,
            policy=self._policy,
        )
        text = _text_for_request(
            request=request,
            profile=self._profile,
            policy=self._policy,
            directive=directive,
        )
        reason = _reason_for_request(request=request, directive=directive)

        if directive.should_warn:
            self._warning_count += 1
        if directive.should_clarify:
            self._clarification_count += 1
        if directive.should_challenge:
            self._challenge_count += 1
        if not directive.should_speak:
            self._silence_count += 1

        return BehaviorRuntimeResult(
            status=BehaviorRuntimeStatus.READY,
            intent=request.intent,
            text=text,
            directive=directive,
            reason=reason,
            created_at=utc_now(),
            metadata=request.metadata,
        )

    def update_profile(
        self,
        profile: PersonalityProfile,
    ) -> BehaviorRuntimeResult:
        self._profile = profile

        directive = BehaviorDirective(
            tone=profile.default_tone,
            stance=BehaviorStance.SUPPORTIVE,
            max_sentences=self._policy.max_reply_sentences,
            should_speak=True,
            should_warn=False,
            should_clarify=False,
            should_challenge=False,
            allow_humor=self._policy.allow_dry_humor,
            created_at=utc_now(),
        )

        return BehaviorRuntimeResult(
            status=BehaviorRuntimeStatus.READY,
            intent=BehaviorIntent.STATUS,
            text="Personality profile updated.",
            directive=directive,
            reason="profile updated",
            created_at=utc_now(),
        )

    def update_policy(
        self,
        policy: BehaviorPolicy,
    ) -> BehaviorRuntimeResult:
        self._policy = policy

        directive = BehaviorDirective(
            tone=self._profile.default_tone,
            stance=BehaviorStance.SUPPORTIVE,
            max_sentences=policy.max_reply_sentences,
            should_speak=True,
            should_warn=False,
            should_clarify=False,
            should_challenge=False,
            allow_humor=policy.allow_dry_humor,
            created_at=utc_now(),
        )

        return BehaviorRuntimeResult(
            status=BehaviorRuntimeStatus.READY,
            intent=BehaviorIntent.STATUS,
            text="Behavior policy updated.",
            directive=directive,
            reason="policy updated",
            created_at=utc_now(),
        )

    def snapshot(self) -> BehaviorRuntimeSnapshot:
        return BehaviorRuntimeSnapshot(
            status=BehaviorRuntimeStatus.READY,
            profile=self._profile,
            policy=self._policy,
            decision_count=self._decision_count,
            warning_count=self._warning_count,
            clarification_count=self._clarification_count,
            challenge_count=self._challenge_count,
            silence_count=self._silence_count,
            created_at=utc_now(),
        )


def default_jarvis_personality() -> PersonalityProfile:
    return PersonalityProfile(
        name="JARVIS",
        traits=(
            "calm",
            "polite",
            "respectful",
            "concise",
            "protective",
            "truthful",
            "patient",
            "socially_aware",
            "slightly_witty",
            "carefully_challenging",
        ),
        default_tone=BehaviorTone.CALM,
        confirmation_phrase="Certainly, sir.",
        warning_phrase="I would advise caution.",
        clarification_phrase="I need one detail before proceeding.",
        created_at=utc_now(),
        metadata={
            "relationship": "Balu-JARVIS",
            "style": "calm executive assistant",
        },
    )


def default_behavior_policy() -> BehaviorPolicy:
    return BehaviorPolicy(
        max_reply_sentences=3,
        interrupt_only_when_important=True,
        ask_when_instruction_incomplete=True,
        allow_dry_humor=True,
        truth_over_comfort=True,
        created_at=utc_now(),
        metadata={
            "no_fake_emotion": True,
            "facts_override_style": True,
            "safety_over_personality": True,
        },
    )


def _directive_for_request(
    *,
    request: BehaviorRequest,
    profile: PersonalityProfile,
    policy: BehaviorPolicy,
) -> BehaviorDirective:
    should_clarify = (
        not request.instruction_complete
        and policy.ask_when_instruction_incomplete
    )
    should_warn = request.risk in {
        BehaviorRisk.HIGH,
        BehaviorRisk.CRITICAL,
    }
    should_challenge = (
        request.requires_truth_challenge
        and policy.truth_over_comfort
    )
    should_speak = request.intent != BehaviorIntent.SILENCE

    if request.user_is_busy and request.risk in {
        BehaviorRisk.NONE,
        BehaviorRisk.LOW,
    }:
        should_speak = False

    tone = _tone_for_request(
        request=request,
        profile=profile,
        should_warn=should_warn,
        should_clarify=should_clarify,
        should_challenge=should_challenge,
    )
    stance = _stance_for_request(
        request=request,
        should_warn=should_warn,
        should_challenge=should_challenge,
        should_speak=should_speak,
    )

    return BehaviorDirective(
        tone=tone,
        stance=stance,
        max_sentences=policy.max_reply_sentences,
        should_speak=should_speak,
        should_warn=should_warn,
        should_clarify=should_clarify,
        should_challenge=should_challenge,
        allow_humor=(
            policy.allow_dry_humor
            and request.allow_humor
            and request.risk in {BehaviorRisk.NONE, BehaviorRisk.LOW}
        ),
        created_at=utc_now(),
        metadata={
            "profile": profile.name,
            "risk": request.risk.value,
        },
    )


def _tone_for_request(
    *,
    request: BehaviorRequest,
    profile: PersonalityProfile,
    should_warn: bool,
    should_clarify: bool,
    should_challenge: bool,
) -> BehaviorTone:
    if should_warn:
        return BehaviorTone.WARNING

    if should_clarify:
        return BehaviorTone.CLARIFYING

    if should_challenge:
        return BehaviorTone.PROTECTIVE

    if request.intent == BehaviorIntent.HUMOR:
        return BehaviorTone.HUMOROUS

    return profile.default_tone


def _stance_for_request(
    *,
    request: BehaviorRequest,
    should_warn: bool,
    should_challenge: bool,
    should_speak: bool,
) -> BehaviorStance:
    if not should_speak:
        return BehaviorStance.SILENT

    if should_warn:
        return BehaviorStance.PROTECTIVE

    if should_challenge:
        return BehaviorStance.CAUTIOUS

    if request.intent == BehaviorIntent.STATUS:
        return BehaviorStance.DIRECT

    return BehaviorStance.SUPPORTIVE


def _text_for_request(
    *,
    request: BehaviorRequest,
    profile: PersonalityProfile,
    policy: BehaviorPolicy,
    directive: BehaviorDirective,
) -> str:
    if not directive.should_speak:
        return ""

    if request.intent == BehaviorIntent.INTERRUPTION:
        return _bounded_text(
            "Stopping. Listening now.",
            max_sentences=policy.max_reply_sentences,
        )

    if directive.should_warn:
        return _bounded_text(
            f"{profile.warning_phrase} {request.message}".strip(),
            max_sentences=policy.max_reply_sentences,
        )

    if directive.should_clarify:
        return _bounded_text(
            f"{profile.clarification_phrase} {request.message}".strip(),
            max_sentences=policy.max_reply_sentences,
        )

    if directive.should_challenge:
        return _bounded_text(
            f"I would challenge that carefully, sir. {request.message}".strip(),
            max_sentences=policy.max_reply_sentences,
        )

    if request.intent == BehaviorIntent.CONFIRMATION:
        return _bounded_text(
            f"{profile.confirmation_phrase} {request.message}".strip(),
            max_sentences=policy.max_reply_sentences,
        )


    if request.intent == BehaviorIntent.HUMOR and directive.allow_humor:
        return _bounded_text(
            _humorous_text(request.message),
            max_sentences=policy.max_reply_sentences,
        )

    if request.message.strip():
        return _bounded_text(
            request.message.strip(),
            max_sentences=policy.max_reply_sentences,
        )

    return _bounded_text(
        profile.confirmation_phrase,
        max_sentences=policy.max_reply_sentences,
    )


def _reason_for_request(
    *,
    request: BehaviorRequest,
    directive: BehaviorDirective,
) -> str:
    if not directive.should_speak:
        return "user is busy and message is not important enough to interrupt"

    if directive.should_warn:
        return "risk requires protective warning behavior"

    if directive.should_clarify:
        return "instruction is incomplete and clarification is required"

    if directive.should_challenge:
        return "truth-over-comfort policy requires careful challenge"

    if request.intent == BehaviorIntent.HUMOR and not directive.allow_humor:
        return "humor suppressed by risk or policy"

    return "behavior response generated"


def _humorous_text(message: str) -> str:
    if message.strip():
        return f"{message.strip()} Small mercy: at least the tests are honest."

    return "Certainly, sir. The code remains dramatic, but manageable."


def _bounded_text(
    text: str,
    *,
    max_sentences: int,
) -> str:
    cleaned = " ".join(text.split())
    if not cleaned:
        return ""

    sentences = _split_sentences(cleaned)
    return " ".join(sentences[:max_sentences])


def _split_sentences(text: str) -> tuple[str, ...]:
    parts: list[str] = []
    current = ""

    for char in text:
        current += char
        if char in {".", "!", "?"}:
            value = current.strip()
            if value:
                parts.append(value)
            current = ""

    if current.strip():
        parts.append(current.strip())

    return tuple(parts)