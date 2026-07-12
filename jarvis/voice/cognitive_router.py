from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol

from jarvis.voice.transcript_attention_gate import TranscriptGateDecision


class VoiceCognitiveState(StrEnum):
    IDLE = "idle"
    LISTENING = "listening"
    CAPTURING_SPEECH = "capturing_speech"
    STABILIZING_TRANSCRIPT = "stabilizing_transcript"
    THINKING = "thinking"
    PLANNING = "planning"
    EXECUTING_TOOLS = "executing_tools"
    RESPONDING = "responding"
    INTERRUPTED = "interrupted"


class VoiceCognitiveRouteAction(StrEnum):
    IGNORE = "ignore"
    WAIT_FOR_STABILITY = "wait_for_stability"
    RESPOND = "respond"
    CLARIFY = "clarify"


class VoiceRouterTranscript(Protocol):
    @property
    def text(self) -> str:
        raise NotImplementedError

    @property
    def confidence(self) -> float:
        raise NotImplementedError

    @property
    def metadata(self) -> dict[str, object]:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class VoiceCognitiveRouterPolicy:
    min_stable_confidence: float = 0.72
    min_stable_words: int = 2
    allow_wake_override: bool = True
    tool_verbs: frozenset[str] = frozenset(
        {
            "open",
            "close",
            "run",
            "create",
            "write",
            "delete",
            "move",
            "copy",
            "send",
            "call",
            "schedule",
            "search",
            "build",
            "install",
        }
    )
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_stable_confidence <= 1.0:
            raise ValueError("min_stable_confidence must be 0..1.")
        if self.min_stable_words < 1:
            raise ValueError("min_stable_words must be positive.")


@dataclass(frozen=True, slots=True)
class VoiceCognitiveRouteRequest:
    transcript: VoiceRouterTranscript
    gate_decision: TranscriptGateDecision
    assistant_speaking: bool = False
    active_playback: bool = False
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class VoiceCognitiveRouteDecision:
    action: VoiceCognitiveRouteAction
    state: VoiceCognitiveState
    reason: str
    confidence: float
    stability: float
    tool_planning_recommended: bool
    requires_confirmation: bool
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def should_enter_cognition(self) -> bool:
        return self.action == VoiceCognitiveRouteAction.RESPOND

    def to_metadata(self) -> dict[str, object]:
        return {
            "action": self.action.value,
            "state": self.state.value,
            "reason": self.reason,
            "confidence": self.confidence,
            "stability": self.stability,
            "tool_planning_recommended": self.tool_planning_recommended,
            "requires_confirmation": self.requires_confirmation,
            **self.metadata,
        }


class VoiceCognitiveRouter:
    """
    Live cognitive gate between perception and cognition.

    This runtime does not call the LLM, execute tools, or synthesize speech.
    It converts transcript/gate facts into a strict cognitive route so noisy
    perception cannot jump directly into reasoning or action planning.
    """

    def __init__(
        self,
        *,
        policy: VoiceCognitiveRouterPolicy | None = None,
    ) -> None:
        self._policy = policy or VoiceCognitiveRouterPolicy()
        self._state = VoiceCognitiveState.IDLE
        self._decisions = 0
        self._ignored = 0
        self._waited = 0
        self._responded = 0
        self._clarified = 0
        self._last_decision: VoiceCognitiveRouteDecision | None = None

    @property
    def state(self) -> VoiceCognitiveState:
        return self._state

    def route(
        self,
        request: VoiceCognitiveRouteRequest,
    ) -> VoiceCognitiveRouteDecision:
        self._decisions += 1
        transcript = request.transcript
        gate = request.gate_decision
        stability = _stability_score(transcript=transcript, gate=gate)
        tool_planning_recommended = _mentions_tool_intent(
            gate.normalized_text,
            self._policy.tool_verbs,
        )

        if request.assistant_speaking or request.active_playback:
            return self._remember(
                VoiceCognitiveRouteDecision(
                    action=VoiceCognitiveRouteAction.WAIT_FOR_STABILITY,
                    state=VoiceCognitiveState.INTERRUPTED,
                    reason="interruption_context_stabilizing",
                    confidence=gate.confidence,
                    stability=stability,
                    tool_planning_recommended=tool_planning_recommended,
                    requires_confirmation=False,
                )
            )

        if not gate.accepted:
            return self._remember(
                VoiceCognitiveRouteDecision(
                    action=VoiceCognitiveRouteAction.IGNORE,
                    state=VoiceCognitiveState.LISTENING,
                    reason=f"attention_gate_{gate.reason}",
                    confidence=gate.confidence,
                    stability=stability,
                    tool_planning_recommended=False,
                    requires_confirmation=False,
                )
            )

        wake_override = self._policy.allow_wake_override and gate.wake_detected
        enough_words = (
            gate.attention_active or gate.word_count >= self._policy.min_stable_words
        )
        stable_enough = stability >= self._policy.min_stable_confidence
        if not wake_override and (not enough_words or not stable_enough):
            return self._remember(
                VoiceCognitiveRouteDecision(
                    action=VoiceCognitiveRouteAction.WAIT_FOR_STABILITY,
                    state=VoiceCognitiveState.STABILIZING_TRANSCRIPT,
                    reason="transcript_not_stable_enough",
                    confidence=gate.confidence,
                    stability=stability,
                    tool_planning_recommended=False,
                    requires_confirmation=False,
                    metadata={"min_stability": self._policy.min_stable_confidence},
                )
            )

        requires_confirmation = tool_planning_recommended and not wake_override
        return self._remember(
            VoiceCognitiveRouteDecision(
                action=VoiceCognitiveRouteAction.RESPOND,
                state=(
                    VoiceCognitiveState.PLANNING
                    if tool_planning_recommended
                    else VoiceCognitiveState.THINKING
                ),
                reason=(
                    "tool_intent_routed_to_planning"
                    if tool_planning_recommended
                    else "stable_dialogue_routed_to_cognition"
                ),
                confidence=gate.confidence,
                stability=stability,
                tool_planning_recommended=tool_planning_recommended,
                requires_confirmation=requires_confirmation,
            )
        )

    def reset(self) -> None:
        self._state = VoiceCognitiveState.IDLE
        self._last_decision = None

    def snapshot(self) -> dict[str, object]:
        return {
            "state": self._state.value,
            "decisions": self._decisions,
            "ignored": self._ignored,
            "waited": self._waited,
            "responded": self._responded,
            "clarified": self._clarified,
            "last_decision": (
                None
                if self._last_decision is None
                else self._last_decision.to_metadata()
            ),
        }

    def _remember(
        self,
        decision: VoiceCognitiveRouteDecision,
    ) -> VoiceCognitiveRouteDecision:
        self._state = decision.state
        self._last_decision = decision
        if decision.action == VoiceCognitiveRouteAction.IGNORE:
            self._ignored += 1
        elif decision.action == VoiceCognitiveRouteAction.WAIT_FOR_STABILITY:
            self._waited += 1
        elif decision.action == VoiceCognitiveRouteAction.RESPOND:
            self._responded += 1
        elif decision.action == VoiceCognitiveRouteAction.CLARIFY:
            self._clarified += 1
        return decision


def _stability_score(
    *,
    transcript: VoiceRouterTranscript,
    gate: TranscriptGateDecision,
) -> float:
    metadata = transcript.metadata
    raw_stability = metadata.get("stability") or metadata.get("transcript_stability")
    if isinstance(raw_stability, int | float):
        return max(0.0, min(1.0, float(raw_stability)))

    confidence = max(0.0, min(1.0, float(transcript.confidence)))
    word_factor = min(1.0, gate.word_count / 4.0)
    wake_bonus = 0.12 if gate.wake_detected else 0.0
    attention_bonus = 0.08 if gate.attention_active else 0.0
    promoted_penalty = 0.15 if metadata.get("promoted_from_partial") is True else 0.0
    local_stability = max(
        0.0,
        min(
            1.0,
            (confidence * 0.70)
            + (word_factor * 0.20)
            + wake_bonus
            + attention_bonus
            - promoted_penalty,
        ),
    )
    perception_stability = metadata.get("perception_stability")
    if isinstance(perception_stability, int | float):
        return max(local_stability, max(0.0, min(1.0, float(perception_stability))))
    return local_stability


def _mentions_tool_intent(text: str, tool_verbs: frozenset[str]) -> bool:
    words = set(text.split())
    return bool(words & tool_verbs)
