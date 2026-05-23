from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from threading import RLock
from typing import Any

from pydantic import Field, field_validator

from jarvis.cognition.models import (
    CognitionModel,
    CognitionPlan,
    CognitionPlanKind,
    CognitionRequest,
    SpokenResponseStyle,
    new_id,
)
from jarvis.runtime.observability.structured_logger import get_logger


class ResponseIntent(StrEnum):
    """
    High-level user intent inferred before response generation.
    """

    GREETING = "greeting"
    QUESTION = "question"
    EXPLANATION = "explanation"
    STATUS = "status"
    COMMAND = "command"
    MEMORY = "memory"
    TOOL_ACTION = "tool_action"
    CLARIFICATION_NEEDED = "clarification_needed"
    UNKNOWN = "unknown"


class ResponseAnswerMode(StrEnum):
    """
    How cognition should answer.
    """

    DIRECT = "direct"
    ASK_CLARIFICATION = "ask_clarification"
    SAFE_REFUSAL = "safe_refusal"
    TOOL_PLANNING = "tool_planning"


class ResponseSafetyPosture(StrEnum):
    """
    Safety posture for response planning.
    """

    NORMAL = "normal"
    CAUTION = "caution"
    REFUSE = "refuse"


class ResponsePlanningDecision(CognitionModel):
    """
    Rich response planning decision.

    This is more detailed than CognitionPlan. The engine can later use this for
    memory routing, tool routing, spoken style, and safety behavior.
    """

    decision_id: str = Field(default_factory=new_id)
    request_id: str
    intent: ResponseIntent = ResponseIntent.UNKNOWN
    answer_mode: ResponseAnswerMode = ResponseAnswerMode.DIRECT
    plan_kind: CognitionPlanKind = CognitionPlanKind.DIRECT_ANSWER
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    needs_clarification: bool = False
    safety_posture: ResponseSafetyPosture = ResponseSafetyPosture.NORMAL
    memory_lookup_recommended: bool = False
    tool_planning_recommended: bool = False
    spoken_style: SpokenResponseStyle = SpokenResponseStyle.CONCISE
    reasons: tuple[str, ...] = ()
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("request_id")
    @classmethod
    def _request_id_required(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("request_id cannot be empty.")

        return value


@dataclass(frozen=True, slots=True)
class ResponsePlannerConfig:
    """
    Configuration for ResponsePlanner.

    This planner is deterministic and conservative. It is not a real LLM
    planner yet; it is the stable policy layer before model execution.
    """

    name: str = "response_planner"
    clarification_min_chars: int = 3
    enable_tool_detection: bool = True
    enable_memory_detection: bool = True
    refuse_phrases: tuple[str, ...] = (
        "bypass password",
        "steal password",
        "keylogger",
        "malware",
        "delete system32",
        "format drive",
        "disable antivirus",
    )
    tool_phrases: tuple[str, ...] = (
        "open ",
        "run ",
        "execute ",
        "delete ",
        "move ",
        "copy ",
        "send ",
        "schedule ",
        "create file",
        "terminal",
        "command prompt",
        "powershell",
    )
    memory_phrases: tuple[str, ...] = (
        "remember",
        "what did we",
        "what have we",
        "last time",
        "previously",
        "recall",
        "my preference",
    )
    status_phrases: tuple[str, ...] = (
        "status",
        "report",
        "diagnostics",
        "health",
        "what is running",
    )

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("name cannot be empty.")

        if self.clarification_min_chars <= 0:
            raise ValueError("clarification_min_chars must be greater than zero.")

        self._validate_phrase_tuple("refuse_phrases", self.refuse_phrases)
        self._validate_phrase_tuple("tool_phrases", self.tool_phrases)
        self._validate_phrase_tuple("memory_phrases", self.memory_phrases)
        self._validate_phrase_tuple("status_phrases", self.status_phrases)

    @staticmethod
    def _validate_phrase_tuple(
        name: str,
        phrases: tuple[str, ...],
    ) -> None:
        for phrase in phrases:
            if not phrase.strip():
                raise ValueError(f"{name} cannot contain empty phrases.")


@dataclass(frozen=True, slots=True)
class ResponsePlannerSnapshot:
    """
    Observable planner diagnostics.
    """

    name: str
    planned_count: int
    direct_count: int
    clarification_count: int
    refusal_count: int
    tool_planning_count: int
    memory_recommended_count: int
    last_request_id: str | None
    last_intent: ResponseIntent | None
    last_answer_mode: ResponseAnswerMode | None
    last_error: str | None


class ResponsePlanner:
    """
    Deterministic response planning layer.

    Responsibilities:
    - infer high-level intent
    - decide answer mode
    - decide if clarification is needed
    - flag memory/tool readiness
    - produce CognitionPlan for engine compatibility

    Non-responsibilities:
    - no LLM calls
    - no tool execution
    - no memory retrieval
    - no audio or Presence internals
    """

    def __init__(
        self,
        *,
        config: ResponsePlannerConfig | None = None,
    ) -> None:
        self._config = config or ResponsePlannerConfig()
        self._config.validate()

        self._lock = RLock()
        self._logger = get_logger("cognition.response_planner")

        self._planned_count = 0
        self._direct_count = 0
        self._clarification_count = 0
        self._refusal_count = 0
        self._tool_planning_count = 0
        self._memory_recommended_count = 0
        self._last_request_id: str | None = None
        self._last_intent: ResponseIntent | None = None
        self._last_answer_mode: ResponseAnswerMode | None = None
        self._last_error: str | None = None

    @property
    def name(self) -> str:
        return self._config.name

    def plan(self, request: CognitionRequest) -> ResponsePlanningDecision:
        """
        Build a rich planning decision for one cognition request.
        """

        normalized_text = self._normalize(request.text)
        reasons: list[str] = []

        intent = self._intent_for(request, normalized_text)
        safety_posture = self._safety_posture_for(normalized_text)

        memory_lookup_recommended = (
            self._config.enable_memory_detection
            and self._contains_any(normalized_text, self._config.memory_phrases)
        )
        tool_planning_recommended = (
            self._config.enable_tool_detection
            and self._contains_any(normalized_text, self._config.tool_phrases)
        )

        if memory_lookup_recommended:
            reasons.append("memory lookup may improve response")

        if tool_planning_recommended:
            reasons.append("request appears action-oriented")

        if safety_posture == ResponseSafetyPosture.REFUSE:
            decision = self._refusal_decision(
                request=request,
                intent=intent,
                memory_lookup_recommended=memory_lookup_recommended,
                tool_planning_recommended=tool_planning_recommended,
                reasons=tuple([*reasons, "request matched refusal policy"]),
            )

        elif self._needs_clarification(normalized_text):
            decision = self._clarification_decision(
                request=request,
                intent=ResponseIntent.CLARIFICATION_NEEDED,
                memory_lookup_recommended=memory_lookup_recommended,
                tool_planning_recommended=tool_planning_recommended,
                reasons=tuple([*reasons, "request is underspecified"]),
            )

        elif tool_planning_recommended and request.policy.allow_tools:
            decision = self._tool_planning_decision(
                request=request,
                intent=ResponseIntent.TOOL_ACTION,
                memory_lookup_recommended=memory_lookup_recommended,
                reasons=tuple([*reasons, "tools are allowed by policy"]),
            )

        else:
            decision = self._direct_decision(
                request=request,
                intent=intent,
                safety_posture=safety_posture,
                memory_lookup_recommended=memory_lookup_recommended,
                tool_planning_recommended=tool_planning_recommended,
                reasons=tuple(reasons),
            )

        self._record_decision(decision)

        self._logger.info(
            "response_planning_completed",
            planner=self.name,
            request_id=request.request_id,
            intent=decision.intent.value,
            answer_mode=decision.answer_mode.value,
            plan_kind=decision.plan_kind.value,
            confidence=decision.confidence,
        )

        return decision

    def create_plan(self, request: CognitionRequest) -> CognitionPlan:
        """
        Produce the engine-compatible CognitionPlan.
        """

        return self.to_cognition_plan(self.plan(request))

    def to_cognition_plan(
        self,
        decision: ResponsePlanningDecision,
    ) -> CognitionPlan:
        """
        Convert a rich planning decision into the existing CognitionPlan model.
        """

        return CognitionPlan(
            request_id=decision.request_id,
            kind=decision.plan_kind,
            confidence=decision.confidence,
            needs_clarification=decision.needs_clarification,
            allowed_tool_names=(),
            notes=decision.reasons,
            metadata={
                "planner": self.name,
                "decision_id": decision.decision_id,
                "intent": decision.intent.value,
                "answer_mode": decision.answer_mode.value,
                "safety_posture": decision.safety_posture.value,
                "memory_lookup_recommended": decision.memory_lookup_recommended,
                "tool_planning_recommended": decision.tool_planning_recommended,
                "spoken_style": decision.spoken_style.value,
                **decision.metadata,
            },
        )

    def snapshot(self) -> ResponsePlannerSnapshot:
        """
        Return planner diagnostics.
        """

        with self._lock:
            return ResponsePlannerSnapshot(
                name=self.name,
                planned_count=self._planned_count,
                direct_count=self._direct_count,
                clarification_count=self._clarification_count,
                refusal_count=self._refusal_count,
                tool_planning_count=self._tool_planning_count,
                memory_recommended_count=self._memory_recommended_count,
                last_request_id=self._last_request_id,
                last_intent=self._last_intent,
                last_answer_mode=self._last_answer_mode,
                last_error=self._last_error,
            )

    def reset(self) -> None:
        """
        Reset planner counters.
        """

        with self._lock:
            self._planned_count = 0
            self._direct_count = 0
            self._clarification_count = 0
            self._refusal_count = 0
            self._tool_planning_count = 0
            self._memory_recommended_count = 0
            self._last_request_id = None
            self._last_intent = None
            self._last_answer_mode = None
            self._last_error = None

        self._logger.info("response_planner_reset", planner=self.name)

    def _direct_decision(
        self,
        *,
        request: CognitionRequest,
        intent: ResponseIntent,
        safety_posture: ResponseSafetyPosture,
        memory_lookup_recommended: bool,
        tool_planning_recommended: bool,
        reasons: tuple[str, ...],
    ) -> ResponsePlanningDecision:
        return ResponsePlanningDecision(
            request_id=request.request_id,
            intent=intent,
            answer_mode=ResponseAnswerMode.DIRECT,
            plan_kind=CognitionPlanKind.DIRECT_ANSWER,
            confidence=0.9,
            needs_clarification=False,
            safety_posture=safety_posture,
            memory_lookup_recommended=memory_lookup_recommended,
            tool_planning_recommended=tool_planning_recommended,
            spoken_style=request.policy.spoken_style,
            reasons=reasons,
        )

    def _clarification_decision(
        self,
        *,
        request: CognitionRequest,
        intent: ResponseIntent,
        memory_lookup_recommended: bool,
        tool_planning_recommended: bool,
        reasons: tuple[str, ...],
    ) -> ResponsePlanningDecision:
        return ResponsePlanningDecision(
            request_id=request.request_id,
            intent=intent,
            answer_mode=ResponseAnswerMode.ASK_CLARIFICATION,
            plan_kind=CognitionPlanKind.ASK_CLARIFICATION,
            confidence=0.65,
            needs_clarification=True,
            safety_posture=ResponseSafetyPosture.NORMAL,
            memory_lookup_recommended=memory_lookup_recommended,
            tool_planning_recommended=tool_planning_recommended,
            spoken_style=request.policy.spoken_style,
            reasons=reasons,
        )

    def _refusal_decision(
        self,
        *,
        request: CognitionRequest,
        intent: ResponseIntent,
        memory_lookup_recommended: bool,
        tool_planning_recommended: bool,
        reasons: tuple[str, ...],
    ) -> ResponsePlanningDecision:
        return ResponsePlanningDecision(
            request_id=request.request_id,
            intent=intent,
            answer_mode=ResponseAnswerMode.SAFE_REFUSAL,
            plan_kind=CognitionPlanKind.SAFE_REFUSAL,
            confidence=0.95,
            needs_clarification=False,
            safety_posture=ResponseSafetyPosture.REFUSE,
            memory_lookup_recommended=memory_lookup_recommended,
            tool_planning_recommended=tool_planning_recommended,
            spoken_style=request.policy.spoken_style,
            reasons=reasons,
        )

    def _tool_planning_decision(
        self,
        *,
        request: CognitionRequest,
        intent: ResponseIntent,
        memory_lookup_recommended: bool,
        reasons: tuple[str, ...],
    ) -> ResponsePlanningDecision:
        return ResponsePlanningDecision(
            request_id=request.request_id,
            intent=intent,
            answer_mode=ResponseAnswerMode.TOOL_PLANNING,
            plan_kind=CognitionPlanKind.TOOL_PLANNING_REQUIRED,
            confidence=0.85,
            needs_clarification=False,
            safety_posture=ResponseSafetyPosture.CAUTION,
            memory_lookup_recommended=memory_lookup_recommended,
            tool_planning_recommended=True,
            spoken_style=request.policy.spoken_style,
            reasons=reasons,
        )

    def _record_decision(self, decision: ResponsePlanningDecision) -> None:
        with self._lock:
            self._planned_count += 1
            self._last_request_id = decision.request_id
            self._last_intent = decision.intent
            self._last_answer_mode = decision.answer_mode
            self._last_error = None

            if decision.answer_mode == ResponseAnswerMode.DIRECT:
                self._direct_count += 1

            elif decision.answer_mode == ResponseAnswerMode.ASK_CLARIFICATION:
                self._clarification_count += 1

            elif decision.answer_mode == ResponseAnswerMode.SAFE_REFUSAL:
                self._refusal_count += 1

            elif decision.answer_mode == ResponseAnswerMode.TOOL_PLANNING:
                self._tool_planning_count += 1

            if decision.memory_lookup_recommended:
                self._memory_recommended_count += 1

    def _intent_for(
        self,
        request: CognitionRequest,
        normalized_text: str,
    ) -> ResponseIntent:
        if self._needs_clarification(normalized_text):
            return ResponseIntent.CLARIFICATION_NEEDED

        if self._contains_any(normalized_text, self._config.refuse_phrases):
            return ResponseIntent.COMMAND

        if normalized_text in {"hello", "hi", "hey", "hello jarvis"}:
            return ResponseIntent.GREETING

        if normalized_text.endswith("?") or normalized_text.startswith(
            ("what ", "why ", "how ", "when ", "where ", "who ")
        ):
            return ResponseIntent.QUESTION

        if normalized_text.startswith(("explain ", "describe ", "teach ")):
            return ResponseIntent.EXPLANATION

        if (
            request.policy.allow_tools
            and self._contains_any(normalized_text, self._config.tool_phrases)
        ):
            return ResponseIntent.TOOL_ACTION

        if self._contains_any(normalized_text, self._config.tool_phrases):
            return ResponseIntent.COMMAND

        if self._contains_any(normalized_text, self._config.status_phrases):
            return ResponseIntent.STATUS

        if self._contains_any(normalized_text, self._config.memory_phrases):
            return ResponseIntent.MEMORY

        return ResponseIntent.UNKNOWN

    def _safety_posture_for(
        self,
        normalized_text: str,
    ) -> ResponseSafetyPosture:
        if self._contains_any(normalized_text, self._config.refuse_phrases):
            return ResponseSafetyPosture.REFUSE

        if self._contains_any(normalized_text, self._config.tool_phrases):
            return ResponseSafetyPosture.CAUTION

        return ResponseSafetyPosture.NORMAL

    def _needs_clarification(self, normalized_text: str) -> bool:
        if len(normalized_text) < self._config.clarification_min_chars:
            return True

        return normalized_text in {
            "what",
            "why",
            "how",
            "this",
            "that",
            "do it",
            "explain",
        }

    @staticmethod
    def _contains_any(
        text: str,
        phrases: tuple[str, ...],
    ) -> bool:
        return any(phrase.casefold() in text for phrase in phrases)

    @staticmethod
    def _normalize(text: str) -> str:
        return " ".join(text.casefold().split())