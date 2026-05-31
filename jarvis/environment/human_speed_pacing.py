from __future__ import annotations

from enum import StrEnum
from math import sqrt
from threading import RLock
from typing import Any
from uuid import uuid4

from pydantic import Field, field_validator, model_validator

from jarvis.environment.human_collaboration import (
    ProgressNarration,
)
from jarvis.environment.models import EnvironmentSource, ScreenPoint, TrustCalibration
from jarvis.environment.physical_input import MouseButton, NaturalMotionPoint
from jarvis.environment.trust_runtime import TrustPolicyClassification
from jarvis.orchestration.ids import utc_now
from jarvis.orchestration.models import OrchestrationModel


class HumanSpeedActionKind(StrEnum):
    MOUSE_MOVE = "mouse_move"
    MOUSE_CLICK = "mouse_click"
    KEYBOARD_TYPE = "keyboard_type"
    KEYBOARD_SHORTCUT = "keyboard_shortcut"
    SPEECH = "speech"
    VERIFY = "verify"
    WAIT = "wait"


class HumanSpeedStatus(StrEnum):
    PLANNED = "planned"
    OPTIMIZED = "optimized"
    SYNC_READY = "sync_ready"
    NATURAL = "natural"
    BLOCKED = "blocked"
    FAILED = "failed"


class HumanSpeedDecision(StrEnum):
    EXECUTE_WITH_PACING = "execute_with_pacing"
    SLOW_DOWN = "slow_down"
    WAIT_FOR_SPEECH = "wait_for_speech"
    WAIT_FOR_ACTION = "wait_for_action"
    BLOCK = "block"


class HumanSpeedReason(StrEnum):
    SESSION_CREATED = "session_created"
    TIMING_MODEL_CREATED = "timing_model_created"
    MOTION_CURVE_CREATED = "motion_curve_created"
    SPEECH_ACTION_SYNC_CREATED = "speech_action_sync_created"
    PACING_OPTIMIZED = "pacing_optimized"
    NATURALNESS_ACCEPTED = "naturalness_accepted"
    ROBOTIC_PACING_BLOCKED = "robotic_pacing_blocked"
    SESSION_NOT_FOUND = "session_not_found"
    RUNTIME_RESET = "runtime_reset"


class HumanSpeedEventKind(StrEnum):
    SESSION_CREATED = "session_created"
    TIMING_PLANNED = "timing_planned"
    MOTION_PLANNED = "motion_planned"
    SPEECH_SYNC_PLANNED = "speech_sync_planned"
    PACING_OPTIMIZED = "pacing_optimized"
    NATURALNESS_EVALUATED = "naturalness_evaluated"
    OPERATION_BLOCKED = "operation_blocked"
    RUNTIME_RESET = "runtime_reset"


class HumanPacingMode(StrEnum):
    CAREFUL = "careful"
    NORMAL = "normal"
    FAST_BUT_NATURAL = "fast_but_natural"
    DEBUG_VISIBLE = "debug_visible"


class MotionCurveKind(StrEnum):
    EASE_IN_OUT = "ease_in_out"
    HUMAN_ARC = "human_arc"
    SMALL_ADJUSTMENT = "small_adjustment"


class SpeechActionSyncKind(StrEnum):
    SPEAK_BEFORE_ACTION = "speak_before_action"
    SPEAK_DURING_WAIT = "speak_during_wait"
    SPEAK_AFTER_ACTION = "speak_after_action"
    NO_SPEECH_NEEDED = "no_speech_needed"


class NaturalnessLevel(StrEnum):
    ROBOTIC = "robotic"
    ACCEPTABLE = "acceptable"
    NATURAL = "natural"
    EXCELLENT = "excellent"


class HumanTimingModel(OrchestrationModel):
    """
    Human-scale timing budget.

    This does not slow JARVIS randomly. It coordinates actions so visible
    automation feels understandable, interruptible, and human-paced.
    """

    model_id: str = Field(default_factory=lambda: f"human_timing_{uuid4().hex}")
    mode: HumanPacingMode = HumanPacingMode.NORMAL
    min_action_gap_ms: int = Field(default=120, ge=0, le=5000)
    min_click_hold_ms: int = Field(default=55, ge=20, le=1000)
    min_pre_action_notice_ms: int = Field(default=250, ge=0, le=5000)
    min_post_action_verify_ms: int = Field(default=180, ge=0, le=5000)
    typing_chars_per_second: float = Field(default=9.0, gt=0.0, le=50.0)
    speech_words_per_minute: int = Field(default=150, ge=80, le=240)
    max_robotic_speed_score: float = Field(default=0.72, ge=0.0, le=1.0)
    created_at: object = Field(default_factory=utc_now)

    @field_validator("model_id")
    @classmethod
    def _required_id(cls, value: str) -> str:
        return _clean_required(value)


class MotionCurvePolicy(OrchestrationModel):
    policy_id: str = Field(default_factory=lambda: f"motion_curve_{uuid4().hex}")
    curve_kind: MotionCurveKind = MotionCurveKind.HUMAN_ARC
    min_duration_ms: int = Field(default=180, ge=0, le=10000)
    max_duration_ms: int = Field(default=1200, ge=1, le=20000)
    samples: int = Field(default=8, ge=2, le=100)
    forbid_teleport: bool = True
    allow_micro_jitter: bool = True
    created_at: object = Field(default_factory=utc_now)

    @field_validator("policy_id")
    @classmethod
    def _required_id(cls, value: str) -> str:
        return _clean_required(value)

    @model_validator(mode="after")
    def _duration_order(self) -> MotionCurvePolicy:
        if self.min_duration_ms > self.max_duration_ms:
            raise ValueError("min_duration_ms cannot exceed max_duration_ms.")

        return self


class HumanSpeedActionRequest(OrchestrationModel):
    request_id: str = Field(default_factory=lambda: f"human_speed_req_{uuid4().hex}")
    session_id: str
    action_kind: HumanSpeedActionKind
    description: str
    start_point: ScreenPoint | None = None
    target_point: ScreenPoint | None = None
    mouse_button: MouseButton | None = None
    text: str | None = None
    narration: ProgressNarration | None = None
    requires_user_visibility: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("request_id", "session_id", "description")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class HumanTimingPlan(OrchestrationModel):
    plan_id: str = Field(default_factory=lambda: f"human_timing_plan_{uuid4().hex}")
    action_kind: HumanSpeedActionKind
    pre_action_delay_ms: int = Field(ge=0)
    action_duration_ms: int = Field(ge=0)
    post_action_delay_ms: int = Field(ge=0)
    total_duration_ms: int = Field(ge=0)
    reason: str
    model: HumanTimingModel
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("plan_id", "reason")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)

    @model_validator(mode="after")
    def _total_matches_parts(self) -> HumanTimingPlan:
        expected = (
            self.pre_action_delay_ms
            + self.action_duration_ms
            + self.post_action_delay_ms
        )
        if self.total_duration_ms != expected:
            raise ValueError("total_duration_ms must match timing parts.")

        return self


class MotionCurvePlan(OrchestrationModel):
    curve_id: str = Field(default_factory=lambda: f"motion_curve_plan_{uuid4().hex}")
    start_point: ScreenPoint
    target_point: ScreenPoint
    points: tuple[NaturalMotionPoint, ...]
    duration_ms: int = Field(ge=0)
    curve_kind: MotionCurveKind
    teleport_detected: bool = False
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("curve_id")
    @classmethod
    def _required_id(cls, value: str) -> str:
        return _clean_required(value)

    @model_validator(mode="after")
    def _requires_points(self) -> MotionCurvePlan:
        if len(self.points) < 2:
            raise ValueError("motion curve requires at least two points.")

        if self.teleport_detected:
            raise ValueError("motion curve cannot contain teleport movement.")

        return self


class SpeechActionSync(OrchestrationModel):
    sync_id: str = Field(default_factory=lambda: f"speech_action_sync_{uuid4().hex}")
    kind: SpeechActionSyncKind
    speech_text: str | None = None
    speech_duration_ms: int = Field(default=0, ge=0)
    action_start_offset_ms: int = Field(default=0, ge=0)
    action_should_wait_for_speech: bool = False
    speech_interruptible: bool = True
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("sync_id")
    @classmethod
    def _required_id(cls, value: str) -> str:
        return _clean_required(value)


class PacingPlan(OrchestrationModel):
    pacing_id: str = Field(default_factory=lambda: f"pacing_plan_{uuid4().hex}")
    request: HumanSpeedActionRequest
    timing: HumanTimingPlan
    motion_curve: MotionCurvePlan | None = None
    speech_sync: SpeechActionSync | None = None
    decision: HumanSpeedDecision
    status: HumanSpeedStatus
    reason: HumanSpeedReason
    naturalness_score: float = Field(ge=0.0, le=1.0)
    robotic_speed_score: float = Field(ge=0.0, le=1.0)
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("pacing_id")
    @classmethod
    def _required_id(cls, value: str) -> str:
        return _clean_required(value)

    @model_validator(mode="after")
    def _mouse_move_requires_motion_curve(self) -> PacingPlan:
        if (
            self.request.action_kind == HumanSpeedActionKind.MOUSE_MOVE
            and self.motion_curve is None
        ):
            raise ValueError("mouse movement pacing requires motion curve.")

        return self


class NaturalnessEvaluation(OrchestrationModel):
    evaluation_id: str = Field(
        default_factory=lambda: f"naturalness_eval_{uuid4().hex}"
    )
    level: NaturalnessLevel
    accepted: bool
    naturalness_score: float = Field(ge=0.0, le=1.0)
    robotic_speed_score: float = Field(ge=0.0, le=1.0)
    reason: str
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("evaluation_id", "reason")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class HumanSpeedPacingResult(OrchestrationModel):
    result_id: str = Field(default_factory=lambda: f"human_speed_result_{uuid4().hex}")
    status: HumanSpeedStatus
    decision: HumanSpeedDecision
    reason: HumanSpeedReason
    pacing_plan: PacingPlan | None = None
    evaluation: NaturalnessEvaluation | None = None
    trust: TrustCalibration
    message: str
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("result_id", "message")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)

    @model_validator(mode="after")
    def _optimized_requires_plan(self) -> HumanSpeedPacingResult:
        if self.status in {
            HumanSpeedStatus.OPTIMIZED,
            HumanSpeedStatus.SYNC_READY,
            HumanSpeedStatus.NATURAL,
        } and self.pacing_plan is None:
            raise ValueError("optimized human-speed result requires pacing plan.")

        return self


class HumanSpeedPacingSession(OrchestrationModel):
    session_id: str = Field(default_factory=lambda: f"human_speed_{uuid4().hex}")
    workspace_id: str
    optimized_count: int = Field(default=0, ge=0)
    natural_count: int = Field(default=0, ge=0)
    blocked_count: int = Field(default=0, ge=0)
    speech_sync_count: int = Field(default=0, ge=0)
    motion_plan_count: int = Field(default=0, ge=0)
    created_at: object = Field(default_factory=utc_now)
    updated_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("session_id", "workspace_id")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class HumanSpeedPacingRuntimeEvent(OrchestrationModel):
    event_id: str = Field(default_factory=lambda: f"human_speed_event_{uuid4().hex}")
    kind: HumanSpeedEventKind
    reason: HumanSpeedReason
    session_id: str | None = None
    result_id: str | None = None
    pacing_id: str | None = None
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("event_id")
    @classmethod
    def _required_id(cls, value: str) -> str:
        return _clean_required(value)


class HumanSpeedPacingRuntimeSnapshot(OrchestrationModel):
    name: str
    session_count: int = Field(ge=0)
    result_count: int = Field(ge=0)
    optimized_count: int = Field(ge=0)
    natural_count: int = Field(ge=0)
    blocked_count: int = Field(ge=0)
    speech_sync_count: int = Field(ge=0)
    motion_plan_count: int = Field(ge=0)
    runtime_event_count: int = Field(ge=0)
    last_reason: HumanSpeedReason | None = None
    created_at: object = Field(default_factory=utc_now)

    @field_validator("name")
    @classmethod
    def _required_name(cls, value: str) -> str:
        return _clean_required(value)


class PacingOptimizer:
    def optimize(
        self,
        *,
        request: HumanSpeedActionRequest,
        model: HumanTimingModel,
        motion_policy: MotionCurvePolicy,
    ) -> PacingPlan:
        timing = _timing_for(request=request, model=model)
        motion = None

        if request.action_kind == HumanSpeedActionKind.MOUSE_MOVE:
            motion = _motion_curve_for(
                request=request,
                timing=timing,
                policy=motion_policy,
            )

        speech_sync = _speech_sync_for(request=request, model=model, timing=timing)
        robotic = _robotic_score(timing)
        naturalness = max(0.0, min(1.0, 1.0 - robotic + 0.10))

        decision = HumanSpeedDecision.EXECUTE_WITH_PACING
        status = HumanSpeedStatus.OPTIMIZED
        reason = HumanSpeedReason.PACING_OPTIMIZED

        if robotic > model.max_robotic_speed_score:
            decision = HumanSpeedDecision.SLOW_DOWN
            reason = HumanSpeedReason.ROBOTIC_PACING_BLOCKED
            status = HumanSpeedStatus.BLOCKED

        if speech_sync is not None:
            status = HumanSpeedStatus.SYNC_READY
            reason = HumanSpeedReason.SPEECH_ACTION_SYNC_CREATED

        return PacingPlan(
            request=request,
            timing=timing,
            motion_curve=motion,
            speech_sync=speech_sync,
            decision=decision,
            status=status,
            reason=reason,
            naturalness_score=naturalness,
            robotic_speed_score=robotic,
        )


class InteractionNaturalnessRuntime:
    def evaluate(
        self,
        *,
        pacing_plan: PacingPlan,
        model: HumanTimingModel,
    ) -> NaturalnessEvaluation:
        score = pacing_plan.naturalness_score
        robotic = pacing_plan.robotic_speed_score
        accepted = robotic <= model.max_robotic_speed_score

        level = NaturalnessLevel.ROBOTIC
        if score >= 0.90:
            level = NaturalnessLevel.EXCELLENT
        elif score >= 0.75:
            level = NaturalnessLevel.NATURAL
        elif score >= 0.55:
            level = NaturalnessLevel.ACCEPTABLE

        return NaturalnessEvaluation(
            level=level,
            accepted=accepted,
            naturalness_score=score,
            robotic_speed_score=robotic,
            reason=(
                "human-paced interaction accepted"
                if accepted
                else "interaction pacing is too robotic"
            ),
        )


class HumanSpeedInteractionRuntime:
    """
    Phase 8 Step 37 Human-Speed Interaction & Pacing Optimizer.

    It coordinates:
    - visible action timing
    - mouse movement curve shape
    - speech/action synchronization
    - naturalness scoring
    - robotic speed blocking

    It does not execute physical input. It produces pacing plans for Step 25
    and collaboration sync for Step 36.
    """

    def __init__(
        self,
        *,
        name: str = "human_speed_interaction_runtime",
        timing_model: HumanTimingModel | None = None,
        motion_policy: MotionCurvePolicy | None = None,
        optimizer: PacingOptimizer | None = None,
        naturalness_runtime: InteractionNaturalnessRuntime | None = None,
    ) -> None:
        cleaned = name.strip()
        if not cleaned:
            raise ValueError("name cannot be empty.")

        self._name = cleaned
        self._timing_model = timing_model or HumanTimingModel()
        self._motion_policy = motion_policy or MotionCurvePolicy()
        self._optimizer = optimizer or PacingOptimizer()
        self._naturalness_runtime = (
            naturalness_runtime or InteractionNaturalnessRuntime()
        )
        self._sessions: dict[str, HumanSpeedPacingSession] = {}
        self._results: list[HumanSpeedPacingResult] = []
        self._events: list[HumanSpeedPacingRuntimeEvent] = []
        self._lock = RLock()
        self._last_reason: HumanSpeedReason | None = None

    @property
    def name(self) -> str:
        return self._name

    def create_session(
        self,
        *,
        workspace_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> HumanSpeedPacingSession:
        session = HumanSpeedPacingSession(
            workspace_id=workspace_id,
            metadata=metadata or {},
        )
        event = self._event(
            kind=HumanSpeedEventKind.SESSION_CREATED,
            reason=HumanSpeedReason.SESSION_CREATED,
            session_id=session.session_id,
        )

        with self._lock:
            self._sessions[session.session_id] = session
            self._events.append(event)
            self._last_reason = event.reason

        return session

    def optimize(
        self,
        request: HumanSpeedActionRequest,
    ) -> HumanSpeedPacingResult:
        if self.session_for(request.session_id) is None:
            result = _blocked_result(
                reason=HumanSpeedReason.SESSION_NOT_FOUND,
                message="human-speed pacing session not found",
            )
            self._record_result(result, request.session_id)
            return result

        plan = self._optimizer.optimize(
            request=request,
            model=self._timing_model,
            motion_policy=self._motion_policy,
        )
        evaluation = self._naturalness_runtime.evaluate(
            pacing_plan=plan,
            model=self._timing_model,
        )

        status = plan.status
        decision = plan.decision
        reason = plan.reason

        if not evaluation.accepted:
            status = HumanSpeedStatus.BLOCKED
            decision = HumanSpeedDecision.SLOW_DOWN
            reason = HumanSpeedReason.ROBOTIC_PACING_BLOCKED

        elif evaluation.level in {
            NaturalnessLevel.NATURAL,
            NaturalnessLevel.EXCELLENT,
        }:
            status = HumanSpeedStatus.NATURAL
            reason = HumanSpeedReason.NATURALNESS_ACCEPTED

        result = HumanSpeedPacingResult(
            status=status,
            decision=decision,
            reason=reason,
            pacing_plan=plan,
            evaluation=evaluation,
            trust=_trust(
                confidence=0.88 if evaluation.accepted else 0.35,
                reason=evaluation.reason,
            ),
            message=evaluation.reason,
        )
        self._record_result(result, request.session_id)

        return result

    def session_for(
        self,
        session_id: str,
    ) -> HumanSpeedPacingSession | None:
        with self._lock:
            return self._sessions.get(session_id)

    def results(self) -> tuple[HumanSpeedPacingResult, ...]:
        with self._lock:
            return tuple(self._results)

    def events(self) -> tuple[HumanSpeedPacingRuntimeEvent, ...]:
        with self._lock:
            return tuple(self._events)

    def snapshot(self) -> HumanSpeedPacingRuntimeSnapshot:
        with self._lock:
            return HumanSpeedPacingRuntimeSnapshot(
                name=self.name,
                session_count=len(self._sessions),
                result_count=len(self._results),
                optimized_count=sum(
                    1
                    for result in self._results
                    if result.status
                    in {
                        HumanSpeedStatus.OPTIMIZED,
                        HumanSpeedStatus.SYNC_READY,
                        HumanSpeedStatus.NATURAL,
                    }
                ),
                natural_count=sum(
                    1
                    for result in self._results
                    if result.status == HumanSpeedStatus.NATURAL
                ),
                blocked_count=sum(
                    1
                    for result in self._results
                    if result.status
                    in {
                        HumanSpeedStatus.BLOCKED,
                        HumanSpeedStatus.FAILED,
                    }
                ),
                speech_sync_count=sum(
                    1
                    for result in self._results
                    if result.pacing_plan is not None
                    and result.pacing_plan.speech_sync is not None
                ),
                motion_plan_count=sum(
                    1
                    for result in self._results
                    if result.pacing_plan is not None
                    and result.pacing_plan.motion_curve is not None
                ),
                runtime_event_count=len(self._events),
                last_reason=self._last_reason,
            )

    def reset(self) -> None:
        event = self._event(
            kind=HumanSpeedEventKind.RUNTIME_RESET,
            reason=HumanSpeedReason.RUNTIME_RESET,
        )

        with self._lock:
            self._sessions.clear()
            self._results.clear()
            self._events.clear()
            self._events.append(event)
            self._last_reason = event.reason

    def _record_result(
        self,
        result: HumanSpeedPacingResult,
        session_id: str,
    ) -> None:
        event = self._event(
            kind=_event_kind_for(result),
            reason=result.reason,
            session_id=session_id,
            result_id=result.result_id,
            pacing_id=(
                result.pacing_plan.pacing_id
                if result.pacing_plan is not None
                else None
            ),
            metadata={"status": result.status.value},
        )

        with self._lock:
            self._results.append(result)
            self._events.append(event)
            self._last_reason = result.reason

            session = self._sessions.get(session_id)
            if session is not None:
                self._sessions[session_id] = session.model_copy(
                    update={
                        "updated_at": utc_now(),
                        "optimized_count": session.optimized_count
                        + (
                            1
                            if result.status
                            in {
                                HumanSpeedStatus.OPTIMIZED,
                                HumanSpeedStatus.SYNC_READY,
                                HumanSpeedStatus.NATURAL,
                            }
                            else 0
                        ),
                        "natural_count": session.natural_count
                        + (
                            1
                            if result.status == HumanSpeedStatus.NATURAL
                            else 0
                        ),
                        "blocked_count": session.blocked_count
                        + (
                            1
                            if result.status
                            in {
                                HumanSpeedStatus.BLOCKED,
                                HumanSpeedStatus.FAILED,
                            }
                            else 0
                        ),
                        "speech_sync_count": session.speech_sync_count
                        + (
                            1
                            if result.pacing_plan is not None
                            and result.pacing_plan.speech_sync is not None
                            else 0
                        ),
                        "motion_plan_count": session.motion_plan_count
                        + (
                            1
                            if result.pacing_plan is not None
                            and result.pacing_plan.motion_curve is not None
                            else 0
                        ),
                    }
                )

    @staticmethod
    def _event(
        *,
        kind: HumanSpeedEventKind,
        reason: HumanSpeedReason,
        session_id: str | None = None,
        result_id: str | None = None,
        pacing_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> HumanSpeedPacingRuntimeEvent:
        return HumanSpeedPacingRuntimeEvent(
            kind=kind,
            reason=reason,
            session_id=session_id,
            result_id=result_id,
            pacing_id=pacing_id,
            metadata=metadata or {},
        )


def _timing_for(
    *,
    request: HumanSpeedActionRequest,
    model: HumanTimingModel,
) -> HumanTimingPlan:
    pre = model.min_action_gap_ms
    post = model.min_post_action_verify_ms
    action = model.min_action_gap_ms

    if request.requires_user_visibility:
        pre = max(pre, model.min_pre_action_notice_ms)

    if request.action_kind == HumanSpeedActionKind.MOUSE_MOVE:
        distance = _distance(request.start_point, request.target_point)
        action = int(min(1200, max(180, distance * 1.3)))

    elif request.action_kind == HumanSpeedActionKind.MOUSE_CLICK:
        action = model.min_click_hold_ms

    elif request.action_kind == HumanSpeedActionKind.KEYBOARD_TYPE:
        text_len = len(request.text or "")
        action = int((text_len / model.typing_chars_per_second) * 1000)

    elif request.action_kind == HumanSpeedActionKind.SPEECH:
        action = _speech_duration_ms(request.text or request.description, model)

    elif request.action_kind == HumanSpeedActionKind.VERIFY:
        action = model.min_post_action_verify_ms
        post = 0

    total = pre + action + post

    return HumanTimingPlan(
        action_kind=request.action_kind,
        pre_action_delay_ms=pre,
        action_duration_ms=action,
        post_action_delay_ms=post,
        total_duration_ms=total,
        reason="human timing model created for visible action",
        model=model,
    )


def _motion_curve_for(
    *,
    request: HumanSpeedActionRequest,
    timing: HumanTimingPlan,
    policy: MotionCurvePolicy,
) -> MotionCurvePlan:
    if request.start_point is None or request.target_point is None:
        raise ValueError("mouse movement requires start_point and target_point.")

    distance = _distance(request.start_point, request.target_point)
    duration = int(min(policy.max_duration_ms, max(policy.min_duration_ms, distance)))

    points: list[NaturalMotionPoint] = []
    for index in range(policy.samples):
        t = index / (policy.samples - 1)
        eased = _ease_in_out(t)
        arc = 0.0
        if policy.curve_kind == MotionCurveKind.HUMAN_ARC:
            arc = 8.0 * (1.0 - (2.0 * t - 1.0) ** 2)

        x = request.start_point.x + (
            request.target_point.x - request.start_point.x
        ) * eased
        y = (
            request.start_point.y
            + (request.target_point.y - request.start_point.y) * eased
            - arc
        )
        points.append(
            NaturalMotionPoint(
                x=int(round(x)),
                y=int(round(y)),
            )
        )

    teleport = False
    if policy.forbid_teleport:
        teleport = _has_teleport(points)

    return MotionCurvePlan(
        start_point=request.start_point,
        target_point=request.target_point,
        points=tuple(points),
        duration_ms=max(duration, timing.action_duration_ms),
        curve_kind=policy.curve_kind,
        teleport_detected=teleport,
    )


def _speech_sync_for(
    *,
    request: HumanSpeedActionRequest,
    model: HumanTimingModel,
    timing: HumanTimingPlan,
) -> SpeechActionSync | None:
    narration = request.narration
    if narration is None:
        return None

    duration = _speech_duration_ms(narration.text, model)

    if request.action_kind in {
        HumanSpeedActionKind.MOUSE_CLICK,
        HumanSpeedActionKind.KEYBOARD_TYPE,
    }:
        return SpeechActionSync(
            kind=SpeechActionSyncKind.SPEAK_BEFORE_ACTION,
            speech_text=narration.text,
            speech_duration_ms=duration,
            action_start_offset_ms=duration + model.min_action_gap_ms,
            action_should_wait_for_speech=True,
            speech_interruptible=narration.interruptible,
            metadata={"phase": narration.phase.value},
        )

    if request.action_kind in {
        HumanSpeedActionKind.VERIFY,
        HumanSpeedActionKind.WAIT,
    }:
        return SpeechActionSync(
            kind=SpeechActionSyncKind.SPEAK_DURING_WAIT,
            speech_text=narration.text,
            speech_duration_ms=duration,
            action_start_offset_ms=0,
            action_should_wait_for_speech=False,
            speech_interruptible=narration.interruptible,
            metadata={"phase": narration.phase.value},
        )

    return SpeechActionSync(
        kind=SpeechActionSyncKind.SPEAK_AFTER_ACTION,
        speech_text=narration.text,
        speech_duration_ms=duration,
        action_start_offset_ms=timing.action_duration_ms,
        action_should_wait_for_speech=False,
        speech_interruptible=narration.interruptible,
        metadata={"phase": narration.phase.value},
    )


def _speech_duration_ms(text: str, model: HumanTimingModel) -> int:
    words = max(1, len(text.split()))
    minutes = words / model.speech_words_per_minute
    return int(max(500, minutes * 60_000))


def _robotic_score(timing: HumanTimingPlan) -> float:
    if timing.total_duration_ms <= 0:
        return 1.0

    if timing.total_duration_ms < 180:
        return 0.95

    if timing.total_duration_ms < 350:
        return 0.75

    if timing.total_duration_ms < 700:
        return 0.45

    return 0.25


def _distance(start: ScreenPoint | None, target: ScreenPoint | None) -> float:
    if start is None or target is None:
        return 0.0

    return sqrt((target.x - start.x) ** 2 + (target.y - start.y) ** 2)


def _ease_in_out(value: float) -> float:
    return value * value * (3.0 - 2.0 * value)


def _has_teleport(
    points: tuple[NaturalMotionPoint, ...] | list[NaturalMotionPoint]
) -> bool:
    if len(points) < 2:
        return False

    previous = points[0]
    for point in points[1:]:
        if sqrt((point.x - previous.x) ** 2 + (point.y - previous.y) ** 2) > 800:
            return True
        previous = point

    return False


def _blocked_result(
    *,
    reason: HumanSpeedReason,
    message: str,
) -> HumanSpeedPacingResult:
    return HumanSpeedPacingResult(
        status=(
            HumanSpeedStatus.FAILED
            if reason == HumanSpeedReason.SESSION_NOT_FOUND
            else HumanSpeedStatus.BLOCKED
        ),
        decision=HumanSpeedDecision.BLOCK,
        reason=reason,
        trust=_trust(confidence=0.20, reason=message),
        message=message,
    )


def _event_kind_for(result: HumanSpeedPacingResult) -> HumanSpeedEventKind:
    if result.status == HumanSpeedStatus.NATURAL:
        return HumanSpeedEventKind.NATURALNESS_EVALUATED

    if result.reason == HumanSpeedReason.SPEECH_ACTION_SYNC_CREATED:
        return HumanSpeedEventKind.SPEECH_SYNC_PLANNED

    if (
        result.pacing_plan is not None
        and result.pacing_plan.motion_curve is not None
    ):
        return HumanSpeedEventKind.MOTION_PLANNED

    if result.status in {
        HumanSpeedStatus.BLOCKED,
        HumanSpeedStatus.FAILED,
    }:
        return HumanSpeedEventKind.OPERATION_BLOCKED

    return HumanSpeedEventKind.PACING_OPTIMIZED


def _trust(
    *,
    confidence: float,
    reason: str,
) -> TrustCalibration:
    return TrustCalibration(
        confidence=confidence,
        stability=max(0.0, min(1.0, confidence + 0.05)),
        ambiguity=1.0 - confidence,
        source=EnvironmentSource.OS_OBSERVER,
        reason=reason,
        metadata={"policy": TrustPolicyClassification.SAFE.value},
    )


def _clean_required(value: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError("field cannot be empty.")
    return cleaned