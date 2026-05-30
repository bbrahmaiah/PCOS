from __future__ import annotations

from enum import StrEnum
from threading import RLock
from typing import Any
from uuid import uuid4

from pydantic import Field, field_validator, model_validator

from jarvis.environment.interaction_policy import (
    InteractionDecision,
    InteractionPolicyResult,
    InteractionVerificationRequirement,
    PhysicalInteractionKind,
)
from jarvis.environment.models import EnvironmentSource, ScreenRegion, TrustCalibration
from jarvis.environment.trust_runtime import TrustPolicyClassification
from jarvis.orchestration.ids import utc_now
from jarvis.orchestration.models import OrchestrationModel


class PhysicalInputKind(StrEnum):
    MOUSE_MOVE = "mouse_move"
    MOUSE_CLICK = "mouse_click"
    MOUSE_DOUBLE_CLICK = "mouse_double_click"
    KEYBOARD_TYPE = "keyboard_type"
    KEYBOARD_SHORTCUT = "keyboard_shortcut"


class PhysicalInputStatus(StrEnum):
    READY = "ready"
    VERIFIED = "verified"
    EXECUTED = "executed"
    CANCELLED = "cancelled"
    PAUSED_BY_USER = "paused_by_user"
    BLOCKED = "blocked"
    FAILED = "failed"


class PhysicalInputDecision(StrEnum):
    ALLOW = "allow"
    REQUIRE_PRECLICK_VERIFICATION = "require_preclick_verification"
    REQUIRE_FOCUS_VERIFICATION = "require_focus_verification"
    PAUSE_FOR_USER = "pause_for_user"
    CANCEL = "cancel"
    BLOCK = "block"


class PhysicalInputReason(StrEnum):
    SESSION_CREATED = "session_created"
    REQUEST_ACCEPTED = "request_accepted"
    POLICY_NOT_ELIGIBLE = "policy_not_eligible"
    PRECLICK_VERIFIED = "preclick_verified"
    PRECLICK_FAILED = "preclick_failed"
    FOCUS_UNCERTAIN = "focus_uncertain"
    UNKNOWN_FIELD = "unknown_field"
    HUMAN_OVERRIDE_DETECTED = "human_override_detected"
    STOP_CANCELLED = "stop_cancelled"
    NATURAL_MOTION_PLANNED = "natural_motion_planned"
    MOUSE_ACTION_EXECUTED = "mouse_action_executed"
    KEYBOARD_ACTION_EXECUTED = "keyboard_action_executed"
    SESSION_NOT_FOUND = "session_not_found"
    RUNTIME_RESET = "runtime_reset"


class PhysicalInputEventKind(StrEnum):
    SESSION_CREATED = "session_created"
    REQUEST_EVALUATED = "request_evaluated"
    MOTION_PLANNED = "motion_planned"
    ACTION_EXECUTED = "action_executed"
    ACTION_CANCELLED = "action_cancelled"
    ACTION_BLOCKED = "action_blocked"
    RUNTIME_RESET = "runtime_reset"


class MouseButton(StrEnum):
    LEFT = "left"
    RIGHT = "right"
    MIDDLE = "middle"


class KeyboardShortcutKind(StrEnum):
    COPY = "copy"
    PASTE = "paste"
    SELECT_ALL = "select_all"
    SAVE = "save"
    ENTER = "enter"
    ESCAPE = "escape"
    TAB = "tab"
    CUSTOM = "custom"


class HumanOverrideKind(StrEnum):
    NONE = "none"
    USER_MOUSE_MOVED = "user_mouse_moved"
    USER_KEYBOARD_INPUT = "user_keyboard_input"
    USER_STOP_COMMAND = "user_stop_command"
    FOCUS_CHANGED_EXTERNALLY = "focus_changed_externally"


class NaturalMotionPoint(OrchestrationModel):
    """
    A single point in a human-like mouse movement path.
    """

    point_id: str = Field(default_factory=lambda: f"motion_point_{uuid4().hex}")
    x: int
    y: int
    duration_ms: int = Field(default=16, ge=1, le=500)
    created_at: object = Field(default_factory=utc_now)

    @field_validator("point_id")
    @classmethod
    def _required_id(cls, value: str) -> str:
        return _clean_required(value)


class NaturalMotionPlan(OrchestrationModel):
    """
    Human-speed motion plan.

    No teleport clicking. Mouse movement must have a path.
    """

    plan_id: str = Field(default_factory=lambda: f"motion_plan_{uuid4().hex}")
    start_x: int
    start_y: int
    end_x: int
    end_y: int
    points: tuple[NaturalMotionPoint, ...]
    total_duration_ms: int = Field(ge=1)
    human_speed: bool = True
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("plan_id")
    @classmethod
    def _required_id(cls, value: str) -> str:
        return _clean_required(value)

    @model_validator(mode="after")
    def _no_teleport_clicking(self) -> NaturalMotionPlan:
        if not self.points:
            raise ValueError("natural motion plan requires movement points.")

        if self.points[-1].x != self.end_x or self.points[-1].y != self.end_y:
            raise ValueError("natural motion plan must end at target.")

        return self


class HumanTimingPolicy(OrchestrationModel):
    """
    Timing policy for human-like physical interaction.
    """

    min_move_duration_ms: int = Field(default=120, ge=1, le=5000)
    max_move_duration_ms: int = Field(default=1200, ge=1, le=10000)
    min_click_settle_ms: int = Field(default=80, ge=0, le=2000)
    min_key_interval_ms: int = Field(default=25, ge=1, le=1000)
    max_key_interval_ms: int = Field(default=120, ge=1, le=2000)
    pause_after_user_override_ms: int = Field(default=1500, ge=0, le=10000)

    @model_validator(mode="after")
    def _valid_ranges(self) -> HumanTimingPolicy:
        if self.max_move_duration_ms < self.min_move_duration_ms:
            raise ValueError("max_move_duration_ms must be >= min_move_duration_ms.")

        if self.max_key_interval_ms < self.min_key_interval_ms:
            raise ValueError("max_key_interval_ms must be >= min_key_interval_ms.")

        return self


class MouseActionRequest(OrchestrationModel):
    """
    Mouse action request after Step 24 policy eligibility.
    """

    request_id: str = Field(default_factory=lambda: f"mouse_req_{uuid4().hex}")
    session_id: str
    policy_result: InteractionPolicyResult
    target_region: ScreenRegion | None = None
    button: MouseButton = MouseButton.LEFT
    double_click: bool = False
    current_x: int = 0
    current_y: int = 0
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("request_id", "session_id")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class KeyboardActionRequest(OrchestrationModel):
    """
    Keyboard action request after Step 24 policy eligibility.
    """

    request_id: str = Field(default_factory=lambda: f"keyboard_req_{uuid4().hex}")
    session_id: str
    policy_result: InteractionPolicyResult
    text: str | None = None
    shortcut: KeyboardShortcutKind | None = None
    custom_keys: tuple[str, ...] = ()
    focus_known: bool = False
    target_field_known: bool = False
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("request_id", "session_id")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)

    @model_validator(mode="after")
    def _requires_text_or_shortcut(self) -> KeyboardActionRequest:
        if self.text is None and self.shortcut is None and not self.custom_keys:
            raise ValueError("keyboard action requires text, shortcut, or custom keys.")

        return self


class PreClickVerification(OrchestrationModel):
    """
    Verification immediately before clicking.
    """

    verification_id: str = Field(default_factory=lambda: f"preclick_{uuid4().hex}")
    target_known: bool
    target_still_valid: bool
    focus_stable: bool
    region: ScreenRegion | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("verification_id", "reason")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)

    @property
    def passed(self) -> bool:
        return (
            self.target_known
            and self.target_still_valid
            and self.focus_stable
            and self.confidence >= 0.70
        )


class PhysicalOverrideSignal(OrchestrationModel):
    """
    User override signal.

    User always wins over JARVIS.
    """

    signal_id: str = Field(default_factory=lambda: f"override_{uuid4().hex}")
    kind: HumanOverrideKind
    active: bool = False
    reason: str = "no override"
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("signal_id", "reason")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class PhysicalInputResult(OrchestrationModel):
    """
    Result for mouse/keyboard physical runtime.

    Executed means runtime accepted and simulated execution event occurred.
    Real OS effect is not performed in this step.
    """

    result_id: str = Field(default_factory=lambda: f"physical_result_{uuid4().hex}")
    status: PhysicalInputStatus
    decision: PhysicalInputDecision
    reason: PhysicalInputReason
    input_kind: PhysicalInputKind
    policy_result_id: str | None = None
    motion_plan: NaturalMotionPlan | None = None
    preclick: PreClickVerification | None = None
    override: PhysicalOverrideSignal | None = None
    trust: TrustCalibration
    audit_message: str
    executed: bool = False
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("result_id", "audit_message")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)

    @model_validator(mode="after")
    def _executed_requires_policy(self) -> PhysicalInputResult:
        if self.executed and self.policy_result_id is None:
            raise ValueError("executed physical input requires policy_result_id.")

        return self


class PhysicalInputSession(OrchestrationModel):
    """
    Physical input runtime session.
    """

    session_id: str = Field(default_factory=lambda: f"physical_session_{uuid4().hex}")
    workspace_id: str
    paused_by_user: bool = False
    cancelled: bool = False
    created_at: object = Field(default_factory=utc_now)
    updated_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("session_id", "workspace_id")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class PhysicalInputRuntimeEvent(OrchestrationModel):
    """
    Runtime event for Step 25.
    """

    event_id: str = Field(default_factory=lambda: f"physical_event_{uuid4().hex}")
    kind: PhysicalInputEventKind
    reason: PhysicalInputReason
    session_id: str | None = None
    result_id: str | None = None
    request_id: str | None = None
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("event_id")
    @classmethod
    def _required_id(cls, value: str) -> str:
        return _clean_required(value)


class PhysicalInputRuntimeSnapshot(OrchestrationModel):
    """
    Diagnostics for Step 25.
    """

    name: str
    session_count: int = Field(ge=0)
    result_count: int = Field(ge=0)
    executed_count: int = Field(ge=0)
    blocked_count: int = Field(ge=0)
    cancelled_count: int = Field(ge=0)
    paused_by_user_count: int = Field(ge=0)
    motion_plan_count: int = Field(ge=0)
    runtime_event_count: int = Field(ge=0)
    last_reason: PhysicalInputReason | None = None
    created_at: object = Field(default_factory=utc_now)

    @field_validator("name")
    @classmethod
    def _required_name(cls, value: str) -> str:
        return _clean_required(value)


class NaturalMotionEngine:
    """
    Generates human-like motion path.

    No teleport clicking. Every mouse action gets a path.
    """

    def __init__(self, *, timing: HumanTimingPolicy | None = None) -> None:
        self._timing = timing or HumanTimingPolicy()

    def plan(
        self,
        *,
        start_x: int,
        start_y: int,
        target: ScreenRegion,
    ) -> NaturalMotionPlan:
        end_x = target.x + target.width // 2
        end_y = target.y + target.height // 2
        distance = abs(end_x - start_x) + abs(end_y - start_y)
        duration = max(
            self._timing.min_move_duration_ms,
            min(self._timing.max_move_duration_ms, 120 + distance // 2),
        )
        steps = max(3, min(24, duration // 40))
        points: list[NaturalMotionPoint] = []

        for index in range(1, steps + 1):
            ratio = index / steps
            eased = ratio * ratio * (3 - 2 * ratio)
            x = round(start_x + (end_x - start_x) * eased)
            y = round(start_y + (end_y - start_y) * eased)
            points.append(
                NaturalMotionPoint(
                    x=x,
                    y=y,
                    duration_ms=max(1, duration // steps),
                )
            )

        return NaturalMotionPlan(
            start_x=start_x,
            start_y=start_y,
            end_x=end_x,
            end_y=end_y,
            points=tuple(points),
            total_duration_ms=sum(point.duration_ms for point in points),
            human_speed=True,
            metadata={"distance": distance},
        )


class PreClickVerifier:
    """
    Verifies target immediately before click.
    """

    def verify(self, request: MouseActionRequest) -> PreClickVerification:
        policy = request.policy_result
        region = request.target_region

        if region is None:
            return PreClickVerification(
                target_known=False,
                target_still_valid=False,
                focus_stable=False,
                region=None,
                confidence=0.0,
                reason="mouse click target region is unknown",
            )

        target_known = policy.request.contract.target_label is not None
        target_still_valid = policy.decision in {
            InteractionDecision.ELIGIBLE_FOR_EXECUTION,
            InteractionDecision.REQUIRES_VERIFICATION_FIRST,
        }
        focus_stable = (
            policy.verification_requirement
            != InteractionVerificationRequirement.VERIFY_AND_RECONCILE
        )
        confidence = policy.trust.confidence

        return PreClickVerification(
            target_known=target_known,
            target_still_valid=target_still_valid,
            focus_stable=focus_stable,
            region=region,
            confidence=confidence,
            reason="pre-click target verification completed",
        )


class PhysicalOverrideDetector:
    """
    Detects human override.

    User mouse/keyboard/stop command always cancels or pauses JARVIS.
    """

    def __init__(self) -> None:
        self._signal = PhysicalOverrideSignal(kind=HumanOverrideKind.NONE)

    def update(self, signal: PhysicalOverrideSignal) -> None:
        self._signal = signal

    def current(self) -> PhysicalOverrideSignal:
        return self._signal

    def clear(self) -> None:
        self._signal = PhysicalOverrideSignal(kind=HumanOverrideKind.NONE)


class MouseRuntime:
    """
    Governed mouse runtime.

    No raw click happens here; this emits accepted execution result only after:
    - Step 24 policy
    - pre-click verification
    - human override check
    - natural motion planning
    """

    def __init__(
        self,
        *,
        motion_engine: NaturalMotionEngine | None = None,
        verifier: PreClickVerifier | None = None,
        override_detector: PhysicalOverrideDetector | None = None,
    ) -> None:
        self._motion_engine = motion_engine or NaturalMotionEngine()
        self._verifier = verifier or PreClickVerifier()
        self._override_detector = override_detector or PhysicalOverrideDetector()

    def execute(self, request: MouseActionRequest) -> PhysicalInputResult:
        override = self._override_detector.current()
        if override.active:
            return _blocked_result(
                input_kind=PhysicalInputKind.MOUSE_CLICK,
                reason=(
                    PhysicalInputReason.STOP_CANCELLED
                    if override.kind == HumanOverrideKind.USER_STOP_COMMAND
                    else PhysicalInputReason.HUMAN_OVERRIDE_DETECTED
                ),
                decision=(
                    PhysicalInputDecision.CANCEL
                    if override.kind == HumanOverrideKind.USER_STOP_COMMAND
                    else PhysicalInputDecision.PAUSE_FOR_USER
                ),
                audit_message=override.reason,
                override=override,
                policy_result_id=request.policy_result.result_id,
            )

        policy = request.policy_result
        if not _policy_allows_physical_attempt(policy):
            return _blocked_result(
                input_kind=PhysicalInputKind.MOUSE_CLICK,
                reason=PhysicalInputReason.POLICY_NOT_ELIGIBLE,
                decision=PhysicalInputDecision.BLOCK,
                audit_message="interaction policy is not eligible for mouse action",
                policy_result_id=policy.result_id,
            )

        preclick = self._verifier.verify(request)
        if not preclick.passed:
            return PhysicalInputResult(
                status=PhysicalInputStatus.BLOCKED,
                decision=PhysicalInputDecision.REQUIRE_PRECLICK_VERIFICATION,
                reason=PhysicalInputReason.PRECLICK_FAILED,
                input_kind=PhysicalInputKind.MOUSE_CLICK,
                policy_result_id=policy.result_id,
                preclick=preclick,
                trust=_trust(confidence=preclick.confidence, reason=preclick.reason),
                audit_message="pre-click verification failed",
                executed=False,
            )

        if preclick.region is None:
            return PhysicalInputResult(
                status=PhysicalInputStatus.BLOCKED,
                decision=PhysicalInputDecision.REQUIRE_PRECLICK_VERIFICATION,
                reason=PhysicalInputReason.PRECLICK_FAILED,
                input_kind=PhysicalInputKind.MOUSE_CLICK,
                policy_result_id=policy.result_id,
                preclick=preclick,
                trust=_trust(
                    confidence=preclick.confidence,
                    reason="pre-click region missing",
                ),
                audit_message="pre-click verification region missing",
                executed=False,
            )

        motion = self._motion_engine.plan(
            start_x=request.current_x,
            start_y=request.current_y,
            target=preclick.region,
        )
        input_kind = (
            PhysicalInputKind.MOUSE_DOUBLE_CLICK
            if request.double_click
            else PhysicalInputKind.MOUSE_CLICK
        )

        return PhysicalInputResult(
            status=PhysicalInputStatus.EXECUTED,
            decision=PhysicalInputDecision.ALLOW,
            reason=PhysicalInputReason.MOUSE_ACTION_EXECUTED,
            input_kind=input_kind,
            policy_result_id=policy.result_id,
            motion_plan=motion,
            preclick=preclick,
            trust=_trust(
                confidence=preclick.confidence,
                reason="mouse action accepted"
            ),
            audit_message="mouse action accepted by governed runtime",
            executed=True,
            metadata={"button": request.button.value},
        )


class KeyboardRuntime:
    """
    Governed keyboard runtime.

    No typing into unknown field.
    No typing if focus uncertain.
    Stop/user override cancels immediately.
    """

    def __init__(
        self,
        *,
        override_detector: PhysicalOverrideDetector | None = None,
        timing: HumanTimingPolicy | None = None,
    ) -> None:
        self._override_detector = override_detector or PhysicalOverrideDetector()
        self._timing = timing or HumanTimingPolicy()

    def execute(self, request: KeyboardActionRequest) -> PhysicalInputResult:
        override = self._override_detector.current()
        if override.active:
            return _blocked_result(
                input_kind=PhysicalInputKind.KEYBOARD_TYPE,
                reason=(
                    PhysicalInputReason.STOP_CANCELLED
                    if override.kind == HumanOverrideKind.USER_STOP_COMMAND
                    else PhysicalInputReason.HUMAN_OVERRIDE_DETECTED
                ),
                decision=(
                    PhysicalInputDecision.CANCEL
                    if override.kind == HumanOverrideKind.USER_STOP_COMMAND
                    else PhysicalInputDecision.PAUSE_FOR_USER
                ),
                audit_message=override.reason,
                override=override,
                policy_result_id=request.policy_result.result_id,
            )

        policy = request.policy_result
        if not _policy_allows_physical_attempt(policy):
            return _blocked_result(
                input_kind=PhysicalInputKind.KEYBOARD_TYPE,
                reason=PhysicalInputReason.POLICY_NOT_ELIGIBLE,
                decision=PhysicalInputDecision.BLOCK,
                audit_message="interaction policy is not eligible for keyboard action",
                policy_result_id=policy.result_id,
            )

        if request.text is not None:
            if not request.target_field_known:
                return _blocked_result(
                    input_kind=PhysicalInputKind.KEYBOARD_TYPE,
                    reason=PhysicalInputReason.UNKNOWN_FIELD,
                    decision=PhysicalInputDecision.REQUIRE_FOCUS_VERIFICATION,
                    audit_message="typing target field is unknown",
                    policy_result_id=policy.result_id,
                )

            if not request.focus_known:
                return _blocked_result(
                    input_kind=PhysicalInputKind.KEYBOARD_TYPE,
                    reason=PhysicalInputReason.FOCUS_UNCERTAIN,
                    decision=PhysicalInputDecision.REQUIRE_FOCUS_VERIFICATION,
                    audit_message="typing focus is uncertain",
                    policy_result_id=policy.result_id,
                )

        input_kind = (
            PhysicalInputKind.KEYBOARD_SHORTCUT
            if request.shortcut is not None or request.custom_keys
            else PhysicalInputKind.KEYBOARD_TYPE
        )

        return PhysicalInputResult(
            status=PhysicalInputStatus.EXECUTED,
            decision=PhysicalInputDecision.ALLOW,
            reason=PhysicalInputReason.KEYBOARD_ACTION_EXECUTED,
            input_kind=input_kind,
            policy_result_id=policy.result_id,
            trust=_trust(
                confidence=policy.trust.confidence,
                reason="keyboard action accepted",
            ),
            audit_message="keyboard action accepted by governed runtime",
            executed=True,
            metadata={
                "text_present": request.text is not None,
                "shortcut": request.shortcut.value if request.shortcut else None,
                "key_interval_ms": self._timing.min_key_interval_ms,
            },
        )


class PhysicalInputRuntime:
    """
    Phase 8 Step 25 Mouse & Keyboard Runtime.

    Responsibilities:
    - route mouse and keyboard actions through Step 24 policy
    - detect human override
    - cancel immediately on stop
    - pause when user moves mouse or keyboard
    - require pre-click verification
    - require known field/focus before typing
    - plan natural mouse movement

    Non-responsibilities:
    - no raw pyautogui calls yet
    - no bypass of InteractionPolicyResult
    - no action without audit trail
    """

    def __init__(
        self,
        *,
        name: str = "physical_input_runtime",
        override_detector: PhysicalOverrideDetector | None = None,
        mouse_runtime: MouseRuntime | None = None,
        keyboard_runtime: KeyboardRuntime | None = None,
    ) -> None:
        cleaned = name.strip()

        if not cleaned:
            raise ValueError("name cannot be empty.")

        self._name = cleaned
        self._override_detector = override_detector or PhysicalOverrideDetector()
        self._mouse_runtime = mouse_runtime or MouseRuntime(
            override_detector=self._override_detector
        )
        self._keyboard_runtime = keyboard_runtime or KeyboardRuntime(
            override_detector=self._override_detector
        )
        self._sessions: dict[str, PhysicalInputSession] = {}
        self._results: list[PhysicalInputResult] = []
        self._events: list[PhysicalInputRuntimeEvent] = []
        self._lock = RLock()
        self._last_reason: PhysicalInputReason | None = None

    @property
    def name(self) -> str:
        return self._name

    def create_session(
        self,
        *,
        workspace_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> PhysicalInputSession:
        session = PhysicalInputSession(
            workspace_id=workspace_id,
            metadata=metadata or {},
        )
        event = self._event(
            kind=PhysicalInputEventKind.SESSION_CREATED,
            reason=PhysicalInputReason.SESSION_CREATED,
            session_id=session.session_id,
        )

        with self._lock:
            self._sessions[session.session_id] = session
            self._events.append(event)
            self._last_reason = event.reason

        return session

    def update_override(self, signal: PhysicalOverrideSignal) -> None:
        self._override_detector.update(signal)

        if signal.active:
            with self._lock:
                for session_id, session in self._sessions.items():
                    self._sessions[session_id] = session.model_copy(
                        update={
                            "paused_by_user": signal.kind
                            != HumanOverrideKind.USER_STOP_COMMAND,
                            "cancelled": signal.kind
                            == HumanOverrideKind.USER_STOP_COMMAND,
                            "updated_at": utc_now(),
                        }
                    )

    def clear_override(self) -> None:
        self._override_detector.clear()

    def execute_mouse(self, request: MouseActionRequest) -> PhysicalInputResult:
        if self.session_for(request.session_id) is None:
            result = _blocked_result(
                input_kind=PhysicalInputKind.MOUSE_CLICK,
                reason=PhysicalInputReason.SESSION_NOT_FOUND,
                decision=PhysicalInputDecision.BLOCK,
                audit_message="physical input session not found",
                policy_result_id=request.policy_result.result_id,
            )
            self._record_result(result, request.session_id, request.request_id)
            return result

        result = self._mouse_runtime.execute(request)
        self._record_result(result, request.session_id, request.request_id)
        self._touch_session(request.session_id)

        return result

    def execute_keyboard(
        self,
        request: KeyboardActionRequest,
    ) -> PhysicalInputResult:
        if self.session_for(request.session_id) is None:
            result = _blocked_result(
                input_kind=PhysicalInputKind.KEYBOARD_TYPE,
                reason=PhysicalInputReason.SESSION_NOT_FOUND,
                decision=PhysicalInputDecision.BLOCK,
                audit_message="physical input session not found",
                policy_result_id=request.policy_result.result_id,
            )
            self._record_result(result, request.session_id, request.request_id)
            return result

        result = self._keyboard_runtime.execute(request)
        self._record_result(result, request.session_id, request.request_id)
        self._touch_session(request.session_id)

        return result

    def session_for(self, session_id: str) -> PhysicalInputSession | None:
        with self._lock:
            return self._sessions.get(session_id)

    def results(self) -> tuple[PhysicalInputResult, ...]:
        with self._lock:
            return tuple(self._results)

    def events(self) -> tuple[PhysicalInputRuntimeEvent, ...]:
        with self._lock:
            return tuple(self._events)

    def snapshot(self) -> PhysicalInputRuntimeSnapshot:
        with self._lock:
            return PhysicalInputRuntimeSnapshot(
                name=self.name,
                session_count=len(self._sessions),
                result_count=len(self._results),
                executed_count=sum(1 for result in self._results if result.executed),
                blocked_count=sum(
                    1
                    for result in self._results
                    if result.status == PhysicalInputStatus.BLOCKED
                ),
                cancelled_count=sum(
                    1
                    for result in self._results
                    if result.status == PhysicalInputStatus.CANCELLED
                ),
                paused_by_user_count=sum(
                    1
                    for session in self._sessions.values()
                    if session.paused_by_user
                ),
                motion_plan_count=sum(
                    1 for result in self._results if result.motion_plan is not None
                ),
                runtime_event_count=len(self._events),
                last_reason=self._last_reason,
            )

    def reset(self) -> None:
        event = self._event(
            kind=PhysicalInputEventKind.RUNTIME_RESET,
            reason=PhysicalInputReason.RUNTIME_RESET,
        )

        with self._lock:
            self._sessions.clear()
            self._results.clear()
            self._events.clear()
            self._events.append(event)
            self._last_reason = event.reason

    def _record_result(
        self,
        result: PhysicalInputResult,
        session_id: str,
        request_id: str,
    ) -> None:
        event = self._event(
            kind=_event_kind_for(result),
            reason=result.reason,
            session_id=session_id,
            result_id=result.result_id,
            request_id=request_id,
            metadata={
                "status": result.status.value,
                "executed": result.executed,
            },
        )

        with self._lock:
            self._results.append(result)
            self._events.append(event)
            self._last_reason = result.reason

    def _touch_session(self, session_id: str) -> None:
        session = self._sessions.get(session_id)

        if session is None:
            return

        self._sessions[session_id] = session.model_copy(
            update={"updated_at": utc_now()}
        )

    @staticmethod
    def _event(
        *,
        kind: PhysicalInputEventKind,
        reason: PhysicalInputReason,
        session_id: str | None = None,
        result_id: str | None = None,
        request_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> PhysicalInputRuntimeEvent:
        return PhysicalInputRuntimeEvent(
            kind=kind,
            reason=reason,
            session_id=session_id,
            result_id=result_id,
            request_id=request_id,
            metadata=metadata or {},
        )


def _policy_allows_physical_attempt(policy: InteractionPolicyResult) -> bool:
    if policy.execution_eligible:
        return True

    if policy.decision in {
        InteractionDecision.ELIGIBLE_FOR_EXECUTION,
        InteractionDecision.REQUIRES_VERIFICATION_FIRST,
    }:
        return True

    contract_kind = policy.request.contract.kind

    if (
        contract_kind == PhysicalInteractionKind.KEYBOARD_TYPE
        and policy.decision == InteractionDecision.WAITING_FOR_APPROVAL
        and policy.request.user_initiated
    ):
        return True

    return False


def _blocked_result(
    *,
    input_kind: PhysicalInputKind,
    reason: PhysicalInputReason,
    decision: PhysicalInputDecision,
    audit_message: str,
    policy_result_id: str | None = None,
    override: PhysicalOverrideSignal | None = None,
) -> PhysicalInputResult:
    status = PhysicalInputStatus.BLOCKED

    if decision == PhysicalInputDecision.CANCEL:
        status = PhysicalInputStatus.CANCELLED

    if decision == PhysicalInputDecision.PAUSE_FOR_USER:
        status = PhysicalInputStatus.PAUSED_BY_USER

    return PhysicalInputResult(
        status=status,
        decision=decision,
        reason=reason,
        input_kind=input_kind,
        policy_result_id=policy_result_id,
        override=override,
        trust=_trust(confidence=0.0, reason=audit_message),
        audit_message=audit_message,
        executed=False,
    )


def _event_kind_for(result: PhysicalInputResult) -> PhysicalInputEventKind:
    if result.executed:
        return PhysicalInputEventKind.ACTION_EXECUTED

    if result.status == PhysicalInputStatus.CANCELLED:
        return PhysicalInputEventKind.ACTION_CANCELLED

    return PhysicalInputEventKind.ACTION_BLOCKED


def _trust(*, confidence: float, reason: str) -> TrustCalibration:
    return TrustCalibration(
        confidence=max(0.0, min(1.0, confidence)),
        stability=max(0.0, min(1.0, confidence + 0.05)),
        ambiguity=1.0 - max(0.0, min(1.0, confidence)),
        source=EnvironmentSource.OS_OBSERVER,
        reason=reason,
        metadata={"policy": TrustPolicyClassification.REVIEW.value},
    )


def _clean_required(value: str) -> str:
    cleaned = value.strip()

    if not cleaned:
        raise ValueError("field cannot be empty.")

    return cleaned