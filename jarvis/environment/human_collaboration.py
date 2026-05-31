from __future__ import annotations

from enum import StrEnum
from threading import RLock
from typing import Any
from uuid import uuid4

from pydantic import Field, field_validator, model_validator

from jarvis.environment.models import EnvironmentSource, TrustCalibration
from jarvis.environment.trust_runtime import TrustPolicyClassification
from jarvis.orchestration.ids import utc_now
from jarvis.orchestration.models import OrchestrationModel


class CollaborationPhase(StrEnum):
    OBSERVING = "observing"
    THINKING = "thinking"
    FOUND_ERROR = "found_error"
    OPENING_FILE = "opening_file"
    RUNNING_TESTS = "running_tests"
    APPLYING_CHANGE = "applying_change"
    VERIFYING = "verifying"
    ISSUE_REMAINS = "issue_remains"
    WAITING_FOR_APPROVAL = "waiting_for_approval"
    COMPLETED = "completed"
    PAUSED = "paused"
    CANCELLED = "cancelled"


class CollaborationStatus(StrEnum):
    READY = "ready"
    NARRATED = "narrated"
    APPROVAL_REQUIRED = "approval_required"
    APPROVED = "approved"
    DENIED = "denied"
    OVERRIDDEN = "overridden"
    PAUSED = "paused"
    CANCELLED = "cancelled"
    BLOCKED = "blocked"
    FAILED = "failed"


class CollaborationDecision(StrEnum):
    CONTINUE = "continue"
    ASK_APPROVAL = "ask_approval"
    WAIT_FOR_USER = "wait_for_user"
    PAUSE = "pause"
    CANCEL = "cancel"
    BLOCK = "block"


class CollaborationReason(StrEnum):
    SESSION_CREATED = "session_created"
    PROGRESS_NARRATED = "progress_narrated"
    APPROVAL_DIALOGUE_CREATED = "approval_dialogue_created"
    APPROVAL_GRANTED = "approval_granted"
    APPROVAL_DENIED = "approval_denied"
    USER_OVERRIDE_PAUSE = "user_override_pause"
    USER_OVERRIDE_CANCEL = "user_override_cancel"
    USER_OVERRIDE_TAKEOVER = "user_override_takeover"
    SILENT_CONTROL_BLOCKED = "silent_control_blocked"
    APPROVAL_REQUIRED_BEFORE_ACTION = "approval_required_before_action"
    SESSION_NOT_FOUND = "session_not_found"
    RUNTIME_RESET = "runtime_reset"


class CollaborationEventKind(StrEnum):
    SESSION_CREATED = "session_created"
    NARRATION_EMITTED = "narration_emitted"
    APPROVAL_REQUESTED = "approval_requested"
    APPROVAL_RECORDED = "approval_recorded"
    USER_OVERRIDE_RECORDED = "user_override_recorded"
    OPERATION_BLOCKED = "operation_blocked"
    RUNTIME_RESET = "runtime_reset"


class NarrationTone(StrEnum):
    CALM = "calm"
    BRIEF = "brief"
    TECHNICAL = "technical"
    REASSURING = "reassuring"


class NarrationVerbosity(StrEnum):
    SILENT_SAFE = "silent_safe"
    BRIEF = "brief"
    NORMAL = "normal"
    DEBUG = "debug"


class UserOverrideKind(StrEnum):
    NONE = "none"
    PAUSE = "pause"
    CANCEL = "cancel"
    TAKE_OVER = "take_over"
    APPROVE = "approve"
    DENY = "deny"


class ApprovalRisk(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ApprovalDialogueStatus(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    DENIED = "denied"
    EXPIRED = "expired"


class NarrationPolicy(OrchestrationModel):
    """
    Policy for user-visible collaboration narration.

    JARVIS should not silently operate the computer. The user should know what
    JARVIS found, what it is doing, what remains, and when approval is required.
    """

    verbosity: NarrationVerbosity = NarrationVerbosity.BRIEF
    tone: NarrationTone = NarrationTone.CALM
    narrate_found_error: bool = True
    narrate_opening_file: bool = True
    narrate_running_tests: bool = True
    narrate_issue_remains: bool = True
    narrate_approval_needed: bool = True
    block_silent_physical_control: bool = True
    max_message_chars: int = Field(default=180, ge=40, le=500)


class CollaborationRequest(OrchestrationModel):
    request_id: str = Field(
        default_factory=lambda: f"collab_request_{uuid4().hex}"
    )
    session_id: str
    phase: CollaborationPhase
    message: str | None = None
    detail: str | None = None
    target: str | None = None
    requires_approval: bool = False
    silent_physical_control: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("request_id", "session_id")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class ProgressNarration(OrchestrationModel):
    narration_id: str = Field(
        default_factory=lambda: f"progress_narration_{uuid4().hex}"
    )
    phase: CollaborationPhase
    text: str
    tone: NarrationTone
    user_visible: bool = True
    interruptible: bool = True
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("narration_id", "text")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)

    @model_validator(mode="after")
    def _must_be_user_visible(self) -> ProgressNarration:
        if not self.user_visible:
            raise ValueError("collaboration narration must be user-visible.")

        return self


class ApprovalDialogueRequest(OrchestrationModel):
    request_id: str = Field(default_factory=lambda: f"approval_req_{uuid4().hex}")
    session_id: str
    proposed_action: str
    reason: str
    risk: ApprovalRisk
    target: str | None = None
    irreversible: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("request_id", "session_id", "proposed_action", "reason")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class ApprovalDialogue(OrchestrationModel):
    dialogue_id: str = Field(
        default_factory=lambda: f"approval_dialogue_{uuid4().hex}"
    )
    request: ApprovalDialogueRequest
    status: ApprovalDialogueStatus = ApprovalDialogueStatus.PENDING
    prompt: str
    requires_explicit_approval: bool
    created_at: object = Field(default_factory=utc_now)
    responded_at: object | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("dialogue_id", "prompt")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class ApprovalDialogueResponse(OrchestrationModel):
    response_id: str = Field(
        default_factory=lambda: f"approval_response_{uuid4().hex}"
    )
    dialogue_id: str
    approved: bool
    user_text: str
    created_at: object = Field(default_factory=utc_now)

    @field_validator("response_id", "dialogue_id", "user_text")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class UserOverrideSignal(OrchestrationModel):
    signal_id: str = Field(default_factory=lambda: f"user_override_{uuid4().hex}")
    session_id: str
    text: str
    kind: UserOverrideKind
    created_at: object = Field(default_factory=utc_now)

    @field_validator("signal_id", "session_id", "text")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class UserOverrideResult(OrchestrationModel):
    result_id: str = Field(
        default_factory=lambda: f"user_override_result_{uuid4().hex}"
    )
    signal: UserOverrideSignal
    status: CollaborationStatus
    decision: CollaborationDecision
    reason: CollaborationReason
    message: str
    trust: TrustCalibration
    created_at: object = Field(default_factory=utc_now)

    @field_validator("result_id", "message")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class CollaborativePacing(OrchestrationModel):
    """
    Step 36 pacing is collaboration-level pacing only.

    Step 37 will own detailed motion/speech/action timing.
    """

    announce_before_risky_action: bool = True
    pause_after_approval_prompt: bool = True
    max_consecutive_narrations: int = Field(default=3, ge=1, le=10)
    user_override_has_priority: bool = True


class CollaborationResult(OrchestrationModel):
    result_id: str = Field(default_factory=lambda: f"collab_result_{uuid4().hex}")
    status: CollaborationStatus
    decision: CollaborationDecision
    reason: CollaborationReason
    narration: ProgressNarration | None = None
    approval_dialogue: ApprovalDialogue | None = None
    approval_response: ApprovalDialogueResponse | None = None
    override_result: UserOverrideResult | None = None
    trust: TrustCalibration
    message: str
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("result_id", "message")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)

    @model_validator(mode="after")
    def _status_requires_payload(self) -> CollaborationResult:
        if self.status == CollaborationStatus.NARRATED and self.narration is None:
            raise ValueError("NARRATED result requires narration.")

        if (
            self.status == CollaborationStatus.APPROVAL_REQUIRED
            and self.approval_dialogue is None
        ):
            raise ValueError("APPROVAL_REQUIRED requires approval_dialogue.")

        if self.status == CollaborationStatus.OVERRIDDEN:
            if self.override_result is None:
                raise ValueError("OVERRIDDEN result requires override_result.")

        return self


class HumanCollaborationSession(OrchestrationModel):
    session_id: str = Field(default_factory=lambda: f"human_collab_{uuid4().hex}")
    workspace_id: str
    active_dialogue_id: str | None = None
    narration_count: int = Field(default=0, ge=0)
    approval_count: int = Field(default=0, ge=0)
    approved_count: int = Field(default=0, ge=0)
    denied_count: int = Field(default=0, ge=0)
    override_count: int = Field(default=0, ge=0)
    blocked_count: int = Field(default=0, ge=0)
    paused: bool = False
    cancelled: bool = False
    created_at: object = Field(default_factory=utc_now)
    updated_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("session_id", "workspace_id")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class HumanCollaborationRuntimeEvent(OrchestrationModel):
    event_id: str = Field(default_factory=lambda: f"human_collab_event_{uuid4().hex}")
    kind: CollaborationEventKind
    reason: CollaborationReason
    session_id: str | None = None
    result_id: str | None = None
    dialogue_id: str | None = None
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("event_id")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class HumanCollaborationRuntimeSnapshot(OrchestrationModel):
    name: str
    session_count: int = Field(ge=0)
    result_count: int = Field(ge=0)
    narration_count: int = Field(ge=0)
    approval_count: int = Field(ge=0)
    approved_count: int = Field(ge=0)
    denied_count: int = Field(ge=0)
    override_count: int = Field(ge=0)
    blocked_count: int = Field(ge=0)
    runtime_event_count: int = Field(ge=0)
    last_reason: CollaborationReason | None = None
    created_at: object = Field(default_factory=utc_now)

    @field_validator("name")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class ProgressNarrator:
    def narrate(
        self,
        *,
        request: CollaborationRequest,
        policy: NarrationPolicy,
    ) -> ProgressNarration:
        text = request.message or _default_message_for(request)

        if request.detail and policy.verbosity in {
            NarrationVerbosity.NORMAL,
            NarrationVerbosity.DEBUG,
        }:
            text = f"{text} {request.detail}"

        return ProgressNarration(
            phase=request.phase,
            text=_limit_text(text, policy.max_message_chars),
            tone=policy.tone,
            metadata={
                "target": request.target,
                "requires_approval": request.requires_approval,
            },
        )


class ApprovalDialogueRuntime:
    def create_dialogue(
        self,
        request: ApprovalDialogueRequest,
    ) -> ApprovalDialogue:
        explicit = request.irreversible or request.risk in {
            ApprovalRisk.HIGH,
            ApprovalRisk.CRITICAL,
        }
        prompt = _approval_prompt(request=request, explicit=explicit)

        return ApprovalDialogue(
            request=request,
            prompt=prompt,
            requires_explicit_approval=explicit,
        )

    def respond(
        self,
        *,
        dialogue: ApprovalDialogue,
        response: ApprovalDialogueResponse,
    ) -> ApprovalDialogue:
        return dialogue.model_copy(
            update={
                "status": (
                    ApprovalDialogueStatus.APPROVED
                    if response.approved
                    else ApprovalDialogueStatus.DENIED
                ),
                "responded_at": utc_now(),
            }
        )


class UserOverrideRuntime:
    def parse(self, *, session_id: str, text: str) -> UserOverrideSignal:
        lowered = text.strip().lower()
        kind = UserOverrideKind.NONE

        if lowered in {"stop", "cancel", "abort", "shut up"}:
            kind = UserOverrideKind.CANCEL
        elif lowered in {"pause", "wait", "hold on"}:
            kind = UserOverrideKind.PAUSE
        elif lowered in {"i will do it", "let me do it", "take over"}:
            kind = UserOverrideKind.TAKE_OVER
        elif lowered in {"yes", "approve", "confirm", "go ahead"}:
            kind = UserOverrideKind.APPROVE
        elif lowered in {"no", "deny", "do not", "don't"}:
            kind = UserOverrideKind.DENY

        return UserOverrideSignal(
            session_id=session_id,
            text=text,
            kind=kind,
        )

    def evaluate(self, signal: UserOverrideSignal) -> UserOverrideResult:
        if signal.kind == UserOverrideKind.CANCEL:
            return _override_result(
                signal=signal,
                status=CollaborationStatus.CANCELLED,
                decision=CollaborationDecision.CANCEL,
                reason=CollaborationReason.USER_OVERRIDE_CANCEL,
                message="User cancelled the operation.",
            )

        if signal.kind in {UserOverrideKind.PAUSE, UserOverrideKind.TAKE_OVER}:
            reason = CollaborationReason.USER_OVERRIDE_PAUSE
            message = "User paused the operation."
            if signal.kind == UserOverrideKind.TAKE_OVER:
                reason = CollaborationReason.USER_OVERRIDE_TAKEOVER
                message = "User took over control."

            return _override_result(
                signal=signal,
                status=CollaborationStatus.PAUSED,
                decision=CollaborationDecision.PAUSE,
                reason=reason,
                message=message,
            )

        return _override_result(
            signal=signal,
            status=CollaborationStatus.READY,
            decision=CollaborationDecision.CONTINUE,
            reason=CollaborationReason.PROGRESS_NARRATED,
            message="No blocking override detected.",
        )


class HumanCollaborationRuntime:
    """
    Phase 8 Step 36 Human Collaboration Runtime.

    JARVIS must collaborate with the user:
    - narrate progress
    - ask approval before risky actions
    - respect user override immediately
    - avoid silent physical control
    - keep the user aware of important state changes

    Step 37 will optimize human-speed timing and natural pacing.
    """

    def __init__(
        self,
        *,
        name: str = "human_collaboration_runtime",
        narration_policy: NarrationPolicy | None = None,
        pacing: CollaborativePacing | None = None,
        narrator: ProgressNarrator | None = None,
        approval_runtime: ApprovalDialogueRuntime | None = None,
        override_runtime: UserOverrideRuntime | None = None,
    ) -> None:
        cleaned = name.strip()
        if not cleaned:
            raise ValueError("name cannot be empty.")

        self._name = cleaned
        self._narration_policy = narration_policy or NarrationPolicy()
        self._pacing = pacing or CollaborativePacing()
        self._narrator = narrator or ProgressNarrator()
        self._approval_runtime = approval_runtime or ApprovalDialogueRuntime()
        self._override_runtime = override_runtime or UserOverrideRuntime()
        self._sessions: dict[str, HumanCollaborationSession] = {}
        self._dialogues: dict[str, ApprovalDialogue] = {}
        self._results: list[CollaborationResult] = []
        self._events: list[HumanCollaborationRuntimeEvent] = []
        self._lock = RLock()
        self._last_reason: CollaborationReason | None = None

    @property
    def name(self) -> str:
        return self._name

    def create_session(
        self,
        *,
        workspace_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> HumanCollaborationSession:
        session = HumanCollaborationSession(
            workspace_id=workspace_id,
            metadata=metadata or {},
        )
        event = self._event(
            kind=CollaborationEventKind.SESSION_CREATED,
            reason=CollaborationReason.SESSION_CREATED,
            session_id=session.session_id,
        )

        with self._lock:
            self._sessions[session.session_id] = session
            self._events.append(event)
            self._last_reason = event.reason

        return session

    def narrate_progress(
        self,
        request: CollaborationRequest,
    ) -> CollaborationResult:
        session = self.session_for(request.session_id)
        if session is None:
            result = _blocked_result(
                reason=CollaborationReason.SESSION_NOT_FOUND,
                message="human collaboration session not found",
            )
            self._record_result(result, request.session_id)
            return result

        if (
            request.silent_physical_control
            and self._narration_policy.block_silent_physical_control
        ):
            result = _blocked_result(
                reason=CollaborationReason.SILENT_CONTROL_BLOCKED,
                message="silent physical control is blocked",
            )
            self._record_result(result, request.session_id)
            return result

        if request.requires_approval:
            approval_request = ApprovalDialogueRequest(
                session_id=request.session_id,
                proposed_action=request.message or _default_message_for(request),
                reason=request.detail or "this action requires your approval",
                risk=ApprovalRisk.MEDIUM,
                target=request.target,
                irreversible=False,
            )
            dialogue = self._approval_runtime.create_dialogue(approval_request)

            with self._lock:
                self._dialogues[dialogue.dialogue_id] = dialogue
                self._sessions[request.session_id] = session.model_copy(
                    update={
                        "active_dialogue_id": dialogue.dialogue_id,
                        "updated_at": utc_now(),
                    }
                )

            result = CollaborationResult(
                status=CollaborationStatus.APPROVAL_REQUIRED,
                decision=CollaborationDecision.ASK_APPROVAL,
                reason=CollaborationReason.APPROVAL_REQUIRED_BEFORE_ACTION,
                approval_dialogue=dialogue,
                trust=_trust(
                    confidence=0.90,
                    reason="approval required before action",
                ),
                message=dialogue.prompt,
            )
            self._record_result(result, request.session_id)
            return result

        narration = self._narrator.narrate(
            request=request,
            policy=self._narration_policy,
        )
        result = CollaborationResult(
            status=CollaborationStatus.NARRATED,
            decision=CollaborationDecision.CONTINUE,
            reason=CollaborationReason.PROGRESS_NARRATED,
            narration=narration,
            trust=_trust(confidence=0.88, reason="progress narrated to user"),
            message=narration.text,
        )
        self._record_result(result, request.session_id)

        return result

    def request_approval(
        self,
        request: ApprovalDialogueRequest,
    ) -> CollaborationResult:
        if self.session_for(request.session_id) is None:
            result = _blocked_result(
                reason=CollaborationReason.SESSION_NOT_FOUND,
                message="human collaboration session not found",
            )
            self._record_result(result, request.session_id)
            return result

        dialogue = self._approval_runtime.create_dialogue(request)
        result = CollaborationResult(
            status=CollaborationStatus.APPROVAL_REQUIRED,
            decision=CollaborationDecision.ASK_APPROVAL,
            reason=CollaborationReason.APPROVAL_DIALOGUE_CREATED,
            approval_dialogue=dialogue,
            trust=_trust(confidence=0.92, reason="approval dialogue created"),
            message=dialogue.prompt,
        )

        with self._lock:
            self._dialogues[dialogue.dialogue_id] = dialogue
            session = self._sessions[request.session_id]
            self._sessions[request.session_id] = session.model_copy(
                update={
                    "active_dialogue_id": dialogue.dialogue_id,
                    "updated_at": utc_now(),
                }
            )

        self._record_result(result, request.session_id)
        return result

    def respond_to_approval(
        self,
        *,
        session_id: str,
        response: ApprovalDialogueResponse,
    ) -> CollaborationResult:
        session = self.session_for(session_id)
        if session is None:
            result = _blocked_result(
                reason=CollaborationReason.SESSION_NOT_FOUND,
                message="human collaboration session not found",
            )
            self._record_result(result, session_id)
            return result

        dialogue = self._dialogues.get(response.dialogue_id)
        if dialogue is None:
            result = _blocked_result(
                reason=CollaborationReason.APPROVAL_REQUIRED_BEFORE_ACTION,
                message="approval dialogue not found",
            )
            self._record_result(result, session_id)
            return result

        updated = self._approval_runtime.respond(
            dialogue=dialogue,
            response=response,
        )
        with self._lock:
            self._dialogues[updated.dialogue_id] = updated

        approved = updated.status == ApprovalDialogueStatus.APPROVED
        result = CollaborationResult(
            status=(
                CollaborationStatus.APPROVED
                if approved
                else CollaborationStatus.DENIED
            ),
            decision=(
                CollaborationDecision.CONTINUE
                if approved
                else CollaborationDecision.BLOCK
            ),
            reason=(
                CollaborationReason.APPROVAL_GRANTED
                if approved
                else CollaborationReason.APPROVAL_DENIED
            ),
            approval_dialogue=updated,
            approval_response=response,
            trust=_trust(
                confidence=0.95,
                reason="approval response recorded",
            ),
            message=(
                "Approval granted."
                if approved
                else "Approval denied. I will not continue that action."
            ),
        )
        self._record_result(result, session_id)
        return result

    def handle_user_override(
        self,
        *,
        session_id: str,
        user_text: str,
    ) -> CollaborationResult:
        if self.session_for(session_id) is None:
            result = _blocked_result(
                reason=CollaborationReason.SESSION_NOT_FOUND,
                message="human collaboration session not found",
            )
            self._record_result(result, session_id)
            return result

        signal = self._override_runtime.parse(
            session_id=session_id,
            text=user_text,
        )
        override = self._override_runtime.evaluate(signal)

        if override.status == CollaborationStatus.READY:
            result = CollaborationResult(
                status=CollaborationStatus.READY,
                decision=CollaborationDecision.CONTINUE,
                reason=CollaborationReason.PROGRESS_NARRATED,
                override_result=override,
                trust=override.trust,
                message="No override action required.",
            )
        else:
            result = CollaborationResult(
                status=CollaborationStatus.OVERRIDDEN,
                decision=override.decision,
                reason=override.reason,
                override_result=override,
                trust=override.trust,
                message=override.message,
            )

        self._record_result(result, session_id)
        return result

    def session_for(
        self,
        session_id: str,
    ) -> HumanCollaborationSession | None:
        with self._lock:
            return self._sessions.get(session_id)

    def results(self) -> tuple[CollaborationResult, ...]:
        with self._lock:
            return tuple(self._results)

    def events(self) -> tuple[HumanCollaborationRuntimeEvent, ...]:
        with self._lock:
            return tuple(self._events)

    def snapshot(self) -> HumanCollaborationRuntimeSnapshot:
        with self._lock:
            return HumanCollaborationRuntimeSnapshot(
                name=self.name,
                session_count=len(self._sessions),
                result_count=len(self._results),
                narration_count=sum(
                    1
                    for result in self._results
                    if result.status == CollaborationStatus.NARRATED
                ),
                approval_count=sum(
                    1
                    for result in self._results
                    if result.status == CollaborationStatus.APPROVAL_REQUIRED
                ),
                approved_count=sum(
                    1
                    for result in self._results
                    if result.status == CollaborationStatus.APPROVED
                ),
                denied_count=sum(
                    1
                    for result in self._results
                    if result.status == CollaborationStatus.DENIED
                ),
                override_count=sum(
                    1
                    for result in self._results
                    if result.status == CollaborationStatus.OVERRIDDEN
                ),
                blocked_count=sum(
                    1
                    for result in self._results
                    if result.status
                    in {
                        CollaborationStatus.BLOCKED,
                        CollaborationStatus.FAILED,
                    }
                ),
                runtime_event_count=len(self._events),
                last_reason=self._last_reason,
            )

    def reset(self) -> None:
        event = self._event(
            kind=CollaborationEventKind.RUNTIME_RESET,
            reason=CollaborationReason.RUNTIME_RESET,
        )

        with self._lock:
            self._sessions.clear()
            self._dialogues.clear()
            self._results.clear()
            self._events.clear()
            self._events.append(event)
            self._last_reason = event.reason

    def _record_result(
        self,
        result: CollaborationResult,
        session_id: str,
    ) -> None:
        event = self._event(
            kind=_event_kind_for(result),
            reason=result.reason,
            session_id=session_id,
            result_id=result.result_id,
            dialogue_id=(
                result.approval_dialogue.dialogue_id
                if result.approval_dialogue is not None
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
                        "narration_count": session.narration_count
                        + (
                            1
                            if result.status == CollaborationStatus.NARRATED
                            else 0
                        ),
                        "approval_count": session.approval_count
                        + (
                            1
                            if result.status
                            == CollaborationStatus.APPROVAL_REQUIRED
                            else 0
                        ),
                        "approved_count": session.approved_count
                        + (
                            1
                            if result.status == CollaborationStatus.APPROVED
                            else 0
                        ),
                        "denied_count": session.denied_count
                        + (
                            1
                            if result.status == CollaborationStatus.DENIED
                            else 0
                        ),
                        "override_count": session.override_count
                        + (
                            1
                            if result.status == CollaborationStatus.OVERRIDDEN
                            else 0
                        ),
                        "blocked_count": session.blocked_count
                        + (
                            1
                            if result.status
                            in {
                                CollaborationStatus.BLOCKED,
                                CollaborationStatus.FAILED,
                            }
                            else 0
                        ),
                        "paused": (
                            True
                            if result.decision == CollaborationDecision.PAUSE
                            else session.paused
                        ),
                        "cancelled": (
                            True
                            if result.decision == CollaborationDecision.CANCEL
                            else session.cancelled
                        ),
                    }
                )

    @staticmethod
    def _event(
        *,
        kind: CollaborationEventKind,
        reason: CollaborationReason,
        session_id: str | None = None,
        result_id: str | None = None,
        dialogue_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> HumanCollaborationRuntimeEvent:
        return HumanCollaborationRuntimeEvent(
            kind=kind,
            reason=reason,
            session_id=session_id,
            result_id=result_id,
            dialogue_id=dialogue_id,
            metadata=metadata or {},
        )


def _default_message_for(request: CollaborationRequest) -> str:
    if request.phase == CollaborationPhase.FOUND_ERROR:
        return "I found the error."

    if request.phase == CollaborationPhase.OPENING_FILE:
        target = f" {request.target}" if request.target else ""
        return f"I'm opening the file now{target}."

    if request.phase == CollaborationPhase.RUNNING_TESTS:
        return "The tests are running."

    if request.phase == CollaborationPhase.ISSUE_REMAINS:
        return "One issue remains."

    if request.phase == CollaborationPhase.WAITING_FOR_APPROVAL:
        return "I need your approval before continuing."

    if request.phase == CollaborationPhase.COMPLETED:
        return "Done. The action is complete."

    if request.phase == CollaborationPhase.VERIFYING:
        return "I'm verifying the result now."

    return "I'm working on it."


def _approval_prompt(
    *,
    request: ApprovalDialogueRequest,
    explicit: bool,
) -> str:
    target = f" Target: {request.target}." if request.target else ""
    approval = "Please explicitly approve or deny." if explicit else "Approve?"

    return (
        f"I need your approval before {request.proposed_action}. "
        f"Reason: {request.reason}. Risk: {request.risk.value}.{target} "
        f"{approval}"
    )


def _override_result(
    *,
    signal: UserOverrideSignal,
    status: CollaborationStatus,
    decision: CollaborationDecision,
    reason: CollaborationReason,
    message: str,
) -> UserOverrideResult:
    return UserOverrideResult(
        signal=signal,
        status=status,
        decision=decision,
        reason=reason,
        message=message,
        trust=_trust(confidence=0.95, reason="user override evaluated"),
    )


def _blocked_result(
    *,
    reason: CollaborationReason,
    message: str,
) -> CollaborationResult:
    return CollaborationResult(
        status=(
            CollaborationStatus.FAILED
            if reason == CollaborationReason.SESSION_NOT_FOUND
            else CollaborationStatus.BLOCKED
        ),
        decision=CollaborationDecision.BLOCK,
        reason=reason,
        trust=_trust(confidence=0.20, reason=message),
        message=message,
    )


def _event_kind_for(result: CollaborationResult) -> CollaborationEventKind:
    if result.status == CollaborationStatus.NARRATED:
        return CollaborationEventKind.NARRATION_EMITTED

    if result.status == CollaborationStatus.APPROVAL_REQUIRED:
        return CollaborationEventKind.APPROVAL_REQUESTED

    if result.status in {
        CollaborationStatus.APPROVED,
        CollaborationStatus.DENIED,
    }:
        return CollaborationEventKind.APPROVAL_RECORDED

    if result.status == CollaborationStatus.OVERRIDDEN:
        return CollaborationEventKind.USER_OVERRIDE_RECORDED

    if result.status in {
        CollaborationStatus.BLOCKED,
        CollaborationStatus.FAILED,
    }:
        return CollaborationEventKind.OPERATION_BLOCKED

    return CollaborationEventKind.NARRATION_EMITTED


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


def _limit_text(value: str, limit: int) -> str:
    cleaned = _clean_required(value)
    if len(cleaned) <= limit:
        return cleaned

    return cleaned[: limit - 3] + "..."


def _clean_required(value: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError("field cannot be empty.")
    return cleaned