from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from threading import RLock

from pydantic import Field, field_validator

from jarvis.tools.approval import HumanApprovalRuntime
from jarvis.tools.audit import (
    ActionAuditActor,
    ActionAuditEventKind,
    ActionAuditLog,
    ActionAuditOutcome,
)
from jarvis.tools.filesystem import (
    FileOperationKind,
    FileOperationRequest,
    FileOperationResult,
    FileSystemRuntime,
)
from jarvis.tools.ide import (
    IdeActionKind,
    IdeActionRequest,
    IdeActionResult,
    IdeRuntime,
)
from jarvis.tools.ids import new_action_result_id, utc_now
from jarvis.tools.memory_integration import (
    ToolMemoryEvent,
    ToolMemoryEventKind,
    ToolMemoryIntegrationRuntime,
    ToolMemoryWriteProposal,
    ToolMemoryWriteResult,
)
from jarvis.tools.models import (
    ActionKind,
    ActionPlan,
    ActionRisk,
    ActionScope,
    ActionStatus,
    ActionStep,
    PermissionDecision,
    ToolCapability,
    ToolModel,
)
from jarvis.tools.planner import (
    ActionPlanningDecision,
    ActionPlanningRequest,
    ActionPlanProposal,
    MultiStepActionPlanner,
)
from jarvis.tools.policy import PermissionPolicy
from jarvis.tools.registry import (
    ToolAvailability,
    ToolDescriptor,
    ToolHealth,
    ToolRegistry,
)
from jarvis.tools.scheduler import (
    ActionScheduleDecision,
    ActionSchedulePriority,
    ParallelActionScheduler,
)
from jarvis.tools.shell import (
    SafeShellRuntime,
    ShellCommandRequest,
    ShellCommandResult,
)
from jarvis.tools.validation import (
    ActionValidationDecision,
    ActionValidationResult,
    ActionValidator,
    ActionValidatorConfig,
)


class RealActionSmokeStatus(StrEnum):
    """
    End-to-end smoke runtime status.
    """

    PLANNED = "planned"
    POLICY_BLOCKED = "policy_blocked"
    VALIDATION_BLOCKED = "validation_blocked"
    APPROVAL_REQUIRED = "approval_required"
    SCHEDULED = "scheduled"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    UNSUPPORTED = "unsupported"


class RealActionSmokeReason(StrEnum):
    """
    Machine-readable smoke runtime reason.
    """

    PIPELINE_SUCCEEDED = "pipeline_succeeded"
    PLANNING_FAILED = "planning_failed"
    POLICY_DENIED = "policy_denied"
    VALIDATION_BLOCKED = "validation_blocked"
    APPROVAL_REQUIRED = "approval_required"
    SCHEDULER_DEFERRED = "scheduler_deferred"
    SCHEDULER_BLOCKED = "scheduler_blocked"
    DISPATCH_UNSUPPORTED = "dispatch_unsupported"
    DISPATCH_FAILED = "dispatch_failed"
    MEMORY_PROPOSED = "memory_proposed"


class SmokeDispatchKind(StrEnum):
    """
    Runtime dispatch kind.
    """

    SHELL = "shell"
    FILE_SYSTEM = "file_system"
    IDE = "ide"
    COGNITIVE = "cognitive"


class SmokeDispatchResult(ToolModel):
    """
    One dispatched step result.

    This wraps a result from an existing governed runtime. The smoke runtime
    never executes raw shell, file, or IDE operations directly.
    """

    dispatch_id: str = Field(default_factory=new_action_result_id)
    action_id: str
    step_order: int = Field(ge=0)
    kind: SmokeDispatchKind
    action_kind: ActionKind
    success: bool
    status: ActionStatus
    output: str
    runtime_name: str
    shell_result: ShellCommandResult | None = None
    file_result: FileOperationResult | None = None
    ide_result: IdeActionResult | None = None
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("dispatch_id", "action_id", "output", "runtime_name")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned


class RealActionSmokeRequest(ToolModel):
    """
    Request for the smoke runtime.

    This is the highest-level safe action request in Step 16.
    """

    request_id: str = Field(default_factory=new_action_result_id)
    planning_request: ActionPlanningRequest
    priority: ActionSchedulePriority = ActionSchedulePriority.NORMAL
    approved: bool = False
    validated: bool = False
    write_memory: bool = False
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("request_id")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("request_id cannot be empty.")

        return cleaned


class RealActionSmokeResult(ToolModel):
    """
    End-to-end smoke runtime result.
    """

    result_id: str = Field(default_factory=new_action_result_id)
    request_id: str
    action_id: str | None = None
    status: RealActionSmokeStatus
    reason: RealActionSmokeReason
    success: bool
    message: str
    proposal: ActionPlanProposal | None = None
    policy_decision: PermissionDecision | None = None
    validation_result: ActionValidationResult | None = None
    dispatch_results: tuple[SmokeDispatchResult, ...] = ()
    memory_proposal: ToolMemoryWriteProposal | None = None
    memory_result: ToolMemoryWriteResult | None = None
    started_at: object = Field(default_factory=utc_now)
    completed_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("result_id", "request_id", "message")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned


@dataclass(frozen=True, slots=True)
class RealActionSmokeRuntimeConfig:
    """
    Smoke runtime configuration.
    """

    name: str = "real_action_smoke_runtime"
    allow_medium_risk_smoke: bool = False
    allow_memory_write: bool = False

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("name cannot be empty.")


@dataclass(frozen=True, slots=True)
class RealActionSmokeRuntimeSnapshot:
    """
    Runtime diagnostics.
    """

    name: str
    request_count: int
    success_count: int
    failed_count: int
    blocked_count: int
    last_status: RealActionSmokeStatus | None
    last_reason: RealActionSmokeReason | None
    last_error: str | None


class RealActionSmokeRuntime:
    """
    First real governed action smoke runtime.

    Responsibilities:
    - accept user intent as an ActionPlanningRequest
    - ask planner for a typed ActionPlan
    - run policy and validation
    - check approval requirements
    - schedule the action
    - dispatch only supported smoke-safe steps to existing governed runtimes
    - audit important lifecycle events
    - optionally propose/write safe memory through ToolMemoryIntegrationRuntime

    Non-responsibilities:
    - no autonomy
    - no direct shell execution
    - no direct file mutation
    - no hidden browser or desktop control
    - no bypass of policy, validation, approval, scheduler, or audit
    """

    def __init__(
        self,
        *,
        config: RealActionSmokeRuntimeConfig | None = None,
        planner: MultiStepActionPlanner | None = None,
        policy: PermissionPolicy | None = None,
        validator: ActionValidator | None = None,
        approval_runtime: HumanApprovalRuntime | None = None,
        scheduler: ParallelActionScheduler | None = None,
        audit_log: ActionAuditLog | None = None,
        memory_runtime: ToolMemoryIntegrationRuntime | None = None,
        shell_runtime: SafeShellRuntime | None = None,
        file_runtime: FileSystemRuntime | None = None,
        ide_runtime: IdeRuntime | None = None,
        registry: ToolRegistry | None = None,
    ) -> None:
        self._config = config or RealActionSmokeRuntimeConfig()
        self._config.validate()

        self._registry = registry or ToolRegistry()
        self._register_smoke_tools()

        self._planner = planner or MultiStepActionPlanner()
        self._policy = policy or PermissionPolicy()
        self._validator = validator or ActionValidator(
            config=ActionValidatorConfig(require_policy_evaluation=False),
            registry=self._registry,
        )
        self._approval_runtime = approval_runtime or HumanApprovalRuntime()
        self._scheduler = scheduler or ParallelActionScheduler()
        self._audit_log = audit_log or ActionAuditLog()
        self._memory_runtime = memory_runtime or ToolMemoryIntegrationRuntime()
        self._shell_runtime = shell_runtime or SafeShellRuntime()
        self._file_runtime = file_runtime or FileSystemRuntime()
        self._ide_runtime = ide_runtime or IdeRuntime()
        self._lock = RLock()

        self._request_count = 0
        self._success_count = 0
        self._failed_count = 0
        self._blocked_count = 0
        self._last_status: RealActionSmokeStatus | None = None
        self._last_reason: RealActionSmokeReason | None = None
        self._last_error: str | None = None

    @property
    def name(self) -> str:
        return self._config.name

    def run(self, request: RealActionSmokeRequest) -> RealActionSmokeResult:
        """
        Run one governed smoke action pipeline.
        """

        with self._lock:
            self._request_count += 1
            self._last_error = None

        self._audit_log.record_intent(
            action_id=request.planning_request.request_id,
            user_intent=request.planning_request.user_intent,
            requested_by="user",
            correlation_id=request.request_id,
        )

        try:
            proposal = self._planner.propose(request.planning_request)

            if proposal.decision != ActionPlanningDecision.PROPOSED:
                result = self._failed(
                    request=request,
                    proposal=proposal,
                    status=RealActionSmokeStatus.FAILED,
                    reason=RealActionSmokeReason.PLANNING_FAILED,
                    message=proposal.summary,
                )
                self._record_result(result)

                return result

            plan = self._require_plan(proposal)
            self._audit_plan(proposal, plan)

            risk_block = self._risk_block_result(
                request=request,
                proposal=proposal,
                plan=plan,
            )

            if risk_block is not None:
                self._record_result(risk_block)

                return risk_block

            policy_evaluation = self._policy.evaluate_plan(plan)

            if policy_evaluation.decision == PermissionDecision.DENY:
                result = self._failed(
                    request=request,
                    proposal=proposal,
                    status=RealActionSmokeStatus.POLICY_BLOCKED,
                    reason=RealActionSmokeReason.POLICY_DENIED,
                    message="permission policy denied smoke action",
                    action_id=plan.action_id,
                    policy_decision=policy_evaluation.decision,
                )
                self._record_result(result)

                return result

            validation = self._validator.validate_plan(plan)

            if validation.decision == ActionValidationDecision.BLOCK:
                result = self._failed(
                    request=request,
                    proposal=proposal,
                    status=RealActionSmokeStatus.VALIDATION_BLOCKED,
                    reason=RealActionSmokeReason.VALIDATION_BLOCKED,
                    message="action validation blocked smoke action",
                    action_id=plan.action_id,
                    policy_decision=policy_evaluation.decision,
                    validation_result=validation,
                )
                self._record_result(result)

                return result

            approval_result = self._approval_runtime.evaluate(
                action_id=plan.action_id,
                risk=plan.risk,
                permission_decision=policy_evaluation.decision,
            )

            approval_required = self._approval_required_for_smoke(
                plan=plan,
                policy_decision=policy_evaluation.decision,
                approval_requires_human=approval_result.requires_human,
            )

            if approval_required and not request.approved:
                result = self._failed(
                    request=request,
                    proposal=proposal,
                    status=RealActionSmokeStatus.APPROVAL_REQUIRED,
                    reason=RealActionSmokeReason.APPROVAL_REQUIRED,
                    message=approval_result.message,
                    action_id=plan.action_id,
                    policy_decision=policy_evaluation.decision,
                    validation_result=validation,
                )
                self._record_result(result)

                return result

            schedule_result = self._scheduler.submit(
                plan,
                priority=request.priority,
                approved=request.approved,
                validated=True,
                metadata={"runtime": self.name},
            )

            if schedule_result.decision != ActionScheduleDecision.ACCEPTED:
                result = self._failed(
                    request=request,
                    proposal=proposal,
                    status=RealActionSmokeStatus.FAILED,
                    reason=RealActionSmokeReason.SCHEDULER_BLOCKED,
                    message=schedule_result.message,
                    action_id=plan.action_id,
                    policy_decision=policy_evaluation.decision,
                    validation_result=validation,
                )
                self._record_result(result)

                return result

            start_result = self._scheduler.start_action(plan.action_id)

            if start_result.decision != ActionScheduleDecision.STARTED:
                result = self._failed(
                    request=request,
                    proposal=proposal,
                    status=RealActionSmokeStatus.FAILED,
                    reason=RealActionSmokeReason.SCHEDULER_DEFERRED,
                    message=start_result.message,
                    action_id=plan.action_id,
                    policy_decision=policy_evaluation.decision,
                    validation_result=validation,
                )
                self._record_result(result)

                return result

            dispatch_results = self._dispatch_plan(plan)
            success = all(item.success for item in dispatch_results)

            self._scheduler.complete(
                plan.action_id,
                success=success,
                message=(
                    "smoke action completed"
                    if success
                    else "smoke action failed"
                ),
            )

            self._audit_log.record_execution_completed(
                action_id=plan.action_id,
                runtime=self.name,
                success=success,
                output_summary=self._summarize_dispatch(dispatch_results),
                status=ActionStatus.SUCCEEDED if success else ActionStatus.FAILED,
            )

            memory_proposal, memory_result = self._memory_step(
                request=request,
                plan=plan,
                success=success,
                dispatch_results=dispatch_results,
            )

            result = RealActionSmokeResult(
                request_id=request.request_id,
                action_id=plan.action_id,
                status=(
                    RealActionSmokeStatus.SUCCEEDED
                    if success
                    else RealActionSmokeStatus.FAILED
                ),
                reason=(
                    RealActionSmokeReason.PIPELINE_SUCCEEDED
                    if success
                    else RealActionSmokeReason.DISPATCH_FAILED
                ),
                success=success,
                message=(
                    "governed smoke action pipeline succeeded"
                    if success
                    else "governed smoke action pipeline failed"
                ),
                proposal=proposal,
                policy_decision=policy_evaluation.decision,
                validation_result=validation,
                dispatch_results=dispatch_results,
                memory_proposal=memory_proposal,
                memory_result=memory_result,
                metadata={"runtime": self.name},
            )
            self._record_result(result)

            return result

        except Exception as exc:
            with self._lock:
                self._last_error = f"{type(exc).__name__}: {exc}"

            result = RealActionSmokeResult(
                request_id=request.request_id,
                status=RealActionSmokeStatus.FAILED,
                reason=RealActionSmokeReason.DISPATCH_FAILED,
                success=False,
                message=f"{type(exc).__name__}: {exc}",
                metadata={"runtime": self.name},
            )
            self._record_result(result)

            return result

    def snapshot(self) -> RealActionSmokeRuntimeSnapshot:
        """
        Return runtime diagnostics.
        """

        with self._lock:
            return RealActionSmokeRuntimeSnapshot(
                name=self.name,
                request_count=self._request_count,
                success_count=self._success_count,
                failed_count=self._failed_count,
                blocked_count=self._blocked_count,
                last_status=self._last_status,
                last_reason=self._last_reason,
                last_error=self._last_error,
            )

    def reset(self) -> None:
        """
        Reset diagnostics only.
        """

        with self._lock:
            self._request_count = 0
            self._success_count = 0
            self._failed_count = 0
            self._blocked_count = 0
            self._last_status = None
            self._last_reason = None
            self._last_error = None

    def _dispatch_plan(self, plan: ActionPlan) -> tuple[SmokeDispatchResult, ...]:
        results: list[SmokeDispatchResult] = []

        for step in plan.steps:
            results.append(self._dispatch_step(step))

        return tuple(results)

    def _dispatch_step(self, step: ActionStep) -> SmokeDispatchResult:
        if step.kind == ActionKind.SHELL_COMMAND:
            return self._dispatch_shell(step)

        if step.kind == ActionKind.READ:
            return self._dispatch_file_read(step)

        if step.kind == ActionKind.SEARCH:
            return self._dispatch_file_search(step)

        if step.kind == ActionKind.IDE_OPEN_FILE:
            return self._dispatch_ide_open_file(step)

        if step.kind == ActionKind.PATCH:
            return self._dispatch_ide_prepare_patch(step)

        return SmokeDispatchResult(
            action_id=step.action_id,
            step_order=step.order,
            kind=SmokeDispatchKind.COGNITIVE,
            action_kind=step.kind,
            success=False,
            status=ActionStatus.BLOCKED,
            output=f"unsupported smoke dispatch step: {step.kind.value}",
            runtime_name=self.name,
            metadata={"reason": RealActionSmokeReason.DISPATCH_UNSUPPORTED.value},
        )

    def _dispatch_shell(self, step: ActionStep) -> SmokeDispatchResult:
        command = self._argument_text(step, "command")
        result = self._shell_runtime.execute(
            ShellCommandRequest(
                action_id=step.action_id,
                command=command,
            )
        )
        output = result.stdout or result.stderr or "shell completed"

        return SmokeDispatchResult(
            action_id=step.action_id,
            step_order=step.order,
            kind=SmokeDispatchKind.SHELL,
            action_kind=step.kind,
            success=result.success,
            status=result.status,
            output=output,
            runtime_name="safe_shell_runtime",
            shell_result=result,
        )

    def _dispatch_file_read(self, step: ActionStep) -> SmokeDispatchResult:
        path = self._argument_text(step, "path")
        result = self._file_runtime.execute(
            FileOperationRequest(
                action_id=step.action_id,
                kind=FileOperationKind.READ_FILE,
                path=path,
            )
        )

        return SmokeDispatchResult(
            action_id=step.action_id,
            step_order=step.order,
            kind=SmokeDispatchKind.FILE_SYSTEM,
            action_kind=step.kind,
            success=result.success,
            status=result.status,
            output=result.output or "file read completed",
            runtime_name="file_system_runtime",
            file_result=result,
        )

    def _dispatch_file_search(self, step: ActionStep) -> SmokeDispatchResult:
        path = str(step.arguments.get("path", "."))
        query = str(step.arguments.get("query", "*")).strip()
        pattern = query if "*" in query else f"*{query}*"

        result = self._file_runtime.execute(
            FileOperationRequest(
                action_id=step.action_id,
                kind=FileOperationKind.SEARCH_FILES,
                path=path,
                pattern=pattern,
                recursive=True,
            )
        )

        return SmokeDispatchResult(
            action_id=step.action_id,
            step_order=step.order,
            kind=SmokeDispatchKind.FILE_SYSTEM,
            action_kind=step.kind,
            success=result.success,
            status=result.status,
            output=result.output or "no matching files found",
            runtime_name="file_system_runtime",
            file_result=result,
        )

    def _dispatch_ide_open_file(self, step: ActionStep) -> SmokeDispatchResult:
        path = self._argument_text(step, "path")
        result = self._ide_runtime.execute(
            IdeActionRequest(
                action_id=step.action_id,
                kind=IdeActionKind.OPEN_FILE,
                path=path,
            )
        )

        return SmokeDispatchResult(
            action_id=step.action_id,
            step_order=step.order,
            kind=SmokeDispatchKind.IDE,
            action_kind=step.kind,
            success=result.success,
            status=result.status,
            output=result.output,
            runtime_name="ide_runtime",
            ide_result=result,
        )

    def _dispatch_ide_prepare_patch(self, step: ActionStep) -> SmokeDispatchResult:
        result = self._ide_runtime.execute(
            IdeActionRequest(
                action_id=step.action_id,
                kind=IdeActionKind.PREPARE_PATCH,
                path=self._argument_text(step, "path"),
                old_text=self._argument_text(step, "old_text"),
                new_text=self._argument_text(step, "new_text"),
            )
        )

        return SmokeDispatchResult(
            action_id=step.action_id,
            step_order=step.order,
            kind=SmokeDispatchKind.IDE,
            action_kind=step.kind,
            success=result.success,
            status=result.status,
            output=result.output,
            runtime_name="ide_runtime",
            ide_result=result,
        )

    def _memory_step(
        self,
        *,
        request: RealActionSmokeRequest,
        plan: ActionPlan,
        success: bool,
        dispatch_results: tuple[SmokeDispatchResult, ...],
    ) -> tuple[ToolMemoryWriteProposal | None, ToolMemoryWriteResult | None]:
        event = ToolMemoryEvent(
            action_id=plan.action_id,
            kind=(
                ToolMemoryEventKind.EXECUTION_COMPLETED
                if success
                else ToolMemoryEventKind.EXECUTION_FAILED
            ),
            summary=self._summarize_dispatch(dispatch_results),
            risk=plan.risk,
            status=ActionStatus.SUCCEEDED if success else ActionStatus.FAILED,
            user_visible=True,
            approved_by_user=request.approved,
            source_runtime=self.name,
            data={
                "request_id": request.request_id,
                "step_count": len(dispatch_results),
            },
        )
        proposal = self._memory_runtime.propose(event)

        if request.write_memory and self._config.allow_memory_write:
            return proposal, self._memory_runtime.write(proposal)

        return proposal, None

    def _risk_block_result(
        self,
        *,
        request: RealActionSmokeRequest,
        proposal: ActionPlanProposal,
        plan: ActionPlan,
    ) -> RealActionSmokeResult | None:
        if plan.risk == ActionRisk.LOW:
            return None

        if plan.risk == ActionRisk.MEDIUM and self._config.allow_medium_risk_smoke:
            return None

        return self._failed(
            request=request,
            proposal=proposal,
            status=RealActionSmokeStatus.APPROVAL_REQUIRED,
            reason=RealActionSmokeReason.APPROVAL_REQUIRED,
            message="smoke runtime only dispatches low-risk actions by default",
            action_id=plan.action_id,
        )

    def _approval_required_for_smoke(
        self,
        *,
        plan: ActionPlan,
        policy_decision: PermissionDecision,
        approval_requires_human: bool,
    ) -> bool:
        """
        Decide whether Step 16 smoke execution needs human approval.

        Low-risk smoke actions are allowed to proceed after policy and
        validation as long as policy did not deny them. This is necessary for
        safe smoke commands like pytest, ruff, mypy, read file, search project,
        and open file.

        Medium/high/critical actions still require approval unless explicitly
        handled by smoke configuration.
        """

        if policy_decision == PermissionDecision.DENY:
            return True

        if plan.risk == ActionRisk.LOW:
            return False

        if plan.risk == ActionRisk.MEDIUM:
            return approval_requires_human and not self._config.allow_medium_risk_smoke

        return approval_requires_human

    def _audit_plan(
        self,
        proposal: ActionPlanProposal,
        plan: ActionPlan,
    ) -> None:
        self._audit_log.record_plan_proposed(
            action_id=plan.action_id,
            summary=proposal.summary,
            plan_steps=len(plan.steps),
            risk=plan.risk,
            requires_approval=plan.requires_approval,
        )

    def _register_smoke_tools(self) -> None:
        """
        Register only the tool contracts needed for Step 16 smoke validation.

        The actual execution still belongs to SafeShellRuntime,
        FileSystemRuntime, and IdeRuntime.
        """

        descriptors = (
            ToolDescriptor(
                tool_id="tool_smoke_shell",
                name="smoke shell runtime",
                description="Smoke validation descriptor for safe shell steps",
                capabilities=(ToolCapability.RUN_SHELL_COMMAND,),
                supported_action_kinds=(ActionKind.SHELL_COMMAND,),
                scopes=(ActionScope.SHELL,),
                max_risk=ActionRisk.LOW,
                required_permission=PermissionDecision.ALLOW,
                availability=ToolAvailability.AVAILABLE,
                health=ToolHealth.HEALTHY,
                enabled=True,
            ),
            ToolDescriptor(
                tool_id="tool_smoke_file_system",
                name="smoke file-system runtime",
                description="Smoke validation descriptor for safe file steps",
                capabilities=(
                    ToolCapability.READ_FILE,
                    ToolCapability.SEARCH_FILES,
                ),
                supported_action_kinds=(ActionKind.READ, ActionKind.SEARCH),
                scopes=(ActionScope.WORKSPACE,),
                max_risk=ActionRisk.LOW,
                required_permission=PermissionDecision.ALLOW,
                availability=ToolAvailability.AVAILABLE,
                health=ToolHealth.HEALTHY,
                enabled=True,
            ),
            ToolDescriptor(
                tool_id="tool_smoke_ide",
                name="smoke IDE runtime",
                description="Smoke validation descriptor for safe IDE steps",
                capabilities=(
                    ToolCapability.READ_FILE,
                    ToolCapability.PATCH_FILE,
                ),
                supported_action_kinds=(
                    ActionKind.IDE_OPEN_FILE,
                    ActionKind.PATCH,
                ),
                scopes=(ActionScope.IDE,),
                max_risk=ActionRisk.MEDIUM,
                required_permission=PermissionDecision.REQUIRE_CONFIRMATION,
                availability=ToolAvailability.AVAILABLE,
                health=ToolHealth.HEALTHY,
                enabled=True,
            ),
        )

        for descriptor in descriptors:
            self._registry.register(descriptor)

    @staticmethod
    def _require_plan(proposal: ActionPlanProposal) -> ActionPlan:
        if proposal.action_plan is None:
            raise ValueError("planner proposal does not contain an action plan.")

        return proposal.action_plan

    @staticmethod
    def _argument_text(step: ActionStep, name: str) -> str:
        value = step.arguments.get(name)

        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"step argument is required: {name}")

        return value.strip()

    @staticmethod
    def _summarize_dispatch(
        dispatch_results: tuple[SmokeDispatchResult, ...],
    ) -> str:
        if not dispatch_results:
            return "no dispatch results"

        lines = []

        for result in dispatch_results:
            status = "ok" if result.success else "failed"
            lines.append(f"{result.step_order}:{result.action_kind.value}:{status}")

        return "; ".join(lines)

    def _failed(
        self,
        *,
        request: RealActionSmokeRequest,
        status: RealActionSmokeStatus,
        reason: RealActionSmokeReason,
        message: str,
        proposal: ActionPlanProposal | None = None,
        action_id: str | None = None,
        policy_decision: PermissionDecision | None = None,
        validation_result: ActionValidationResult | None = None,
    ) -> RealActionSmokeResult:
        if action_id is not None:
            self._audit_log.record(
                action_id=action_id,
                event_kind=ActionAuditEventKind.EXECUTION_FAILED,
                actor=ActionAuditActor.RUNTIME,
                outcome=ActionAuditOutcome.FAILED,
                message=message,
                source_runtime=self.name,
            )

        return RealActionSmokeResult(
            request_id=request.request_id,
            action_id=action_id,
            status=status,
            reason=reason,
            success=False,
            message=message,
            proposal=proposal,
            policy_decision=policy_decision,
            validation_result=validation_result,
            metadata={"runtime": self.name},
        )

    def _record_result(self, result: RealActionSmokeResult) -> None:
        with self._lock:
            self._last_status = result.status
            self._last_reason = result.reason

            if result.success:
                self._success_count += 1

            elif result.status in {
                RealActionSmokeStatus.POLICY_BLOCKED,
                RealActionSmokeStatus.VALIDATION_BLOCKED,
                RealActionSmokeStatus.APPROVAL_REQUIRED,
                RealActionSmokeStatus.UNSUPPORTED,
            }:
                self._blocked_count += 1

            else:
                self._failed_count += 1