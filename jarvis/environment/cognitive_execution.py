from __future__ import annotations

from enum import StrEnum
from threading import RLock
from typing import Any
from uuid import uuid4

from pydantic import Field, field_validator, model_validator

from jarvis.environment.app_control import AppControlResult
from jarvis.environment.models import EnvironmentSource, TrustCalibration
from jarvis.environment.trust_runtime import TrustPolicyClassification
from jarvis.orchestration.ids import utc_now
from jarvis.orchestration.models import OrchestrationModel


class CognitiveEnvironmentKind(StrEnum):
    IDE = "ide"
    BROWSER = "browser"
    TERMINAL = "terminal"
    DOCUMENT = "document"
    FILE_EXPLORER = "file_explorer"


class CognitiveExecutionCapability(StrEnum):
    OPEN_FILE = "open_file"
    JUMP_TO_LINE = "jump_to_line"
    READ_DIAGNOSTICS = "read_diagnostics"
    RUN_TESTS = "run_tests"
    PARSE_TERMINAL_OUTPUT = "parse_terminal_output"
    NAVIGATE_BROWSER = "navigate_browser"
    EXTRACT_ARTICLE = "extract_article"
    COMPARE_SOURCES = "compare_sources"
    EDIT_DOCUMENT = "edit_document"
    RESTORE_WORKSPACE = "restore_workspace"
    LIST_FILES = "list_files"
    OPEN_SELECTED_FILE = "open_selected_file"


class CognitiveExecutionStatus(StrEnum):
    EXECUTED = "executed"
    NEEDS_VERIFICATION = "needs_verification"
    NEEDS_APP_CONTROL = "needs_app_control"
    BLOCKED = "blocked"
    FAILED = "failed"


class CognitiveExecutionDecision(StrEnum):
    ALLOW = "allow"
    REQUIRE_VERIFICATION = "require_verification"
    REQUIRE_APP_CONTROL = "require_app_control"
    BLOCK = "block"


class CognitiveExecutionReason(StrEnum):
    SESSION_CREATED = "session_created"
    IDE_ACTION_EXECUTED = "ide_action_executed"
    BROWSER_ACTION_EXECUTED = "browser_action_executed"
    TERMINAL_ACTION_EXECUTED = "terminal_action_executed"
    DOCUMENT_ACTION_EXECUTED = "document_action_executed"
    FILE_EXPLORER_ACTION_EXECUTED = "file_explorer_action_executed"
    APP_CONTROL_NOT_ELIGIBLE = "app_control_not_eligible"
    WRONG_ENVIRONMENT = "wrong_environment"
    UNSAFE_DOCUMENT_EDIT = "unsafe_document_edit"
    COMMAND_REQUIRES_VERIFICATION = "command_requires_verification"
    UNKNOWN_CAPABILITY = "unknown_capability"
    SESSION_NOT_FOUND = "session_not_found"
    RUNTIME_RESET = "runtime_reset"


class CognitiveExecutionEventKind(StrEnum):
    SESSION_CREATED = "session_created"
    EXECUTION_COMPLETED = "execution_completed"
    EXECUTION_BLOCKED = "execution_blocked"
    RUNTIME_RESET = "runtime_reset"


class CognitiveExecutionRisk(StrEnum):
    SAFE = "safe"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    BLOCKED = "blocked"


class CognitiveExecutionStep(OrchestrationModel):
    step_id: str = Field(default_factory=lambda: f"cog_exec_step_{uuid4().hex}")
    description: str
    capability: CognitiveExecutionCapability
    requires_verification: bool = True
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("step_id", "description")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class CognitiveExecutionPlan(OrchestrationModel):
    plan_id: str = Field(default_factory=lambda: f"cog_exec_plan_{uuid4().hex}")
    environment: CognitiveEnvironmentKind
    capability: CognitiveExecutionCapability
    steps: tuple[CognitiveExecutionStep, ...]
    risk: CognitiveExecutionRisk
    requires_verification: bool = True
    expected_result: str
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("plan_id", "expected_result")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)

    @model_validator(mode="after")
    def _requires_steps(self) -> CognitiveExecutionPlan:
        if not self.steps:
            raise ValueError("cognitive execution plan requires steps.")
        return self


class CognitiveExecutionRequest(OrchestrationModel):
    request_id: str = Field(default_factory=lambda: f"cog_exec_req_{uuid4().hex}")
    session_id: str
    workspace_id: str
    environment: CognitiveEnvironmentKind
    capability: CognitiveExecutionCapability
    instruction: str
    app_control: AppControlResult
    target_path: str | None = None
    line_number: int | None = None
    text: str | None = None
    command: str | None = None
    url: str | None = None
    query: str | None = None
    sources: tuple[str, ...] = ()
    user_initiated: bool = True
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("request_id", "session_id", "workspace_id", "instruction")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)

    @field_validator("line_number")
    @classmethod
    def _positive_line(cls, value: int | None) -> int | None:
        if value is not None and value <= 0:
            raise ValueError("line_number must be positive.")
        return value


class CognitiveExecutionOutput(OrchestrationModel):
    output_id: str = Field(default_factory=lambda: f"cog_exec_out_{uuid4().hex}")
    summary: str
    extracted_text: str | None = None
    diagnostics: tuple[str, ...] = ()
    sources: tuple[str, ...] = ()
    changed_paths: tuple[str, ...] = ()
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("output_id", "summary")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class CognitiveExecutionAuditRecord(OrchestrationModel):
    audit_id: str = Field(default_factory=lambda: f"cog_exec_audit_{uuid4().hex}")
    request_id: str
    environment: CognitiveEnvironmentKind
    capability: CognitiveExecutionCapability
    status: CognitiveExecutionStatus
    decision: CognitiveExecutionDecision
    reason: CognitiveExecutionReason
    app_control_result_id: str
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("audit_id", "request_id", "app_control_result_id")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class CognitiveExecutionResult(OrchestrationModel):
    result_id: str = Field(default_factory=lambda: f"cog_exec_result_{uuid4().hex}")
    status: CognitiveExecutionStatus
    decision: CognitiveExecutionDecision
    reason: CognitiveExecutionReason
    request: CognitiveExecutionRequest
    plan: CognitiveExecutionPlan | None = None
    output: CognitiveExecutionOutput | None = None
    audit: CognitiveExecutionAuditRecord
    trust: TrustCalibration
    safe_for_followup_action: bool = False
    message: str
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("result_id", "message")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)

    @model_validator(mode="after")
    def _safe_requires_output(self) -> CognitiveExecutionResult:
        if self.safe_for_followup_action and self.output is None:
            raise ValueError("safe follow-up execution requires output.")
        return self


class CognitiveExecutionSession(OrchestrationModel):
    session_id: str = Field(default_factory=lambda: f"cog_exec_session_{uuid4().hex}")
    workspace_id: str
    last_environment: CognitiveEnvironmentKind | None = None
    execution_count: int = Field(default=0, ge=0)
    created_at: object = Field(default_factory=utc_now)
    updated_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("session_id", "workspace_id")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class CognitiveExecutionRuntimeEvent(OrchestrationModel):
    event_id: str = Field(default_factory=lambda: f"cog_exec_event_{uuid4().hex}")
    kind: CognitiveExecutionEventKind
    reason: CognitiveExecutionReason
    session_id: str | None = None
    result_id: str | None = None
    request_id: str | None = None
    audit_id: str | None = None
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("event_id")
    @classmethod
    def _required_id(cls, value: str) -> str:
        return _clean_required(value)


class CognitiveExecutionRuntimeSnapshot(OrchestrationModel):
    name: str
    session_count: int = Field(ge=0)
    result_count: int = Field(ge=0)
    executed_count: int = Field(ge=0)
    verification_count: int = Field(ge=0)
    blocked_count: int = Field(ge=0)
    audit_count: int = Field(ge=0)
    runtime_event_count: int = Field(ge=0)
    last_reason: CognitiveExecutionReason | None = None
    created_at: object = Field(default_factory=utc_now)

    @field_validator("name")
    @classmethod
    def _required_name(cls, value: str) -> str:
        return _clean_required(value)


class IDEExecutionRuntime:
    def execute(self, request: CognitiveExecutionRequest) -> CognitiveExecutionResult:
        blocked = _precheck(request, CognitiveEnvironmentKind.IDE)
        if blocked is not None:
            return blocked

        if request.capability == CognitiveExecutionCapability.OPEN_FILE:
            return _allowed(
                request=request,
                reason=CognitiveExecutionReason.IDE_ACTION_EXECUTED,
                risk=CognitiveExecutionRisk.LOW,
                steps=(
                    _step("open target file in IDE", request.capability),
                    _step("verify editor tab opened", request.capability),
                ),
                summary=f"opened file: {request.target_path or 'unknown'}",
                expected="file is visible in editor",
            )

        if request.capability == CognitiveExecutionCapability.JUMP_TO_LINE:
            return _allowed(
                request=request,
                reason=CognitiveExecutionReason.IDE_ACTION_EXECUTED,
                risk=CognitiveExecutionRisk.LOW,
                steps=(
                    _step("open file before line jump", request.capability),
                    _step("jump cursor to requested line", request.capability),
                    _step("verify line is visible", request.capability),
                ),
                summary=f"jumped to line {request.line_number}",
                expected="requested line is visible and focused",
            )

        if request.capability == CognitiveExecutionCapability.READ_DIAGNOSTICS:
            return _allowed(
                request=request,
                reason=CognitiveExecutionReason.IDE_ACTION_EXECUTED,
                risk=CognitiveExecutionRisk.SAFE,
                steps=(
                    _step("read diagnostics panel", request.capability),
                    _step("extract visible error messages", request.capability),
                ),
                summary="read IDE diagnostics",
                diagnostics=("diagnostics extraction planned",),
                expected="diagnostics are available as structured text",
            )

        if request.capability == CognitiveExecutionCapability.RUN_TESTS:
            return _needs_verification(
                request=request,
                reason=CognitiveExecutionReason.COMMAND_REQUIRES_VERIFICATION,
                risk=CognitiveExecutionRisk.MEDIUM,
                steps=(
                    _step("select project test command", request.capability),
                    _step("run tests through terminal/task runner", request.capability),
                    _step("parse test output", request.capability),
                ),
                summary="test run requires verification before follow-up action",
                expected="test output appears and is parsed",
            )

        if request.capability == CognitiveExecutionCapability.RESTORE_WORKSPACE:
            return _allowed(
                request=request,
                reason=CognitiveExecutionReason.IDE_ACTION_EXECUTED,
                risk=CognitiveExecutionRisk.LOW,
                steps=(
                    _step("restore IDE workspace", request.capability),
                    _step(
                        "verify project files and terminal state",
                         request.capability
                    ),
                ),
                summary="IDE workspace restore planned",
                expected="workspace state is restored and visible",
            )

        return _unknown(request)


class BrowserExecutionRuntime:
    def execute(self, request: CognitiveExecutionRequest) -> CognitiveExecutionResult:
        blocked = _precheck(request, CognitiveEnvironmentKind.BROWSER)
        if blocked is not None:
            return blocked

        if request.capability == CognitiveExecutionCapability.NAVIGATE_BROWSER:
            return _needs_verification(
                request=request,
                reason=CognitiveExecutionReason.BROWSER_ACTION_EXECUTED,
                risk=CognitiveExecutionRisk.MEDIUM,
                steps=(
                    _step(
                        "navigate browser to requested URL/query",
                         request.capability
                    ),
                    _step("verify final URL and page title", request.capability),
                ),
                summary=f"browser navigation planned: {request.url or request.query}",
                expected="browser reaches requested page",
            )

        if request.capability == CognitiveExecutionCapability.EXTRACT_ARTICLE:
            return _allowed(
                request=request,
                reason=CognitiveExecutionReason.BROWSER_ACTION_EXECUTED,
                risk=CognitiveExecutionRisk.SAFE,
                steps=(
                    _step("identify main article content", request.capability),
                    _step("extract article text", request.capability),
                    _step("preserve source URL/title metadata", request.capability),
                ),
                summary="article extraction planned",
                extracted_text="article extraction placeholder",
                expected="main article content is extracted",
            )

        if request.capability == CognitiveExecutionCapability.COMPARE_SOURCES:
            return _allowed(
                request=request,
                reason=CognitiveExecutionReason.BROWSER_ACTION_EXECUTED,
                risk=CognitiveExecutionRisk.SAFE,
                steps=(
                    _step("collect visible source summaries", request.capability),
                    _step("compare claims across sources", request.capability),
                    _step("flag contradictions", request.capability),
                ),
                summary="source comparison planned",
                sources=request.sources,
                expected="sources are compared with contradiction notes",
            )

        return _unknown(request)


class TerminalExecutionRuntime:
    def execute(self, request: CognitiveExecutionRequest) -> CognitiveExecutionResult:
        blocked = _precheck(request, CognitiveEnvironmentKind.TERMINAL)
        if blocked is not None:
            return blocked

        if request.capability == CognitiveExecutionCapability.RUN_TESTS:
            return _needs_verification(
                request=request,
                reason=CognitiveExecutionReason.TERMINAL_ACTION_EXECUTED,
                risk=CognitiveExecutionRisk.MEDIUM,
                steps=(
                    _step("prepare terminal command", request.capability),
                    _step(
                        "execute command through governed terminal path",
                         request.capability
                    ),
                    _step("parse terminal output", request.capability),
                ),
                summary=f"terminal test command planned: {request.command or 'pytest'}",
                expected="terminal returns test result output",
            )

        if request.capability == CognitiveExecutionCapability.PARSE_TERMINAL_OUTPUT:
            return _allowed(
                request=request,
                reason=CognitiveExecutionReason.TERMINAL_ACTION_EXECUTED,
                risk=CognitiveExecutionRisk.SAFE,
                steps=(
                    _step("read terminal buffer", request.capability),
                    _step(
                        "extract errors, warnings, and summaries",
                         request.capability
                    ),
                ),
                summary="terminal output parsing planned",
                extracted_text=request.text,
                diagnostics=("terminal output parsed",),
                expected="terminal output is structured",
            )

        return _unknown(request)


class DocumentExecutionRuntime:
    def execute(self, request: CognitiveExecutionRequest) -> CognitiveExecutionResult:
        blocked = _precheck(request, CognitiveEnvironmentKind.DOCUMENT)
        if blocked is not None:
            return blocked

        if request.capability == CognitiveExecutionCapability.EDIT_DOCUMENT:
            if not request.user_initiated:
                return _blocked(
                    request=request,
                    reason=CognitiveExecutionReason.UNSAFE_DOCUMENT_EDIT,
                    message="document edit must be user initiated",
                )

            return _needs_verification(
                request=request,
                reason=CognitiveExecutionReason.DOCUMENT_ACTION_EXECUTED,
                risk=CognitiveExecutionRisk.MEDIUM,
                steps=(
                    _step("locate target document region", request.capability),
                    _step("apply edit through governed input path", request.capability),
                    _step(
                        "verify document content changed as expected",
                         request.capability
                    ),
                ),
                summary="document edit planned",
                changed_paths=(request.target_path,) if request.target_path else (),
                expected="document edit is visible and verified",
            )

        if request.capability == CognitiveExecutionCapability.EXTRACT_ARTICLE:
            return _allowed(
                request=request,
                reason=CognitiveExecutionReason.DOCUMENT_ACTION_EXECUTED,
                risk=CognitiveExecutionRisk.SAFE,
                steps=(
                    _step("read visible document text", request.capability),
                    _step("extract structured sections", request.capability),
                ),
                summary="document text extraction planned",
                extracted_text=request.text,
                expected="document content is extracted",
            )

        return _unknown(request)


class FileExplorerExecutionRuntime:
    def execute(self, request: CognitiveExecutionRequest) -> CognitiveExecutionResult:
        blocked = _precheck(request, CognitiveEnvironmentKind.FILE_EXPLORER)
        if blocked is not None:
            return blocked

        if request.capability == CognitiveExecutionCapability.LIST_FILES:
            return _allowed(
                request=request,
                reason=CognitiveExecutionReason.FILE_EXPLORER_ACTION_EXECUTED,
                risk=CognitiveExecutionRisk.SAFE,
                steps=(
                    _step("read current folder view", request.capability),
                    _step("extract file names and metadata", request.capability),
                ),
                summary="file list extraction planned",
                expected="visible files are listed",
            )

        if request.capability == CognitiveExecutionCapability.OPEN_SELECTED_FILE:
            return _needs_verification(
                request=request,
                reason=CognitiveExecutionReason.FILE_EXPLORER_ACTION_EXECUTED,
                risk=CognitiveExecutionRisk.LOW,
                steps=(
                    _step("verify selected file", request.capability),
                    _step("open selected file", request.capability),
                    _step("verify owning app opened", request.capability),
                ),
                summary="selected file open planned",
                expected="selected file opens in associated app",
            )

        return _unknown(request)


class CognitiveExecutionRuntime:
    def __init__(
        self,
        *,
        name: str = "cognitive_execution_runtime",
        ide: IDEExecutionRuntime | None = None,
        browser: BrowserExecutionRuntime | None = None,
        terminal: TerminalExecutionRuntime | None = None,
        document: DocumentExecutionRuntime | None = None,
        file_explorer: FileExplorerExecutionRuntime | None = None,
    ) -> None:
        cleaned = name.strip()
        if not cleaned:
            raise ValueError("name cannot be empty.")

        self._name = cleaned
        self._ide = ide or IDEExecutionRuntime()
        self._browser = browser or BrowserExecutionRuntime()
        self._terminal = terminal or TerminalExecutionRuntime()
        self._document = document or DocumentExecutionRuntime()
        self._file_explorer = file_explorer or FileExplorerExecutionRuntime()
        self._sessions: dict[str, CognitiveExecutionSession] = {}
        self._results: list[CognitiveExecutionResult] = []
        self._audits: list[CognitiveExecutionAuditRecord] = []
        self._events: list[CognitiveExecutionRuntimeEvent] = []
        self._lock = RLock()
        self._last_reason: CognitiveExecutionReason | None = None

    @property
    def name(self) -> str:
        return self._name

    def create_session(
        self,
        *,
        workspace_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> CognitiveExecutionSession:
        session = CognitiveExecutionSession(
            workspace_id=workspace_id,
            metadata=metadata or {},
        )
        event = self._event(
            kind=CognitiveExecutionEventKind.SESSION_CREATED,
            reason=CognitiveExecutionReason.SESSION_CREATED,
            session_id=session.session_id,
        )

        with self._lock:
            self._sessions[session.session_id] = session
            self._events.append(event)
            self._last_reason = event.reason

        return session

    def execute(
        self,
        request: CognitiveExecutionRequest,
    ) -> CognitiveExecutionResult:
        if self.session_for(request.session_id) is None:
            result = _blocked(
                request=request,
                reason=CognitiveExecutionReason.SESSION_NOT_FOUND,
                status=CognitiveExecutionStatus.FAILED,
                message="cognitive execution session not found",
            )
            self._record_result(result)
            return result

        result = self._route(request)
        self._record_result(result)
        self._touch_session_from_result(result)

        return result

    def session_for(
        self,
        session_id: str,
    ) -> CognitiveExecutionSession | None:
        with self._lock:
            return self._sessions.get(session_id)

    def results(self) -> tuple[CognitiveExecutionResult, ...]:
        with self._lock:
            return tuple(self._results)

    def audits(self) -> tuple[CognitiveExecutionAuditRecord, ...]:
        with self._lock:
            return tuple(self._audits)

    def events(self) -> tuple[CognitiveExecutionRuntimeEvent, ...]:
        with self._lock:
            return tuple(self._events)

    def snapshot(self) -> CognitiveExecutionRuntimeSnapshot:
        with self._lock:
            return CognitiveExecutionRuntimeSnapshot(
                name=self.name,
                session_count=len(self._sessions),
                result_count=len(self._results),
                executed_count=sum(
                    1
                    for result in self._results
                    if result.status == CognitiveExecutionStatus.EXECUTED
                ),
                verification_count=sum(
                    1
                    for result in self._results
                    if result.status == CognitiveExecutionStatus.NEEDS_VERIFICATION
                ),
                blocked_count=sum(
                    1
                    for result in self._results
                    if result.status
                    in {
                        CognitiveExecutionStatus.BLOCKED,
                        CognitiveExecutionStatus.FAILED,
                        CognitiveExecutionStatus.NEEDS_APP_CONTROL,
                    }
                ),
                audit_count=len(self._audits),
                runtime_event_count=len(self._events),
                last_reason=self._last_reason,
            )

    def reset(self) -> None:
        event = self._event(
            kind=CognitiveExecutionEventKind.RUNTIME_RESET,
            reason=CognitiveExecutionReason.RUNTIME_RESET,
        )

        with self._lock:
            self._sessions.clear()
            self._results.clear()
            self._audits.clear()
            self._events.clear()
            self._events.append(event)
            self._last_reason = event.reason

    def _route(
        self,
        request: CognitiveExecutionRequest,
    ) -> CognitiveExecutionResult:
        if request.environment == CognitiveEnvironmentKind.IDE:
            return self._ide.execute(request)

        if request.environment == CognitiveEnvironmentKind.BROWSER:
            return self._browser.execute(request)

        if request.environment == CognitiveEnvironmentKind.TERMINAL:
            return self._terminal.execute(request)

        if request.environment == CognitiveEnvironmentKind.DOCUMENT:
            return self._document.execute(request)

        return self._file_explorer.execute(request)

    def _record_result(self, result: CognitiveExecutionResult) -> None:
        event = self._event(
            kind=(
                CognitiveExecutionEventKind.EXECUTION_COMPLETED
                if result.status
                in {
                    CognitiveExecutionStatus.EXECUTED,
                    CognitiveExecutionStatus.NEEDS_VERIFICATION,
                }
                else CognitiveExecutionEventKind.EXECUTION_BLOCKED
            ),
            reason=result.reason,
            session_id=result.request.session_id,
            result_id=result.result_id,
            request_id=result.request.request_id,
            audit_id=result.audit.audit_id,
            metadata={
                "status": result.status.value,
                "decision": result.decision.value,
            },
        )

        with self._lock:
            self._results.append(result)
            self._audits.append(result.audit)
            self._events.append(event)
            self._last_reason = result.reason

    def _touch_session_from_result(self, result: CognitiveExecutionResult) -> None:
        session = self._sessions.get(result.request.session_id)
        if session is None:
            return

        count_increment = 1
        if result.status in {
            CognitiveExecutionStatus.BLOCKED,
            CognitiveExecutionStatus.FAILED,
            CognitiveExecutionStatus.NEEDS_APP_CONTROL,
        }:
            count_increment = 0

        self._sessions[result.request.session_id] = session.model_copy(
            update={
                "updated_at": utc_now(),
                "last_environment": result.request.environment,
                "execution_count": session.execution_count + count_increment,
            }
        )

    @staticmethod
    def _event(
        *,
        kind: CognitiveExecutionEventKind,
        reason: CognitiveExecutionReason,
        session_id: str | None = None,
        result_id: str | None = None,
        request_id: str | None = None,
        audit_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> CognitiveExecutionRuntimeEvent:
        return CognitiveExecutionRuntimeEvent(
            kind=kind,
            reason=reason,
            session_id=session_id,
            result_id=result_id,
            request_id=request_id,
            audit_id=audit_id,
            metadata=metadata or {},
        )


def _precheck(
    request: CognitiveExecutionRequest,
    expected: CognitiveEnvironmentKind,
) -> CognitiveExecutionResult | None:
    if not request.app_control.control_eligible:
        return _blocked(
            request=request,
            reason=CognitiveExecutionReason.APP_CONTROL_NOT_ELIGIBLE,
            status=CognitiveExecutionStatus.NEEDS_APP_CONTROL,
            decision=CognitiveExecutionDecision.REQUIRE_APP_CONTROL,
            message="app control must be eligible before cognitive execution",
        )

    if request.environment != expected:
        return _blocked(
            request=request,
            reason=CognitiveExecutionReason.WRONG_ENVIRONMENT,
            message=f"request environment must be {expected.value}",
        )

    return None


def _allowed(
    *,
    request: CognitiveExecutionRequest,
    reason: CognitiveExecutionReason,
    risk: CognitiveExecutionRisk,
    steps: tuple[CognitiveExecutionStep, ...],
    summary: str,
    expected: str,
    extracted_text: str | None = None,
    diagnostics: tuple[str, ...] = (),
    sources: tuple[str, ...] = (),
    changed_paths: tuple[str, ...] = (),
) -> CognitiveExecutionResult:
    plan = CognitiveExecutionPlan(
        environment=request.environment,
        capability=request.capability,
        steps=steps,
        risk=risk,
        requires_verification=False,
        expected_result=expected,
    )
    output = CognitiveExecutionOutput(
        summary=summary,
        extracted_text=extracted_text,
        diagnostics=diagnostics,
        sources=sources,
        changed_paths=changed_paths,
    )

    return _result(
        request=request,
        status=CognitiveExecutionStatus.EXECUTED,
        decision=CognitiveExecutionDecision.ALLOW,
        reason=reason,
        plan=plan,
        output=output,
        safe_for_followup_action=True,
        message=summary,
    )


def _needs_verification(
    *,
    request: CognitiveExecutionRequest,
    reason: CognitiveExecutionReason,
    risk: CognitiveExecutionRisk,
    steps: tuple[CognitiveExecutionStep, ...],
    summary: str,
    expected: str,
    changed_paths: tuple[str, ...] = (),
) -> CognitiveExecutionResult:
    plan = CognitiveExecutionPlan(
        environment=request.environment,
        capability=request.capability,
        steps=steps,
        risk=risk,
        requires_verification=True,
        expected_result=expected,
    )
    output = CognitiveExecutionOutput(
        summary=summary,
        changed_paths=changed_paths,
    )

    return _result(
        request=request,
        status=CognitiveExecutionStatus.NEEDS_VERIFICATION,
        decision=CognitiveExecutionDecision.REQUIRE_VERIFICATION,
        reason=reason,
        plan=plan,
        output=output,
        safe_for_followup_action=False,
        message=summary,
    )


def _unknown(request: CognitiveExecutionRequest) -> CognitiveExecutionResult:
    return _blocked(
        request=request,
        reason=CognitiveExecutionReason.UNKNOWN_CAPABILITY,
        message="capability is not supported by this cognitive environment",
    )


def _blocked(
    *,
    request: CognitiveExecutionRequest,
    reason: CognitiveExecutionReason,
    message: str,
    status: CognitiveExecutionStatus = CognitiveExecutionStatus.BLOCKED,
    decision: CognitiveExecutionDecision = CognitiveExecutionDecision.BLOCK,
) -> CognitiveExecutionResult:
    return _result(
        request=request,
        status=status,
        decision=decision,
        reason=reason,
        plan=None,
        output=None,
        safe_for_followup_action=False,
        message=message,
    )


def _result(
    *,
    request: CognitiveExecutionRequest,
    status: CognitiveExecutionStatus,
    decision: CognitiveExecutionDecision,
    reason: CognitiveExecutionReason,
    plan: CognitiveExecutionPlan | None,
    output: CognitiveExecutionOutput | None,
    safe_for_followup_action: bool,
    message: str,
) -> CognitiveExecutionResult:
    audit = CognitiveExecutionAuditRecord(
        request_id=request.request_id,
        environment=request.environment,
        capability=request.capability,
        status=status,
        decision=decision,
        reason=reason,
        app_control_result_id=request.app_control.result_id,
    )
    confidence = 0.86 if output is not None else 0.20

    return CognitiveExecutionResult(
        status=status,
        decision=decision,
        reason=reason,
        request=request,
        plan=plan,
        output=output,
        audit=audit,
        trust=TrustCalibration(
            confidence=confidence,
            stability=max(0.0, min(1.0, confidence + 0.05)),
            ambiguity=1.0 - confidence,
            source=EnvironmentSource.OS_OBSERVER,
            reason="specialized cognitive execution runtime",
            metadata={"policy": TrustPolicyClassification.REVIEW.value},
        ),
        safe_for_followup_action=safe_for_followup_action,
        message=message,
    )


def _step(
    description: str,
    capability: CognitiveExecutionCapability,
) -> CognitiveExecutionStep:
    return CognitiveExecutionStep(
        description=description,
        capability=capability,
    )


def _clean_required(value: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError("field cannot be empty.")
    return cleaned