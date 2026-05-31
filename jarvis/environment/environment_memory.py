from __future__ import annotations

from enum import StrEnum
from hashlib import sha256
from threading import RLock
from typing import Any
from uuid import uuid4

from pydantic import Field, field_validator, model_validator

from jarvis.environment.models import EnvironmentSource, TrustCalibration
from jarvis.environment.trust_runtime import TrustPolicyClassification
from jarvis.orchestration.ids import utc_now
from jarvis.orchestration.models import OrchestrationModel


class EnvironmentMemoryScope(StrEnum):
    SESSION = "session"
    PROJECT = "project"
    WORKSPACE = "workspace"


class WorkflowStage(StrEnum):
    UNKNOWN = "unknown"
    EXPLORING = "exploring"
    CODING = "coding"
    DEBUGGING = "debugging"
    TESTING = "testing"
    RESEARCHING = "researching"
    WRITING = "writing"
    REVIEWING = "reviewing"
    BLOCKED = "blocked"
    PAUSED = "paused"
    COMPLETE = "complete"


class EnvironmentMemoryStatus(StrEnum):
    STORED = "stored"
    RECALLED = "recalled"
    CONTINUITY_READY = "continuity_ready"
    BLOCKED = "blocked"
    FAILED = "failed"


class EnvironmentMemoryDecision(StrEnum):
    STORE = "store"
    RECALL = "recall"
    CONTINUE_WORKFLOW = "continue_workflow"
    BLOCK = "block"


class EnvironmentMemoryReason(StrEnum):
    SESSION_CREATED = "session_created"
    WORKFLOW_ENTRY_STORED = "workflow_entry_stored"
    SESSION_MEMORY_RECALLED = "session_memory_recalled"
    PROJECT_MEMORY_RECALLED = "project_memory_recalled"
    WORKSPACE_MEMORY_RECALLED = "workspace_memory_recalled"
    CONTINUITY_TOKEN_CREATED = "continuity_token_created"
    RAW_SCREEN_MEMORY_BLOCKED = "raw_screen_memory_blocked"
    EMPTY_MEMORY_BLOCKED = "empty_memory_blocked"
    SESSION_NOT_FOUND = "session_not_found"
    RUNTIME_RESET = "runtime_reset"


class EnvironmentMemoryEventKind(StrEnum):
    SESSION_CREATED = "session_created"
    MEMORY_STORED = "memory_stored"
    MEMORY_RECALLED = "memory_recalled"
    CONTINUITY_READY = "continuity_ready"
    MEMORY_BLOCKED = "memory_blocked"
    RUNTIME_RESET = "runtime_reset"


class WorkflowMemoryKind(StrEnum):
    APP_CONTEXT = "app_context"
    PROJECT_CONTEXT = "project_context"
    FILE_CONTEXT = "file_context"
    CURSOR_CONTEXT = "cursor_context"
    TERMINAL_CONTEXT = "terminal_context"
    ERROR_CONTEXT = "error_context"
    TODO_CONTEXT = "todo_context"
    STAGE_CONTEXT = "stage_context"


class CursorPosition(OrchestrationModel):
    file_path: str
    line: int = Field(ge=1)
    column: int = Field(default=1, ge=1)
    symbol: str | None = None
    created_at: object = Field(default_factory=utc_now)

    @field_validator("file_path")
    @classmethod
    def _required_path(cls, value: str) -> str:
        return _clean_required(value)


class WorkspaceMemoryEntry(OrchestrationModel):
    """
    Semantic workflow memory.

    This is intentionally not a screenshot, screen recording, raw OCR dump, or
    raw desktop transcript. It stores compact workflow continuity.
    """

    entry_id: str = Field(default_factory=lambda: f"workspace_mem_{uuid4().hex}")
    session_id: str
    workspace_id: str
    app_name: str
    project_path: str | None = None
    active_files: tuple[str, ...] = ()
    cursor_positions: tuple[CursorPosition, ...] = ()
    terminal_directory: str | None = None
    recent_commands: tuple[str, ...] = ()
    visible_errors: tuple[str, ...] = ()
    pending_todos: tuple[str, ...] = ()
    workflow_stage: WorkflowStage = WorkflowStage.UNKNOWN
    continuity_token: str
    source: EnvironmentSource = EnvironmentSource.OS_OBSERVER
    trust: TrustCalibration
    policy: TrustPolicyClassification = TrustPolicyClassification.SAFE
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("entry_id", "session_id", "workspace_id", "app_name")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)

    @model_validator(mode="after")
    def _semantic_memory_only(self) -> WorkspaceMemoryEntry:
        if _looks_like_raw_screen_memory(self.metadata):
            raise ValueError("environment memory cannot store raw screen data.")

        has_semantic_state = any(
            (
                self.project_path,
                self.active_files,
                self.cursor_positions,
                self.terminal_directory,
                self.recent_commands,
                self.visible_errors,
                self.pending_todos,
                self.workflow_stage != WorkflowStage.UNKNOWN,
            )
        )

        if not has_semantic_state:
            raise ValueError("workspace memory entry requires semantic state.")

        return self


class SessionContinuity(OrchestrationModel):
    continuity_id: str = Field(default_factory=lambda: f"session_cont_{uuid4().hex}")
    session_id: str
    workspace_id: str
    continuity_token: str
    active_entry: WorkspaceMemoryEntry
    workflow_stage: WorkflowStage
    resume_summary: str
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator(
        "continuity_id",
        "session_id",
        "workspace_id",
        "continuity_token",
        "resume_summary",
    )
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class ProjectMemory(OrchestrationModel):
    project_id: str = Field(default_factory=lambda: f"project_memory_{uuid4().hex}")
    workspace_id: str
    project_path: str
    latest_entry: WorkspaceMemoryEntry
    entry_count: int = Field(ge=1)
    active_files: tuple[str, ...] = ()
    visible_errors: tuple[str, ...] = ()
    pending_todos: tuple[str, ...] = ()
    workflow_stage: WorkflowStage
    updated_at: object = Field(default_factory=utc_now)

    @field_validator("project_id", "workspace_id", "project_path")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class WorkflowMemoryWrite(OrchestrationModel):
    write_id: str = Field(default_factory=lambda: f"workflow_write_{uuid4().hex}")
    entry: WorkspaceMemoryEntry
    scope: EnvironmentMemoryScope
    accepted: bool
    reason: EnvironmentMemoryReason
    created_at: object = Field(default_factory=utc_now)

    @field_validator("write_id")
    @classmethod
    def _required_id(cls, value: str) -> str:
        return _clean_required(value)


class WorkflowMemoryRead(OrchestrationModel):
    read_id: str = Field(default_factory=lambda: f"workflow_read_{uuid4().hex}")
    scope: EnvironmentMemoryScope
    entries: tuple[WorkspaceMemoryEntry, ...]
    reason: EnvironmentMemoryReason
    created_at: object = Field(default_factory=utc_now)

    @field_validator("read_id")
    @classmethod
    def _required_id(cls, value: str) -> str:
        return _clean_required(value)


class EnvironmentMemoryAuditRecord(OrchestrationModel):
    audit_id: str = Field(default_factory=lambda: f"env_memory_audit_{uuid4().hex}")
    status: EnvironmentMemoryStatus
    decision: EnvironmentMemoryDecision
    reason: EnvironmentMemoryReason
    entry_id: str | None = None
    session_id: str | None = None
    workspace_id: str | None = None
    raw_screen_logged: bool = False
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("audit_id")
    @classmethod
    def _required_id(cls, value: str) -> str:
        return _clean_required(value)

    @model_validator(mode="after")
    def _no_raw_screen_log(self) -> EnvironmentMemoryAuditRecord:
        if self.raw_screen_logged:
            raise ValueError("environment memory audit must not log raw screen.")

        return self


class EnvironmentMemoryResult(OrchestrationModel):
    result_id: str = Field(default_factory=lambda: f"env_memory_result_{uuid4().hex}")
    status: EnvironmentMemoryStatus
    decision: EnvironmentMemoryDecision
    reason: EnvironmentMemoryReason
    entry: WorkspaceMemoryEntry | None = None
    entries: tuple[WorkspaceMemoryEntry, ...] = ()
    continuity: SessionContinuity | None = None
    project_memory: ProjectMemory | None = None
    audit: EnvironmentMemoryAuditRecord
    trust: TrustCalibration
    message: str
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("result_id", "message")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class EnvironmentMemorySession(OrchestrationModel):
    session_id: str = Field(default_factory=lambda: f"env_memory_session_{uuid4().hex}")
    workspace_id: str
    store_count: int = Field(default=0, ge=0)
    recall_count: int = Field(default=0, ge=0)
    continuity_count: int = Field(default=0, ge=0)
    blocked_count: int = Field(default=0, ge=0)
    created_at: object = Field(default_factory=utc_now)
    updated_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("session_id", "workspace_id")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class EnvironmentMemoryRuntimeEvent(OrchestrationModel):
    event_id: str = Field(default_factory=lambda: f"env_memory_event_{uuid4().hex}")
    kind: EnvironmentMemoryEventKind
    reason: EnvironmentMemoryReason
    session_id: str | None = None
    result_id: str | None = None
    audit_id: str | None = None
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("event_id")
    @classmethod
    def _required_id(cls, value: str) -> str:
        return _clean_required(value)


class EnvironmentMemoryRuntimeSnapshot(OrchestrationModel):
    name: str
    session_count: int = Field(ge=0)
    stored_entry_count: int = Field(ge=0)
    result_count: int = Field(ge=0)
    store_count: int = Field(ge=0)
    recall_count: int = Field(ge=0)
    continuity_count: int = Field(ge=0)
    blocked_count: int = Field(ge=0)
    audit_count: int = Field(ge=0)
    runtime_event_count: int = Field(ge=0)
    last_reason: EnvironmentMemoryReason | None = None
    created_at: object = Field(default_factory=utc_now)

    @field_validator("name")
    @classmethod
    def _required_name(cls, value: str) -> str:
        return _clean_required(value)


class WorkflowMemoryGateway:
    """
    Gateway boundary for environment memory.

    In production this is where Phase 4 Memory Gateway integration plugs in.
    The runtime does not write directly to a database or vector store.
    """

    def __init__(self) -> None:
        self._entries: list[WorkspaceMemoryEntry] = []
        self._lock = RLock()

    def store(
        self,
        *,
        entry: WorkspaceMemoryEntry,
        scope: EnvironmentMemoryScope,
    ) -> WorkflowMemoryWrite:
        with self._lock:
            self._entries.append(entry)

        return WorkflowMemoryWrite(
            entry=entry,
            scope=scope,
            accepted=True,
            reason=EnvironmentMemoryReason.WORKFLOW_ENTRY_STORED,
        )

    def recall_session(self, *, session_id: str) -> WorkflowMemoryRead:
        with self._lock:
            entries = tuple(
                entry for entry in self._entries if entry.session_id == session_id
            )

        return WorkflowMemoryRead(
            scope=EnvironmentMemoryScope.SESSION,
            entries=entries,
            reason=EnvironmentMemoryReason.SESSION_MEMORY_RECALLED,
        )

    def recall_project(
        self,
        *,
        workspace_id: str,
        project_path: str,
    ) -> WorkflowMemoryRead:
        with self._lock:
            entries = tuple(
                entry
                for entry in self._entries
                if entry.workspace_id == workspace_id
                and entry.project_path == project_path
            )

        return WorkflowMemoryRead(
            scope=EnvironmentMemoryScope.PROJECT,
            entries=entries,
            reason=EnvironmentMemoryReason.PROJECT_MEMORY_RECALLED,
        )

    def recall_workspace(self, *, workspace_id: str) -> WorkflowMemoryRead:
        with self._lock:
            entries = tuple(
                entry for entry in self._entries if entry.workspace_id == workspace_id
            )

        return WorkflowMemoryRead(
            scope=EnvironmentMemoryScope.WORKSPACE,
            entries=entries,
            reason=EnvironmentMemoryReason.WORKSPACE_MEMORY_RECALLED,
        )

    def snapshot_entries(self) -> tuple[WorkspaceMemoryEntry, ...]:
        with self._lock:
            return tuple(self._entries)


class EnvironmentMemoryRuntime:
    """
    Phase 8 Step 33 Environment Memory Runtime.

    Stores semantic workflow continuity:
    - app
    - project path
    - active files
    - cursor positions
    - terminal directory
    - recent commands
    - visible errors
    - pending todos
    - workflow stage
    - continuity token

    It intentionally does not store screenshots, raw screen recordings, or raw
    OCR dumps.
    """

    def __init__(
        self,
        *,
        name: str = "environment_memory_runtime",
        gateway: WorkflowMemoryGateway | None = None,
    ) -> None:
        cleaned = name.strip()
        if not cleaned:
            raise ValueError("name cannot be empty.")

        self._name = cleaned
        self._gateway = gateway or WorkflowMemoryGateway()
        self._sessions: dict[str, EnvironmentMemorySession] = {}
        self._results: list[EnvironmentMemoryResult] = []
        self._audits: list[EnvironmentMemoryAuditRecord] = []
        self._events: list[EnvironmentMemoryRuntimeEvent] = []
        self._lock = RLock()
        self._last_reason: EnvironmentMemoryReason | None = None

    @property
    def name(self) -> str:
        return self._name

    def create_session(
        self,
        *,
        workspace_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> EnvironmentMemorySession:
        session = EnvironmentMemorySession(
            workspace_id=workspace_id,
            metadata=metadata or {},
        )
        event = self._event(
            kind=EnvironmentMemoryEventKind.SESSION_CREATED,
            reason=EnvironmentMemoryReason.SESSION_CREATED,
            session_id=session.session_id,
        )

        with self._lock:
            self._sessions[session.session_id] = session
            self._events.append(event)
            self._last_reason = event.reason

        return session

    def store_workflow(
        self,
        *,
        session_id: str,
        app_name: str,
        project_path: str | None = None,
        active_files: tuple[str, ...] = (),
        cursor_positions: tuple[CursorPosition, ...] = (),
        terminal_directory: str | None = None,
        recent_commands: tuple[str, ...] = (),
        visible_errors: tuple[str, ...] = (),
        pending_todos: tuple[str, ...] = (),
        workflow_stage: WorkflowStage = WorkflowStage.UNKNOWN,
        metadata: dict[str, Any] | None = None,
    ) -> EnvironmentMemoryResult:
        session = self.session_for(session_id)
        if session is None:
            result = _blocked_result(
                status=EnvironmentMemoryStatus.FAILED,
                decision=EnvironmentMemoryDecision.BLOCK,
                reason=EnvironmentMemoryReason.SESSION_NOT_FOUND,
                message="environment memory session not found",
            )
            self._record_result(result, session_id)
            return result

        if _looks_like_raw_screen_memory(metadata or {}):
            result = _blocked_result(
                status=EnvironmentMemoryStatus.BLOCKED,
                decision=EnvironmentMemoryDecision.BLOCK,
                reason=EnvironmentMemoryReason.RAW_SCREEN_MEMORY_BLOCKED,
                message="raw screen memory is not allowed",
            )
            self._record_result(result, session_id)
            return result

        continuity_token = _continuity_token(
            workspace_id=session.workspace_id,
            project_path=project_path,
            app_name=app_name,
            active_files=active_files,
            stage=workflow_stage,
        )
        entry = WorkspaceMemoryEntry(
            session_id=session_id,
            workspace_id=session.workspace_id,
            app_name=app_name,
            project_path=project_path,
            active_files=tuple(_clean_required(path) for path in active_files),
            cursor_positions=cursor_positions,
            terminal_directory=terminal_directory,
            recent_commands=tuple(_sanitize_command(cmd) for cmd in recent_commands),
            visible_errors=tuple(_trim_text(error) for error in visible_errors),
            pending_todos=tuple(_trim_text(todo) for todo in pending_todos),
            workflow_stage=workflow_stage,
            continuity_token=continuity_token,
            trust=_trust(
                confidence=0.86,
                reason="semantic workflow memory entry",
            ),
            metadata=metadata or {},
        )

        write = self._gateway.store(
            entry=entry,
            scope=EnvironmentMemoryScope.WORKSPACE,
        )
        result = _result(
            status=EnvironmentMemoryStatus.STORED,
            decision=EnvironmentMemoryDecision.STORE,
            reason=write.reason,
            entry=entry,
            message="semantic workflow memory stored",
        )
        self._record_result(result, session_id)

        return result

    def recall_session(self, *, session_id: str) -> EnvironmentMemoryResult:
        if self.session_for(session_id) is None:
            result = _blocked_result(
                status=EnvironmentMemoryStatus.FAILED,
                decision=EnvironmentMemoryDecision.BLOCK,
                reason=EnvironmentMemoryReason.SESSION_NOT_FOUND,
                message="environment memory session not found",
            )
            self._record_result(result, session_id)
            return result

        read = self._gateway.recall_session(session_id=session_id)
        result = _result(
            status=EnvironmentMemoryStatus.RECALLED,
            decision=EnvironmentMemoryDecision.RECALL,
            reason=read.reason,
            entries=read.entries,
            message="session workflow memory recalled",
        )
        self._record_result(result, session_id)

        return result

    def recall_project(
        self,
        *,
        session_id: str,
        project_path: str,
    ) -> EnvironmentMemoryResult:
        session = self.session_for(session_id)
        if session is None:
            result = _blocked_result(
                status=EnvironmentMemoryStatus.FAILED,
                decision=EnvironmentMemoryDecision.BLOCK,
                reason=EnvironmentMemoryReason.SESSION_NOT_FOUND,
                message="environment memory session not found",
            )
            self._record_result(result, session_id)
            return result

        read = self._gateway.recall_project(
            workspace_id=session.workspace_id,
            project_path=project_path,
        )
        project_memory = None
        if read.entries:
            latest = read.entries[-1]
            project_memory = ProjectMemory(
                workspace_id=session.workspace_id,
                project_path=project_path,
                latest_entry=latest,
                entry_count=len(read.entries),
                active_files=latest.active_files,
                visible_errors=latest.visible_errors,
                pending_todos=latest.pending_todos,
                workflow_stage=latest.workflow_stage,
            )

        result = _result(
            status=EnvironmentMemoryStatus.RECALLED,
            decision=EnvironmentMemoryDecision.RECALL,
            reason=read.reason,
            entries=read.entries,
            project_memory=project_memory,
            message="project workflow memory recalled",
        )
        self._record_result(result, session_id)

        return result

    def continue_workflow(
        self,
        *,
        session_id: str,
    ) -> EnvironmentMemoryResult:
        read = self._gateway.recall_session(session_id=session_id)
        if self.session_for(session_id) is None:
            result = _blocked_result(
                status=EnvironmentMemoryStatus.FAILED,
                decision=EnvironmentMemoryDecision.BLOCK,
                reason=EnvironmentMemoryReason.SESSION_NOT_FOUND,
                message="environment memory session not found",
            )
            self._record_result(result, session_id)
            return result

        if not read.entries:
            result = _blocked_result(
                status=EnvironmentMemoryStatus.BLOCKED,
                decision=EnvironmentMemoryDecision.BLOCK,
                reason=EnvironmentMemoryReason.EMPTY_MEMORY_BLOCKED,
                message="no workflow memory exists for continuation",
            )
            self._record_result(result, session_id)
            return result

        latest = read.entries[-1]
        continuity = SessionContinuity(
            session_id=session_id,
            workspace_id=latest.workspace_id,
            continuity_token=latest.continuity_token,
            active_entry=latest,
            workflow_stage=latest.workflow_stage,
            resume_summary=_resume_summary(latest),
        )
        result = _result(
            status=EnvironmentMemoryStatus.CONTINUITY_READY,
            decision=EnvironmentMemoryDecision.CONTINUE_WORKFLOW,
            reason=EnvironmentMemoryReason.CONTINUITY_TOKEN_CREATED,
            entry=latest,
            entries=read.entries,
            continuity=continuity,
            message="workflow continuity is ready",
        )
        self._record_result(result, session_id)

        return result

    def session_for(
        self,
        session_id: str,
    ) -> EnvironmentMemorySession | None:
        with self._lock:
            return self._sessions.get(session_id)

    def results(self) -> tuple[EnvironmentMemoryResult, ...]:
        with self._lock:
            return tuple(self._results)

    def audits(self) -> tuple[EnvironmentMemoryAuditRecord, ...]:
        with self._lock:
            return tuple(self._audits)

    def events(self) -> tuple[EnvironmentMemoryRuntimeEvent, ...]:
        with self._lock:
            return tuple(self._events)

    def gateway_entries(self) -> tuple[WorkspaceMemoryEntry, ...]:
        return self._gateway.snapshot_entries()

    def snapshot(self) -> EnvironmentMemoryRuntimeSnapshot:
        with self._lock:
            return EnvironmentMemoryRuntimeSnapshot(
                name=self.name,
                session_count=len(self._sessions),
                stored_entry_count=len(self._gateway.snapshot_entries()),
                result_count=len(self._results),
                store_count=sum(
                    1
                    for result in self._results
                    if result.decision == EnvironmentMemoryDecision.STORE
                ),
                recall_count=sum(
                    1
                    for result in self._results
                    if result.decision == EnvironmentMemoryDecision.RECALL
                ),
                continuity_count=sum(
                    1
                    for result in self._results
                    if result.decision
                    == EnvironmentMemoryDecision.CONTINUE_WORKFLOW
                ),
                blocked_count=sum(
                    1
                    for result in self._results
                    if result.status
                    in {
                        EnvironmentMemoryStatus.BLOCKED,
                        EnvironmentMemoryStatus.FAILED,
                    }
                ),
                audit_count=len(self._audits),
                runtime_event_count=len(self._events),
                last_reason=self._last_reason,
            )

    def reset(self) -> None:
        event = self._event(
            kind=EnvironmentMemoryEventKind.RUNTIME_RESET,
            reason=EnvironmentMemoryReason.RUNTIME_RESET,
        )

        with self._lock:
            self._sessions.clear()
            self._results.clear()
            self._audits.clear()
            self._events.clear()
            self._events.append(event)
            self._last_reason = event.reason

    def _record_result(
        self,
        result: EnvironmentMemoryResult,
        session_id: str,
    ) -> None:
        event = self._event(
            kind=_event_kind_for(result),
            reason=result.reason,
            session_id=session_id,
            result_id=result.result_id,
            audit_id=result.audit.audit_id,
            metadata={"status": result.status.value},
        )

        with self._lock:
            self._results.append(result)
            self._audits.append(result.audit)
            self._events.append(event)
            self._last_reason = result.reason

            session = self._sessions.get(session_id)
            if session is not None:
                self._sessions[session_id] = session.model_copy(
                    update={
                        "updated_at": utc_now(),
                        "store_count": session.store_count
                        + (
                            1
                            if result.decision == EnvironmentMemoryDecision.STORE
                            else 0
                        ),
                        "recall_count": session.recall_count
                        + (
                            1
                            if result.decision == EnvironmentMemoryDecision.RECALL
                            else 0
                        ),
                        "continuity_count": session.continuity_count
                        + (
                            1
                            if result.decision
                            == EnvironmentMemoryDecision.CONTINUE_WORKFLOW
                            else 0
                        ),
                        "blocked_count": session.blocked_count
                        + (
                            1
                            if result.status
                            in {
                                EnvironmentMemoryStatus.BLOCKED,
                                EnvironmentMemoryStatus.FAILED,
                            }
                            else 0
                        ),
                    }
                )

    @staticmethod
    def _event(
        *,
        kind: EnvironmentMemoryEventKind,
        reason: EnvironmentMemoryReason,
        session_id: str | None = None,
        result_id: str | None = None,
        audit_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> EnvironmentMemoryRuntimeEvent:
        return EnvironmentMemoryRuntimeEvent(
            kind=kind,
            reason=reason,
            session_id=session_id,
            result_id=result_id,
            audit_id=audit_id,
            metadata=metadata or {},
        )


def _result(
    *,
    status: EnvironmentMemoryStatus,
    decision: EnvironmentMemoryDecision,
    reason: EnvironmentMemoryReason,
    message: str,
    entry: WorkspaceMemoryEntry | None = None,
    entries: tuple[WorkspaceMemoryEntry, ...] = (),
    continuity: SessionContinuity | None = None,
    project_memory: ProjectMemory | None = None,
) -> EnvironmentMemoryResult:
    audit = EnvironmentMemoryAuditRecord(
        status=status,
        decision=decision,
        reason=reason,
        entry_id=entry.entry_id if entry is not None else None,
        session_id=entry.session_id if entry is not None else None,
        workspace_id=entry.workspace_id if entry is not None else None,
        raw_screen_logged=False,
    )

    return EnvironmentMemoryResult(
        status=status,
        decision=decision,
        reason=reason,
        entry=entry,
        entries=entries,
        continuity=continuity,
        project_memory=project_memory,
        audit=audit,
        trust=_trust(
            confidence=0.86 if status != EnvironmentMemoryStatus.FAILED else 0.20,
            reason=message,
        ),
        message=message,
    )


def _blocked_result(
    *,
    status: EnvironmentMemoryStatus,
    decision: EnvironmentMemoryDecision,
    reason: EnvironmentMemoryReason,
    message: str,
) -> EnvironmentMemoryResult:
    audit = EnvironmentMemoryAuditRecord(
        status=status,
        decision=decision,
        reason=reason,
        raw_screen_logged=False,
    )

    return EnvironmentMemoryResult(
        status=status,
        decision=decision,
        reason=reason,
        audit=audit,
        trust=_trust(confidence=0.25, reason=message),
        message=message,
    )


def _event_kind_for(result: EnvironmentMemoryResult) -> EnvironmentMemoryEventKind:
    if result.status in {
        EnvironmentMemoryStatus.BLOCKED,
        EnvironmentMemoryStatus.FAILED,
    }:
        return EnvironmentMemoryEventKind.MEMORY_BLOCKED

    if result.decision == EnvironmentMemoryDecision.STORE:
        return EnvironmentMemoryEventKind.MEMORY_STORED

    if result.decision == EnvironmentMemoryDecision.CONTINUE_WORKFLOW:
        return EnvironmentMemoryEventKind.CONTINUITY_READY

    return EnvironmentMemoryEventKind.MEMORY_RECALLED


def _continuity_token(
    *,
    workspace_id: str,
    project_path: str | None,
    app_name: str,
    active_files: tuple[str, ...],
    stage: WorkflowStage,
) -> str:
    payload = "|".join(
        (
            workspace_id,
            project_path or "",
            app_name,
            ",".join(active_files),
            stage.value,
        )
    )
    return sha256(payload.encode("utf-8")).hexdigest()


def _resume_summary(entry: WorkspaceMemoryEntry) -> str:
    project = entry.project_path or "unknown project"
    files = ", ".join(entry.active_files[:3]) or "no active files"
    errors = "; ".join(entry.visible_errors[:2]) or "no visible errors"
    todos = "; ".join(entry.pending_todos[:2]) or "no pending todos"

    return (
        f"{entry.app_name} in {project}; "
        f"stage={entry.workflow_stage.value}; "
        f"files={files}; errors={errors}; todos={todos}"
    )


def _sanitize_command(command: str) -> str:
    cleaned = _trim_text(command)

    lowered = cleaned.lower()
    sensitive_terms = (
        "password",
        "passwd",
        "token",
        "api_key",
        "apikey",
        "secret",
        "bearer",
    )

    if any(term in lowered for term in sensitive_terms):
        return f"<redacted-command:{sha256(cleaned.encode('utf-8')).hexdigest()}>"

    return cleaned


def _trim_text(value: str) -> str:
    cleaned = value.strip()
    if len(cleaned) > 500:
        return cleaned[:497] + "..."

    return cleaned


def _looks_like_raw_screen_memory(metadata: dict[str, Any]) -> bool:
    forbidden = {
        "screenshot",
        "screenshot_bytes",
        "screen_recording",
        "raw_ocr_dump",
        "raw_screen_text",
        "pixel_buffer",
        "image_bytes",
    }
    return any(key in metadata for key in forbidden)


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