from __future__ import annotations

import difflib
import fnmatch
import shutil
import time
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from threading import RLock

from pydantic import Field, field_validator, model_validator

from jarvis.tools.ids import new_action_id, new_action_result_id, utc_now
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
from jarvis.tools.policy import PermissionPolicy
from jarvis.tools.registry import (
    ToolAvailability,
    ToolDescriptor,
    ToolHealth,
    ToolRegistry,
)
from jarvis.tools.validation import (
    ActionValidationDecision,
    ActionValidationResult,
    ActionValidator,
    ActionValidatorConfig,
)


class FileOperationKind(StrEnum):
    """
    Supported file-system operation kinds.
    """

    READ_FILE = "read_file"
    LIST_DIRECTORY = "list_directory"
    SEARCH_FILES = "search_files"
    CREATE_DRAFT = "create_draft"
    WRITE_FILE = "write_file"
    PATCH_FILE = "patch_file"
    COPY_FILE = "copy_file"
    MOVE_FILE = "move_file"
    DELETE_FILE = "delete_file"


class FileOperationDecision(StrEnum):
    """
    File-system policy decision.
    """

    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_CONFIRMATION = "require_confirmation"
    REQUIRE_APPROVAL = "require_approval"


class FileOperationReason(StrEnum):
    """
    Machine-readable file-system policy/runtime reason.
    """

    SAFE_READ_ALLOWED = "safe_read_allowed"
    SAFE_LIST_ALLOWED = "safe_list_allowed"
    SAFE_SEARCH_ALLOWED = "safe_search_allowed"
    DRAFT_CREATION_ALLOWED = "draft_creation_allowed"
    WRITE_REQUIRES_CONFIRMATION = "write_requires_confirmation"
    PATCH_REQUIRES_CONFIRMATION = "patch_requires_confirmation"
    COPY_REQUIRES_CONFIRMATION = "copy_requires_confirmation"
    MOVE_REQUIRES_CONFIRMATION = "move_requires_confirmation"
    DELETE_REQUIRES_APPROVAL = "delete_requires_approval"
    CONFIRMATION_MISSING = "confirmation_missing"
    APPROVAL_MISSING = "approval_missing"
    PATH_OUT_OF_BOUNDS = "path_out_of_bounds"
    PATH_TRAVERSAL_BLOCKED = "path_traversal_blocked"
    ABSOLUTE_PATH_BLOCKED = "absolute_path_blocked"
    SOURCE_NOT_FOUND = "source_not_found"
    DESTINATION_EXISTS = "destination_exists"
    VALIDATION_BLOCKED = "validation_blocked"
    OPERATION_SUCCEEDED = "operation_succeeded"
    OPERATION_FAILED = "operation_failed"


class FileOperationRequest(ToolModel):
    """
    Typed request for a governed file-system operation.

    This is not raw file access. FileSystemRuntime still applies policy,
    validation, workspace boundary checks, backup rules, and observable result
    capture.
    """

    action_id: str = Field(default_factory=new_action_id)
    kind: FileOperationKind
    path: str
    content: str | None = None
    destination_path: str | None = None
    pattern: str = "*"
    old_text: str | None = None
    new_text: str | None = None
    recursive: bool = False
    overwrite: bool = False
    confirmed: bool = False
    approved: bool = False
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("action_id", "path", "pattern")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned

    @field_validator("content", "destination_path", "old_text", "new_text")
    @classmethod
    def _clean_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None

        cleaned = value.strip()

        return cleaned or None

    @model_validator(mode="after")
    def _validate_operation_arguments(self) -> FileOperationRequest:
        if self.kind in {
            FileOperationKind.CREATE_DRAFT,
            FileOperationKind.WRITE_FILE,
        }:
            if self.content is None:
                raise ValueError("content is required for write operations.")

        if self.kind == FileOperationKind.PATCH_FILE:
            if self.old_text is None or self.new_text is None:
                raise ValueError("old_text and new_text are required for patch.")

        if self.kind in {
            FileOperationKind.COPY_FILE,
            FileOperationKind.MOVE_FILE,
        }:
            if self.destination_path is None:
                raise ValueError("destination_path is required.")

        return self


class FileOperationPolicyResult(ToolModel):
    """
    File-system policy result.
    """

    decision: FileOperationDecision
    permission_decision: PermissionDecision
    reason: FileOperationReason
    risk: ActionRisk
    explanation: str
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("explanation")
    @classmethod
    def _explanation_required(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("explanation cannot be empty.")

        return cleaned

    @property
    def allowed(self) -> bool:
        return self.decision == FileOperationDecision.ALLOW


class FileOperationResult(ToolModel):
    """
    Observable result of a governed file-system operation.
    """

    result_id: str = Field(default_factory=new_action_result_id)
    action_id: str
    kind: FileOperationKind
    path: str
    status: ActionStatus
    success: bool
    output: str = ""
    content: str | None = None
    backup_path: str | None = None
    rollback_supported: bool = False
    diff: str | None = None
    policy_result: FileOperationPolicyResult
    validation_result: ActionValidationResult | None = None
    started_at: object = Field(default_factory=utc_now)
    completed_at: object = Field(default_factory=utc_now)
    duration_ms: int = Field(default=0, ge=0)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("result_id", "action_id", "path")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned


@dataclass(frozen=True, slots=True)
class FileSystemRuntimeConfig:
    """
    Configuration for FileSystemRuntime.
    """

    name: str = "file_system_runtime"
    workspace_root: str = "."
    backup_directory: str = ".jarvis_backups"
    max_read_chars: int = 200_000
    max_search_results: int = 200
    register_default_file_tool: bool = True

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("name cannot be empty.")

        if self.max_read_chars <= 0:
            raise ValueError("max_read_chars must be positive.")

        if self.max_search_results <= 0:
            raise ValueError("max_search_results must be positive.")


@dataclass(frozen=True, slots=True)
class FileSystemRuntimeSnapshot:
    """
    Observable diagnostics for FileSystemRuntime.
    """

    name: str
    operation_count: int
    success_count: int
    blocked_count: int
    failed_count: int
    rollback_supported_count: int
    last_status: ActionStatus | None
    last_reason: FileOperationReason | None
    last_error: str | None


class FileSystemPolicy:
    """
    Conservative file-system policy.

    Read/list/search are allowed inside workspace.
    Mutating operations require confirmation.
    Delete requires explicit approval.
    """

    def evaluate(
        self,
        request: FileOperationRequest,
    ) -> FileOperationPolicyResult:
        if request.kind == FileOperationKind.READ_FILE:
            return self._allow(
                reason=FileOperationReason.SAFE_READ_ALLOWED,
                explanation="workspace-bounded file read is allowed",
            )

        if request.kind == FileOperationKind.LIST_DIRECTORY:
            return self._allow(
                reason=FileOperationReason.SAFE_LIST_ALLOWED,
                explanation="workspace-bounded directory listing is allowed",
            )

        if request.kind == FileOperationKind.SEARCH_FILES:
            return self._allow(
                reason=FileOperationReason.SAFE_SEARCH_ALLOWED,
                explanation="workspace-bounded file search is allowed",
            )

        if request.kind == FileOperationKind.CREATE_DRAFT:
            return self._allow(
                reason=FileOperationReason.DRAFT_CREATION_ALLOWED,
                explanation="draft file creation is allowed",
            )

        if request.kind == FileOperationKind.WRITE_FILE:
            if not request.confirmed:
                return self._confirmation_required(
                    reason=FileOperationReason.WRITE_REQUIRES_CONFIRMATION,
                    explanation="write operation requires confirmation",
                )

            return self._allow(
                reason=FileOperationReason.WRITE_REQUIRES_CONFIRMATION,
                explanation="confirmed write operation is allowed",
                risk=ActionRisk.MEDIUM,
            )

        if request.kind == FileOperationKind.PATCH_FILE:
            if not request.confirmed:
                return self._confirmation_required(
                    reason=FileOperationReason.PATCH_REQUIRES_CONFIRMATION,
                    explanation="patch operation requires confirmation",
                )

            return self._allow(
                reason=FileOperationReason.PATCH_REQUIRES_CONFIRMATION,
                explanation="confirmed patch operation is allowed",
                risk=ActionRisk.MEDIUM,
            )

        if request.kind == FileOperationKind.COPY_FILE:
            if not request.confirmed:
                return self._confirmation_required(
                    reason=FileOperationReason.COPY_REQUIRES_CONFIRMATION,
                    explanation="copy operation requires confirmation",
                )

            return self._allow(
                reason=FileOperationReason.COPY_REQUIRES_CONFIRMATION,
                explanation="confirmed copy operation is allowed",
                risk=ActionRisk.MEDIUM,
            )

        if request.kind == FileOperationKind.MOVE_FILE:
            if not request.confirmed:
                return self._confirmation_required(
                    reason=FileOperationReason.MOVE_REQUIRES_CONFIRMATION,
                    explanation="move operation requires confirmation",
                )

            return self._allow(
                reason=FileOperationReason.MOVE_REQUIRES_CONFIRMATION,
                explanation="confirmed move operation is allowed",
                risk=ActionRisk.MEDIUM,
            )

        if request.kind == FileOperationKind.DELETE_FILE:
            if not request.approved:
                return FileOperationPolicyResult(
                    decision=FileOperationDecision.REQUIRE_APPROVAL,
                    permission_decision=PermissionDecision.REQUIRE_APPROVAL,
                    reason=FileOperationReason.DELETE_REQUIRES_APPROVAL,
                    risk=ActionRisk.HIGH,
                    explanation="delete operation requires explicit approval",
                )

            return self._allow(
                reason=FileOperationReason.DELETE_REQUIRES_APPROVAL,
                explanation="approved delete operation is allowed",
                risk=ActionRisk.HIGH,
                permission=PermissionDecision.REQUIRE_APPROVAL,
            )

        return FileOperationPolicyResult(
            decision=FileOperationDecision.DENY,
            permission_decision=PermissionDecision.DENY,
            reason=FileOperationReason.OPERATION_FAILED,
            risk=ActionRisk.HIGH,
            explanation="unknown file operation denied",
        )

    @staticmethod
    def _allow(
        *,
        reason: FileOperationReason,
        explanation: str,
        risk: ActionRisk = ActionRisk.LOW,
        permission: PermissionDecision = PermissionDecision.ALLOW,
    ) -> FileOperationPolicyResult:
        return FileOperationPolicyResult(
            decision=FileOperationDecision.ALLOW,
            permission_decision=permission,
            reason=reason,
            risk=risk,
            explanation=explanation,
        )

    @staticmethod
    def _confirmation_required(
        *,
        reason: FileOperationReason,
        explanation: str,
    ) -> FileOperationPolicyResult:
        return FileOperationPolicyResult(
            decision=FileOperationDecision.REQUIRE_CONFIRMATION,
            permission_decision=PermissionDecision.REQUIRE_CONFIRMATION,
            reason=reason,
            risk=ActionRisk.MEDIUM,
            explanation=explanation,
        )


class FileSystemRuntime:
    """
    Governed file-system runtime.

    Responsibilities:
    - normalize and enforce workspace boundaries
    - policy-gate file operations
    - build typed action plans
    - validate before mutation
    - backup before write/patch/move/delete
    - produce rollback-aware observable results

    Non-responsibilities:
    - no hidden file mutation
    - no direct cognition-to-file access
    - no approval UI
    - no bypass of policy or validation
    """

    def __init__(
        self,
        *,
        config: FileSystemRuntimeConfig | None = None,
        registry: ToolRegistry | None = None,
        policy: FileSystemPolicy | None = None,
        validator: ActionValidator | None = None,
    ) -> None:
        self._config = config or FileSystemRuntimeConfig()
        self._config.validate()

        self._registry = registry or ToolRegistry()
        self._policy = policy or FileSystemPolicy()

        if self._config.register_default_file_tool:
            self._register_default_file_tool()

        self._validator = validator or ActionValidator(
            config=ActionValidatorConfig(require_policy_evaluation=False),
            registry=self._registry,
            policy=PermissionPolicy(),
        )
        self._lock = RLock()

        self._operation_count = 0
        self._success_count = 0
        self._blocked_count = 0
        self._failed_count = 0
        self._rollback_supported_count = 0
        self._last_status: ActionStatus | None = None
        self._last_reason: FileOperationReason | None = None
        self._last_error: str | None = None

    @property
    def name(self) -> str:
        return self._config.name

    def execute(self, request: FileOperationRequest) -> FileOperationResult:
        """
        Execute one governed file-system operation.
        """

        with self._lock:
            self._operation_count += 1
            self._last_error = None

        started = utc_now()
        monotonic_start = time.monotonic()

        try:
            policy_result = self._policy.evaluate(request)

            if policy_result.decision != FileOperationDecision.ALLOW:
                result = self._blocked_result(
                    request=request,
                    policy_result=policy_result,
                    started_at=started,
                    monotonic_start=monotonic_start,
                )
                self._record(result)

                return result

            target_path = self._resolve_relative_path(request.path)
            destination_path = (
                self._resolve_relative_path(request.destination_path)
                if request.destination_path is not None
                else None
            )
            plan = self._build_plan(
                request=request,
                policy_result=policy_result,
            )
            validation = self._validator.validate_plan(plan)

            if validation.decision == ActionValidationDecision.BLOCK:
                result = self._blocked_result(
                    request=request,
                    policy_result=policy_result,
                    started_at=started,
                    monotonic_start=monotonic_start,
                    validation_result=validation,
                    reason=FileOperationReason.VALIDATION_BLOCKED,
                )
                self._record(result)

                return result

            result = self._execute_allowed(
                request=request,
                target_path=target_path,
                destination_path=destination_path,
                policy_result=policy_result,
                validation_result=validation,
                started_at=started,
                monotonic_start=monotonic_start,
            )
            self._record(result)

            return result

        except Exception as exc:
            self._last_error = f"{type(exc).__name__}: {exc}"
            policy_result = FileOperationPolicyResult(
                decision=FileOperationDecision.DENY,
                permission_decision=PermissionDecision.DENY,
                reason=FileOperationReason.OPERATION_FAILED,
                risk=ActionRisk.HIGH,
                explanation=f"{type(exc).__name__}: {exc}",
            )
            result = self._failed_result(
                request=request,
                policy_result=policy_result,
                started_at=started,
                monotonic_start=monotonic_start,
                output=f"{type(exc).__name__}: {exc}",
            )
            self._record(result)

            return result

    def snapshot(self) -> FileSystemRuntimeSnapshot:
        """
        Return runtime diagnostics.
        """

        with self._lock:
            return FileSystemRuntimeSnapshot(
                name=self.name,
                operation_count=self._operation_count,
                success_count=self._success_count,
                blocked_count=self._blocked_count,
                failed_count=self._failed_count,
                rollback_supported_count=self._rollback_supported_count,
                last_status=self._last_status,
                last_reason=self._last_reason,
                last_error=self._last_error,
            )

    def reset(self) -> None:
        """
        Reset runtime diagnostics only.
        """

        with self._lock:
            self._operation_count = 0
            self._success_count = 0
            self._blocked_count = 0
            self._failed_count = 0
            self._rollback_supported_count = 0
            self._last_status = None
            self._last_reason = None
            self._last_error = None

    def _execute_allowed(
        self,
        *,
        request: FileOperationRequest,
        target_path: Path,
        destination_path: Path | None,
        policy_result: FileOperationPolicyResult,
        validation_result: ActionValidationResult,
        started_at: object,
        monotonic_start: float,
    ) -> FileOperationResult:
        if request.kind == FileOperationKind.READ_FILE:
            return self._read_file(
                request=request,
                target_path=target_path,
                policy_result=policy_result,
                validation_result=validation_result,
                started_at=started_at,
                monotonic_start=monotonic_start,
            )

        if request.kind == FileOperationKind.LIST_DIRECTORY:
            return self._list_directory(
                request=request,
                target_path=target_path,
                policy_result=policy_result,
                validation_result=validation_result,
                started_at=started_at,
                monotonic_start=monotonic_start,
            )

        if request.kind == FileOperationKind.SEARCH_FILES:
            return self._search_files(
                request=request,
                target_path=target_path,
                policy_result=policy_result,
                validation_result=validation_result,
                started_at=started_at,
                monotonic_start=monotonic_start,
            )

        if request.kind == FileOperationKind.CREATE_DRAFT:
            return self._create_draft(
                request=request,
                target_path=target_path,
                policy_result=policy_result,
                validation_result=validation_result,
                started_at=started_at,
                monotonic_start=monotonic_start,
            )

        if request.kind == FileOperationKind.WRITE_FILE:
            return self._write_file(
                request=request,
                target_path=target_path,
                policy_result=policy_result,
                validation_result=validation_result,
                started_at=started_at,
                monotonic_start=monotonic_start,
            )

        if request.kind == FileOperationKind.PATCH_FILE:
            return self._patch_file(
                request=request,
                target_path=target_path,
                policy_result=policy_result,
                validation_result=validation_result,
                started_at=started_at,
                monotonic_start=monotonic_start,
            )

        if request.kind == FileOperationKind.COPY_FILE:
            if destination_path is None:
                raise ValueError("destination_path is required.")

            return self._copy_file(
                request=request,
                target_path=target_path,
                destination_path=destination_path,
                policy_result=policy_result,
                validation_result=validation_result,
                started_at=started_at,
                monotonic_start=monotonic_start,
            )

        if request.kind == FileOperationKind.MOVE_FILE:
            if destination_path is None:
                raise ValueError("destination_path is required.")

            return self._move_file(
                request=request,
                target_path=target_path,
                destination_path=destination_path,
                policy_result=policy_result,
                validation_result=validation_result,
                started_at=started_at,
                monotonic_start=monotonic_start,
            )

        if request.kind == FileOperationKind.DELETE_FILE:
            return self._delete_file(
                request=request,
                target_path=target_path,
                policy_result=policy_result,
                validation_result=validation_result,
                started_at=started_at,
                monotonic_start=monotonic_start,
            )

        raise ValueError(f"unsupported file operation: {request.kind.value}")

    def _read_file(
        self,
        *,
        request: FileOperationRequest,
        target_path: Path,
        policy_result: FileOperationPolicyResult,
        validation_result: ActionValidationResult,
        started_at: object,
        monotonic_start: float,
    ) -> FileOperationResult:
        if not target_path.exists():
            return self._failed_result(
                request=request,
                policy_result=policy_result,
                validation_result=validation_result,
                started_at=started_at,
                monotonic_start=monotonic_start,
                reason=FileOperationReason.SOURCE_NOT_FOUND,
                output="file not found",
            )

        content = target_path.read_text(encoding="utf-8")
        truncated = content[: self._config.max_read_chars]

        return self._success_result(
            request=request,
            policy_result=policy_result,
            validation_result=validation_result,
            started_at=started_at,
            monotonic_start=monotonic_start,
            content=truncated,
            output=f"read {len(truncated)} characters",
        )

    def _list_directory(
        self,
        *,
        request: FileOperationRequest,
        target_path: Path,
        policy_result: FileOperationPolicyResult,
        validation_result: ActionValidationResult,
        started_at: object,
        monotonic_start: float,
    ) -> FileOperationResult:
        if not target_path.exists() or not target_path.is_dir():
            return self._failed_result(
                request=request,
                policy_result=policy_result,
                validation_result=validation_result,
                started_at=started_at,
                monotonic_start=monotonic_start,
                reason=FileOperationReason.SOURCE_NOT_FOUND,
                output="directory not found",
            )

        entries = sorted(item.name for item in target_path.iterdir())

        return self._success_result(
            request=request,
            policy_result=policy_result,
            validation_result=validation_result,
            started_at=started_at,
            monotonic_start=monotonic_start,
            output="\n".join(entries),
        )

    def _search_files(
        self,
        *,
        request: FileOperationRequest,
        target_path: Path,
        policy_result: FileOperationPolicyResult,
        validation_result: ActionValidationResult,
        started_at: object,
        monotonic_start: float,
    ) -> FileOperationResult:
        if not target_path.exists() or not target_path.is_dir():
            return self._failed_result(
                request=request,
                policy_result=policy_result,
                validation_result=validation_result,
                started_at=started_at,
                monotonic_start=monotonic_start,
                reason=FileOperationReason.SOURCE_NOT_FOUND,
                output="search root not found",
            )

        iterator = (
            target_path.rglob("*")
            if request.recursive
            else target_path.glob("*")
        )
        matches: list[str] = []
        root = self._workspace_root()

        for item in iterator:
            if len(matches) >= self._config.max_search_results:
                break

            if item.is_file() and fnmatch.fnmatch(item.name, request.pattern):
                matches.append(str(item.relative_to(root)))

        return self._success_result(
            request=request,
            policy_result=policy_result,
            validation_result=validation_result,
            started_at=started_at,
            monotonic_start=monotonic_start,
            output="\n".join(matches),
        )

    def _create_draft(
        self,
        *,
        request: FileOperationRequest,
        target_path: Path,
        policy_result: FileOperationPolicyResult,
        validation_result: ActionValidationResult,
        started_at: object,
        monotonic_start: float,
    ) -> FileOperationResult:
        if target_path.exists() and not request.overwrite:
            return self._failed_result(
                request=request,
                policy_result=policy_result,
                validation_result=validation_result,
                started_at=started_at,
                monotonic_start=monotonic_start,
                reason=FileOperationReason.DESTINATION_EXISTS,
                output="draft destination already exists",
            )

        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(request.content or "", encoding="utf-8")

        return self._success_result(
            request=request,
            policy_result=policy_result,
            validation_result=validation_result,
            started_at=started_at,
            monotonic_start=monotonic_start,
            output="draft file created",
            rollback_supported=True,
        )

    def _write_file(
        self,
        *,
        request: FileOperationRequest,
        target_path: Path,
        policy_result: FileOperationPolicyResult,
        validation_result: ActionValidationResult,
        started_at: object,
        monotonic_start: float,
    ) -> FileOperationResult:
        backup_path = self._backup_file(target_path) if target_path.exists() else None
        before = target_path.read_text(encoding="utf-8") if target_path.exists() else ""
        after = request.content or ""

        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(after, encoding="utf-8")

        return self._success_result(
            request=request,
            policy_result=policy_result,
            validation_result=validation_result,
            started_at=started_at,
            monotonic_start=monotonic_start,
            output="file written with backup support",
            backup_path=backup_path,
            rollback_supported=backup_path is not None,
            diff=self._diff(before=before, after=after, path=request.path),
        )

    def _patch_file(
        self,
        *,
        request: FileOperationRequest,
        target_path: Path,
        policy_result: FileOperationPolicyResult,
        validation_result: ActionValidationResult,
        started_at: object,
        monotonic_start: float,
    ) -> FileOperationResult:
        if not target_path.exists():
            return self._failed_result(
                request=request,
                policy_result=policy_result,
                validation_result=validation_result,
                started_at=started_at,
                monotonic_start=monotonic_start,
                reason=FileOperationReason.SOURCE_NOT_FOUND,
                output="patch source file not found",
            )

        before = target_path.read_text(encoding="utf-8")
        old_text = request.old_text or ""
        new_text = request.new_text or ""

        if old_text not in before:
            return self._failed_result(
                request=request,
                policy_result=policy_result,
                validation_result=validation_result,
                started_at=started_at,
                monotonic_start=monotonic_start,
                output="patch old_text not found",
            )

        backup_path = self._backup_file(target_path)
        after = before.replace(old_text, new_text, 1)
        target_path.write_text(after, encoding="utf-8")

        return self._success_result(
            request=request,
            policy_result=policy_result,
            validation_result=validation_result,
            started_at=started_at,
            monotonic_start=monotonic_start,
            output="file patched with backup support",
            backup_path=backup_path,
            rollback_supported=True,
            diff=self._diff(before=before, after=after, path=request.path),
        )

    def _copy_file(
        self,
        *,
        request: FileOperationRequest,
        target_path: Path,
        destination_path: Path,
        policy_result: FileOperationPolicyResult,
        validation_result: ActionValidationResult,
        started_at: object,
        monotonic_start: float,
    ) -> FileOperationResult:
        if not target_path.exists():
            return self._failed_result(
                request=request,
                policy_result=policy_result,
                validation_result=validation_result,
                started_at=started_at,
                monotonic_start=monotonic_start,
                reason=FileOperationReason.SOURCE_NOT_FOUND,
                output="copy source file not found",
            )

        if destination_path.exists() and not request.overwrite:
            return self._failed_result(
                request=request,
                policy_result=policy_result,
                validation_result=validation_result,
                started_at=started_at,
                monotonic_start=monotonic_start,
                reason=FileOperationReason.DESTINATION_EXISTS,
                output="copy destination exists",
            )

        backup_path = (
            self._backup_file(destination_path) if destination_path.exists() else None
        )
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(target_path, destination_path)

        return self._success_result(
            request=request,
            policy_result=policy_result,
            validation_result=validation_result,
            started_at=started_at,
            monotonic_start=monotonic_start,
            output="file copied",
            backup_path=backup_path,
            rollback_supported=True,
        )

    def _move_file(
        self,
        *,
        request: FileOperationRequest,
        target_path: Path,
        destination_path: Path,
        policy_result: FileOperationPolicyResult,
        validation_result: ActionValidationResult,
        started_at: object,
        monotonic_start: float,
    ) -> FileOperationResult:
        if not target_path.exists():
            return self._failed_result(
                request=request,
                policy_result=policy_result,
                validation_result=validation_result,
                started_at=started_at,
                monotonic_start=monotonic_start,
                reason=FileOperationReason.SOURCE_NOT_FOUND,
                output="move source file not found",
            )

        if destination_path.exists() and not request.overwrite:
            return self._failed_result(
                request=request,
                policy_result=policy_result,
                validation_result=validation_result,
                started_at=started_at,
                monotonic_start=monotonic_start,
                reason=FileOperationReason.DESTINATION_EXISTS,
                output="move destination exists",
            )

        backup_path = self._backup_file(target_path)
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(target_path), str(destination_path))

        return self._success_result(
            request=request,
            policy_result=policy_result,
            validation_result=validation_result,
            started_at=started_at,
            monotonic_start=monotonic_start,
            output="file moved with source backup",
            backup_path=backup_path,
            rollback_supported=True,
        )

    def _delete_file(
        self,
        *,
        request: FileOperationRequest,
        target_path: Path,
        policy_result: FileOperationPolicyResult,
        validation_result: ActionValidationResult,
        started_at: object,
        monotonic_start: float,
    ) -> FileOperationResult:
        if not target_path.exists():
            return self._failed_result(
                request=request,
                policy_result=policy_result,
                validation_result=validation_result,
                started_at=started_at,
                monotonic_start=monotonic_start,
                reason=FileOperationReason.SOURCE_NOT_FOUND,
                output="delete source file not found",
            )

        backup_path = self._backup_file(target_path)
        target_path.unlink()

        return self._success_result(
            request=request,
            policy_result=policy_result,
            validation_result=validation_result,
            started_at=started_at,
            monotonic_start=monotonic_start,
            output="file deleted with backup support",
            backup_path=backup_path,
            rollback_supported=True,
        )

    def _build_plan(
        self,
        *,
        request: FileOperationRequest,
        policy_result: FileOperationPolicyResult,
    ) -> ActionPlan:
        action_kind = self._action_kind(request.kind)
        capability = self._capability(request.kind)
        risk = policy_result.risk
        scope = ActionScope.WORKSPACE
        timeout_ms = 30_000 if risk in {ActionRisk.HIGH, ActionRisk.CRITICAL} else None
        requires_approval = (
            request.approved
            or policy_result.decision == FileOperationDecision.REQUIRE_APPROVAL
            or policy_result.permission_decision == PermissionDecision.REQUIRE_APPROVAL
            or risk in {ActionRisk.HIGH, ActionRisk.CRITICAL}
        )

        arguments: dict[str, object] = {
            "path": request.path,
        }

        if request.destination_path is not None:
            arguments["destination_path"] = request.destination_path

        step = ActionStep(
            action_id=request.action_id,
            order=0,
            kind=action_kind,
            capability=capability,
            scope=scope,
            risk=risk,
            description=f"execute governed file operation: {request.kind.value}",
            arguments=arguments,
            timeout_ms=timeout_ms,
            interruptible=True,
            rollback_supported=self._rollback_expected(request.kind),
        )

        return ActionPlan(
            action_id=request.action_id,
            goal=f"execute file operation: {request.kind.value}",
            steps=(step,),
            risk=risk,
            scope=scope,
            requires_approval=requires_approval,
            permission_decision=policy_result.permission_decision,
            status=ActionStatus.PLANNED,
        )

    def _register_default_file_tool(self) -> None:
        self._registry.register(
            ToolDescriptor(
                tool_id="tool_file_system",
                name="file system runtime",
                description="Governed workspace-bounded file-system runtime",
                capabilities=(
                    ToolCapability.READ_FILE,
                    ToolCapability.LIST_DIRECTORY,
                    ToolCapability.SEARCH_FILES,
                    ToolCapability.WRITE_FILE,
                    ToolCapability.PATCH_FILE,
                    ToolCapability.DELETE_FILE,
                ),
                supported_action_kinds=(
                    ActionKind.READ,
                    ActionKind.SEARCH,
                    ActionKind.WRITE,
                    ActionKind.PATCH,
                    ActionKind.COPY,
                    ActionKind.MOVE,
                    ActionKind.DELETE,
                ),
                scopes=(ActionScope.WORKSPACE,),
                max_risk=ActionRisk.HIGH,
                required_permission=PermissionDecision.REQUIRE_APPROVAL,
                availability=ToolAvailability.AVAILABLE,
                health=ToolHealth.HEALTHY,
                enabled=True,
            )
        )

    def _resolve_relative_path(self, value: str | None) -> Path:
        if value is None:
            raise ValueError("path is required.")

        raw = value.strip()

        if not raw:
            raise ValueError("path cannot be empty.")

        if self._is_absolute(raw):
            raise ValueError("absolute paths are blocked.")

        normalized = raw.replace("\\", "/")
        parts = Path(normalized).parts

        if ".." in parts:
            raise ValueError("path traversal is blocked.")

        root = self._workspace_root()
        path = (root / normalized).resolve()

        if not self._is_within_root(path=path, root=root):
            raise ValueError("path must stay inside workspace root.")

        return path

    def _backup_file(self, path: Path) -> str:
        if not path.exists():
            raise ValueError("cannot backup missing file.")

        root = self._workspace_root()
        backup_root = root / self._config.backup_directory
        backup_root.mkdir(parents=True, exist_ok=True)

        relative = path.relative_to(root)
        stamp = str(int(time.time() * 1000))
        backup_path = backup_root / f"{stamp}_{relative.as_posix()}"
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, backup_path)

        return str(backup_path.relative_to(root))

    def _workspace_root(self) -> Path:
        return Path(self._config.workspace_root).resolve()

    @staticmethod
    def _is_absolute(value: str) -> bool:
        candidate = Path(value)

        return candidate.is_absolute() or value.startswith(("/", "\\"))

    @staticmethod
    def _is_within_root(*, path: Path, root: Path) -> bool:
        return path == root or root in path.parents

    @staticmethod
    def _action_kind(kind: FileOperationKind) -> ActionKind:
        return {
            FileOperationKind.READ_FILE: ActionKind.READ,
            FileOperationKind.LIST_DIRECTORY: ActionKind.READ,
            FileOperationKind.SEARCH_FILES: ActionKind.SEARCH,
            FileOperationKind.CREATE_DRAFT: ActionKind.WRITE,
            FileOperationKind.WRITE_FILE: ActionKind.WRITE,
            FileOperationKind.PATCH_FILE: ActionKind.PATCH,
            FileOperationKind.COPY_FILE: ActionKind.COPY,
            FileOperationKind.MOVE_FILE: ActionKind.MOVE,
            FileOperationKind.DELETE_FILE: ActionKind.DELETE,
        }[kind]

    @staticmethod
    def _capability(kind: FileOperationKind) -> ToolCapability:
        return {
            FileOperationKind.READ_FILE: ToolCapability.READ_FILE,
            FileOperationKind.LIST_DIRECTORY: ToolCapability.LIST_DIRECTORY,
            FileOperationKind.SEARCH_FILES: ToolCapability.SEARCH_FILES,
            FileOperationKind.CREATE_DRAFT: ToolCapability.WRITE_FILE,
            FileOperationKind.WRITE_FILE: ToolCapability.WRITE_FILE,
            FileOperationKind.PATCH_FILE: ToolCapability.PATCH_FILE,
            FileOperationKind.COPY_FILE: ToolCapability.WRITE_FILE,
            FileOperationKind.MOVE_FILE: ToolCapability.WRITE_FILE,
            FileOperationKind.DELETE_FILE: ToolCapability.DELETE_FILE,
        }[kind]

    @staticmethod
    def _rollback_expected(kind: FileOperationKind) -> bool:
        return kind in {
            FileOperationKind.WRITE_FILE,
            FileOperationKind.PATCH_FILE,
            FileOperationKind.COPY_FILE,
            FileOperationKind.MOVE_FILE,
            FileOperationKind.DELETE_FILE,
        }

    @staticmethod
    def _diff(*, before: str, after: str, path: str) -> str:
        return "\n".join(
            difflib.unified_diff(
                before.splitlines(),
                after.splitlines(),
                fromfile=f"a/{path}",
                tofile=f"b/{path}",
                lineterm="",
            )
        )

    def _success_result(
        self,
        *,
        request: FileOperationRequest,
        policy_result: FileOperationPolicyResult,
        validation_result: ActionValidationResult,
        started_at: object,
        monotonic_start: float,
        output: str,
        content: str | None = None,
        backup_path: str | None = None,
        rollback_supported: bool = False,
        diff: str | None = None,
    ) -> FileOperationResult:
        return FileOperationResult(
            action_id=request.action_id,
            kind=request.kind,
            path=request.path,
            status=ActionStatus.SUCCEEDED,
            success=True,
            output=output,
            content=content,
            backup_path=backup_path,
            rollback_supported=rollback_supported,
            diff=diff,
            policy_result=policy_result,
            validation_result=validation_result,
            started_at=started_at,
            completed_at=utc_now(),
            duration_ms=self._duration_ms(monotonic_start),
            metadata={
                "runtime": self.name,
                "reason": FileOperationReason.OPERATION_SUCCEEDED.value,
            },
        )

    def _blocked_result(
        self,
        *,
        request: FileOperationRequest,
        policy_result: FileOperationPolicyResult,
        started_at: object,
        monotonic_start: float,
        validation_result: ActionValidationResult | None = None,
        reason: FileOperationReason | None = None,
    ) -> FileOperationResult:
        final_reason = reason or policy_result.reason

        return FileOperationResult(
            action_id=request.action_id,
            kind=request.kind,
            path=request.path,
            status=ActionStatus.BLOCKED,
            success=False,
            output=policy_result.explanation,
            policy_result=policy_result,
            validation_result=validation_result,
            started_at=started_at,
            completed_at=utc_now(),
            duration_ms=self._duration_ms(monotonic_start),
            metadata={
                "runtime": self.name,
                "reason": final_reason.value,
            },
        )

    def _failed_result(
        self,
        *,
        request: FileOperationRequest,
        policy_result: FileOperationPolicyResult,
        started_at: object,
        monotonic_start: float,
        output: str,
        validation_result: ActionValidationResult | None = None,
        reason: FileOperationReason = FileOperationReason.OPERATION_FAILED,
    ) -> FileOperationResult:
        return FileOperationResult(
            action_id=request.action_id,
            kind=request.kind,
            path=request.path,
            status=ActionStatus.FAILED,
            success=False,
            output=output,
            policy_result=policy_result,
            validation_result=validation_result,
            started_at=started_at,
            completed_at=utc_now(),
            duration_ms=self._duration_ms(monotonic_start),
            metadata={
                "runtime": self.name,
                "reason": reason.value,
            },
        )

    def _record(self, result: FileOperationResult) -> None:
        with self._lock:
            self._last_status = result.status
            self._last_reason = FileOperationReason(
                str(result.metadata.get("reason", "operation_failed"))
            )

            if result.success:
                self._success_count += 1

            elif result.status == ActionStatus.BLOCKED:
                self._blocked_count += 1

            else:
                self._failed_count += 1

            if result.rollback_supported:
                self._rollback_supported_count += 1

    @staticmethod
    def _duration_ms(start: float) -> int:
        return max(0, int((time.monotonic() - start) * 1000))