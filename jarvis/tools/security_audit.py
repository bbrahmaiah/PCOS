from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from threading import RLock
from typing import Any

from pydantic import Field, field_validator

from jarvis.tools.ids import new_action_result_id, utc_now
from jarvis.tools.models import (
    ActionPlan,
    ActionRisk,
    ActionStatus,
    PermissionDecision,
    ToolModel,
)


class SecurityAuditDecision(StrEnum):
    """
    Final security hardening decision.
    """

    PASS = "pass"
    WARN = "warn"
    BLOCK = "block"


class SecurityAuditSeverity(StrEnum):
    """
    Security finding severity.
    """

    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SecurityAuditSubjectKind(StrEnum):
    """
    Auditable subject kind.
    """

    ACTION_PLAN = "action_plan"
    COGNITION_INTENT = "cognition_intent"
    MEMORY_PAYLOAD = "memory_payload"
    RUNTIME_EVENT = "runtime_event"
    TEXT_PAYLOAD = "text_payload"


class SecurityAuditFindingKind(StrEnum):
    """
    Machine-readable security finding kind.
    """

    PROMPT_INJECTION = "prompt_injection"
    TOOL_INJECTION = "tool_injection"
    APPROVAL_BYPASS = "approval_bypass"
    POLICY_BYPASS = "policy_bypass"
    VALIDATION_BYPASS = "validation_bypass"
    UNSAFE_SHELL = "unsafe_shell"
    PATH_TRAVERSAL = "path_traversal"
    HIDDEN_EXECUTION = "hidden_execution"
    MEMORY_POISONING = "memory_poisoning"
    AUTONOMY_ESCALATION = "autonomy_escalation"
    SCHEDULER_COLLISION_BYPASS = "scheduler_collision_bypass"
    AUDIT_OMISSION = "audit_omission"
    DIRECT_COGNITION_TOOL_EXECUTION = "direct_cognition_tool_execution"
    UNSAFE_RISK_ESCALATION = "unsafe_risk_escalation"
    MISSING_TIMEOUT = "missing_timeout"
    MISSING_INTERRUPTIBILITY = "missing_interruptibility"
    MISSING_ROLLBACK_EXPLANATION = "missing_rollback_explanation"


class SecurityAuditFinding(ToolModel):
    """
    One security hardening finding.

    Findings are observable contracts. They explain what was detected and why
    the runtime should warn or block.
    """

    finding_id: str = Field(default_factory=new_action_result_id)
    kind: SecurityAuditFindingKind
    severity: SecurityAuditSeverity
    message: str
    evidence: str
    blocked: bool = False
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("finding_id", "message", "evidence")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned


class SecurityAuditSubject(ToolModel):
    """
    Generic subject for security hardening audit.

    This allows the audit runtime to inspect text, plans, cognition intents,
    memory payloads, and runtime events without coupling to every concrete
    runtime class.
    """

    subject_id: str = Field(default_factory=new_action_result_id)
    kind: SecurityAuditSubjectKind
    title: str
    text: str = ""
    data: dict[str, Any] = Field(default_factory=dict)
    risk: ActionRisk = ActionRisk.LOW
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("subject_id", "title")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned


class SecurityAuditResult(ToolModel):
    """
    Result of one security hardening audit.
    """

    result_id: str = Field(default_factory=new_action_result_id)
    subject_id: str
    subject_kind: SecurityAuditSubjectKind
    decision: SecurityAuditDecision
    findings: tuple[SecurityAuditFinding, ...] = ()
    passed: bool
    blocked: bool
    message: str
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("result_id", "subject_id", "message")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned


@dataclass(frozen=True, slots=True)
class SecurityHardeningAuditConfig:
    """
    Security hardening audit configuration.
    """

    name: str = "security_hardening_audit"
    block_prompt_injection: bool = True
    block_tool_injection: bool = True
    block_approval_bypass: bool = True
    block_policy_bypass: bool = True
    block_unsafe_shell: bool = True
    block_path_traversal: bool = True
    block_hidden_execution: bool = True
    block_memory_poisoning: bool = True
    block_autonomy_escalation: bool = True
    require_audit_marker_for_execution: bool = True

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("name cannot be empty.")


@dataclass(frozen=True, slots=True)
class SecurityHardeningAuditSnapshot:
    """
    Security audit runtime diagnostics.
    """

    name: str
    audit_count: int
    pass_count: int
    warn_count: int
    block_count: int
    finding_count: int
    last_decision: SecurityAuditDecision | None
    last_error: str | None


class SecurityHardeningAudit:
    """
    Phase 5 security hardening audit.

    Responsibilities:
    - inspect action plans before mature Phase 5 completion
    - inspect cognition/tool text for injection and bypass patterns
    - inspect memory payloads for poisoning and secrets
    - detect unsafe shell, unsafe paths, hidden execution, audit omission
    - produce deterministic findings and final pass/warn/block decisions

    Non-responsibilities:
    - no execution
    - no approval granting
    - no policy replacement
    - no validation replacement
    - no memory writes
    """

    _PROMPT_INJECTION_PATTERNS = (
        r"ignore\s+(all\s+)?previous\s+instructions",
        r"disregard\s+(all\s+)?(prior|previous)\s+instructions",
        r"system\s+prompt",
        r"developer\s+message",
        r"you\s+are\s+now\s+in\s+developer\s+mode",
        r"bypass\s+(safety|policy|guardrails)",
        r"do\s+not\s+tell\s+the\s+user",
    )
    _TOOL_INJECTION_PATTERNS = (
        r"call\s+tool\s+directly",
        r"execute\s+without\s+approval",
        r"run\s+without\s+validation",
        r"skip\s+the\s+policy",
        r"disable\s+audit",
        r"hide\s+this\s+action",
    )
    _MEMORY_POISONING_PATTERNS = (
        r"remember\s+that\s+security\s+is\s+disabled",
        r"store\s+this\s+as\s+trusted",
        r"always\s+approve\s+my\s+actions",
        r"never\s+ask\s+for\s+approval",
        r"policy\s+is\s+no\s+longer\s+required",
    )
    _DANGEROUS_SHELL_PATTERNS = (
        r"\bdel\s+",
        r"\brmdir\s+",
        r"\brm\s+-rf\b",
        r"\bformat\s+",
        r"\bshutdown\b",
        r"\brestart-computer\b",
        r"\breg\s+(add|delete)\b",
        r"\bset-executionpolicy\b",
        r"\binvoke-webrequest\b",
        r"\bcurl\s+.*\|\s*(sh|bash|powershell)",
        r"\bchmod\s+777\b",
    )
    _SENSITIVE_KEYS = {
        "password",
        "secret",
        "token",
        "api_key",
        "apikey",
        "authorization",
        "cookie",
        "otp",
        "private_key",
    }

    def __init__(
        self,
        *,
        config: SecurityHardeningAuditConfig | None = None,
    ) -> None:
        self._config = config or SecurityHardeningAuditConfig()
        self._config.validate()

        self._lock = RLock()
        self._audit_count = 0
        self._pass_count = 0
        self._warn_count = 0
        self._block_count = 0
        self._finding_count = 0
        self._last_decision: SecurityAuditDecision | None = None
        self._last_error: str | None = None

    @property
    def name(self) -> str:
        return self._config.name

    def audit_subject(self, subject: SecurityAuditSubject) -> SecurityAuditResult:
        """
        Audit a generic security subject.
        """

        findings: list[SecurityAuditFinding] = []
        text = self._combined_subject_text(subject)

        findings.extend(self._scan_text(text))

        if subject.kind == SecurityAuditSubjectKind.MEMORY_PAYLOAD:
            findings.extend(self._scan_memory_payload(subject))

        if subject.kind == SecurityAuditSubjectKind.COGNITION_INTENT:
            findings.extend(self._scan_cognition_subject(subject))

        result = self._result(
            subject_id=subject.subject_id,
            subject_kind=subject.kind,
            findings=tuple(findings),
        )
        self._record_result(result)

        return result

    def audit_plan(self, plan: ActionPlan) -> SecurityAuditResult:
        """
        Audit an ActionPlan for Phase 5 security hardening.
        """

        findings: list[SecurityAuditFinding] = []

        for step in plan.steps:
            findings.extend(self._scan_step_arguments(step.arguments))
            findings.extend(self._scan_action_step(step_kind=step.kind.value))
            findings.extend(
                self._scan_shell_command(
                    command=step.arguments.get("command"),
                )
            )
            findings.extend(
                self._scan_paths(
                    path=step.arguments.get("path"),
                    destination_path=step.arguments.get("destination_path"),
                )
            )

            if step.risk in {ActionRisk.HIGH, ActionRisk.CRITICAL}:
                if step.timeout_ms is None:
                    findings.append(
                        self._finding(
                            kind=SecurityAuditFindingKind.MISSING_TIMEOUT,
                            severity=SecurityAuditSeverity.HIGH,
                            message="high-risk step is missing timeout",
                            evidence=step.description,
                            blocked=True,
                        )
                    )

            if not step.interruptible:
                findings.append(
                    self._finding(
                        kind=SecurityAuditFindingKind.MISSING_INTERRUPTIBILITY,
                        severity=SecurityAuditSeverity.HIGH,
                        message="action step is not interruptible",
                        evidence=step.description,
                        blocked=True,
                    )
                )

        if plan.risk in {ActionRisk.HIGH, ActionRisk.CRITICAL}:
            if not plan.requires_approval:
                findings.append(
                    self._finding(
                        kind=SecurityAuditFindingKind.APPROVAL_BYPASS,
                        severity=SecurityAuditSeverity.CRITICAL,
                        message="high-risk plan does not require approval",
                        evidence=plan.goal,
                        blocked=True,
                    )
                )

        if plan.permission_decision == PermissionDecision.DENY:
            findings.append(
                self._finding(
                    kind=SecurityAuditFindingKind.POLICY_BYPASS,
                    severity=SecurityAuditSeverity.CRITICAL,
                    message="plan carries denied permission decision",
                    evidence=plan.goal,
                    blocked=True,
                )
            )

        if plan.status not in {ActionStatus.PLANNED, ActionStatus.BLOCKED}:
            findings.append(
                self._finding(
                    kind=SecurityAuditFindingKind.HIDDEN_EXECUTION,
                    severity=SecurityAuditSeverity.HIGH,
                    message="plan is not in a pre-execution status",
                    evidence=plan.status.value,
                    blocked=True,
                )
            )

        result = self._result(
            subject_id=plan.action_id,
            subject_kind=SecurityAuditSubjectKind.ACTION_PLAN,
            findings=tuple(findings),
        )
        self._record_result(result)

        return result

    def snapshot(self) -> SecurityHardeningAuditSnapshot:
        """
        Return audit diagnostics.
        """

        with self._lock:
            return SecurityHardeningAuditSnapshot(
                name=self.name,
                audit_count=self._audit_count,
                pass_count=self._pass_count,
                warn_count=self._warn_count,
                block_count=self._block_count,
                finding_count=self._finding_count,
                last_decision=self._last_decision,
                last_error=self._last_error,
            )

    def reset(self) -> None:
        """
        Reset diagnostics only.
        """

        with self._lock:
            self._audit_count = 0
            self._pass_count = 0
            self._warn_count = 0
            self._block_count = 0
            self._finding_count = 0
            self._last_decision = None
            self._last_error = None

    def _scan_text(self, text: str) -> tuple[SecurityAuditFinding, ...]:
        findings: list[SecurityAuditFinding] = []

        findings.extend(
            self._scan_patterns(
                text=text,
                patterns=self._PROMPT_INJECTION_PATTERNS,
                kind=SecurityAuditFindingKind.PROMPT_INJECTION,
                severity=SecurityAuditSeverity.CRITICAL,
                message="prompt injection pattern detected",
                blocked=self._config.block_prompt_injection,
            )
        )
        findings.extend(
            self._scan_patterns(
                text=text,
                patterns=self._TOOL_INJECTION_PATTERNS,
                kind=SecurityAuditFindingKind.TOOL_INJECTION,
                severity=SecurityAuditSeverity.CRITICAL,
                message="tool injection or bypass pattern detected",
                blocked=self._config.block_tool_injection,
            )
        )
        findings.extend(
            self._scan_patterns(
                text=text,
                patterns=self._MEMORY_POISONING_PATTERNS,
                kind=SecurityAuditFindingKind.MEMORY_POISONING,
                severity=SecurityAuditSeverity.HIGH,
                message="memory poisoning pattern detected",
                blocked=self._config.block_memory_poisoning,
            )
        )

        return tuple(findings)

    def _scan_memory_payload(
        self,
        subject: SecurityAuditSubject,
    ) -> tuple[SecurityAuditFinding, ...]:
        findings: list[SecurityAuditFinding] = []

        for key in self._flatten_keys(subject.data):
            if self._is_sensitive_key(key):
                findings.append(
                    self._finding(
                        kind=SecurityAuditFindingKind.MEMORY_POISONING,
                        severity=SecurityAuditSeverity.HIGH,
                        message="sensitive memory payload key detected",
                        evidence=key,
                        blocked=self._config.block_memory_poisoning,
                    )
                )

        return tuple(findings)

    def _scan_cognition_subject(
        self,
        subject: SecurityAuditSubject,
    ) -> tuple[SecurityAuditFinding, ...]:
        findings: list[SecurityAuditFinding] = []
        direct_execution = bool(subject.data.get("direct_execution", False))
        bypass_pipeline = bool(subject.data.get("bypass_pipeline", False))

        if direct_execution:
            findings.append(
                self._finding(
                    kind=SecurityAuditFindingKind.DIRECT_COGNITION_TOOL_EXECUTION,
                    severity=SecurityAuditSeverity.CRITICAL,
                    message="direct cognition-to-tool execution detected",
                    evidence=subject.title,
                    blocked=True,
                )
            )

        if bypass_pipeline:
            findings.append(
                self._finding(
                    kind=SecurityAuditFindingKind.POLICY_BYPASS,
                    severity=SecurityAuditSeverity.CRITICAL,
                    message="cognition intent attempts to bypass pipeline",
                    evidence=subject.title,
                    blocked=True,
                )
            )

        return tuple(findings)

    def _scan_action_step(
        self,
        *,
        step_kind: str,
    ) -> tuple[SecurityAuditFinding, ...]:
        normalized = step_kind.casefold()
        findings: list[SecurityAuditFinding] = []

        if "apply_patch" in normalized or normalized in {"write", "delete", "move"}:
            findings.append(
                self._finding(
                    kind=SecurityAuditFindingKind.MISSING_ROLLBACK_EXPLANATION,
                    severity=SecurityAuditSeverity.MEDIUM,
                    message="mutating step requires rollback explanation",
                    evidence=step_kind,
                    blocked=False,
                )
            )

        if "autonomous" in normalized and self._config.block_autonomy_escalation:
            findings.append(
                self._finding(
                    kind=SecurityAuditFindingKind.AUTONOMY_ESCALATION,
                    severity=SecurityAuditSeverity.CRITICAL,
                    message="unsafe autonomy escalation detected",
                    evidence=step_kind,
                    blocked=True,
                )
            )

        return tuple(findings)

    def _scan_step_arguments(
        self,
        arguments: dict[str, Any],
    ) -> tuple[SecurityAuditFinding, ...]:
        text = " ".join(str(value) for value in arguments.values())

        return self._scan_text(text)

    def _scan_shell_command(
        self,
        *,
        command: object,
    ) -> tuple[SecurityAuditFinding, ...]:
        if not isinstance(command, str):
            return ()

        return self._scan_patterns(
            text=command,
            patterns=self._DANGEROUS_SHELL_PATTERNS,
            kind=SecurityAuditFindingKind.UNSAFE_SHELL,
            severity=SecurityAuditSeverity.CRITICAL,
            message="unsafe shell command pattern detected",
            blocked=self._config.block_unsafe_shell,
        )

    def _scan_paths(
        self,
        *,
        path: object,
        destination_path: object,
    ) -> tuple[SecurityAuditFinding, ...]:
        findings: list[SecurityAuditFinding] = []

        for value in (path, destination_path):
            if not isinstance(value, str):
                continue

            normalized = value.replace("\\", "/")

            if "../" in normalized or normalized.startswith("../"):
                findings.append(
                    self._finding(
                        kind=SecurityAuditFindingKind.PATH_TRAVERSAL,
                        severity=SecurityAuditSeverity.CRITICAL,
                        message="path traversal detected",
                        evidence=value,
                        blocked=self._config.block_path_traversal,
                    )
                )

            if re.match(r"^[A-Za-z]:/", normalized) or normalized.startswith("/"):
                findings.append(
                    self._finding(
                        kind=SecurityAuditFindingKind.PATH_TRAVERSAL,
                        severity=SecurityAuditSeverity.HIGH,
                        message="absolute path detected",
                        evidence=value,
                        blocked=self._config.block_path_traversal,
                    )
                )

        return tuple(findings)

    def _scan_patterns(
        self,
        *,
        text: str,
        patterns: tuple[str, ...],
        kind: SecurityAuditFindingKind,
        severity: SecurityAuditSeverity,
        message: str,
        blocked: bool,
    ) -> tuple[SecurityAuditFinding, ...]:
        findings: list[SecurityAuditFinding] = []

        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)

            if match is None:
                continue

            findings.append(
                self._finding(
                    kind=kind,
                    severity=severity,
                    message=message,
                    evidence=match.group(0),
                    blocked=blocked,
                )
            )

        return tuple(findings)

    def _result(
        self,
        *,
        subject_id: str,
        subject_kind: SecurityAuditSubjectKind,
        findings: tuple[SecurityAuditFinding, ...],
    ) -> SecurityAuditResult:
        blocked = any(finding.blocked for finding in findings)

        if blocked:
            decision = SecurityAuditDecision.BLOCK
            message = "security hardening audit blocked the subject"

        elif findings:
            decision = SecurityAuditDecision.WARN
            message = "security hardening audit found warnings"

        else:
            decision = SecurityAuditDecision.PASS
            message = "security hardening audit passed"

        return SecurityAuditResult(
            subject_id=subject_id,
            subject_kind=subject_kind,
            decision=decision,
            findings=findings,
            passed=decision == SecurityAuditDecision.PASS,
            blocked=blocked,
            message=message,
            metadata={"runtime": self.name},
        )

    def _record_result(self, result: SecurityAuditResult) -> None:
        with self._lock:
            self._audit_count += 1
            self._finding_count += len(result.findings)
            self._last_decision = result.decision

            if result.decision == SecurityAuditDecision.PASS:
                self._pass_count += 1

            elif result.decision == SecurityAuditDecision.WARN:
                self._warn_count += 1

            else:
                self._block_count += 1

    @staticmethod
    def _combined_subject_text(subject: SecurityAuditSubject) -> str:
        values = [
            subject.title,
            subject.text,
            " ".join(str(value) for value in subject.data.values()),
        ]

        return "\n".join(values)

    @staticmethod
    def _flatten_keys(data: dict[str, Any]) -> tuple[str, ...]:
        keys: list[str] = []

        def walk(prefix: str, value: Any) -> None:
            if isinstance(value, dict):
                for key, item in value.items():
                    key_text = str(key)
                    next_prefix = f"{prefix}.{key_text}" if prefix else key_text
                    keys.append(next_prefix)
                    walk(next_prefix, item)

        walk("", data)

        return tuple(keys)

    def _is_sensitive_key(self, key: str) -> bool:
        normalized = key.casefold()

        return any(token in normalized for token in self._SENSITIVE_KEYS)

    @staticmethod
    def _finding(
        *,
        kind: SecurityAuditFindingKind,
        severity: SecurityAuditSeverity,
        message: str,
        evidence: str,
        blocked: bool,
    ) -> SecurityAuditFinding:
        return SecurityAuditFinding(
            kind=kind,
            severity=severity,
            message=message,
            evidence=evidence,
            blocked=blocked,
        )