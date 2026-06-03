from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from jarvis.developer.code_context import (
    CodeContextEngine,
    CodeContextRequest,
    CodeContextSnapshot,
)
from jarvis.developer.error_intelligence import (
    ErrorIntelligenceEngine,
    ErrorIntelligenceReport,
    ErrorIntelligenceRequest,
    ErrorIntelligenceStatus,
    ErrorSourceKind,
)
from jarvis.developer.fix_suggestion import (
    FixSuggestionEngine,
    FixSuggestionReport,
    FixSuggestionRequest,
)
from jarvis.developer.project_memory import (
    ProjectMemoryEngine,
    ProjectMemoryKind,
    ProjectMemoryResult,
)
from jarvis.developer.test_runner import (
    TestRunnerEngine,
    TestRunRequest,
    TestRunResult,
    TestRunStatus,
)


def utc_now() -> datetime:
    return datetime.now(UTC)


class BuildWatchStatus(StrEnum):
    HEALTHY = "healthy"
    FAILING = "failing"
    BLOCKED = "blocked"
    DEGRADED = "degraded"


class BuildWatchEventKind(StrEnum):
    CONTEXT_CAPTURED = "context_captured"
    CHANGES_DETECTED = "changes_detected"
    TEST_RUN_COMPLETED = "test_run_completed"
    ERROR_ANALYZED = "error_analyzed"
    FIX_SUGGESTED = "fix_suggested"
    MEMORY_UPDATED = "memory_updated"
    WATCH_BLOCKED = "watch_blocked"


class BuildWatchDecision(StrEnum):
    NO_ACTION = "no_action"
    RUN_TESTS = "run_tests"
    ANALYZE_FAILURE = "analyze_failure"
    SUGGEST_FIX = "suggest_fix"
    REMEMBER_ONLY = "remember_only"
    BLOCKED = "blocked"


@dataclass(frozen=True, slots=True)
class BuildWatchRequest:
    project_root: Path
    active_file: Path | None = None
    run_tests: bool = True
    remember: bool = True
    timeout_seconds: float = 120.0
    max_files: int = 300
    previous_fingerprint: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive.")
        if self.max_files < 1:
            raise ValueError("max_files must be at least 1.")


@dataclass(frozen=True, slots=True)
class BuildWatchEvent:
    kind: BuildWatchEventKind
    message: str
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class BuildWatchSnapshot:
    project_root: str
    fingerprint: str
    changed: bool
    status: BuildWatchStatus
    decision: BuildWatchDecision
    code_context: CodeContextSnapshot | None
    test_result: TestRunResult | None
    error_report: ErrorIntelligenceReport | None
    fix_report: FixSuggestionReport | None
    memory_results: tuple[ProjectMemoryResult, ...]
    events: tuple[BuildWatchEvent, ...]
    summary: str
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def healthy(self) -> bool:
        return self.status == BuildWatchStatus.HEALTHY


class BuildWatchEngine:
    """
    Step 47F Build Watch Engine.

    Controlled developer workflow coordinator:
    - captures code context
    - computes project fingerprint
    - detects changes
    - optionally runs safe tests
    - analyzes failures
    - suggests fixes
    - stores project memory

    It never mutates source files.
    It never runs arbitrary commands.
    It never applies fixes.
    It performs one bounded watch cycle per call.
    """

    def __init__(
        self,
        *,
        code_context_engine: CodeContextEngine | None = None,
        test_runner: TestRunnerEngine | None = None,
        error_engine: ErrorIntelligenceEngine | None = None,
        fix_engine: FixSuggestionEngine | None = None,
        memory_engine: ProjectMemoryEngine | None = None,
    ) -> None:
        self._code_context = code_context_engine or CodeContextEngine()
        self._test_runner = test_runner or TestRunnerEngine()
        self._error_engine = error_engine or ErrorIntelligenceEngine()
        self._fix_engine = fix_engine or FixSuggestionEngine()
        self._memory = memory_engine or ProjectMemoryEngine()

    def check_once(self, request: BuildWatchRequest) -> BuildWatchSnapshot:
        root = request.project_root.expanduser().resolve()
        events: list[BuildWatchEvent] = []
        memory_results: list[ProjectMemoryResult] = []

        if not root.exists() or not root.is_dir():
            _record(
                events,
                kind=BuildWatchEventKind.WATCH_BLOCKED,
                message="project root is invalid",
                metadata={"project_root": str(root)},
            )
            return BuildWatchSnapshot(
                project_root=str(root),
                fingerprint="",
                changed=False,
                status=BuildWatchStatus.BLOCKED,
                decision=BuildWatchDecision.BLOCKED,
                code_context=None,
                test_result=None,
                error_report=None,
                fix_report=None,
                memory_results=(),
                events=tuple(events),
                summary=f"Build watch blocked: invalid project root {root}",
                created_at=utc_now(),
                metadata=request.metadata,
            )

        code_context = self._code_context.build_context(
            CodeContextRequest(
                project_root=root,
                active_file=request.active_file,
                max_files=request.max_files,
                metadata=request.metadata,
            )
        )
        _record(
            events,
            kind=BuildWatchEventKind.CONTEXT_CAPTURED,
            message="code context captured",
            metadata={
                "status": code_context.status.value,
                "signal_count": len(code_context.signals),
            },
        )

        fingerprint = _fingerprint(code_context)
        changed = (
            request.previous_fingerprint is None
            or request.previous_fingerprint != fingerprint
        )
        _record(
            events,
            kind=BuildWatchEventKind.CHANGES_DETECTED,
            message="project fingerprint evaluated",
            metadata={
                "changed": changed,
                "fingerprint": fingerprint,
                "previous_fingerprint": request.previous_fingerprint or "",
            },
        )

        if request.remember:
            remembered_context = self._memory.remember_code_context(
                project_root=root,
                code_context=code_context,
            )
            memory_results.append(remembered_context)
            _record(
                events,
                kind=BuildWatchEventKind.MEMORY_UPDATED,
                message="code context memory updated",
                metadata={
                    "status": remembered_context.status.value,
                    "kind": ProjectMemoryKind.CODE_CONTEXT.value,
                },
            )

        if not request.run_tests:
            return BuildWatchSnapshot(
                project_root=str(root),
                fingerprint=fingerprint,
                changed=changed,
                status=BuildWatchStatus.HEALTHY,
                decision=BuildWatchDecision.REMEMBER_ONLY,
                code_context=code_context,
                test_result=None,
                error_report=None,
                fix_report=None,
                memory_results=tuple(memory_results),
                events=tuple(events),
                summary="Build watch captured context without running tests.",
                created_at=utc_now(),
                metadata=request.metadata,
            )

        test_result = self._test_runner.run(
            TestRunRequest(
                project_root=root,
                timeout_seconds=request.timeout_seconds,
                metadata=request.metadata,
            )
        )
        _record(
            events,
            kind=BuildWatchEventKind.TEST_RUN_COMPLETED,
            message="test run completed",
            metadata={
                "status": test_result.status.value,
                "exit_code": test_result.exit_code,
                "duration_ms": test_result.duration_ms,
            },
        )

        if request.remember:
            remembered_test = self._memory.remember_test_result(
                project_root=root,
                test_result=test_result,
            )
            memory_results.append(remembered_test)
            _record(
                events,
                kind=BuildWatchEventKind.MEMORY_UPDATED,
                message="test result memory updated",
                metadata={
                    "status": remembered_test.status.value,
                    "kind": ProjectMemoryKind.TEST_RESULT.value,
                },
            )

        if test_result.status == TestRunStatus.PASSED:
            return BuildWatchSnapshot(
                project_root=str(root),
                fingerprint=fingerprint,
                changed=changed,
                status=BuildWatchStatus.HEALTHY,
                decision=BuildWatchDecision.RUN_TESTS,
                code_context=code_context,
                test_result=test_result,
                error_report=None,
                fix_report=None,
                memory_results=tuple(memory_results),
                events=tuple(events),
                summary="Build watch healthy: tests passed.",
                created_at=utc_now(),
                metadata=request.metadata,
            )

        if test_result.status == TestRunStatus.BLOCKED:
            return BuildWatchSnapshot(
                project_root=str(root),
                fingerprint=fingerprint,
                changed=changed,
                status=BuildWatchStatus.BLOCKED,
                decision=BuildWatchDecision.BLOCKED,
                code_context=code_context,
                test_result=test_result,
                error_report=None,
                fix_report=None,
                memory_results=tuple(memory_results),
                events=tuple(events),
                summary=f"Build watch blocked: {test_result.summary}",
                created_at=utc_now(),
                metadata=request.metadata,
            )

        error_report = self._error_engine.analyze(
            ErrorIntelligenceRequest(
                stdout=test_result.stdout,
                stderr=test_result.stderr,
                source=_source_for_test_result(test_result),
                exit_code=test_result.exit_code,
                project_root=root,
                command=test_result.plan.command,
                metadata=request.metadata,
            )
        )
        _record(
            events,
            kind=BuildWatchEventKind.ERROR_ANALYZED,
            message="test failure output analyzed",
            metadata={
                "status": error_report.status.value,
                "has_errors": error_report.has_errors,
                "signal_count": len(error_report.signals),
            },
        )

        if request.remember:
            remembered_error = self._memory.remember_error_report(
                project_root=root,
                error_report=error_report,
            )
            memory_results.append(remembered_error)
            _record(
                events,
                kind=BuildWatchEventKind.MEMORY_UPDATED,
                message="error diagnosis memory updated",
                metadata={
                    "status": remembered_error.status.value,
                    "kind": ProjectMemoryKind.ERROR_DIAGNOSIS.value,
                },
            )

        fix_report = self._fix_engine.suggest(
            FixSuggestionRequest(
                error_report=error_report,
                code_context=code_context,
                metadata=request.metadata,
            )
        )
        _record(
            events,
            kind=BuildWatchEventKind.FIX_SUGGESTED,
            message="fix suggestions generated",
            metadata={
                "status": fix_report.status.value,
                "suggestion_count": len(fix_report.suggestions),
            },
        )

        if request.remember:
            remembered_fix = self._memory.remember_fix_report(
                project_root=root,
                fix_report=fix_report,
            )
            memory_results.append(remembered_fix)
            _record(
                events,
                kind=BuildWatchEventKind.MEMORY_UPDATED,
                message="fix suggestion memory updated",
                metadata={
                    "status": remembered_fix.status.value,
                    "kind": ProjectMemoryKind.FIX_SUGGESTION.value,
                },
            )

        status = (
            BuildWatchStatus.FAILING
            if error_report.status
            in {
                ErrorIntelligenceStatus.ANALYZED,
                ErrorIntelligenceStatus.PARTIAL,
            }
            else BuildWatchStatus.DEGRADED
        )

        return BuildWatchSnapshot(
            project_root=str(root),
            fingerprint=fingerprint,
            changed=changed,
            status=status,
            decision=BuildWatchDecision.SUGGEST_FIX,
            code_context=code_context,
            test_result=test_result,
            error_report=error_report,
            fix_report=fix_report,
            memory_results=tuple(memory_results),
            events=tuple(events),
            summary=_failure_summary(
                test_result=test_result,
                error_report=error_report,
                fix_report=fix_report,
            ),
            created_at=utc_now(),
            metadata=request.metadata,
        )


def _record(
    events: list[BuildWatchEvent],
    *,
    kind: BuildWatchEventKind,
    message: str,
    metadata: dict[str, object] | None = None,
) -> None:
    events.append(
        BuildWatchEvent(
            kind=kind,
            message=message,
            created_at=utc_now(),
            metadata=metadata or {},
        )
    )


def _fingerprint(code_context: CodeContextSnapshot) -> str:
    parts: list[str] = []

    for signal in code_context.signals:
        parts.append(
            "|".join(
                (
                    signal.path,
                    signal.kind.value,
                    str(signal.size_bytes),
                    (
                        signal.modified_at.isoformat()
                        if signal.modified_at is not None
                        else ""
                    ),
                )
            )
        )

    return str(abs(hash("\n".join(parts))))


def _source_for_test_result(test_result: TestRunResult) -> ErrorSourceKind:
    command_text = " ".join(test_result.plan.command).lower()

    if "pytest" in command_text:
        return ErrorSourceKind.PYTEST

    return ErrorSourceKind.TERMINAL


def _failure_summary(
    *,
    test_result: TestRunResult,
    error_report: ErrorIntelligenceReport,
    fix_report: FixSuggestionReport,
) -> str:
    if error_report.diagnosis is not None:
        return (
            f"Build watch failing: {error_report.diagnosis.title}. "
            f"{len(fix_report.suggestions)} fix suggestion(s) available."
        )

    return (
        f"Build watch failing: {test_result.summary}. "
        f"{len(fix_report.suggestions)} fix suggestion(s) available."
    )