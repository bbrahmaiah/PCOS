from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from jarvis.developer.build_watch import (
    BuildWatchEngine,
    BuildWatchRequest,
    BuildWatchStatus,
)
from jarvis.developer.code_context import (
    CodeContextEngine,
    CodeContextRequest,
    CodeContextStatus,
)
from jarvis.developer.code_navigation import (
    CodeNavigationEngine,
    CodeNavigationQueryKind,
    CodeNavigationRequest,
    CodeNavigationStatus,
)
from jarvis.developer.error_intelligence import (
    ErrorIntelligenceEngine,
    ErrorIntelligenceRequest,
    ErrorIntelligenceStatus,
    ErrorSourceKind,
)
from jarvis.developer.fix_suggestion import (
    FixSuggestionEngine,
    FixSuggestionRequest,
    FixSuggestionStatus,
)
from jarvis.developer.project_memory import (
    ProjectMemoryEngine,
    ProjectMemoryKind,
    ProjectMemoryRecallRequest,
    ProjectMemoryStatus,
    ProjectMemoryStoreRequest,
)
from jarvis.developer.test_runner import (
    TestDiscoveryStatus,
    TestRunnerEngine,
    TestRunRequest,
    TestRunStatus,
)


def utc_now() -> datetime:
    return datetime.now(UTC)


class DeveloperFeaturePackGateStatus(StrEnum):
    PASSED = "passed"
    FAILED = "failed"


class DeveloperFeaturePackCheckKind(StrEnum):
    PROJECT_VALID = "project_valid"
    CODE_CONTEXT = "code_context"
    CODE_NAVIGATION_INDEX = "code_navigation_index"
    CODE_NAVIGATION_SEARCH = "code_navigation_search"
    TEST_DISCOVERY = "test_discovery"
    TEST_RUNNER = "test_runner"
    ERROR_INTELLIGENCE = "error_intelligence"
    FIX_SUGGESTION = "fix_suggestion"
    PROJECT_MEMORY = "project_memory"
    BUILD_WATCH = "build_watch"
    SOURCE_IMMUTABILITY = "source_immutability"


@dataclass(frozen=True, slots=True)
class DeveloperFeaturePackGateConfig:
    project_root: Path
    active_file: Path | None = None
    navigation_query: str = "run"
    max_files: int = 500
    timeout_seconds: float = 120.0
    remember: bool = True
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.max_files < 1:
            raise ValueError("max_files must be at least 1.")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive.")


@dataclass(frozen=True, slots=True)
class DeveloperFeaturePackCheck:
    kind: DeveloperFeaturePackCheckKind
    passed: bool
    message: str
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class DeveloperFeaturePackGateReport:
    status: DeveloperFeaturePackGateStatus
    checks: tuple[DeveloperFeaturePackCheck, ...]
    started_at: datetime
    finished_at: datetime
    error: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.status == DeveloperFeaturePackGateStatus.PASSED

    @property
    def passed_count(self) -> int:
        return sum(1 for check in self.checks if check.passed)

    @property
    def failed_count(self) -> int:
        return sum(1 for check in self.checks if not check.passed)


class DeveloperFeaturePackCompletionGate:
    """
    Step 48 Developer Feature Pack Completion Gate.

    This proves the developer engines operate together as one safe pack:
    - CodeContextEngine
    - CodeNavigationEngine
    - TestRunnerEngine
    - ErrorIntelligenceEngine
    - FixSuggestionEngine
    - ProjectMemoryEngine
    - BuildWatchEngine

    It never mutates source files.
    It never applies fixes.
    It never executes arbitrary commands.
    """

    def __init__(
        self,
        *,
        config: DeveloperFeaturePackGateConfig,
        code_context: CodeContextEngine | None = None,
        code_navigation: CodeNavigationEngine | None = None,
        test_runner: TestRunnerEngine | None = None,
        error_intelligence: ErrorIntelligenceEngine | None = None,
        fix_suggestion: FixSuggestionEngine | None = None,
        project_memory: ProjectMemoryEngine | None = None,
        build_watch: BuildWatchEngine | None = None,
    ) -> None:
        self._config = config
        self._code_context = code_context or CodeContextEngine()
        self._code_navigation = code_navigation or CodeNavigationEngine()
        self._test_runner = test_runner or TestRunnerEngine()
        self._error_intelligence = (
            error_intelligence or ErrorIntelligenceEngine()
        )
        self._fix_suggestion = fix_suggestion or FixSuggestionEngine()
        self._project_memory = project_memory or ProjectMemoryEngine()
        self._build_watch = build_watch or BuildWatchEngine(
            code_context_engine=self._code_context,
            test_runner=self._test_runner,
            error_engine=self._error_intelligence,
            fix_engine=self._fix_suggestion,
            memory_engine=self._project_memory,
        )

    def run(self) -> DeveloperFeaturePackGateReport:
        started_at = utc_now()
        checks: list[DeveloperFeaturePackCheck] = []
        root = self._config.project_root.expanduser().resolve()
        source_fingerprints_before: dict[str, str] = {}

        try:
            project_valid = root.exists() and root.is_dir()
            _record(
                checks,
                kind=DeveloperFeaturePackCheckKind.PROJECT_VALID,
                passed=project_valid,
                message="project root is valid",
                metadata={"project_root": str(root)},
            )

            if not project_valid:
                return _report(
                    checks=checks,
                    started_at=started_at,
                    error=f"invalid project root: {root}",
                    metadata=self._config.metadata,
                )

            source_fingerprints_before = _source_fingerprints(root)

            context = self._code_context.build_context(
                CodeContextRequest(
                    project_root=root,
                    active_file=self._config.active_file,
                    max_files=self._config.max_files,
                    metadata=self._config.metadata,
                )
            )
            _record(
                checks,
                kind=DeveloperFeaturePackCheckKind.CODE_CONTEXT,
                passed=(
                    context.status
                    in {CodeContextStatus.READY, CodeContextStatus.PARTIAL}
                    and context.summary is not None
                ),
                message="code context engine produced project context",
                metadata={
                    "status": context.status.value,
                    "signal_count": len(context.signals),
                },
            )

            nav_index = self._code_navigation.build_index(
                CodeNavigationRequest(
                    project_root=root,
                    max_files=self._config.max_files,
                    metadata=self._config.metadata,
                )
            )
            _record(
                checks,
                kind=DeveloperFeaturePackCheckKind.CODE_NAVIGATION_INDEX,
                passed=(
                    nav_index.status
                    in {CodeNavigationStatus.READY, CodeNavigationStatus.PARTIAL}
                    and nav_index.index is not None
                ),
                message="code navigation index built",
                metadata={
                    "status": nav_index.status.value,
                    "file_count": (
                        nav_index.index.file_count
                        if nav_index.index is not None
                        else 0
                    ),
                    "symbol_count": (
                        nav_index.index.symbol_count
                        if nav_index.index is not None
                        else 0
                    ),
                },
            )

            nav_search = self._code_navigation.search(
                CodeNavigationRequest(
                    project_root=root,
                    query=self._config.navigation_query,
                    query_kind=CodeNavigationQueryKind.REFERENCE,
                    max_files=self._config.max_files,
                    metadata=self._config.metadata,
                )
            )
            _record(
                checks,
                kind=DeveloperFeaturePackCheckKind.CODE_NAVIGATION_SEARCH,
                passed=nav_search.status
                in {CodeNavigationStatus.READY, CodeNavigationStatus.NOT_FOUND},
                message="code navigation search completed safely",
                metadata={
                    "status": nav_search.status.value,
                    "match_count": len(nav_search.matches),
                    "query": self._config.navigation_query,
                },
            )

            return self._run_after_navigation(
                checks=checks,
                root=root,
                source_fingerprints_before=source_fingerprints_before,
                started_at=started_at,
            )

        except Exception as exc:
            return _report(
                checks=checks,
                started_at=started_at,
                error=f"{type(exc).__name__}: {exc}",
                metadata=self._config.metadata,
            )

    def _run_after_navigation(
        self,
        *,
        checks: list[DeveloperFeaturePackCheck],
        root: Path,
        source_fingerprints_before: dict[str, str],
        started_at: datetime,
    ) -> DeveloperFeaturePackGateReport:
        from jarvis.developer.test_runner import TestDiscoveryRequest

        discovery = self._test_runner.discover(
            TestDiscoveryRequest(
                project_root=root,
                max_files=self._config.max_files,
                metadata=self._config.metadata,
            )
        )
        _record(
            checks,
            kind=DeveloperFeaturePackCheckKind.TEST_DISCOVERY,
            passed=discovery.status
            in {TestDiscoveryStatus.READY, TestDiscoveryStatus.PARTIAL},
            message="test workflow discovery completed",
            metadata={
                "status": discovery.status.value,
                "test_file_count": len(discovery.test_files),
                "has_plan": discovery.suggested_plan is not None,
            },
        )

        test_result = self._test_runner.run(
            TestRunRequest(
                project_root=root,
                timeout_seconds=self._config.timeout_seconds,
                metadata=self._config.metadata,
            )
        )
        _record(
            checks,
            kind=DeveloperFeaturePackCheckKind.TEST_RUNNER,
            passed=test_result.status
            in {
                TestRunStatus.PASSED,
                TestRunStatus.FAILED,
                TestRunStatus.TIMEOUT,
                TestRunStatus.BLOCKED,
            },
            message="test runner completed with controlled status",
            metadata={
                "status": test_result.status.value,
                "exit_code": test_result.exit_code,
            },
        )

        error_report = self._error_intelligence.analyze(
            ErrorIntelligenceRequest(
                stdout=(
                    test_result.stdout
                    if test_result.stdout.strip()
                    else "No structured failure output."
                ),
                stderr=test_result.stderr,
                source=_source_for_test_status(test_result.status),
                exit_code=test_result.exit_code,
                project_root=root,
                command=test_result.plan.command,
                metadata=self._config.metadata,
            )
        )
        _record(
            checks,
            kind=DeveloperFeaturePackCheckKind.ERROR_INTELLIGENCE,
            passed=error_report.status
            in {
                ErrorIntelligenceStatus.ANALYZED,
                ErrorIntelligenceStatus.PARTIAL,
                ErrorIntelligenceStatus.NO_ERROR,
            },
            message="error intelligence completed",
            metadata={
                "status": error_report.status.value,
                "has_errors": error_report.has_errors,
                "signal_count": len(error_report.signals),
            },
        )

        if error_report.status == ErrorIntelligenceStatus.NO_ERROR:
            synthetic_error_report = self._error_intelligence.analyze(
                ErrorIntelligenceRequest(
                    stdout="jarvis/example.py:1: error: synthetic check [misc]",
                    source=ErrorSourceKind.MYPY,
                    exit_code=1,
                    project_root=root,
                    metadata={"synthetic": True},
                )
            )
        else:
            synthetic_error_report = error_report

        fix_report = self._fix_suggestion.suggest(
            FixSuggestionRequest(
                error_report=synthetic_error_report,
                code_context=None,
                metadata=self._config.metadata,
            )
        )
        _record(
            checks,
            kind=DeveloperFeaturePackCheckKind.FIX_SUGGESTION,
            passed=fix_report.status == FixSuggestionStatus.READY
            and bool(fix_report.suggestions),
            message="fix suggestion engine produced reviewable suggestions",
            metadata={
                "status": fix_report.status.value,
                "suggestion_count": len(fix_report.suggestions),
            },
        )

        memory_result = self._project_memory.store(
            ProjectMemoryStoreRequest(
                project_root=root,
                kind=ProjectMemoryKind.SESSION_STATE,
                title="Developer feature pack completion gate",
                summary="Developer feature pack gate executed.",
                details="All developer engines were exercised together.",
                source="developer_feature_pack_completion_gate",
                metadata=self._config.metadata,
            )
        )
        recall_result = self._project_memory.recall(
            ProjectMemoryRecallRequest(
                project_root=root,
                query="completion gate",
                kinds=(ProjectMemoryKind.SESSION_STATE,),
                limit=5,
            )
        )
        _record(
            checks,
            kind=DeveloperFeaturePackCheckKind.PROJECT_MEMORY,
            passed=(
                memory_result.status == ProjectMemoryStatus.STORED
                and recall_result.status == ProjectMemoryStatus.LOADED
                and bool(recall_result.records)
            ),
            message="project memory stored and recalled gate state",
            metadata={
                "store_status": memory_result.status.value,
                "recall_status": recall_result.status.value,
                "recall_count": len(recall_result.records),
            },
        )

        build_snapshot = self._build_watch.check_once(
            BuildWatchRequest(
                project_root=root,
                run_tests=True,
                remember=self._config.remember,
                timeout_seconds=self._config.timeout_seconds,
                max_files=self._config.max_files,
                metadata=self._config.metadata,
            )
        )
        _record(
            checks,
            kind=DeveloperFeaturePackCheckKind.BUILD_WATCH,
            passed=build_snapshot.status
            in {
                BuildWatchStatus.HEALTHY,
                BuildWatchStatus.FAILING,
                BuildWatchStatus.BLOCKED,
                BuildWatchStatus.DEGRADED,
            },
            message="build watch completed one controlled cycle",
            metadata={
                "status": build_snapshot.status.value,
                "decision": build_snapshot.decision.value,
                "event_count": len(build_snapshot.events),
            },
        )

        source_fingerprints_after = _source_fingerprints(root)
        unchanged = source_fingerprints_before == source_fingerprints_after
        _record(
            checks,
            kind=DeveloperFeaturePackCheckKind.SOURCE_IMMUTABILITY,
            passed=unchanged,
            message="source files were not mutated by developer gate",
            metadata={
                "before_count": len(source_fingerprints_before),
                "after_count": len(source_fingerprints_after),
            },
        )

        return _report(
            checks=checks,
            started_at=started_at,
            error=None,
            metadata=self._config.metadata,
        )


def _record(
    checks: list[DeveloperFeaturePackCheck],
    *,
    kind: DeveloperFeaturePackCheckKind,
    passed: bool,
    message: str,
    metadata: dict[str, object] | None = None,
) -> None:
    checks.append(
        DeveloperFeaturePackCheck(
            kind=kind,
            passed=passed,
            message=message,
            created_at=utc_now(),
            metadata=metadata or {},
        )
    )


def _report(
    *,
    checks: list[DeveloperFeaturePackCheck],
    started_at: datetime,
    error: str | None,
    metadata: dict[str, object],
) -> DeveloperFeaturePackGateReport:
    status = (
        DeveloperFeaturePackGateStatus.PASSED
        if checks and all(check.passed for check in checks) and error is None
        else DeveloperFeaturePackGateStatus.FAILED
    )
    return DeveloperFeaturePackGateReport(
        status=status,
        checks=tuple(checks),
        started_at=started_at,
        finished_at=utc_now(),
        error=error,
        metadata=metadata,
    )


def _source_for_test_status(status: TestRunStatus) -> ErrorSourceKind:
    if status in {TestRunStatus.PASSED, TestRunStatus.FAILED}:
        return ErrorSourceKind.PYTEST

    return ErrorSourceKind.TERMINAL


def _source_fingerprints(root: Path) -> dict[str, str]:
    fingerprints: dict[str, str] = {}

    for path in root.rglob("*"):
        if not path.is_file():
            continue

        if ".jarvis" in path.parts:
            continue

        if path.suffix.lower() not in {".py", ".toml", ".md", ".json", ".txt"}:
            continue

        relative = str(path.relative_to(root))
        fingerprints[relative] = path.read_text(
            encoding="utf-8",
            errors="ignore",
        )

    return fingerprints