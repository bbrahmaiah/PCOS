from __future__ import annotations

import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum

from jarvis.developer.code_context import CodeContextSnapshot
from jarvis.developer.error_intelligence import (
    ErrorCategory,
    ErrorDiagnosis,
    ErrorIntelligenceReport,
    ErrorIntelligenceStatus,
    ErrorSeverity,
    ErrorSignal,
)


def utc_now() -> datetime:
    return datetime.now(UTC)


class FixSuggestionStatus(StrEnum):
    READY = "ready"
    NO_SUGGESTION = "no_suggestion"
    BLOCKED = "blocked"


class FixSuggestionKind(StrEnum):
    TYPE_ALIGNMENT = "type_alignment"
    LINT_FORMATTING = "lint_formatting"
    TEST_EXPECTATION_REVIEW = "test_expectation_review"
    IMPORT_PATH_REVIEW = "import_path_review"
    SYNTAX_CORRECTION = "syntax_correction"
    RUNTIME_GUARD = "runtime_guard"
    TIMEOUT_BOUNDARY = "timeout_boundary"
    CONFIGURATION_REVIEW = "configuration_review"
    UNKNOWN_REVIEW = "unknown_review"


class FixSuggestionRisk(StrEnum):
    SAFE_REVIEW = "safe_review"
    NEEDS_HUMAN_REVIEW = "needs_human_review"
    HIGH_RISK_REVIEW = "high_risk_review"


class FixActionKind(StrEnum):
    INSPECT_FILE = "inspect_file"
    REVIEW_SIGNATURE = "review_signature"
    REVIEW_IMPORT = "review_import"
    REVIEW_ASSERTION = "review_assertion"
    APPLY_FORMATTING = "apply_formatting"
    FIX_SYNTAX = "fix_syntax"
    ADD_GUARD = "add_guard"
    CHECK_TIMEOUT = "check_timeout"
    RUN_VALIDATION = "run_validation"


@dataclass(frozen=True, slots=True)
class FixSuggestionRequest:
    error_report: ErrorIntelligenceReport
    code_context: CodeContextSnapshot | None = None
    max_suggestions: int = 5
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.max_suggestions < 1:
            raise ValueError("max_suggestions must be at least 1.")


@dataclass(frozen=True, slots=True)
class FixAction:
    kind: FixActionKind
    title: str
    description: str
    file_path: str | None = None
    line: int | None = None
    command_preview: tuple[str, ...] | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.title.strip():
            raise ValueError("fix action title cannot be empty.")
        if not self.description.strip():
            raise ValueError("fix action description cannot be empty.")


@dataclass(frozen=True, slots=True)
class FixSuggestion:
    kind: FixSuggestionKind
    risk: FixSuggestionRisk
    title: str
    rationale: str
    proposed_change: str
    confidence: float
    evidence: tuple[str, ...]
    affected_files: tuple[str, ...]
    actions: tuple[FixAction, ...]
    validation_command: tuple[str, ...] | None
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.title.strip():
            raise ValueError("fix suggestion title cannot be empty.")
        if not self.rationale.strip():
            raise ValueError("fix suggestion rationale cannot be empty.")
        if not self.proposed_change.strip():
            raise ValueError("fix suggestion proposed_change cannot be empty.")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("fix suggestion confidence must be between 0 and 1.")


@dataclass(frozen=True, slots=True)
class FixSuggestionReport:
    status: FixSuggestionStatus
    suggestions: tuple[FixSuggestion, ...]
    reason: str
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def has_suggestions(self) -> bool:
        return self.status == FixSuggestionStatus.READY and bool(
            self.suggestions
        )


class FixSuggestionEngine:
    """
    Step 47D Fix Suggestion Engine.

    This engine converts structured error intelligence into safe, reviewable
    fix suggestions.

    It never mutates files.
    It never executes commands.
    It never applies patches.
    It only produces evidence-grounded suggestions and validation guidance.
    """

    def suggest(
        self,
        request: FixSuggestionRequest,
    ) -> FixSuggestionReport:
        report = request.error_report

        if report.status == ErrorIntelligenceStatus.NO_ERROR:
            return FixSuggestionReport(
                status=FixSuggestionStatus.NO_SUGGESTION,
                suggestions=(),
                reason="no error report signals to fix",
                created_at=utc_now(),
                metadata=request.metadata,
            )

        if report.diagnosis is None and not report.signals:
            return FixSuggestionReport(
                status=FixSuggestionStatus.NO_SUGGESTION,
                suggestions=(),
                reason="error report has no diagnosis or signals",
                created_at=utc_now(),
                metadata=request.metadata,
            )

        suggestions = _build_suggestions(
            report=report,
            code_context=request.code_context,
        )[: request.max_suggestions]

        if not suggestions:
            return FixSuggestionReport(
                status=FixSuggestionStatus.NO_SUGGESTION,
                suggestions=(),
                reason="no safe suggestion could be generated",
                created_at=utc_now(),
                metadata=request.metadata,
            )

        return FixSuggestionReport(
            status=FixSuggestionStatus.READY,
            suggestions=tuple(suggestions),
            reason="fix suggestions generated from error evidence",
            created_at=utc_now(),
            metadata={
                **request.metadata,
                "suggestion_count": len(suggestions),
                "source": report.source.value,
            },
        )


def _build_suggestions(
    *,
    report: ErrorIntelligenceReport,
    code_context: CodeContextSnapshot | None,
) -> list[FixSuggestion]:
    diagnosis = report.diagnosis
    signals = report.signals

    if diagnosis is not None:
        return [
            _suggestion_from_diagnosis(
                diagnosis=diagnosis,
                signals=signals,
                code_context=code_context,
            )
        ]

    return [
        _suggestion_from_signal(
            signal=signal,
            code_context=code_context,
        )
        for signal in signals
    ]


def _suggestion_from_diagnosis(
    *,
    diagnosis: ErrorDiagnosis,
    signals: tuple[ErrorSignal, ...],
    code_context: CodeContextSnapshot | None,
) -> FixSuggestion:
    primary = _primary_signal_from_diagnosis(
        diagnosis=diagnosis,
        signals=signals,
    )
    kind = _kind_for_category(diagnosis.primary_category)
    risk = _risk_for_diagnosis(diagnosis)
    actions = _actions_for_category(
        category=diagnosis.primary_category,
        signal=primary,
    )

    return FixSuggestion(
        kind=kind,
        risk=risk,
        title=_title_for_kind(kind),
        rationale=diagnosis.likely_cause,
        proposed_change=_proposed_change(
            category=diagnosis.primary_category,
            signal=primary,
        ),
        confidence=_bounded_confidence(diagnosis.confidence, risk),
        evidence=diagnosis.evidence[:5],
        affected_files=diagnosis.affected_files,
        actions=actions,
        validation_command=_validation_command(code_context),
        created_at=utc_now(),
        metadata={
            "diagnosis_title": diagnosis.title,
            "severity": diagnosis.severity.value,
        },
    )


def _suggestion_from_signal(
    *,
    signal: ErrorSignal,
    code_context: CodeContextSnapshot | None,
) -> FixSuggestion:
    kind = _kind_for_category(signal.category)
    risk = _risk_for_signal(signal)

    return FixSuggestion(
        kind=kind,
        risk=risk,
        title=_title_for_kind(kind),
        rationale=_rationale_for_signal(signal),
        proposed_change=_proposed_change(
            category=signal.category,
            signal=signal,
        ),
        confidence=_bounded_confidence(signal.confidence, risk),
        evidence=(signal.evidence,),
        affected_files=_affected_files_from_signal(signal),
        actions=_actions_for_category(
            category=signal.category,
            signal=signal,
        ),
        validation_command=_validation_command(code_context),
        created_at=utc_now(),
        metadata={
            "source": signal.source.value,
            "severity": signal.severity.value,
        },
    )


def _primary_signal_from_diagnosis(
    *,
    diagnosis: ErrorDiagnosis,
    signals: tuple[ErrorSignal, ...],
) -> ErrorSignal | None:
    for signal in signals:
        if signal.category == diagnosis.primary_category:
            return signal

    return signals[0] if signals else None


def _kind_for_category(category: ErrorCategory) -> FixSuggestionKind:
    mapping = {
        ErrorCategory.TYPE_ERROR: FixSuggestionKind.TYPE_ALIGNMENT,
        ErrorCategory.LINT_ERROR: FixSuggestionKind.LINT_FORMATTING,
        ErrorCategory.TEST_FAILURE: FixSuggestionKind.TEST_EXPECTATION_REVIEW,
        ErrorCategory.ASSERTION_ERROR: (
            FixSuggestionKind.TEST_EXPECTATION_REVIEW
        ),
        ErrorCategory.IMPORT_ERROR: FixSuggestionKind.IMPORT_PATH_REVIEW,
        ErrorCategory.SYNTAX_ERROR: FixSuggestionKind.SYNTAX_CORRECTION,
        ErrorCategory.RUNTIME_ERROR: FixSuggestionKind.RUNTIME_GUARD,
        ErrorCategory.TIMEOUT: FixSuggestionKind.TIMEOUT_BOUNDARY,
        ErrorCategory.CONFIGURATION_ERROR: (
            FixSuggestionKind.CONFIGURATION_REVIEW
        ),
        ErrorCategory.UNKNOWN: FixSuggestionKind.UNKNOWN_REVIEW,
    }
    return mapping[category]


def _risk_for_diagnosis(diagnosis: ErrorDiagnosis) -> FixSuggestionRisk:
    if diagnosis.severity in {
        ErrorSeverity.HIGH,
        ErrorSeverity.CRITICAL,
    }:
        return FixSuggestionRisk.NEEDS_HUMAN_REVIEW

    if diagnosis.primary_category in {
        ErrorCategory.TEST_FAILURE,
        ErrorCategory.ASSERTION_ERROR,
        ErrorCategory.RUNTIME_ERROR,
        ErrorCategory.UNKNOWN,
    }:
        return FixSuggestionRisk.NEEDS_HUMAN_REVIEW

    return FixSuggestionRisk.SAFE_REVIEW


def _risk_for_signal(signal: ErrorSignal) -> FixSuggestionRisk:
    if signal.severity in {
        ErrorSeverity.HIGH,
        ErrorSeverity.CRITICAL,
    }:
        return FixSuggestionRisk.NEEDS_HUMAN_REVIEW

    if signal.category in {
        ErrorCategory.TEST_FAILURE,
        ErrorCategory.ASSERTION_ERROR,
        ErrorCategory.RUNTIME_ERROR,
        ErrorCategory.UNKNOWN,
    }:
        return FixSuggestionRisk.NEEDS_HUMAN_REVIEW

    return FixSuggestionRisk.SAFE_REVIEW


def _actions_for_category(
    *,
    category: ErrorCategory,
    signal: ErrorSignal | None,
) -> tuple[FixAction, ...]:
    file_path = (
        signal.location.file_path
        if signal is not None
        and signal.location is not None
        else None
    )
    line = (
        signal.location.line
        if signal is not None
        and signal.location is not None
        else None
    )

    common = (
        FixAction(
            kind=FixActionKind.INSPECT_FILE,
            title="Inspect the evidence location",
            description=(
                "Open the affected file and review the exact line connected "
                "to the diagnostic evidence."
            ),
            file_path=file_path,
            line=line,
        ),
    )

    if category == ErrorCategory.TYPE_ERROR:
        return common + (
            FixAction(
                kind=FixActionKind.REVIEW_SIGNATURE,
                title="Review the type contract",
                description=(
                    "Compare the reported value, argument, return type, or "
                    "annotation with the expected type contract."
                ),
                file_path=file_path,
                line=line,
            ),
        )

    if category == ErrorCategory.LINT_ERROR:
        return common + (
            FixAction(
                kind=FixActionKind.APPLY_FORMATTING,
                title="Apply the lint correction",
                description=(
                    "Make the smallest formatting or static-rule correction "
                    "required by the lint diagnostic."
                ),
                file_path=file_path,
                line=line,
            ),
        )

    if category in {
        ErrorCategory.TEST_FAILURE,
        ErrorCategory.ASSERTION_ERROR,
    }:
        return common + (
            FixAction(
                kind=FixActionKind.REVIEW_ASSERTION,
                title="Review expected versus observed behavior",
                description=(
                    "Check whether the test expectation is correct, then "
                    "trace the implementation path that produced the result."
                ),
                file_path=file_path,
                line=line,
            ),
        )

    if category == ErrorCategory.IMPORT_ERROR:
        return common + (
            FixAction(
                kind=FixActionKind.REVIEW_IMPORT,
                title="Review import path and export",
                description=(
                    "Verify that the module exists, the symbol is exported, "
                    "and the dependency is available in the environment."
                ),
                file_path=file_path,
                line=line,
            ),
        )

    if category == ErrorCategory.SYNTAX_ERROR:
        return common + (
            FixAction(
                kind=FixActionKind.FIX_SYNTAX,
                title="Fix syntax first",
                description=(
                    "Correct the syntax error before running deeper analysis."
                ),
                file_path=file_path,
                line=line,
            ),
        )

    if category == ErrorCategory.TIMEOUT:
        return (
            FixAction(
                kind=FixActionKind.CHECK_TIMEOUT,
                title="Inspect timeout boundary",
                description=(
                    "Look for blocking loops, unbounded waits, slow external "
                    "calls, or missing cancellation handling."
                ),
                file_path=file_path,
                line=line,
            ),
        )

    return common + (
        FixAction(
            kind=FixActionKind.ADD_GUARD,
            title="Add diagnostic guard if needed",
            description=(
                "Add a minimal guard, clearer error, or validation boundary "
                "only after confirming the root cause."
            ),
            file_path=file_path,
            line=line,
        ),
    )


def _proposed_change(
    *,
    category: ErrorCategory,
    signal: ErrorSignal | None,
) -> str:
    message = signal.message if signal is not None else ""

    if category == ErrorCategory.TYPE_ERROR:
        return (
            "Align the implementation with the reported type contract. "
            "Prefer changing the narrowest incorrect annotation, argument, "
            "or value construction rather than weakening types globally."
        )

    if category == ErrorCategory.LINT_ERROR:
        return (
            "Apply the smallest lint-safe formatting or style correction. "
            f"Diagnostic: {message}"
        )

    if category in {
        ErrorCategory.TEST_FAILURE,
        ErrorCategory.ASSERTION_ERROR,
    }:
        return (
            "Compare the failing assertion with the intended behavior. "
            "Fix the implementation if the test is correct; update the test "
            "only if the requirement has intentionally changed."
        )

    if category == ErrorCategory.IMPORT_ERROR:
        return (
            "Fix the import path, exported symbol, or missing dependency. "
            "Avoid broad sys.path changes unless the package boundary is "
            "actually wrong."
        )

    if category == ErrorCategory.SYNTAX_ERROR:
        return (
            "Correct the syntax at the reported location before attempting "
            "runtime or type-level fixes."
        )

    if category == ErrorCategory.RUNTIME_ERROR:
        return (
            "Trace the exception path and fix the root cause. Add validation "
            "or a guard only if invalid input can legitimately reach this path."
        )

    if category == ErrorCategory.TIMEOUT:
        return (
            "Add or repair cancellation, timeout, batching, or bounded-wait "
            "logic around the slow path."
        )

    if category == ErrorCategory.CONFIGURATION_ERROR:
        return (
            "Inspect configuration names, required fields, environment paths, "
            "and dependency availability."
        )

    return (
        "Do not patch blindly. Inspect the raw output, recent changes, and "
        "nearest affected files before choosing a fix."
    )


def _rationale_for_signal(signal: ErrorSignal) -> str:
    return (
        f"The diagnostic reported {signal.category.value} from "
        f"{signal.source.value} with confidence {signal.confidence:.2f}."
    )


def _title_for_kind(kind: FixSuggestionKind) -> str:
    titles = {
        FixSuggestionKind.TYPE_ALIGNMENT: "Align the type contract",
        FixSuggestionKind.LINT_FORMATTING: "Apply lint-safe correction",
        FixSuggestionKind.TEST_EXPECTATION_REVIEW: (
            "Review failing test behavior"
        ),
        FixSuggestionKind.IMPORT_PATH_REVIEW: "Review import boundary",
        FixSuggestionKind.SYNTAX_CORRECTION: "Correct syntax first",
        FixSuggestionKind.RUNTIME_GUARD: "Trace runtime exception",
        FixSuggestionKind.TIMEOUT_BOUNDARY: "Repair timeout boundary",
        FixSuggestionKind.CONFIGURATION_REVIEW: "Review configuration",
        FixSuggestionKind.UNKNOWN_REVIEW: "Inspect unknown failure",
    }
    return titles[kind]


def _affected_files_from_signal(signal: ErrorSignal) -> tuple[str, ...]:
    if signal.location is None or signal.location.file_path is None:
        return ()

    return (signal.location.file_path,)


def _validation_command(
    code_context: CodeContextSnapshot | None,
) -> tuple[str, ...] | None:
    if code_context is None or code_context.summary is None:
        return None

    if code_context.summary.test_count > 0:
        return (sys.executable, "-m", "pytest")

    return None


def _bounded_confidence(
    confidence: float,
    risk: FixSuggestionRisk,
) -> float:
    bounded = max(0.0, min(confidence, 1.0))

    if risk == FixSuggestionRisk.NEEDS_HUMAN_REVIEW:
        return min(bounded, 0.85)

    if risk == FixSuggestionRisk.HIGH_RISK_REVIEW:
        return min(bounded, 0.65)

    return bounded