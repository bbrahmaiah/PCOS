from __future__ import annotations

from pathlib import Path

import pytest

from jarvis.developer import (
    CodeContextEngine,
    CodeContextRequest,
    ErrorCategory,
    ErrorIntelligenceEngine,
    ErrorIntelligenceRequest,
    ErrorSourceKind,
    FixActionKind,
    FixSuggestionEngine,
    FixSuggestionKind,
    FixSuggestionRequest,
    FixSuggestionRisk,
    FixSuggestionStatus,
)


def test_fix_suggestion_request_rejects_invalid_max_suggestions() -> None:
    report = ErrorIntelligenceEngine().analyze(
        ErrorIntelligenceRequest(
            stdout="x.py:1: error: bad type [assignment]",
            source=ErrorSourceKind.MYPY,
            exit_code=1,
        )
    )

    with pytest.raises(ValueError):
        FixSuggestionRequest(
            error_report=report,
            max_suggestions=0,
        )


def test_fix_suggestion_returns_no_suggestion_for_clean_report() -> None:
    report = ErrorIntelligenceEngine().analyze(
        ErrorIntelligenceRequest(
            stdout="12 passed in 1.2s",
            source=ErrorSourceKind.PYTEST,
            exit_code=0,
        )
    )

    result = FixSuggestionEngine().suggest(
        FixSuggestionRequest(error_report=report)
    )

    assert result.status == FixSuggestionStatus.NO_SUGGESTION
    assert result.has_suggestions is False


def test_fix_suggestion_for_mypy_type_error(
    tmp_path: Path,
) -> None:
    _write(tmp_path / "tests" / "test_demo.py", "def test_demo(): pass\n")
    _write(tmp_path / "jarvis" / "system.py", "x: str = 1\n")

    context = CodeContextEngine().build_context(
        CodeContextRequest(project_root=tmp_path)
    )
    report = ErrorIntelligenceEngine().analyze(
        ErrorIntelligenceRequest(
            stdout=(
                "jarvis/system.py:1: error: Incompatible types in assignment "
                "[assignment]\n"
            ),
            source=ErrorSourceKind.MYPY,
            exit_code=1,
        )
    )

    result = FixSuggestionEngine().suggest(
        FixSuggestionRequest(
            error_report=report,
            code_context=context,
        )
    )

    assert result.status == FixSuggestionStatus.READY
    assert result.has_suggestions is True

    suggestion = result.suggestions[0]

    assert suggestion.kind == FixSuggestionKind.TYPE_ALIGNMENT
    assert suggestion.risk == FixSuggestionRisk.SAFE_REVIEW
    assert suggestion.validation_command is not None
    assert suggestion.affected_files == ("jarvis/system.py",)
    assert any(
        action.kind == FixActionKind.REVIEW_SIGNATURE
        for action in suggestion.actions
    )


def test_fix_suggestion_for_ruff_lint_error() -> None:
    report = ErrorIntelligenceEngine().analyze(
        ErrorIntelligenceRequest(
            stderr="""E501 Line too long (92 > 88)
   --> scripts\\smoke.py:194:89
""",
            source=ErrorSourceKind.RUFF,
            exit_code=1,
        )
    )

    result = FixSuggestionEngine().suggest(
        FixSuggestionRequest(error_report=report)
    )

    assert result.status == FixSuggestionStatus.READY
    suggestion = result.suggestions[0]

    assert suggestion.kind == FixSuggestionKind.LINT_FORMATTING
    assert suggestion.risk == FixSuggestionRisk.SAFE_REVIEW
    assert any(
        action.kind == FixActionKind.APPLY_FORMATTING
        for action in suggestion.actions
    )


def test_fix_suggestion_for_pytest_failure_needs_review() -> None:
    report = ErrorIntelligenceEngine().analyze(
        ErrorIntelligenceRequest(
            stdout="""________________ test_example ________________

    def test_example():
>       assert 1 == 2
E       assert 1 == 2

tests/test_demo.py:3: AssertionError
""",
            source=ErrorSourceKind.PYTEST,
            exit_code=1,
        )
    )

    result = FixSuggestionEngine().suggest(
        FixSuggestionRequest(error_report=report)
    )

    assert result.status == FixSuggestionStatus.READY
    suggestion = result.suggestions[0]

    assert suggestion.kind == FixSuggestionKind.TEST_EXPECTATION_REVIEW
    assert suggestion.risk == FixSuggestionRisk.NEEDS_HUMAN_REVIEW
    assert suggestion.confidence <= 0.85
    assert any(
        action.kind == FixActionKind.REVIEW_ASSERTION
        for action in suggestion.actions
    )


def test_fix_suggestion_for_import_error() -> None:
    report = ErrorIntelligenceEngine().analyze(
        ErrorIntelligenceRequest(
            stderr="""Traceback (most recent call last):
  File "main.py", line 1, in <module>
    import missing_module
ModuleNotFoundError: No module named 'missing_module'
""",
            source=ErrorSourceKind.PYTHON_TRACEBACK,
            exit_code=1,
        )
    )

    result = FixSuggestionEngine().suggest(
        FixSuggestionRequest(error_report=report)
    )

    assert result.status == FixSuggestionStatus.READY
    suggestion = result.suggestions[0]

    assert suggestion.kind == FixSuggestionKind.IMPORT_PATH_REVIEW
    assert suggestion.risk == FixSuggestionRisk.NEEDS_HUMAN_REVIEW
    assert any(
        action.kind == FixActionKind.REVIEW_IMPORT
        for action in suggestion.actions
    )


def test_fix_suggestion_does_not_mutate_files(
    tmp_path: Path,
) -> None:
    target = tmp_path / "jarvis" / "system.py"
    original = "x: str = 1\n"
    _write(target, original)

    context = CodeContextEngine().build_context(
        CodeContextRequest(project_root=tmp_path)
    )
    report = ErrorIntelligenceEngine().analyze(
        ErrorIntelligenceRequest(
            stdout=(
                "jarvis/system.py:1: error: Incompatible types in assignment "
                "[assignment]\n"
            ),
            source=ErrorSourceKind.MYPY,
            exit_code=1,
        )
    )

    result = FixSuggestionEngine().suggest(
        FixSuggestionRequest(
            error_report=report,
            code_context=context,
        )
    )

    assert result.status == FixSuggestionStatus.READY
    assert target.read_text(encoding="utf-8") == original


def test_fix_suggestion_respects_max_suggestions() -> None:
    report = ErrorIntelligenceEngine().analyze(
        ErrorIntelligenceRequest(
            stdout=(
                "a.py:1: error: bad type [assignment]\n"
                "b.py:2: error: bad type [assignment]\n"
            ),
            source=ErrorSourceKind.MYPY,
            exit_code=1,
        )
    )

    result = FixSuggestionEngine().suggest(
        FixSuggestionRequest(
            error_report=report,
            max_suggestions=1,
        )
    )

    assert len(result.suggestions) == 1


def test_fix_suggestion_unknown_partial_report() -> None:
    report = ErrorIntelligenceEngine().analyze(
        ErrorIntelligenceRequest(
            stdout="Something went wrong without known parser format.",
            source=ErrorSourceKind.UNKNOWN,
            exit_code=1,
        )
    )

    result = FixSuggestionEngine().suggest(
        FixSuggestionRequest(error_report=report)
    )

    assert result.status == FixSuggestionStatus.READY
    assert result.suggestions[0].kind == FixSuggestionKind.UNKNOWN_REVIEW
    assert result.suggestions[0].metadata["diagnosis_title"]


def test_fix_suggestion_enum_values_are_stable() -> None:
    assert FixSuggestionStatus.READY.value == "ready"
    assert FixSuggestionKind.TYPE_ALIGNMENT.value == "type_alignment"
    assert ErrorCategory.TYPE_ERROR.value == "type_error"


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")