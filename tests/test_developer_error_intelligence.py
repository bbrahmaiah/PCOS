from __future__ import annotations

import pytest

from jarvis.developer import (
    ErrorCategory,
    ErrorIntelligenceEngine,
    ErrorIntelligenceRequest,
    ErrorIntelligenceStatus,
    ErrorSeverity,
    ErrorSourceKind,
)


def test_error_intelligence_request_requires_output() -> None:
    with pytest.raises(ValueError):
        ErrorIntelligenceRequest()


def test_error_intelligence_parses_mypy_error() -> None:
    output = (
        'jarvis/system/assembly.py:214: error: Missing positional arguments '
        '"memory_write", "wrote_memory" in call to "JarvisSystemResponse" '
        '[call-arg]\n'
    )

    report = ErrorIntelligenceEngine().analyze(
        ErrorIntelligenceRequest(
            stdout=output,
            source=ErrorSourceKind.MYPY,
            exit_code=1,
        )
    )

    assert report.status == ErrorIntelligenceStatus.ANALYZED
    assert report.has_errors is True
    assert len(report.signals) == 1
    assert report.signals[0].category == ErrorCategory.TYPE_ERROR
    assert report.signals[0].location is not None
    assert report.signals[0].location.file_path == "jarvis/system/assembly.py"
    assert report.signals[0].location.line == 214
    assert report.diagnosis is not None
    assert report.diagnosis.primary_category == ErrorCategory.TYPE_ERROR
    assert report.diagnosis.severity == ErrorSeverity.MEDIUM


def test_error_intelligence_parses_ruff_error() -> None:
    output = """E501 Line too long (92 > 88)
   --> scripts\\smoke_environment_cognition.py:194:89
    |
192 |             bool(results)
"""

    report = ErrorIntelligenceEngine().analyze(
        ErrorIntelligenceRequest(
            stderr=output,
            source=ErrorSourceKind.RUFF,
            exit_code=1,
        )
    )

    assert report.status == ErrorIntelligenceStatus.ANALYZED
    assert len(report.signals) == 1
    signal = report.signals[0]

    assert signal.category == ErrorCategory.LINT_ERROR
    assert signal.location is not None
    assert signal.location.file_path == (
        "scripts\\smoke_environment_cognition.py"
    )
    assert signal.location.line == 194
    assert signal.location.column == 89
    assert report.diagnosis is not None
    assert "Lint error" in report.diagnosis.title


def test_error_intelligence_parses_python_traceback() -> None:
    output = """Traceback (most recent call last):
  File "main.py", line 10, in <module>
    run()
  File "main.py", line 5, in run
    raise RuntimeError("boom")
RuntimeError: boom
"""

    report = ErrorIntelligenceEngine().analyze(
        ErrorIntelligenceRequest(
            stderr=output,
            source=ErrorSourceKind.PYTHON_TRACEBACK,
            exit_code=1,
        )
    )

    assert report.status == ErrorIntelligenceStatus.ANALYZED
    assert len(report.signals) == 1
    signal = report.signals[0]

    assert signal.category == ErrorCategory.RUNTIME_ERROR
    assert signal.severity == ErrorSeverity.HIGH
    assert signal.location is not None
    assert signal.location.file_path == "main.py"
    assert signal.location.line == 5
    assert signal.location.symbol == "run"


def test_error_intelligence_parses_pytest_failure() -> None:
    output = """________________ test_example ________________

    def test_example():
>       assert 1 == 2
E       assert 1 == 2

tests/test_demo.py:3: AssertionError
=========================== short test summary info ===========================
FAILED tests/test_demo.py::test_example - AssertionError
"""

    report = ErrorIntelligenceEngine().analyze(
        ErrorIntelligenceRequest(
            stdout=output,
            source=ErrorSourceKind.PYTEST,
            exit_code=1,
        )
    )

    assert report.status == ErrorIntelligenceStatus.ANALYZED
    assert report.has_errors is True
    assert report.diagnosis is not None
    assert report.diagnosis.primary_category in {
        ErrorCategory.TEST_FAILURE,
        ErrorCategory.ASSERTION_ERROR,
    }
    assert "tests/test_demo.py" in report.diagnosis.affected_files


def test_error_intelligence_handles_timeout_terminal_output() -> None:
    report = ErrorIntelligenceEngine().analyze(
        ErrorIntelligenceRequest(
            stderr="Test run timed out before completion.",
            source=ErrorSourceKind.TERMINAL,
            exit_code=None,
        )
    )

    assert report.status == ErrorIntelligenceStatus.ANALYZED
    assert len(report.signals) == 1
    assert report.signals[0].category == ErrorCategory.TIMEOUT


def test_error_intelligence_no_error_when_exit_zero() -> None:
    report = ErrorIntelligenceEngine().analyze(
        ErrorIntelligenceRequest(
            stdout="12 passed in 1.2s",
            source=ErrorSourceKind.PYTEST,
            exit_code=0,
        )
    )

    assert report.status == ErrorIntelligenceStatus.NO_ERROR
    assert report.has_errors is False
    assert report.diagnosis is None


def test_error_intelligence_partial_when_unknown_failing_output() -> None:
    report = ErrorIntelligenceEngine().analyze(
        ErrorIntelligenceRequest(
            stdout="Something went wrong but no parser knows this format.",
            source=ErrorSourceKind.UNKNOWN,
            exit_code=1,
        )
    )

    assert report.status == ErrorIntelligenceStatus.PARTIAL
    assert report.diagnosis is not None
    assert report.diagnosis.primary_category == ErrorCategory.UNKNOWN


def test_error_signal_validates_confidence() -> None:
    from jarvis.developer import ErrorLocation, ErrorSignal

    with pytest.raises(ValueError):
        ErrorSignal(
            category=ErrorCategory.UNKNOWN,
            severity=ErrorSeverity.LOW,
            source=ErrorSourceKind.UNKNOWN,
            message="bad",
            location=ErrorLocation(
                file_path=None,
                line=None,
                column=None,
            ),
            evidence="evidence",
            confidence=2.0,
        )


def test_error_intelligence_enum_values_are_stable() -> None:
    assert ErrorIntelligenceStatus.ANALYZED.value == "analyzed"
    assert ErrorSourceKind.MYPY.value == "mypy"
    assert ErrorCategory.TYPE_ERROR.value == "type_error"