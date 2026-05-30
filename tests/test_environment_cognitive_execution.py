from __future__ import annotations

import pytest
from pydantic import ValidationError

from jarvis.environment import (
    AppControlDecision,
    AppControlReason,
    AppControlResult,
    AppControlStatus,
    CognitiveEnvironmentKind,
    CognitiveExecutionCapability,
    CognitiveExecutionDecision,
    CognitiveExecutionReason,
    CognitiveExecutionRequest,
    CognitiveExecutionRisk,
    CognitiveExecutionRuntime,
    CognitiveExecutionStatus,
)


def test_runtime_rejects_empty_name() -> None:
    with pytest.raises(ValueError):
        CognitiveExecutionRuntime(name=" ")


def test_request_rejects_bad_line_number() -> None:
    with pytest.raises(ValidationError):
        CognitiveExecutionRequest(
            session_id="session",
            workspace_id="workspace",
            environment=CognitiveEnvironmentKind.IDE,
            capability=CognitiveExecutionCapability.JUMP_TO_LINE,
            instruction="jump",
            app_control=_app_control(),
            line_number=0,
        )


def test_create_session() -> None:
    runtime = CognitiveExecutionRuntime()
    session = runtime.create_session(workspace_id="workspace")

    assert session.workspace_id == "workspace"
    assert runtime.snapshot().session_count == 1


def test_blocks_without_app_control_eligibility() -> None:
    runtime = CognitiveExecutionRuntime()
    session = runtime.create_session(workspace_id="workspace")

    result = runtime.execute(
        CognitiveExecutionRequest(
            session_id=session.session_id,
            workspace_id="workspace",
            environment=CognitiveEnvironmentKind.IDE,
            capability=CognitiveExecutionCapability.OPEN_FILE,
            instruction="open file",
            app_control=_app_control(eligible=False),
            target_path="main.py",
        )
    )

    assert result.status == CognitiveExecutionStatus.NEEDS_APP_CONTROL
    assert result.decision == CognitiveExecutionDecision.REQUIRE_APP_CONTROL
    assert result.reason == CognitiveExecutionReason.APP_CONTROL_NOT_ELIGIBLE


def test_ide_open_file_executes() -> None:
    runtime = CognitiveExecutionRuntime()
    session = runtime.create_session(workspace_id="workspace")

    result = runtime.execute(
        CognitiveExecutionRequest(
            session_id=session.session_id,
            workspace_id="workspace",
            environment=CognitiveEnvironmentKind.IDE,
            capability=CognitiveExecutionCapability.OPEN_FILE,
            instruction="open main.py",
            app_control=_app_control(),
            target_path="main.py",
        )
    )

    assert result.status == CognitiveExecutionStatus.EXECUTED
    assert result.safe_for_followup_action is True
    assert result.output is not None
    assert "main.py" in result.output.summary


def test_ide_jump_to_line_executes() -> None:
    runtime = CognitiveExecutionRuntime()
    session = runtime.create_session(workspace_id="workspace")

    result = runtime.execute(
        CognitiveExecutionRequest(
            session_id=session.session_id,
            workspace_id="workspace",
            environment=CognitiveEnvironmentKind.IDE,
            capability=CognitiveExecutionCapability.JUMP_TO_LINE,
            instruction="jump to line 47",
            app_control=_app_control(),
            target_path="main.py",
            line_number=47,
        )
    )

    assert result.status == CognitiveExecutionStatus.EXECUTED
    assert result.plan is not None
    assert result.plan.risk == CognitiveExecutionRisk.LOW


def test_ide_run_tests_requires_verification() -> None:
    runtime = CognitiveExecutionRuntime()
    session = runtime.create_session(workspace_id="workspace")

    result = runtime.execute(
        CognitiveExecutionRequest(
            session_id=session.session_id,
            workspace_id="workspace",
            environment=CognitiveEnvironmentKind.IDE,
            capability=CognitiveExecutionCapability.RUN_TESTS,
            instruction="run tests",
            app_control=_app_control(),
        )
    )

    assert result.status == CognitiveExecutionStatus.NEEDS_VERIFICATION
    assert result.decision == CognitiveExecutionDecision.REQUIRE_VERIFICATION
    assert result.safe_for_followup_action is False


def test_browser_extract_article_executes() -> None:
    runtime = CognitiveExecutionRuntime()
    session = runtime.create_session(workspace_id="workspace")

    result = runtime.execute(
        CognitiveExecutionRequest(
            session_id=session.session_id,
            workspace_id="workspace",
            environment=CognitiveEnvironmentKind.BROWSER,
            capability=CognitiveExecutionCapability.EXTRACT_ARTICLE,
            instruction="extract article",
            app_control=_app_control(),
            url="https://example.test/article",
        )
    )

    assert result.status == CognitiveExecutionStatus.EXECUTED
    assert result.output is not None
    assert result.output.extracted_text is not None


def test_browser_compare_sources_executes() -> None:
    runtime = CognitiveExecutionRuntime()
    session = runtime.create_session(workspace_id="workspace")

    result = runtime.execute(
        CognitiveExecutionRequest(
            session_id=session.session_id,
            workspace_id="workspace",
            environment=CognitiveEnvironmentKind.BROWSER,
            capability=CognitiveExecutionCapability.COMPARE_SOURCES,
            instruction="compare sources",
            app_control=_app_control(),
            sources=("source-a", "source-b"),
        )
    )

    assert result.status == CognitiveExecutionStatus.EXECUTED
    assert result.output is not None
    assert result.output.sources == ("source-a", "source-b")


def test_browser_navigation_requires_verification() -> None:
    runtime = CognitiveExecutionRuntime()
    session = runtime.create_session(workspace_id="workspace")

    result = runtime.execute(
        CognitiveExecutionRequest(
            session_id=session.session_id,
            workspace_id="workspace",
            environment=CognitiveEnvironmentKind.BROWSER,
            capability=CognitiveExecutionCapability.NAVIGATE_BROWSER,
            instruction="open docs",
            app_control=_app_control(),
            url="https://example.test",
        )
    )

    assert result.status == CognitiveExecutionStatus.NEEDS_VERIFICATION


def test_terminal_parse_output_executes() -> None:
    runtime = CognitiveExecutionRuntime()
    session = runtime.create_session(workspace_id="workspace")

    result = runtime.execute(
        CognitiveExecutionRequest(
            session_id=session.session_id,
            workspace_id="workspace",
            environment=CognitiveEnvironmentKind.TERMINAL,
            capability=CognitiveExecutionCapability.PARSE_TERMINAL_OUTPUT,
            instruction="parse output",
            app_control=_app_control(),
            text="2 failed, 10 passed",
        )
    )

    assert result.status == CognitiveExecutionStatus.EXECUTED
    assert result.output is not None
    assert result.output.diagnostics == ("terminal output parsed",)


def test_terminal_run_tests_requires_verification() -> None:
    runtime = CognitiveExecutionRuntime()
    session = runtime.create_session(workspace_id="workspace")

    result = runtime.execute(
        CognitiveExecutionRequest(
            session_id=session.session_id,
            workspace_id="workspace",
            environment=CognitiveEnvironmentKind.TERMINAL,
            capability=CognitiveExecutionCapability.RUN_TESTS,
            instruction="run pytest",
            app_control=_app_control(),
            command="pytest",
        )
    )

    assert result.status == CognitiveExecutionStatus.NEEDS_VERIFICATION


def test_document_edit_requires_user_initiated() -> None:
    runtime = CognitiveExecutionRuntime()
    session = runtime.create_session(workspace_id="workspace")

    result = runtime.execute(
        CognitiveExecutionRequest(
            session_id=session.session_id,
            workspace_id="workspace",
            environment=CognitiveEnvironmentKind.DOCUMENT,
            capability=CognitiveExecutionCapability.EDIT_DOCUMENT,
            instruction="edit document",
            app_control=_app_control(),
            target_path="notes.md",
            user_initiated=False,
        )
    )

    assert result.status == CognitiveExecutionStatus.BLOCKED
    assert result.reason == CognitiveExecutionReason.UNSAFE_DOCUMENT_EDIT


def test_document_edit_requires_verification() -> None:
    runtime = CognitiveExecutionRuntime()
    session = runtime.create_session(workspace_id="workspace")

    result = runtime.execute(
        CognitiveExecutionRequest(
            session_id=session.session_id,
            workspace_id="workspace",
            environment=CognitiveEnvironmentKind.DOCUMENT,
            capability=CognitiveExecutionCapability.EDIT_DOCUMENT,
            instruction="edit document",
            app_control=_app_control(),
            target_path="notes.md",
            user_initiated=True,
        )
    )

    assert result.status == CognitiveExecutionStatus.NEEDS_VERIFICATION
    assert result.output is not None
    assert result.output.changed_paths == ("notes.md",)


def test_file_explorer_list_files_executes() -> None:
    runtime = CognitiveExecutionRuntime()
    session = runtime.create_session(workspace_id="workspace")

    result = runtime.execute(
        CognitiveExecutionRequest(
            session_id=session.session_id,
            workspace_id="workspace",
            environment=CognitiveEnvironmentKind.FILE_EXPLORER,
            capability=CognitiveExecutionCapability.LIST_FILES,
            instruction="list files",
            app_control=_app_control(),
        )
    )

    assert result.status == CognitiveExecutionStatus.EXECUTED


def test_wrong_environment_blocks() -> None:
    runtime = CognitiveExecutionRuntime()
    session = runtime.create_session(workspace_id="workspace")

    result = runtime.execute(
        CognitiveExecutionRequest(
            session_id=session.session_id,
            workspace_id="workspace",
            environment=CognitiveEnvironmentKind.BROWSER,
            capability=CognitiveExecutionCapability.OPEN_FILE,
            instruction="open file in browser",
            app_control=_app_control(),
            target_path="main.py",
        )
    )

    assert result.status == CognitiveExecutionStatus.BLOCKED
    assert result.reason == CognitiveExecutionReason.UNKNOWN_CAPABILITY


def test_missing_session_fails() -> None:
    runtime = CognitiveExecutionRuntime()

    result = runtime.execute(
        CognitiveExecutionRequest(
            session_id="missing",
            workspace_id="workspace",
            environment=CognitiveEnvironmentKind.IDE,
            capability=CognitiveExecutionCapability.OPEN_FILE,
            instruction="open main",
            app_control=_app_control(),
            target_path="main.py",
        )
    )

    assert result.status == CognitiveExecutionStatus.FAILED
    assert result.reason == CognitiveExecutionReason.SESSION_NOT_FOUND


def test_snapshot_tracks_counts() -> None:
    runtime = CognitiveExecutionRuntime()
    session = runtime.create_session(workspace_id="workspace")

    runtime.execute(
        CognitiveExecutionRequest(
            session_id=session.session_id,
            workspace_id="workspace",
            environment=CognitiveEnvironmentKind.IDE,
            capability=CognitiveExecutionCapability.OPEN_FILE,
            instruction="open main",
            app_control=_app_control(),
            target_path="main.py",
        )
    )
    runtime.execute(
        CognitiveExecutionRequest(
            session_id=session.session_id,
            workspace_id="workspace",
            environment=CognitiveEnvironmentKind.TERMINAL,
            capability=CognitiveExecutionCapability.RUN_TESTS,
            instruction="run tests",
            app_control=_app_control(),
        )
    )
    snapshot = runtime.snapshot()

    assert snapshot.session_count == 1
    assert snapshot.result_count == 2
    assert snapshot.executed_count == 1
    assert snapshot.verification_count == 1
    assert snapshot.audit_count == 2


def test_session_tracks_last_environment_and_execution_count() -> None:
    runtime = CognitiveExecutionRuntime()
    session = runtime.create_session(workspace_id="workspace")

    runtime.execute(
        CognitiveExecutionRequest(
            session_id=session.session_id,
            workspace_id="workspace",
            environment=CognitiveEnvironmentKind.IDE,
            capability=CognitiveExecutionCapability.OPEN_FILE,
            instruction="open main",
            app_control=_app_control(),
            target_path="main.py",
        )
    )
    stored = runtime.session_for(session.session_id)

    assert stored is not None
    assert stored.last_environment == CognitiveEnvironmentKind.IDE
    assert stored.execution_count == 1


def test_reset_clears_runtime() -> None:
    runtime = CognitiveExecutionRuntime()
    runtime.create_session(workspace_id="workspace")

    runtime.reset()
    snapshot = runtime.snapshot()

    assert snapshot.session_count == 0
    assert snapshot.runtime_event_count == 1
    assert snapshot.last_reason == CognitiveExecutionReason.RUNTIME_RESET


def test_enum_values_are_stable() -> None:
    assert CognitiveEnvironmentKind.IDE.value == "ide"
    assert CognitiveExecutionCapability.RUN_TESTS.value == "run_tests"
    assert CognitiveExecutionStatus.EXECUTED.value == "executed"


def _app_control(*, eligible: bool = True) -> AppControlResult:
    return AppControlResult.model_construct(
        result_id="app_control_result_test",
        status=(
            AppControlStatus.READY
            if eligible
            else AppControlStatus.BLOCKED
        ),
        decision=(
            AppControlDecision.ALLOW
            if eligible
            else AppControlDecision.BLOCK
        ),
        reason=(
            AppControlReason.APP_RESPONSIVE_VERIFIED
            if eligible
            else AppControlReason.POLICY_NOT_ELIGIBLE
        ),
        control_eligible=eligible,
        message=(
            "test app control eligible"
            if eligible
            else "test app control blocked"
        ),
    )