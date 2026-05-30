from __future__ import annotations

import pytest
from pydantic import ValidationError

from jarvis.environment import (
    AppIdentityConfidenceLevel,
    AppIdentityObservation,
    AppIdentityReason,
    AppIdentityRuntime,
    AppIdentityStatus,
    AppInteractionReadiness,
    AppProfile,
    AppResponsiveness,
    AppRuntimeState,
    DetectedAppKind,
    ProcessState,
    ProcessStatus,
    SpoofRiskLevel,
    default_app_profiles,
)


def test_runtime_rejects_empty_name() -> None:
    with pytest.raises(ValueError):
        AppIdentityRuntime(name=" ")


def test_profile_requires_identity_hint() -> None:
    with pytest.raises(ValidationError):
        AppProfile(
            canonical_name="Unknown",
            kind=DetectedAppKind.UNKNOWN,
        )


def test_create_session() -> None:
    runtime = AppIdentityRuntime()
    session = runtime.create_session(workspace_id="workspace")

    assert session.workspace_id == "workspace"
    assert runtime.snapshot().session_count == 1


def test_register_default_profiles() -> None:
    runtime = AppIdentityRuntime()

    runtime.register_default_profiles()

    assert runtime.snapshot().profile_count >= 3


def test_identifies_vscode_from_profile() -> None:
    runtime = AppIdentityRuntime()
    runtime.register_default_profiles()
    session = runtime.create_session(workspace_id="workspace")

    result = runtime.identify_app(
        session_id=session.session_id,
        observation=AppIdentityObservation(
            app_id="app-vscode",
            process_name="code.exe",
            window_title="JARVIS_OS - Visual Studio Code",
            publisher="Microsoft Corporation",
        ),
    )

    assert result.status == AppIdentityStatus.IDENTIFIED
    assert result.reason == AppIdentityReason.APP_IDENTIFIED
    assert result.classification.kind == DetectedAppKind.IDE
    assert result.confidence_level == AppIdentityConfidenceLevel.VERIFIED
    assert result.verified is True


def test_unknown_app_is_blocked() -> None:
    runtime = AppIdentityRuntime()
    runtime.register_default_profiles()
    session = runtime.create_session(workspace_id="workspace")

    result = runtime.identify_app(
        session_id=session.session_id,
        observation=AppIdentityObservation(
            app_id="mystery",
            process_name="strange.exe",
            window_title="Unknown App",
        ),
    )

    assert result.status == AppIdentityStatus.UNKNOWN
    assert result.reason == AppIdentityReason.UNKNOWN_APP_BLOCKED
    assert result.verified is False
    assert result.spoof_hints


def test_spoof_publisher_hint_blocks_verified_identity() -> None:
    runtime = AppIdentityRuntime()
    runtime.register_default_profiles()
    session = runtime.create_session(workspace_id="workspace")

    result = runtime.identify_app(
        session_id=session.session_id,
        observation=AppIdentityObservation(
            app_id="app-vscode",
            process_name="code.exe",
            window_title="Visual Studio Code",
            publisher="Unknown Publisher",
        ),
    )

    assert any(hint.risk == SpoofRiskLevel.HIGH for hint in result.spoof_hints)
    assert result.verified is False


def test_temp_path_spoof_hint() -> None:
    runtime = AppIdentityRuntime()
    runtime.register_default_profiles()
    session = runtime.create_session(workspace_id="workspace")

    result = runtime.identify_app(
        session_id=session.session_id,
        observation=AppIdentityObservation(
            app_id="app-vscode",
            process_name="code.exe",
            window_title="Visual Studio Code",
            publisher="Microsoft Corporation",
            executable_path=r"C:\Temp\code.exe",
        ),
    )

    assert any(
        hint.reason == "app executable launched from temporary path"
        for hint in result.spoof_hints
    )


def test_modal_state_model_requires_verify_first() -> None:
    runtime = AppIdentityRuntime()
    runtime.register_default_profiles()
    session = runtime.create_session(workspace_id="workspace")

    model = runtime.build_state_model(
        session_id=session.session_id,
        observation=AppIdentityObservation(
            app_id="app-vscode",
            process_name="code.exe",
            window_title="Visual Studio Code",
            publisher="Microsoft Corporation",
            modal_title="Save changes?",
        ),
    )

    assert model.runtime_state == AppRuntimeState.MODAL_BLOCKED
    assert model.modal is not None
    assert model.interaction_readiness == AppInteractionReadiness.VERIFY_FIRST


def test_unresponsive_process_blocks_interaction() -> None:
    runtime = AppIdentityRuntime()
    runtime.register_default_profiles()
    session = runtime.create_session(workspace_id="workspace")

    model = runtime.build_state_model(
        session_id=session.session_id,
        observation=AppIdentityObservation(
            app_id="app-vscode",
            process_name="code.exe",
            window_title="Visual Studio Code",
            publisher="Microsoft Corporation",
        ),
        process=ProcessState(
            process_id="process",
            app_id="app-vscode",
            name="code.exe",
            status=ProcessStatus.UNRESPONSIVE,
        ),
    )

    assert model.responsiveness == AppResponsiveness.UNRESPONSIVE
    assert model.runtime_state == AppRuntimeState.UNRESPONSIVE
    assert model.interaction_readiness == AppInteractionReadiness.BLOCKED


def test_crashed_process_blocks_interaction() -> None:
    runtime = AppIdentityRuntime()
    runtime.register_default_profiles()
    session = runtime.create_session(workspace_id="workspace")

    model = runtime.build_state_model(
        session_id=session.session_id,
        observation=AppIdentityObservation(
            app_id="app-vscode",
            process_name="code.exe",
            window_title="Visual Studio Code",
            publisher="Microsoft Corporation",
        ),
        process=ProcessState(
            process_id="process",
            app_id="app-vscode",
            name="code.exe",
            status=ProcessStatus.CRASHED,
        ),
    )

    assert model.runtime_state == AppRuntimeState.CRASHED
    assert model.interaction_readiness == AppInteractionReadiness.BLOCKED


def test_loading_state_detected() -> None:
    runtime = AppIdentityRuntime()
    runtime.register_default_profiles()
    session = runtime.create_session(workspace_id="workspace")

    model = runtime.build_state_model(
        session_id=session.session_id,
        observation=AppIdentityObservation(
            app_id="browser",
            process_name="chrome.exe",
            window_title="Chrome",
            publisher="Google LLC",
            loading=True,
        ),
    )

    assert model.runtime_state == AppRuntimeState.LOADING
    assert model.loading is True


def test_browser_profile_verify_first_not_ready() -> None:
    runtime = AppIdentityRuntime()
    runtime.register_default_profiles()
    session = runtime.create_session(workspace_id="workspace")

    model = runtime.build_state_model(
        session_id=session.session_id,
        observation=AppIdentityObservation(
            app_id="browser",
            process_name="chrome.exe",
            window_title="Chrome",
            publisher="Google LLC",
        ),
    )

    assert model.identity.classification.kind == DetectedAppKind.BROWSER
    assert model.interaction_readiness == AppInteractionReadiness.VERIFY_FIRST


def test_missing_session_fails_identity() -> None:
    runtime = AppIdentityRuntime()

    result = runtime.identify_app(
        session_id="missing",
        observation=AppIdentityObservation(
            app_id="app",
            process_name="code.exe",
        ),
    )

    assert result.status == AppIdentityStatus.FAILED
    assert result.reason == AppIdentityReason.SESSION_NOT_FOUND


def test_snapshot_tracks_counts() -> None:
    runtime = AppIdentityRuntime()
    runtime.register_default_profiles()
    session = runtime.create_session(workspace_id="workspace")

    runtime.build_state_model(
        session_id=session.session_id,
        observation=AppIdentityObservation(
            app_id="app-vscode",
            process_name="code.exe",
            window_title="Visual Studio Code",
            publisher="Microsoft Corporation",
        ),
    )
    runtime.build_state_model(
        session_id=session.session_id,
        observation=AppIdentityObservation(
            app_id="unknown",
            process_name="unknown.exe",
            window_title="Unknown",
        ),
    )
    snapshot = runtime.snapshot()

    assert snapshot.session_count == 1
    assert snapshot.profile_count >= 3
    assert snapshot.identity_result_count == 2
    assert snapshot.state_model_count == 2
    assert snapshot.unknown_count == 1
    assert snapshot.blocked_count >= 1


def test_reset_clears_runtime() -> None:
    runtime = AppIdentityRuntime()
    runtime.create_session(workspace_id="workspace")

    runtime.reset()
    snapshot = runtime.snapshot()

    assert snapshot.session_count == 0
    assert snapshot.runtime_event_count == 1
    assert snapshot.last_reason == AppIdentityReason.RUNTIME_RESET


def test_default_profiles_are_available() -> None:
    profiles = default_app_profiles()

    assert any(profile.kind == DetectedAppKind.IDE for profile in profiles)
    assert any(profile.kind == DetectedAppKind.BROWSER for profile in profiles)
    assert any(profile.kind == DetectedAppKind.TERMINAL for profile in profiles)


def test_enum_values_are_stable() -> None:
    assert DetectedAppKind.IDE.value == "ide"
    assert AppRuntimeState.CRASHED.value == "crashed"
    assert AppInteractionReadiness.BLOCKED.value == "blocked"
    assert SpoofRiskLevel.HIGH.value == "high"