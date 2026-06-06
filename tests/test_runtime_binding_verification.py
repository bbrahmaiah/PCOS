from __future__ import annotations

import sys
from pathlib import Path

from jarvis.runtime import (
    JarvisOrganKind,
    JarvisRuntimeBindingCheckKind,
    JarvisRuntimeBindingVerificationMode,
    JarvisRuntimeBindingVerificationStatus,
    JarvisRuntimeBindingVerifier,
    JarvisRuntimeBindingVerifierConfig,
    summarize_binding_report,
)


def _write_module(tmp_path: Path) -> str:
    package = tmp_path / "binding_test_package"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "factories.py").write_text(
        "\n".join(
            (
                "class Runtime:",
                "    def __init__(self):",
                "        self.started = False",
                "    def start(self):",
                "        self.started = True",
                "    def stop(self):",
                "        self.started = False",
                "    def health(self):",
                "        return object()",
                "",
                "def create_runtime():",
                "    return Runtime()",
                "",
                "def create_runtime_with_required_arg(value):",
                "    return Runtime()",
            )
        ),
        encoding="utf-8",
    )
    sys.path.insert(0, str(tmp_path))
    return "binding_test_package.factories:create_runtime"


def _write_bindings(
    path: Path,
    *,
    import_path: str,
    include_all: bool = True,
) -> None:
    phases = (
        (
            JarvisOrganKind.PHASE1_KERNEL,
            JarvisOrganKind.PHASE1_EVENTS,
            JarvisOrganKind.PHASE1_OBSERVABILITY,
        )
        if include_all
        else (JarvisOrganKind.PHASE1_KERNEL,)
    )

    path.write_text(
        "\n".join(f"{phase.value}={import_path}" for phase in phases),
        encoding="utf-8",
    )


def test_runtime_binding_verifier_fails_when_file_missing(tmp_path: Path) -> None:
    verifier = JarvisRuntimeBindingVerifier(
        config=JarvisRuntimeBindingVerifierConfig(
            bindings_path=tmp_path / "missing.env",
            required_phases=(JarvisOrganKind.PHASE1_KERNEL,),
        )
    )

    report = verifier.verify()

    assert report.status == JarvisRuntimeBindingVerificationStatus.FAILED
    assert report.failed_count == 1
    assert report.checks[0].kind == JarvisRuntimeBindingCheckKind.BINDING_FILE_EXISTS


def test_runtime_binding_verifier_passes_resolve_only(tmp_path: Path) -> None:
    import_path = _write_module(tmp_path)
    bindings = tmp_path / "runtime_bindings.env"
    _write_bindings(path=bindings, import_path=import_path)

    verifier = JarvisRuntimeBindingVerifier(
        config=JarvisRuntimeBindingVerifierConfig(
            bindings_path=bindings,
            required_phases=(
                JarvisOrganKind.PHASE1_KERNEL,
                JarvisOrganKind.PHASE1_EVENTS,
                JarvisOrganKind.PHASE1_OBSERVABILITY,
            ),
        )
    )

    report = verifier.verify()

    assert report.status == JarvisRuntimeBindingVerificationStatus.PASSED
    assert report.failed_count == 0
    assert any(
        check.kind == JarvisRuntimeBindingCheckKind.MODULE_IMPORTABLE
        for check in report.checks
    )


def test_runtime_binding_verifier_fails_missing_required_phase(
    tmp_path: Path,
) -> None:
    import_path = _write_module(tmp_path)
    bindings = tmp_path / "runtime_bindings.env"
    _write_bindings(path=bindings, import_path=import_path, include_all=False)

    verifier = JarvisRuntimeBindingVerifier(
        config=JarvisRuntimeBindingVerifierConfig(
            bindings_path=bindings,
            required_phases=(
                JarvisOrganKind.PHASE1_KERNEL,
                JarvisOrganKind.PHASE1_EVENTS,
            ),
        )
    )

    report = verifier.verify()

    assert report.status == JarvisRuntimeBindingVerificationStatus.FAILED
    assert any(
        check.kind == JarvisRuntimeBindingCheckKind.REQUIRED_PHASE_PRESENT
        and not check.passed
        and check.phase == JarvisOrganKind.PHASE1_EVENTS
        for check in report.checks
    )


def test_runtime_binding_verifier_fails_bad_import_format(
    tmp_path: Path,
) -> None:
    bindings = tmp_path / "runtime_bindings.env"
    _write_bindings(path=bindings, import_path="bad.import.path")

    verifier = JarvisRuntimeBindingVerifier(
        config=JarvisRuntimeBindingVerifierConfig(
            bindings_path=bindings,
            required_phases=(JarvisOrganKind.PHASE1_KERNEL,),
        )
    )

    report = verifier.verify()

    assert report.status == JarvisRuntimeBindingVerificationStatus.FAILED
    assert any(
        check.kind == JarvisRuntimeBindingCheckKind.IMPORT_PATH_FORMAT
        and not check.passed
        for check in report.checks
    )


def test_runtime_binding_verifier_fails_missing_factory(tmp_path: Path) -> None:
    _write_module(tmp_path)
    bindings = tmp_path / "runtime_bindings.env"
    _write_bindings(
        path=bindings,
        import_path="binding_test_package.factories:missing_factory",
    )

    verifier = JarvisRuntimeBindingVerifier(
        config=JarvisRuntimeBindingVerifierConfig(
            bindings_path=bindings,
            required_phases=(JarvisOrganKind.PHASE1_KERNEL,),
        )
    )

    report = verifier.verify()

    assert report.status == JarvisRuntimeBindingVerificationStatus.FAILED
    assert any(
        check.kind == JarvisRuntimeBindingCheckKind.FACTORY_CALLABLE
        and not check.passed
        for check in report.checks
    )


def test_runtime_binding_verifier_checks_factory_signature(
    tmp_path: Path,
) -> None:
    _write_module(tmp_path)
    bindings = tmp_path / "runtime_bindings.env"
    _write_bindings(
        path=bindings,
        import_path=(
            "binding_test_package.factories:"
            "create_runtime_with_required_arg"
        ),
    )

    verifier = JarvisRuntimeBindingVerifier(
        config=JarvisRuntimeBindingVerifierConfig(
            bindings_path=bindings,
            required_phases=(JarvisOrganKind.PHASE1_KERNEL,),
        )
    )

    report = verifier.verify()

    assert report.status == JarvisRuntimeBindingVerificationStatus.FAILED
    assert any(
        check.kind == JarvisRuntimeBindingCheckKind.FACTORY_SIGNATURE_SAFE
        and not check.passed
        for check in report.checks
    )


def test_runtime_binding_verifier_factory_dry_run(tmp_path: Path) -> None:
    import_path = _write_module(tmp_path)
    bindings = tmp_path / "runtime_bindings.env"
    _write_bindings(path=bindings, import_path=import_path)

    verifier = JarvisRuntimeBindingVerifier(
        config=JarvisRuntimeBindingVerifierConfig(
            bindings_path=bindings,
            mode=JarvisRuntimeBindingVerificationMode.FACTORY_DRY_RUN,
            required_phases=(JarvisOrganKind.PHASE1_KERNEL,),
        )
    )

    report = verifier.verify()

    assert report.status == JarvisRuntimeBindingVerificationStatus.PASSED
    assert any(
        check.kind == JarvisRuntimeBindingCheckKind.FACTORY_DRY_RUN
        and check.passed
        for check in report.checks
    )


def test_runtime_binding_report_summary(tmp_path: Path) -> None:
    import_path = _write_module(tmp_path)
    bindings = tmp_path / "runtime_bindings.env"
    _write_bindings(path=bindings, import_path=import_path)

    verifier = JarvisRuntimeBindingVerifier(
        config=JarvisRuntimeBindingVerifierConfig(
            bindings_path=bindings,
            required_phases=(JarvisOrganKind.PHASE1_KERNEL,),
        )
    )

    report = verifier.verify()
    summary = summarize_binding_report(report)

    assert "status=passed" in summary
    assert "phase1_kernel" in summary