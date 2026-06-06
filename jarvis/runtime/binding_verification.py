from __future__ import annotations

import importlib
import inspect
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path

from jarvis.runtime.phase_adapters import (
    default_phase_runtime_specs,
    read_runtime_binding_imports,
)
from jarvis.runtime.start_control import JarvisOrganKind, utc_now


class JarvisRuntimeBindingVerificationStatus(StrEnum):
    PASSED = "passed"
    FAILED = "failed"


class JarvisRuntimeBindingCheckKind(StrEnum):
    BINDING_FILE_EXISTS = "binding_file_exists"
    REQUIRED_PHASE_PRESENT = "required_phase_present"
    IMPORT_PATH_FORMAT = "import_path_format"
    MODULE_IMPORTABLE = "module_importable"
    FACTORY_CALLABLE = "factory_callable"
    FACTORY_SIGNATURE_SAFE = "factory_signature_safe"
    FACTORY_DRY_RUN = "factory_dry_run"


class JarvisRuntimeBindingVerificationMode(StrEnum):
    RESOLVE_ONLY = "resolve_only"
    FACTORY_DRY_RUN = "factory_dry_run"


@dataclass(frozen=True, slots=True)
class JarvisRuntimeBindingVerifierConfig:
    bindings_path: Path
    mode: JarvisRuntimeBindingVerificationMode = (
        JarvisRuntimeBindingVerificationMode.RESOLVE_ONLY
    )
    required_phases: tuple[JarvisOrganKind, ...] = field(
        default_factory=default_phase_runtime_specs
    )
    allow_factory_parameters: bool = False
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.required_phases:
            raise ValueError("required_phases cannot be empty.")


@dataclass(frozen=True, slots=True)
class JarvisRuntimeBindingCheck:
    kind: JarvisRuntimeBindingCheckKind
    phase: JarvisOrganKind | None
    status: JarvisRuntimeBindingVerificationStatus
    message: str
    import_path: str | None
    latency_ms: float
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.status == JarvisRuntimeBindingVerificationStatus.PASSED


@dataclass(frozen=True, slots=True)
class JarvisRuntimeBindingVerificationReport:
    status: JarvisRuntimeBindingVerificationStatus
    checks: tuple[JarvisRuntimeBindingCheck, ...]
    bindings_path: Path
    mode: JarvisRuntimeBindingVerificationMode
    started_at: datetime
    finished_at: datetime
    latency_ms: float
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.status == JarvisRuntimeBindingVerificationStatus.PASSED

    @property
    def failed_count(self) -> int:
        return sum(1 for check in self.checks if not check.passed)

    @property
    def passed_count(self) -> int:
        return sum(1 for check in self.checks if check.passed)


class JarvisRuntimeBindingVerifier:
    """
    51N.0 binding verifier.

    Verifies that runtime_bindings.env points to real factories for
    Phase 1-9 organs. It does not start the full JARVIS runtime and does
    not generate conversational output.
    """

    def __init__(
        self,
        *,
        config: JarvisRuntimeBindingVerifierConfig,
    ) -> None:
        self._config = config

    def verify(self) -> JarvisRuntimeBindingVerificationReport:
        started_at = utc_now()
        started_perf = time.perf_counter()
        checks: list[JarvisRuntimeBindingCheck] = []

        file_check = self._check_file_exists()
        checks.append(file_check)

        if not file_check.passed:
            return self._report(
                checks=tuple(checks),
                started_at=started_at,
                started_perf=started_perf,
            )

        try:
            bindings = read_runtime_binding_imports(self._config.bindings_path)
        except Exception as exc:
            checks.append(
                _check(
                    kind=JarvisRuntimeBindingCheckKind.IMPORT_PATH_FORMAT,
                    phase=None,
                    status=JarvisRuntimeBindingVerificationStatus.FAILED,
                    message="binding file parse failed",
                    import_path=None,
                    started=time.perf_counter(),
                    metadata={"error": str(exc)},
                )
            )
            return self._report(
                checks=tuple(checks),
                started_at=started_at,
                started_perf=started_perf,
            )

        for phase in self._config.required_phases:
            import_path = bindings.get(phase)

            checks.append(
                self._check_required_phase_present(
                    phase=phase,
                    import_path=import_path,
                )
            )

            if import_path is None:
                continue

            format_check = self._check_import_path_format(
                phase=phase,
                import_path=import_path,
            )
            checks.append(format_check)

            if not format_check.passed:
                continue

            resolved = self._resolve_factory(
                phase=phase,
                import_path=import_path,
            )
            checks.extend(resolved.checks)

            if resolved.factory is None:
                continue

            checks.append(
                self._check_factory_signature(
                    phase=phase,
                    import_path=import_path,
                    factory=resolved.factory,
                )
            )

            if (
                self._config.mode
                == JarvisRuntimeBindingVerificationMode.FACTORY_DRY_RUN
            ):
                checks.append(
                    self._check_factory_dry_run(
                        phase=phase,
                        import_path=import_path,
                        factory=resolved.factory,
                    )
                )

        return self._report(
            checks=tuple(checks),
            started_at=started_at,
            started_perf=started_perf,
        )

    def _check_file_exists(self) -> JarvisRuntimeBindingCheck:
        started = time.perf_counter()
        exists = self._config.bindings_path.exists()
        return _check(
            kind=JarvisRuntimeBindingCheckKind.BINDING_FILE_EXISTS,
            phase=None,
            status=(
                JarvisRuntimeBindingVerificationStatus.PASSED
                if exists
                else JarvisRuntimeBindingVerificationStatus.FAILED
            ),
            message=(
                "runtime binding file exists"
                if exists
                else "runtime binding file missing"
            ),
            import_path=None,
            started=started,
            metadata={"path": str(self._config.bindings_path)},
        )

    def _check_required_phase_present(
        self,
        *,
        phase: JarvisOrganKind,
        import_path: str | None,
    ) -> JarvisRuntimeBindingCheck:
        started = time.perf_counter()
        return _check(
            kind=JarvisRuntimeBindingCheckKind.REQUIRED_PHASE_PRESENT,
            phase=phase,
            status=(
                JarvisRuntimeBindingVerificationStatus.PASSED
                if import_path
                else JarvisRuntimeBindingVerificationStatus.FAILED
            ),
            message=(
                "required phase binding present"
                if import_path
                else "required phase binding missing"
            ),
            import_path=import_path,
            started=started,
        )

    def _check_import_path_format(
        self,
        *,
        phase: JarvisOrganKind,
        import_path: str,
    ) -> JarvisRuntimeBindingCheck:
        started = time.perf_counter()
        valid = _parse_import_path(import_path) is not None
        return _check(
            kind=JarvisRuntimeBindingCheckKind.IMPORT_PATH_FORMAT,
            phase=phase,
            status=(
                JarvisRuntimeBindingVerificationStatus.PASSED
                if valid
                else JarvisRuntimeBindingVerificationStatus.FAILED
            ),
            message=(
                "runtime factory import path format valid"
                if valid
                else "runtime factory import path must be module:factory"
            ),
            import_path=import_path,
            started=started,
        )

    def _resolve_factory(
        self,
        *,
        phase: JarvisOrganKind,
        import_path: str,
    ) -> _ResolvedFactory:
        checks: list[JarvisRuntimeBindingCheck] = []
        parsed = _parse_import_path(import_path)

        if parsed is None:
            return _ResolvedFactory(checks=(), factory=None)

        module_name, factory_name = parsed

        module_started = time.perf_counter()
        try:
            module = importlib.import_module(module_name)
            checks.append(
                _check(
                    kind=JarvisRuntimeBindingCheckKind.MODULE_IMPORTABLE,
                    phase=phase,
                    status=JarvisRuntimeBindingVerificationStatus.PASSED,
                    message="runtime factory module importable",
                    import_path=import_path,
                    started=module_started,
                    metadata={"module": module_name},
                )
            )
        except Exception as exc:
            checks.append(
                _check(
                    kind=JarvisRuntimeBindingCheckKind.MODULE_IMPORTABLE,
                    phase=phase,
                    status=JarvisRuntimeBindingVerificationStatus.FAILED,
                    message="runtime factory module import failed",
                    import_path=import_path,
                    started=module_started,
                    metadata={"module": module_name, "error": str(exc)},
                )
            )
            return _ResolvedFactory(checks=tuple(checks), factory=None)

        callable_started = time.perf_counter()
        factory = getattr(module, factory_name, None)
        is_callable = callable(factory)

        checks.append(
            _check(
                kind=JarvisRuntimeBindingCheckKind.FACTORY_CALLABLE,
                phase=phase,
                status=(
                    JarvisRuntimeBindingVerificationStatus.PASSED
                    if is_callable
                    else JarvisRuntimeBindingVerificationStatus.FAILED
                ),
                message=(
                    "runtime factory callable"
                    if is_callable
                    else "runtime factory missing or not callable"
                ),
                import_path=import_path,
                started=callable_started,
                metadata={"factory": factory_name},
            )
        )

        if not is_callable:
            return _ResolvedFactory(checks=tuple(checks), factory=None)

        return _ResolvedFactory(
            checks=tuple(checks),
            factory=factory,
        )

    def _check_factory_signature(
        self,
        *,
        phase: JarvisOrganKind,
        import_path: str,
        factory: Callable[..., object],
    ) -> JarvisRuntimeBindingCheck:
        started = time.perf_counter()

        try:
            signature = inspect.signature(factory)
            required_parameters = tuple(
                parameter.name
                for parameter in signature.parameters.values()
                if parameter.default is inspect.Parameter.empty
                and parameter.kind
                in {
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    inspect.Parameter.KEYWORD_ONLY,
                }
            )
            safe = (
                self._config.allow_factory_parameters
                or not required_parameters
            )
            metadata: dict[str, object] = {
                "required_parameters": required_parameters,
                "signature": str(signature),
            }
        except Exception as exc:
            safe = False
            metadata = {"error": str(exc)}

        return _check(
            kind=JarvisRuntimeBindingCheckKind.FACTORY_SIGNATURE_SAFE,
            phase=phase,
            status=(
                JarvisRuntimeBindingVerificationStatus.PASSED
                if safe
                else JarvisRuntimeBindingVerificationStatus.FAILED
            ),
            message=(
                "runtime factory signature safe"
                if safe
                else "runtime factory requires unsupported parameters"
            ),
            import_path=import_path,
            started=started,
            metadata=metadata,
        )

    def _check_factory_dry_run(
        self,
        *,
        phase: JarvisOrganKind,
        import_path: str,
        factory: Callable[..., object],
    ) -> JarvisRuntimeBindingCheck:
        started = time.perf_counter()

        try:
            runtime = factory()
            ok = runtime is not None
            metadata: dict[str, object] = {
                "runtime_type": type(runtime).__name__
            }
        except Exception as exc:
            ok = False
            metadata = {"error": str(exc)}

        return _check(
            kind=JarvisRuntimeBindingCheckKind.FACTORY_DRY_RUN,
            phase=phase,
            status=(
                JarvisRuntimeBindingVerificationStatus.PASSED
                if ok
                else JarvisRuntimeBindingVerificationStatus.FAILED
            ),
            message=(
                "runtime factory dry run created runtime object"
                if ok
                else "runtime factory dry run failed"
            ),
            import_path=import_path,
            started=started,
            metadata=metadata,
        )

    def _report(
        self,
        *,
        checks: tuple[JarvisRuntimeBindingCheck, ...],
        started_at: datetime,
        started_perf: float,
    ) -> JarvisRuntimeBindingVerificationReport:
        failed = any(not check.passed for check in checks)
        return JarvisRuntimeBindingVerificationReport(
            status=(
                JarvisRuntimeBindingVerificationStatus.FAILED
                if failed
                else JarvisRuntimeBindingVerificationStatus.PASSED
            ),
            checks=checks,
            bindings_path=self._config.bindings_path,
            mode=self._config.mode,
            started_at=started_at,
            finished_at=utc_now(),
            latency_ms=(time.perf_counter() - started_perf) * 1000.0,
            metadata=self._config.metadata,
        )


@dataclass(frozen=True, slots=True)
class _ResolvedFactory:
    checks: tuple[JarvisRuntimeBindingCheck, ...]
    factory: Callable[..., object] | None


def _parse_import_path(import_path: str) -> tuple[str, str] | None:
    if ":" not in import_path:
        return None

    module_name, factory_name = import_path.split(":", 1)
    module_name = module_name.strip()
    factory_name = factory_name.strip()

    if not module_name or not factory_name:
        return None

    return module_name, factory_name


def _check(
    *,
    kind: JarvisRuntimeBindingCheckKind,
    phase: JarvisOrganKind | None,
    status: JarvisRuntimeBindingVerificationStatus,
    message: str,
    import_path: str | None,
    started: float,
    metadata: dict[str, object] | None = None,
) -> JarvisRuntimeBindingCheck:
    return JarvisRuntimeBindingCheck(
        kind=kind,
        phase=phase,
        status=status,
        message=message,
        import_path=import_path,
        latency_ms=(time.perf_counter() - started) * 1000.0,
        created_at=utc_now(),
        metadata=metadata or {},
    )


def summarize_binding_report(
    report: JarvisRuntimeBindingVerificationReport,
) -> str:
    lines = [
        f"status={report.status.value}",
        f"passed={report.passed_count}",
        f"failed={report.failed_count}",
        f"mode={report.mode.value}",
        f"path={report.bindings_path}",
    ]

    for check in report.checks:
        phase = check.phase.value if check.phase is not None else "global"
        lines.append(
            f"{check.status.value}: {phase}: "
            f"{check.kind.value}: {check.message}"
        )

    return "\n".join(lines)