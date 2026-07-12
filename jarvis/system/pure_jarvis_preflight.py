from __future__ import annotations

import hashlib
import importlib
import json
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Protocol
from urllib import request as http_request
from urllib.error import URLError

from jarvis.system.pure_jarvis_contract import (
    PureJarvisRequirementStatus,
    default_pure_jarvis_manifest,
    pure_jarvis_requirement_justifications,
)
from jarvis.voice.contracts import utc_now


class PureJarvisPreflightStatus(StrEnum):
    PASSED = "passed"
    WARNING = "warning"
    FAILED = "failed"


class PureJarvisPreflightCheckKind(StrEnum):
    PURE_MANIFEST = "pure_manifest"
    REALTIME_EXPECTATION = "realtime_expectation"
    BINDINGS_FILE = "bindings_file"
    REAL_TTS_EXECUTABLE = "real_tts_executable"
    REAL_TTS_MODEL = "real_tts_model"
    REAL_TTS_CONFIG = "real_tts_config"
    OLLAMA_MODEL = "ollama_model"
    REQUIRED_IMPORTS = "required_imports"
    SOURCE_FINGERPRINT = "source_fingerprint"


@dataclass(frozen=True, slots=True)
class PureJarvisPreflightCheck:
    kind: PureJarvisPreflightCheckKind
    status: PureJarvisPreflightStatus
    message: str
    latency_ms: float
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.status == PureJarvisPreflightStatus.PASSED

    @property
    def failed(self) -> bool:
        return self.status == PureJarvisPreflightStatus.FAILED


@dataclass(frozen=True, slots=True)
class PureJarvisPreflightReport:
    status: PureJarvisPreflightStatus
    checks: tuple[PureJarvisPreflightCheck, ...]
    source_fingerprint: str
    started_at: datetime
    finished_at: datetime
    latency_ms: float
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.status == PureJarvisPreflightStatus.PASSED

    @property
    def failed_count(self) -> int:
        return sum(1 for check in self.checks if check.failed)

    @property
    def warning_count(self) -> int:
        return sum(
            1
            for check in self.checks
            if check.status == PureJarvisPreflightStatus.WARNING
        )

    @property
    def passed_count(self) -> int:
        return sum(1 for check in self.checks if check.passed)


@dataclass(frozen=True, slots=True)
class PureJarvisPreflightConfig:
    bindings_path: Path
    piper_executable_path: Path
    piper_model_path: Path
    piper_config_path: Path
    source_fingerprint_paths: tuple[Path, ...]
    ollama_model: str = "llama3.2:3b"
    ollama_base_url: str = "http://localhost:11434"
    require_ollama: bool = True
    required_imports: tuple[str, ...] = (
        "jarvis.voice.session_loop",
        "jarvis.voice.stt_runtime",
        "jarvis.voice.cognition_response",
        "jarvis.voice.awareness_cognition_bridge",
        "jarvis.voice.tts_runtime",
        "jarvis.voice.windows_audible_playback",
        "jarvis.live.response_boundary",
        "jarvis.live.session_runner",
    )
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.bindings_path.name:
            raise ValueError("bindings_path cannot be empty.")
        if not self.piper_executable_path.name:
            raise ValueError("piper_executable_path cannot be empty.")
        if not self.piper_model_path.name:
            raise ValueError("piper_model_path cannot be empty.")
        if not self.piper_config_path.name:
            raise ValueError("piper_config_path cannot be empty.")
        if not self.source_fingerprint_paths:
            raise ValueError("source_fingerprint_paths cannot be empty.")
        if not self.ollama_model.strip():
            raise ValueError("ollama_model cannot be empty.")
        if not self.ollama_base_url.strip():
            raise ValueError("ollama_base_url cannot be empty.")
        if not self.required_imports:
            raise ValueError("required_imports cannot be empty.")


class PureJarvisOllamaProbe(Protocol):
    def list_models(self) -> tuple[str, ...]:
        raise NotImplementedError


class UrllibOllamaProbe:
    def __init__(self, *, base_url: str, timeout_seconds: float = 2.0) -> None:
        self._base_url = base_url
        self._timeout_seconds = timeout_seconds

    def list_models(self) -> tuple[str, ...]:
        req = http_request.Request(
            url=f"{self._base_url.rstrip('/')}/api/tags",
            method="GET",
        )
        try:
            with http_request.urlopen(req, timeout=self._timeout_seconds) as response:
                raw = response.read().decode("utf-8")
        except URLError as exc:
            raise RuntimeError("Ollama is not reachable.") from exc

        data = json.loads(raw)
        models = data.get("models", ())
        names: list[str] = []
        if isinstance(models, list):
            for item in models:
                if isinstance(item, dict) and isinstance(item.get("name"), str):
                    names.append(item["name"])
        return tuple(names)


class PureJarvisPreflightRuntime:
    """
    Production launcher preflight for the connected Pure JARVIS path.

    This runtime does not generate user-facing speech. It verifies that the
    real launcher has the assets, imports, model backend, and source identity
    needed before entering the always-on voice loop.
    """

    def __init__(
        self,
        *,
        config: PureJarvisPreflightConfig,
        ollama_probe: PureJarvisOllamaProbe | None = None,
    ) -> None:
        self._config = config
        self._ollama_probe = ollama_probe or UrllibOllamaProbe(
            base_url=config.ollama_base_url
        )

    def run(self) -> PureJarvisPreflightReport:
        started_at = utc_now()
        started = time.perf_counter()
        checks = (
            self._pure_manifest_check(),
            self._realtime_expectation_check(),
            self._file_check(
                kind=PureJarvisPreflightCheckKind.BINDINGS_FILE,
                path=self._config.bindings_path,
                message="runtime bindings file is present",
            ),
            self._piper_executable_check(),
            self._file_check(
                kind=PureJarvisPreflightCheckKind.REAL_TTS_MODEL,
                path=self._config.piper_model_path,
                message="real Piper voice model is present",
            ),
            self._file_check(
                kind=PureJarvisPreflightCheckKind.REAL_TTS_CONFIG,
                path=self._config.piper_config_path,
                message="real Piper voice config is present",
            ),
            self._ollama_check(),
            self._required_imports_check(),
            self._source_fingerprint_check(),
        )
        fingerprint = _source_fingerprint(self._config.source_fingerprint_paths)
        status = _report_status(checks)
        return PureJarvisPreflightReport(
            status=status,
            checks=checks,
            source_fingerprint=fingerprint,
            started_at=started_at,
            finished_at=utc_now(),
            latency_ms=(time.perf_counter() - started) * 1000.0,
            metadata=self._config.metadata,
        )

    def _realtime_expectation_check(self) -> PureJarvisPreflightCheck:
        started = time.perf_counter()
        justifications = pure_jarvis_requirement_justifications()
        physics_limited = tuple(
            item
            for item in justifications
            if item.status == PureJarvisRequirementStatus.PHYSICS_LIMITED
        )
        speed_related = tuple(
            item
            for item in justifications
            if "millisecond" in item.request_line
            or "instantly" in item.request_line
            or item.status
            in {
                PureJarvisRequirementStatus.HARDWARE_LIMITED,
                PureJarvisRequirementStatus.PHYSICS_LIMITED,
            }
        )
        passed = bool(physics_limited) and all(
            item.remaining_work for item in speed_related if not item.fully_satisfied
        )
        return _check(
            kind=PureJarvisPreflightCheckKind.REALTIME_EXPECTATION,
            status=(
                PureJarvisPreflightStatus.PASSED
                if passed
                else PureJarvisPreflightStatus.FAILED
            ),
            message=(
                "Pure JARVIS speed contract is physics-honest"
                if passed
                else "Pure JARVIS speed contract hides an unsatisfied promise"
            ),
            started=started,
            metadata={
                "speed_related_requirements": tuple(
                    item.request_line for item in speed_related
                ),
                "physics_limited_requirements": tuple(
                    item.request_line for item in physics_limited
                ),
                "complete_answer_true_milliseconds_available": False,
                "millisecond_interruption_target_available": True,
            },
        )

    def _pure_manifest_check(self) -> PureJarvisPreflightCheck:
        started = time.perf_counter()
        manifest = default_pure_jarvis_manifest()
        missing = tuple(kind.value for kind in manifest.missing_required_capabilities())
        passed = manifest.ready_for_pure_runtime and not missing
        return _check(
            kind=PureJarvisPreflightCheckKind.PURE_MANIFEST,
            status=(
                PureJarvisPreflightStatus.PASSED
                if passed
                else PureJarvisPreflightStatus.FAILED
            ),
            message=(
                "Pure JARVIS capability manifest is ready"
                if passed
                else "Pure JARVIS capability manifest is incomplete"
            ),
            started=started,
            metadata={
                "capability_count": len(manifest.capabilities),
                "missing_required": missing,
            },
        )

    def _file_check(
        self,
        *,
        kind: PureJarvisPreflightCheckKind,
        path: Path,
        message: str,
    ) -> PureJarvisPreflightCheck:
        started = time.perf_counter()
        exists = path.exists()
        size = path.stat().st_size if exists and path.is_file() else 0
        passed = exists and size > 0
        return _check(
            kind=kind,
            status=(
                PureJarvisPreflightStatus.PASSED
                if passed
                else PureJarvisPreflightStatus.FAILED
            ),
            message=message if passed else f"missing or empty file: {path}",
            started=started,
            metadata={"path": str(path), "size_bytes": size},
        )

    def _piper_executable_check(self) -> PureJarvisPreflightCheck:
        started = time.perf_counter()
        executable = str(self._config.piper_executable_path)
        resolved = shutil.which(executable)
        path_exists = self._config.piper_executable_path.is_file()
        passed = path_exists or resolved is not None
        return _check(
            kind=PureJarvisPreflightCheckKind.REAL_TTS_EXECUTABLE,
            status=(
                PureJarvisPreflightStatus.PASSED
                if passed
                else PureJarvisPreflightStatus.FAILED
            ),
            message=(
                "real Piper executable is available"
                if passed
                else f"Piper executable is missing: {executable}"
            ),
            started=started,
            metadata={
                "path": executable,
                "path_exists": path_exists,
                "resolved": resolved,
            },
        )

    def _ollama_check(self) -> PureJarvisPreflightCheck:
        started = time.perf_counter()
        try:
            models = self._ollama_probe.list_models()
        except Exception as exc:
            return _check(
                kind=PureJarvisPreflightCheckKind.OLLAMA_MODEL,
                status=(
                    PureJarvisPreflightStatus.FAILED
                    if self._config.require_ollama
                    else PureJarvisPreflightStatus.WARNING
                ),
                message="Ollama model probe failed",
                started=started,
                metadata={"error": str(exc), "required": self._config.require_ollama},
            )

        has_model = self._config.ollama_model in models
        return _check(
            kind=PureJarvisPreflightCheckKind.OLLAMA_MODEL,
            status=(
                PureJarvisPreflightStatus.PASSED
                if has_model
                else PureJarvisPreflightStatus.FAILED
                if self._config.require_ollama
                else PureJarvisPreflightStatus.WARNING
            ),
            message=(
                "required Ollama model is available"
                if has_model
                else "required Ollama model is missing"
            ),
            started=started,
            metadata={
                "required_model": self._config.ollama_model,
                "available_models": models,
                "required": self._config.require_ollama,
            },
        )

    def _required_imports_check(self) -> PureJarvisPreflightCheck:
        started = time.perf_counter()
        missing: list[str] = []
        for module_name in self._config.required_imports:
            try:
                importlib.import_module(module_name)
            except Exception:
                missing.append(module_name)

        passed = not missing
        return _check(
            kind=PureJarvisPreflightCheckKind.REQUIRED_IMPORTS,
            status=(
                PureJarvisPreflightStatus.PASSED
                if passed
                else PureJarvisPreflightStatus.FAILED
            ),
            message=(
                "required Pure JARVIS runtime imports are available"
                if passed
                else "required Pure JARVIS runtime imports are missing"
            ),
            started=started,
            metadata={"missing_imports": tuple(missing)},
        )

    def _source_fingerprint_check(self) -> PureJarvisPreflightCheck:
        started = time.perf_counter()
        missing = tuple(
            str(path)
            for path in self._config.source_fingerprint_paths
            if not path.is_file()
        )
        fingerprint = _source_fingerprint(self._config.source_fingerprint_paths)
        passed = not missing and fingerprint != "unavailable"
        return _check(
            kind=PureJarvisPreflightCheckKind.SOURCE_FINGERPRINT,
            status=(
                PureJarvisPreflightStatus.PASSED
                if passed
                else PureJarvisPreflightStatus.FAILED
            ),
            message=(
                "launcher source fingerprint generated"
                if passed
                else "launcher source fingerprint could not be generated"
            ),
            started=started,
            metadata={"fingerprint": fingerprint, "missing_paths": missing},
        )


def summarize_pure_jarvis_preflight(report: PureJarvisPreflightReport) -> str:
    lines = [
        f"status={report.status.value}",
        f"fingerprint={report.source_fingerprint[:16]}",
        f"passed={report.passed_count}",
        f"warnings={report.warning_count}",
        f"failed={report.failed_count}",
    ]
    for check in report.checks:
        lines.append(f"{check.status.value}: {check.kind.value}: {check.message}")
    return "\n".join(lines)


def _source_fingerprint(paths: tuple[Path, ...]) -> str:
    digest = hashlib.sha256()
    found = False
    for path in sorted(paths, key=lambda item: str(item)):
        if not path.is_file():
            continue
        found = True
        digest.update(str(path).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    if not found:
        return "unavailable"
    return digest.hexdigest()


def _report_status(
    checks: tuple[PureJarvisPreflightCheck, ...],
) -> PureJarvisPreflightStatus:
    if any(check.status == PureJarvisPreflightStatus.FAILED for check in checks):
        return PureJarvisPreflightStatus.FAILED
    if any(check.status == PureJarvisPreflightStatus.WARNING for check in checks):
        return PureJarvisPreflightStatus.WARNING
    return PureJarvisPreflightStatus.PASSED


def _check(
    *,
    kind: PureJarvisPreflightCheckKind,
    status: PureJarvisPreflightStatus,
    message: str,
    started: float,
    metadata: dict[str, object] | None = None,
) -> PureJarvisPreflightCheck:
    return PureJarvisPreflightCheck(
        kind=kind,
        status=status,
        message=message,
        latency_ms=(time.perf_counter() - started) * 1000.0,
        created_at=utc_now(),
        metadata=metadata or {},
    )
