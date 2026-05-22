from __future__ import annotations

import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Phase2CommandResult:
    """
    Result for one Phase 2 completion command.
    """

    name: str
    command: tuple[str, ...]
    passed: bool
    return_code: int
    duration_ms: float
    stdout: str
    stderr: str

    @property
    def command_text(self) -> str:
        return " ".join(self.command)


@dataclass(frozen=True, slots=True)
class Phase2CompletionReport:
    """
    Final Phase 2 completion report.
    """

    passed: bool
    started_at: datetime
    finished_at: datetime
    duration_ms: float
    results: tuple[Phase2CommandResult, ...]
    known_good_voice_command: str

    @property
    def total_count(self) -> int:
        return len(self.results)

    @property
    def passed_count(self) -> int:
        return sum(1 for result in self.results if result.passed)

    @property
    def failed_count(self) -> int:
        return sum(1 for result in self.results if not result.passed)

    @property
    def failed_results(self) -> tuple[Phase2CommandResult, ...]:
        return tuple(result for result in self.results if not result.passed)


@dataclass(frozen=True, slots=True)
class Phase2CompletionConfig:
    """
    Configuration for the Phase 2 completion gate.
    """

    project_root: Path
    timeout_seconds: float = 180.0
    include_pytest: bool = True
    include_validation: bool = True
    include_latency_profile: bool = True

    def validate(self) -> None:
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero.")

        if not self.project_root.exists():
            raise ValueError(f"project_root does not exist: {self.project_root}")

        if not self.project_root.is_dir():
            raise ValueError(f"project_root is not a directory: {self.project_root}")


class Phase2CompletionGate:
    """
    Final completion gate for Phase 2 Presence Runtime.

    This proves the phase can pass the same checks a senior engineer would run
    before declaring the layer stable.
    """

    known_good_voice_command = (
        "python .\\scripts\\run_jarvis_voice_smoke.py --preset fast"
    )

    def __init__(self, config: Phase2CompletionConfig) -> None:
        config.validate()
        self._config = config

    def run(self) -> Phase2CompletionReport:
        started_at = datetime.now(UTC)
        started_perf = time.perf_counter()
        results: list[Phase2CommandResult] = []

        for name, command in self._commands():
            results.append(self._run_command(name=name, command=command))

        finished_at = datetime.now(UTC)
        duration_ms = (time.perf_counter() - started_perf) * 1000.0

        return Phase2CompletionReport(
            passed=all(result.passed for result in results),
            started_at=started_at,
            finished_at=finished_at,
            duration_ms=duration_ms,
            results=tuple(results),
            known_good_voice_command=self.known_good_voice_command,
        )

    def _commands(self) -> tuple[tuple[str, tuple[str, ...]], ...]:
        commands: list[tuple[str, tuple[str, ...]]] = [
            ("ruff_fix_check", ("ruff", "check", ".", "--fix")),
            ("ruff_check", ("ruff", "check", ".")),
            ("mypy", ("mypy", ".")),
        ]

        if self._config.include_pytest:
            commands.append(("pytest", ("pytest",)))

        if self._config.include_validation:
            commands.append(
                (
                    "presence_validation",
                    (
                        sys.executable,
                        ".\\scripts\\validate_presence.py",
                    ),
                )
            )

        if self._config.include_latency_profile:
            commands.append(
                (
                    "presence_latency_profile",
                    (
                        sys.executable,
                        ".\\scripts\\profile_presence_latency.py",
                    ),
                )
            )

        return tuple(commands)

    def _run_command(
        self,
        *,
        name: str,
        command: tuple[str, ...],
    ) -> Phase2CommandResult:
        started = time.perf_counter()

        try:
            completed = subprocess.run(
                command,
                cwd=self._config.project_root,
                capture_output=True,
                text=True,
                timeout=self._config.timeout_seconds,
                check=False,
            )
            return_code = completed.returncode
            stdout = completed.stdout
            stderr = completed.stderr

        except subprocess.TimeoutExpired as exc:
            return_code = 124
            stdout = exc.stdout if isinstance(exc.stdout, str) else ""
            stderr = exc.stderr if isinstance(exc.stderr, str) else ""
            stderr = f"{stderr}\nTimed out after {self._config.timeout_seconds}s."

        duration_ms = (time.perf_counter() - started) * 1000.0

        return Phase2CommandResult(
            name=name,
            command=command,
            passed=return_code == 0,
            return_code=return_code,
            duration_ms=duration_ms,
            stdout=stdout,
            stderr=stderr,
        )


def default_phase2_completion_config() -> Phase2CompletionConfig:
    """
    Build the default config from repository root.
    """

    return Phase2CompletionConfig(
        project_root=Path(__file__).resolve().parents[2],
    )


def format_phase2_completion_report(report: Phase2CompletionReport) -> str:
    """
    Render a terminal-friendly completion report.
    """

    lines = [
        "",
        "JARVIS Phase 2 Completion Gate",
        "------------------------------",
        f"Passed: {report.passed}",
        f"Duration: {report.duration_ms:.2f} ms",
        f"Commands: {report.passed_count}/{report.total_count} passed",
        "",
        "Command Results:",
    ]

    for result in report.results:
        status = "PASS" if result.passed else "FAIL"
        lines.append(
            f" [{status}] {result.name}: "
            f"{result.duration_ms:.2f} ms "
            f"(return code {result.return_code})"
        )

    lines.extend(
        [
            "",
            "Known-good live voice command:",
            f" {report.known_good_voice_command}",
        ]
    )

    if report.failed_results:
        lines.append("")
        lines.append("Failures:")

        for result in report.failed_results:
            lines.append("")
            lines.append(f"## {result.name}")
            lines.append(f"Command: {result.command_text}")

            if result.stdout.strip():
                lines.append("stdout:")
                lines.append(result.stdout.strip())

            if result.stderr.strip():
                lines.append("stderr:")
                lines.append(result.stderr.strip())

    if report.passed:
        lines.extend(
            [
                "",
                "PHASE 2 STATUS: COMPLETE",
                (
                    "Presence runtime is validated and ready for "
                    "Phase 3 cognition integration."
                ),
            ]
        )

    return "\n".join(lines)