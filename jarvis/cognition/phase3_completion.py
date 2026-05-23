from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import RLock

from jarvis.cognition.local_llm_adapter import LocalLLMAdapter
from jarvis.cognition.ollama_backend import OllamaBackendConfig
from jarvis.cognition.phase3_validation import (
    ValidationLocalLLMBackend,
    validate_phase3_cognition,
)
from jarvis.cognition.runtime import CognitionRuntime, create_cognition_runtime
from jarvis.cognition.voice_cognition_smoke import (
    VoiceCognitionPlaybackResult,
    VoiceCognitionSmokeConfig,
    VoiceCognitionSmokeRunner,
    VoiceCognitionTranscript,
)
from jarvis.runtime.observability.structured_logger import get_logger


@dataclass(frozen=True, slots=True)
class Phase3CompletionCheck:
    """
    One Phase 3 completion check.
    """

    name: str
    passed: bool
    detail: str


@dataclass(frozen=True, slots=True)
class Phase3CompletionReport:
    """
    Final Phase 3 completion report.
    """

    passed: bool
    checks: tuple[Phase3CompletionCheck, ...]

    @property
    def total_count(self) -> int:
        return len(self.checks)

    @property
    def passed_count(self) -> int:
        return sum(1 for check in self.checks if check.passed)

    @property
    def failed_count(self) -> int:
        return sum(1 for check in self.checks if not check.passed)


class Phase3CompletionVoiceIO:
    """
    Deterministic voice I/O used by the completion gate.

    It proves the voice+cognition contract path without requiring the user to
    speak during automated validation.
    """

    def __init__(
        self,
        *,
        transcript: str,
    ) -> None:
        self._transcript = transcript
        self.spoken_text: str | None = None

    @property
    def name(self) -> str:
        return "phase3_completion_voice_io"

    def listen_once(self) -> VoiceCognitionTranscript:
        return VoiceCognitionTranscript(
            text=self._transcript,
            confidence=1.0,
            source="phase3_completion_text_fallback",
        )

    def speak(self, text: str) -> VoiceCognitionPlaybackResult:
        self.spoken_text = text

        return VoiceCognitionPlaybackResult(
            text=text,
            started=True,
            completed=True,
            metadata={
                "completion_gate": True,
            },
        )


class Phase3CompletionGate:
    """
    Final Phase 3 completion gate.

    Responsibilities:
    - run deterministic cognition integration validation
    - verify runtime assembly can process text
    - verify streaming path works
    - verify voice+cognition smoke path works
    - verify tool planning remains plan-only
    - verify real local LLM smoke scripts exist
    - verify Ollama backend config is constructible

    Non-responsibilities:
    - no real microphone requirement
    - no real Ollama server requirement
    - no laptop action execution
    - no shell/file/browser control
    """

    def __init__(
        self,
        *,
        project_root: Path | None = None,
    ) -> None:
        self._project_root = project_root or Path.cwd()
        self._lock = RLock()
        self._logger = get_logger("cognition.phase3_completion")
        self._last_report: Phase3CompletionReport | None = None

    def complete(self) -> Phase3CompletionReport:
        checks: list[Phase3CompletionCheck] = []

        checks.append(self._check_phase3_validation())
        checks.append(self._check_runtime_text_turn())
        checks.append(self._check_runtime_streaming_turn())
        checks.append(self._check_voice_cognition_smoke())
        checks.append(self._check_tool_planning_is_plan_only())
        checks.append(self._check_ollama_backend_boundary())
        checks.append(self._check_real_local_llm_script_exists())
        checks.append(self._check_real_voice_cognition_script_exists())

        report = Phase3CompletionReport(
            passed=all(check.passed for check in checks),
            checks=tuple(checks),
        )

        with self._lock:
            self._last_report = report

        self._logger.info(
            "phase3_completion_gate_completed",
            passed=report.passed,
            passed_count=report.passed_count,
            failed_count=report.failed_count,
            total_count=report.total_count,
        )

        return report

    def last_report(self) -> Phase3CompletionReport | None:
        with self._lock:
            return self._last_report

    def _check_phase3_validation(self) -> Phase3CompletionCheck:
        report = validate_phase3_cognition()

        return self._check(
            name="phase3_cognition_validation_passed",
            passed=report.passed and report.failed_count == 0,
            detail=(
                f"phase3 validation checks passed "
                f"{report.passed_count}/{report.total_count}"
            ),
        )

    def _check_runtime_text_turn(self) -> Phase3CompletionCheck:
        runtime = self._runtime()
        result = runtime.process_text(
            "Confirm the Phase 3 cognition runtime is assembled.",
            request_id="phase3-completion-text",
        )

        return self._check(
            name="cognition_runtime_text_turn_passed",
            passed=result.succeeded
            and result.response is not None
            and runtime.snapshot().success_count == 1,
            detail="assembled runtime completed one generated text turn",
        )

    def _check_runtime_streaming_turn(self) -> Phase3CompletionCheck:
        runtime = self._runtime()
        result = runtime.process_text_streaming(
            "Stream the Phase 3 cognition runtime status.",
            request_id="phase3-completion-streaming",
        )

        return self._check(
            name="cognition_runtime_streaming_turn_passed",
            passed=result.succeeded
            and result.streamed
            and result.streaming_result is not None
            and result.streaming_result.completed,
            detail="assembled runtime completed one streaming cognition turn",
        )

    def _check_voice_cognition_smoke(self) -> Phase3CompletionCheck:
        runtime = self._runtime()
        voice_io = Phase3CompletionVoiceIO(
            transcript="Hello Jarvis, confirm Phase 3 cognition is complete.",
        )
        runner = VoiceCognitionSmokeRunner(
            runtime=runtime,
            voice_io=voice_io,
            config=VoiceCognitionSmokeConfig(
                streaming=False,
                allow_tools=False,
                allow_memory_lookup=True,
            ),
        )

        report = runner.run_once()

        return self._check(
            name="voice_cognition_smoke_path_passed",
            passed=report.passed
            and report.response_text is not None
            and voice_io.spoken_text == report.response_text,
            detail="voice transcript to cognition to voice playback path passed",
        )

    def _check_tool_planning_is_plan_only(self) -> Phase3CompletionCheck:
        runtime = self._runtime()
        result = runtime.process_text(
            "open diagnostics",
            request_id="phase3-completion-action-plan",
            allow_tools=True,
        )

        action_plan = result.action_plan

        return self._check(
            name="tool_planning_is_plan_only",
            passed=result.succeeded
            and action_plan is not None
            and action_plan.metadata["llm_direct_execution_allowed"] is False
            and action_plan.executable is True,
            detail="action intent creates safe plan without direct execution",
        )

    def _check_ollama_backend_boundary(self) -> Phase3CompletionCheck:
        config = OllamaBackendConfig(
            model="llama3.2:3b",
            num_predict=64,
        )
        config.validate()

        return self._check(
            name="ollama_backend_boundary_available",
            passed=config.model == "llama3.2:3b",
            detail="real local LLM backend config is constructible",
        )

    def _check_real_local_llm_script_exists(self) -> Phase3CompletionCheck:
        path = self._project_root / "scripts" / "smoke_real_local_llm.py"

        return self._check(
            name="real_local_llm_smoke_script_exists",
            passed=path.exists(),
            detail="real local LLM smoke script is present",
        )

    def _check_real_voice_cognition_script_exists(self) -> Phase3CompletionCheck:
        path = self._project_root / "scripts" / "smoke_real_voice_cognition.py"

        return self._check(
            name="real_voice_cognition_smoke_script_exists",
            passed=path.exists(),
            detail="real voice+cognition smoke script is present",
        )

    @staticmethod
    def _runtime() -> CognitionRuntime:
        backend = ValidationLocalLLMBackend()
        adapter = LocalLLMAdapter(backend=backend)

        return create_cognition_runtime(adapter=adapter)

    @staticmethod
    def _check(
        *,
        name: str,
        passed: bool,
        detail: str,
    ) -> Phase3CompletionCheck:
        return Phase3CompletionCheck(
            name=name,
            passed=passed,
            detail=detail,
        )


def complete_phase3_cognition(
    *,
    project_root: Path | None = None,
) -> Phase3CompletionReport:
    """
    Run the final Phase 3 completion gate.
    """

    return Phase3CompletionGate(project_root=project_root).complete()